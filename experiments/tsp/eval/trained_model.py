import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tyro
import wandb

from experiments.tsp.train.reinforce_multibinary import Policy
from problems.tsp.tsp_env_multibinary import TSPEnvironmentMultiBinary


@dataclass
class Args:
    instance_name: Optional[str] = None
    """the instance on which the method will be run"""
    instances_dir_name: str = "generated/train"
    """the name of instances directory"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    wandb_project_name: str = "engineering-thesis"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    debug: bool = True
    """if toggled, extra logs will be printed to the console"""
    log_every_n_step: int = 10
    """the logging interval"""
    model_name: Optional[str] = None
    """the name of the model"""
    model_tag: str = "latest"
    """the tag of the model"""

    # Environment specific arguments
    max_t: Optional[int] = None
    """the maximum number of steps during one episode"""
    solver: str = "gecode"
    """the solver to use to find an initial solution and repair the subsequent"""

    # Algorithm specific arguments
    k: int = 4


if __name__ == '__main__':
    args = tyro.cli(Args)

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", args.instances_dir_name)

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init_circuit.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair_circuit.mzn")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.debug:
        print(f"Device: {device}")

    if args.instance_name:
        problem_instances_paths = [os.path.join(TSP_DATA_DIR, f"{args.instance_name}.json")]
    else:
        problem_instances_paths = [os.path.join(TSP_DATA_DIR, path) for path in os.listdir(TSP_DATA_DIR) if path.endswith(".json")]

    # ==== Loading the model ====
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"model__k_{args.k}"

    wandb_api = wandb.Api()
    artifact = wandb_api.artifact(f"{args.wandb_project_name}/{model_name}:{args.model_tag}")
    artifact_dir = artifact.download()

    model_path = os.path.join(artifact_dir, "model.pt")
    model_hparams_path = os.path.join(artifact_dir, "hparams.json")

    with open(model_hparams_path, "r") as f:
        model_hparams = json.load(f)

    model = Policy(
        features_extractor_kwargs=model_hparams,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()


    # ==== Loop over problem instances ====
    for instance_idx, instance_path in enumerate(problem_instances_paths):
        results = []

        # ==== Seeding ====
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        instance_name = os.path.basename(instance_path).rstrip(".json")

        if args.debug:
            print(f"Instance {instance_idx}: {instance_name}")

        # ==== Environment Creation ====
        env = TSPEnvironmentMultiBinary(
            problem_instance_path=instance_path,
            init_model_path=TSP_INIT_SOLVER_PATH,
            repair_model_path=TSP_REPAIR_SOLVER_PATH,
            solver_name=args.solver,
            max_episode_length=args.max_t,
        )

        k = args.k

        # ==== Main Loop ====
        observation, info = env.reset()

        initial_objective_value = info["best_objective_value"]
        if args.debug:
           print(f"Initial solution objective value: {initial_objective_value}")

        step = 0
        episode_time = 0.0
        while True:
            step_start_time = time.perf_counter()

            graph_data = env.preprocess(observation).to(device)

            action, log_prob, entropy = model.get_action(graph_data, k=k)
            action = action.cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)

            episode_time += time.perf_counter() - step_start_time

            # ==== Logging ====
            if args.debug:
                print(f"Step {step}")
                print(f"Reward: {reward}, Step objective value: {info['step_objective_value']}, k: {k}")

            step += 1

            # ==== Save to the best results ====
            if step % args.log_every_n_step == 0 or step < 10:
                method_name = f"model({args.k}):{args.model_tag}"

                new_record = {
                    "instance": instance_name,
                    "subset": "test" if "test" in args.instances_dir_name else "train",
                    "method": method_name,
                    "seed": args.seed,
                    "steps": step,
                    "initial_objective_value": initial_objective_value,
                    "objective_value": info["best_objective_value"],
                    "time": f"{episode_time:.3f}",
                    "avg_time_per_step": f"{(episode_time / step):.3f}"
                }

                DB_PATH = os.path.join(BASE_PATH, "experiments", "results.db")
                conn = sqlite3.connect(DB_PATH)

                # Check if the record already exists
                cursor = conn.cursor()
                cursor.execute(
                    """
                        SELECT * FROM best_results
                        WHERE instance = ? AND subset = ? AND method = ? AND seed = ? AND steps = ?
                    """, (
                    new_record["instance"], new_record["subset"], new_record["method"], new_record["seed"],
                    new_record["steps"])
                )

                existing_record = cursor.fetchone()

                if existing_record:
                    # If the record exists, update it
                    cursor.execute(
                        """
                            UPDATE best_results
                            SET objective_value = ?, time = ?, avg_time_per_step = ?
                            WHERE instance = ? AND subset = ? AND method = ? AND seed = ? AND steps = ?
                        """, (
                            new_record["objective_value"],
                            new_record["time"],
                            new_record["avg_time_per_step"],
                            new_record["instance"],
                            new_record["subset"],
                            new_record["method"],
                            new_record["seed"],
                            new_record["steps"]
                        )
                    )
                else:
                    # If the record doesn't exist, insert a new record
                    cursor.execute(
                        """
                            INSERT INTO best_results (instance, subset, method, seed, steps, initial_objective_value, objective_value, time, avg_time_per_step)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            new_record["instance"],
                            new_record["subset"],
                            new_record["method"],
                            new_record["seed"],
                            new_record["steps"],
                            new_record["initial_objective_value"],
                            new_record["objective_value"],
                            new_record["time"],
                            new_record["avg_time_per_step"]
                        )
                    )

                # Commit the changes and close the connection
                conn.commit()
                conn.close()

                if args.debug:
                    print(f"Solution: {env.lns.best_solution.next}")
                    print(f"Objective value: {info['best_objective_value']}")
                    print(f"Measured time: {episode_time:.3f} s")
                    print(f"Average time per one step: {(episode_time / step):.3f} s")

            if terminated or truncated:
                break

    wandb.finish()
