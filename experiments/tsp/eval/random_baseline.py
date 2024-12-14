import os
import time
from collections import deque
from dataclasses import dataclass
import random
from typing import Callable, Tuple, Optional

import gymnasium
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Bernoulli

import torch_geometric as pyg

import tyro
import wandb

from general.ml.features_extractor import GraphFeaturesExtractor
from problems.tsp.tsp_env_multibinary import TSPEnvironmentMultiBinary

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "engineering-thesis"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    debug: bool = True
    """if toggled, extra logs will be printed to the console"""

    # Environment specific arguments
    max_t: Optional[int] = None
    """the maximum number of steps during one episode"""
    solver: str = "gecode"

    # Algorithm specific arguments
    proportion: float = 0.2


if __name__ == '__main__':
    args = tyro.cli(Args)

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", "generated")

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair.mzn")

    run_name = f"TSP__eval__{args.exp_name}__{args.seed}__{int(time.time())}"

    problem_instances_paths = [os.path.join(TSP_DATA_DIR, path) for path in os.listdir(TSP_DATA_DIR) if path.endswith(".json")]

    # ==== Tracking Initialisation ====
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            save_code=True,
            config={
                "env": {
                    "data_dir": TSP_DATA_DIR,
                    "init_solver": TSP_INIT_SOLVER_PATH,
                    "repair_solver": TSP_REPAIR_SOLVER_PATH,
                    "solver": args.solver,
                    "max_t": args.max_t,
                },
                "algorithm": {
                    "proportion": args.proportion,
                }
            },
            job_type="eval"
        )

    # ==== Seeding ====
    np.random.seed(args.seed)

    # ==== Environment Creation ====
    env = TSPEnvironmentMultiBinary(
        problem_instance_path=problem_instances_paths[0],
        init_model_path=TSP_INIT_SOLVER_PATH,
        repair_model_path=TSP_REPAIR_SOLVER_PATH,
        solver_name=args.solver,
        max_episode_length=args.max_t,
        action_bounds=None,
    )

    proportion = args.proportion
    if proportion <= 0.0 or proportion > 1.0:
        raise ValueError("proportion must be between 0.0 and 1.0")

    k = int(env.problem.num_nodes * proportion)

    # ==== Main Loop ====
    obs, info = env.reset()

    if args.track:
        wandb.log({
            "best_objective_value": info["best_objective_value"],
        })

    measured_time = 0.0

    step = 0
    done = False
    while not done:
        start_time = time.perf_counter()

        action = env.action_space.sample_limited(k=k)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        measured_time += time.perf_counter() - start_time

        if args.track:
            wandb.log({
                "reward": reward,
                "best_objective_value": info["best_objective_value"],
                "step_objective_value": info["step_objective_value"],
            })

        if args.debug:
            print(f"Step {step}")
            print(f"Reward {reward}, Step Objective Value {info['step_objective_value']}")

        step += 1

    print(f"Final solution: {env.lns.best_solution.route}")
    print(f"Objective value: {info['best_objective_value']}")
    print(f"Measured time: {measured_time:.3f} s")
    print(f"Average time per one step: {(measured_time / step):.3f} s")

    # ==== Update best results ====
    TSP_BEST_RESULTS_PATH = os.path.join(TSP_DATA_DIR, "best_results.csv")

    best_results = pd.read_csv(TSP_BEST_RESULTS_PATH)

    instance_name = os.path.basename(problem_instances_paths[0]).rstrip(".json")
    row_index = best_results[best_results["instance"] == instance_name].index

    if not row_index.empty:
        best_result = best_results.loc[row_index, "random"].iloc[0]
        if info["best_objective_value"] < best_result or best_result is None:
            best_results.loc[row_index, "random"] = info["best_objective_value"]
    else:
        new_record = {
            "instance": instance_name,
            "random": info["best_objective_value"],
            "adaptive": None,
            "mine": None,
        }

        best_results = pd.concat([best_results, pd.DataFrame([new_record])], ignore_index=True)

    best_results.to_csv(TSP_BEST_RESULTS_PATH, index=False)
