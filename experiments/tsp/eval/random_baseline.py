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
import torchmetrics.aggregation
from torch import nn
from torch.distributions import Bernoulli

import torch_geometric as pyg

import tyro
import wandb

from experiments.playground import action
from general.ml.features_extractor import GraphFeaturesExtractor
from problems.tsp.tsp_env_multibinary import TSPEnvironmentMultiBinary

@dataclass
class Args:
    instance_name: Optional[str] = None
    """the instance on which the method will be run"""
    instances_dir_name: Optional[str] = None
    """the name of instances directory"""
    seed: int = 1
    """seed of the experiment"""
    debug: bool = True
    """if toggled, extra logs will be printed to the console"""

    # Environment specific arguments
    max_t: Optional[int] = None
    """the maximum number of steps during one episode"""
    solver: str = "gecode"
    """the solver to use to find an initial solution and repair the subsequent"""

    # Algorithm specific arguments
    proportion: float = 0.2
    """the proportion of the nodes in the problem to destroy in one step"""


if __name__ == '__main__':
    args = tyro.cli(Args)

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", args.instances_dir_name)

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair.mzn")

    # ==== Environment Creation ====
    if args.instance_name is None:
        problem_instances_paths = [os.path.join(TSP_DATA_DIR, path) for path in os.listdir(TSP_DATA_DIR) if path.endswith(".json")]
    else:
        problem_instances_paths = [os.path.join(TSP_DATA_DIR, f"{args.instance_name}.json")]

    # ==== Loop over problem instances ====
    for instance_idx, instance_path in enumerate(problem_instances_paths):

        # ==== Seeding ====
        random.seed(args.seed)
        np.random.seed(args.seed)

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
            action_bounds=None,
        )

        proportion = args.proportion
        if proportion <= 0.0 or proportion > 1.0:
            raise ValueError("proportion must be between 0.0 and 1.0")
        k = int(env.problem.num_nodes * proportion)

        # ==== Main Loop ====
        obs, info = env.reset()

        initial_objective_value = info["best_objective_value"]
        if args.debug:
           print(f"Initial solution objective value: {initial_objective_value}")

        episode_time = 0.0

        step = 0
        done = False
        while not done:
            step_start_time = time.perf_counter()

            action = env.action_space.sample_limited(k=k)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            episode_time += time.perf_counter() - step_start_time

            if args.debug:
                print(f"Step {step}")
                print(f"Reward {reward}, Step Objective Value {info['step_objective_value']}")

            step += 1

            # ==== Update the best results ====
            if step % 10 == 0:
                TSP_BEST_RESULTS_PATH = os.path.join(BASE_PATH, "problems", "tsp", "data", "best_results.csv")
                method_name = f"random({args.proportion})"

                df = pd.read_csv(TSP_BEST_RESULTS_PATH)

                new_record = {
                    "instance": instance_name,
                    "subset": args.instances_dir_name,
                    "method": method_name,
                    "seed": args.seed,
                    "steps": step,
                    "initial_objective_value": initial_objective_value,
                    "objective_value": info["best_objective_value"],
                    "time": f"{episode_time:.3f}",
                }
                df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

                df.to_csv(TSP_BEST_RESULTS_PATH, index=False)

                if args.debug:
                    print(f"Solution: {env.lns.best_solution.route}")
                    print(f"Objective value: {info['best_objective_value']}")
                    print(f"Measured time: {episode_time:.3f} s")
                    print(f"Average time per one step: {(episode_time / step):.3f} s")
