import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

from problems.cvrp.cvrp_env import CVRPEnvironment
from problems.tsp.tsp_env_multibinary import TSPEnvironmentMultiBinary

from config import *


@dataclass
class Args:
    instance_name: Optional[str] = None
    """the instance on which the method will be run"""
    instances_dir_name: str = "generated/train"
    """the name of instances directory"""
    seed: int = 1
    """seed of the experiment"""
    debug: bool = True
    """if toggled, extra logs will be printed to the console"""
    log_every_n_step: int = 1
    """the logging interval"""

    # Environment specific arguments
    max_t: Optional[int] = None
    """the maximum number of steps during one episode"""
    solver: str = "gecode"
    """the solver to use to find an initial solution and repair the subsequent"""
    processes: int = 1
    """the number of processes to use in MiniZinc"""

    # Algorithm specific arguments
    k: int = 5


if __name__ == '__main__':
    args = tyro.cli(Args)

    data_dir = os.path.join(CVRP_DATA_DIR, args.instances_dir_name)
    if not args.instance_name:
        problem_instances_paths = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path.endswith(".json")]
    else:
        problem_instances_paths = [os.path.join(data_dir, f"{args.instance_name}.json")]

    print(problem_instances_paths)

    # ==== Loop over problem instances ====
    for instance_idx, instance_path in enumerate(problem_instances_paths):

        # ==== Seeding ====
        random.seed(args.seed)
        np.random.seed(args.seed)

        instance_name = os.path.basename(instance_path).rstrip(".json")

        if args.debug:
            print(f"Instance {instance_idx}: {instance_name}")

        # ==== Environment Creation ====
        env = CVRPEnvironment(
            problem_instance_path=instance_path,
            init_model_path=CVRP_INIT_SOLVER_PATH,
            repair_model_path=CVRP_REPAIR_SOLVER_PATH,
            solver_name=args.solver,
            max_episode_length=args.max_t,
            processes=args.processes,
        )

        k = args.k

        # ==== Main Loop ====
        obs, info = env.reset()

        initial_objective_value = info["best_objective_value"]
        if args.debug:
           print(f"Initial solution objective value: {initial_objective_value}")

        step = 0
        episode_time = 0.0
        done = False
        while not done:
            step_start_time = time.perf_counter()

            action = np.where(env.action_space.sample_limited(k=k) == 1.0)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_time += time.perf_counter() - step_start_time

            # ==== Logging ====
            if args.debug:
                print(f"Step {step}")
                print(f"Action: {action}, Reward {reward:.3f}, Step Objective Value {info['step_objective_value']}")

            step += 1

            # ==== Save to the best results ====
            if step % args.log_every_n_step == 0 or step < 10:
                method_name = f"random({args.k})"

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

                conn = sqlite3.connect(DB_PATH)

                # Check if the record already exists
                cursor = conn.cursor()
                cursor.execute(
                """
                        SELECT * FROM cvrp_best_results
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
                            UPDATE cvrp_best_results
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
                            INSERT INTO cvrp_best_results (instance, subset, method, seed, steps, initial_objective_value, objective_value, time, avg_time_per_step)
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
                    print(f"Objective value: {info['best_objective_value']}")
                    print(f"Measured time: {episode_time:.3f} s")
                    print(f"Average time per one step: {(episode_time / step):.3f} s")
