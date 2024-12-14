from __future__ import annotations

from functools import partial, wraps
from itertools import combinations
from types import MethodType
from typing import Optional, List, Self, Dict, Tuple

import numpy as np

import gymnasium as gym
import torch

import torch_geometric as pyg

from general.lns_env import LNSEnvironment
from general.utils import MultiBinaryWithLimitedSampling
from problems.tsp.tsp import TravelingSalesmanProblem
from problems.tsp.tsp_lns import TSPSolver


class TSPEnvironmentDiscrete(LNSEnvironment):
    def __init__(
        self,
        problem_instance_path: str,
        init_model_path: str,
        repair_model_path: str,
        solver_name: str = "gecode",
        max_episode_length: Optional[int] = None,
        k: int = 4,
    ):
        super().__init__(
            problem_cls=TravelingSalesmanProblem,
            solver_cls=TSPSolver,
            problem_instance_path=problem_instance_path,
            init_model_path=init_model_path,
            repair_model_path=repair_model_path,
            solver_name=solver_name,
            max_episode_length=max_episode_length,
        )

        num_nodes = self.problem.num_nodes

        # ==== Observation Space ====
        self.observation_space = gym.spaces.Dict({
            "problem": gym.spaces.Dict({
                "node_positions": gym.spaces.Box(
                    low=0,
                    high=self.problem.config["max_coordinate"],
                    shape=(num_nodes, 2),
                    dtype=int
                )
            }),
            "solution": gym.spaces.Dict({
                "route": gym.spaces.Sequence(gym.spaces.Discrete(num_nodes, start=1))
            })
        })
        # Override sampling with a mask for a fixed Sequence length
        self.observation_space.sample = partial(self.observation_space.sample, mask={"problem": None, "solution" : {"route": (num_nodes, None)}})

        # ==== Action Space ====
        self.k = k
        self.n = num_nodes

        all_combinations = list(combinations(range(self.n), self.k))
        self.action_space = gym.spaces.Discrete(len(all_combinations))
        self.action_to_combination = {i: combination for i, combination in enumerate(all_combinations)}

    def step(self, action: int):
        """
        Args:
            action

        Returns:
            observation
            reward
            terminated (bool)
            truncated (bool)
            info (dict)
        """
        info = {
            "is_action_ignored": False,
            "best_objective_value": None,
            "step_objective_value": None,
        }

        solution, score, terminated, truncated = self.lns.step(self.action_to_vector(action))

        self.episode_length += 1
        if self.max_episode_length:
            if self.episode_length >= self.max_episode_length:
                truncated = True

        observation = self._observation(solution)
        reward = self._reward(score, action)

        info["best_objective_value"] = self.lns.best_solution.objective_value
        info["step_objective_value"] = self.lns.step_objective_value

        return observation, reward, terminated, truncated, info

    def action_to_vector(self, action: int):
        combination = self.action_to_combination[action]
        vector = np.zeros(self.n, dtype=int)
        vector[list(combination)] = 1
        return vector

    def _observation(self, solution) -> dict:
        result = {
            "problem": {
                "node_positions": [{"x": position.x, "y": position.y} for position in self.problem.node_positions],
            },
            "solution": {
                "route": solution.route,
            },
        }
        return result

    def _reward(self, score, action):
        return score

    @staticmethod
    def preprocess(observation: dict) -> pyg.data.Data:
        node_positions = observation['problem']['node_positions']
        node_features = torch.tensor([[node['x'], node['y']] for node in node_positions], dtype=torch.float)
        node_positions = node_features.clone()

        route = observation['solution']['route']
        edges = []
        for i in range(len(route) - 1):
            src = route[i] - 1
            dst = route[i + 1] - 1
            edges.append((src, dst))
            edges.append((dst, src))

        src = route[-1] - 1
        dst = route[0] - 1
        edges.append((src, dst))
        edges.append((dst, src))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        edge_attr = torch.tensor([[1] for _ in edges], dtype=torch.float)

        return pyg.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=node_positions)


if __name__ == "__main__":
    problem_path = "data/generated/20_1000_0.json"
    init_model_path = "minizinc/tsp_init.mzn"
    repair_model_path = "minizinc/tsp_repair.mzn"
    solver_name = "gecode"

    env = TSPEnvironmentDiscrete(
        problem_path,
        init_model_path,
        repair_model_path,
        solver_name,
        k = 4
    )
    print(env.problem)

    print("==== Observation Space ====")
    print(env.observation_space)

    print("==== Action Space ====")
    print(env.action_space)

    obs, _ = env.reset()
    print(obs)

    action = env.action_space.sample()
    print(f"A sample action: {action}")
    print(f"The sample action as a vector: {env.action_to_vector(action)}")
