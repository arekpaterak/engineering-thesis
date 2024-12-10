from __future__ import annotations

from functools import partial, wraps
from types import MethodType
from typing import Optional, List, Self, Dict

import numpy as np

import gymnasium as gym
import torch

import torch_geometric as pyg

from general.lns_env import LNSEnvironment
from general.utils import MultiBinaryWithLimitedSampling


class CVRPEnvironment(LNSEnvironment):
    EXPENSIVE_ACTION_PENALTY: int = -1000

    def __init__(
        self,
        problem_instance_path: str,
        init_model_path: str,
        repair_model_path: str,
        solver_name: str = "gecode",
        max_episode_length: Optional[int] = None,
        expensive_action_threshold: Optional[float] = 0.5,
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

        self.expensive_action_threshold = expensive_action_threshold

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
        # self.action_space = gym.spaces.MultiBinary(num_nodes)
        self.action_space = MultiBinaryWithLimitedSampling(num_nodes)

        # Extend sampling with possibility to sample 1s on exactly n random positions
        # def add_new_arg_wrapper(method):
        #     @wraps(method)
        #     def wrapper(self, n=None):
        #         if n:
        #             size = self.shape[0]
        #             action = np.zeros(size, dtype="int")
        #             choices = np.random.choice(size, size=n, replace=False)
        #             action[choices] = 1
        #             return action
        #         else:
        #             return method(self)
        #     return wrapper
        # self.action_space.sample = MethodType(
        #     add_new_arg_wrapper(gym.spaces.MultiBinary.sample),
        #     self.action_space
        # )

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
        if self.is_action_expensive(action):
            return score + self.EXPENSIVE_ACTION_PENALTY
        else:
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

    def is_action_expensive(self, action: list[int]) -> bool:
        if self.expensive_action_threshold is None:
            return False

        proportion = sum(action) / len(action)
        return proportion > self.expensive_action_threshold


if __name__ == "__main__":
    pass
