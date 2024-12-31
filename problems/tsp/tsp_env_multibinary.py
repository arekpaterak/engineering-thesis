from __future__ import annotations

from functools import partial, wraps
from itertools import combinations, permutations
from types import MethodType
from typing import Optional, List, Self, Dict, Tuple

import numpy as np

import gymnasium as gym
import torch

import torch_geometric as pyg

from general.lns_env import LNSEnvironment
from general.utils import MultiBinaryWithLimitedSampling, draw_graph, route_from_circuit, minizinc_circuit_to_python
from problems.tsp.tsp import TravelingSalesmanProblem
from problems.tsp.tsp_lns import TSPSolver


class TSPEnvironmentMultiBinary(LNSEnvironment):
    def __init__(
        self,
        problem_instance_path: str,
        init_model_path: str,
        repair_model_path: str,
        solver_name: str = "gecode",
        max_episode_length: Optional[int] = None,
        processes: int = 1,
        fully_connected: bool = False,
    ):
        super().__init__(
            problem_cls=TravelingSalesmanProblem,
            solver_cls=TSPSolver,
            problem_instance_path=problem_instance_path,
            init_model_path=init_model_path,
            repair_model_path=repair_model_path,
            solver_name=solver_name,
            max_episode_length=max_episode_length,
            processes=processes
        )

        num_nodes = self.problem.num_nodes
        self.fully_connected = fully_connected

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
                "route": gym.spaces.Sequence(gym.spaces.Discrete(num_nodes, start=1)),
                "circuit": gym.spaces.Sequence(gym.spaces.Discrete(num_nodes, start=1))
            })
        })
        # Override sampling with a mask for a fixed Sequence length
        self.observation_space.sample = partial(self.observation_space.sample, mask={"problem": None, "solution" : {"route": (num_nodes, None), "circuit": (num_nodes, None)}})

        # ==== Action Space ====
        self.action_space: MultiBinaryWithLimitedSampling = MultiBinaryWithLimitedSampling(num_nodes)

        self.action_counter = np.array([0.0 for _ in range(num_nodes)])

    def reset(self):
        self.action_counter = np.array([0.0 for _ in range(self.problem.num_nodes)])

        observation, info = super().reset()
        return observation, info

    def step(self, action: np.ndarray):
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
        solution, score, terminated, truncated, lns_info = self.lns.step(action)

        if score != 0:
            self.action_counter *= 0
        self.action_counter += action

        self.episode_length += 1
        if self.max_episode_length:
            if self.episode_length >= self.max_episode_length:
                truncated = True

        observation = self._observation(solution)
        reward = self._reward(score, action)

        info = {
            "best_objective_value": self.lns.best_solution.objective_value,
            "step_objective_value": self.lns.step_objective_value,
            "partial_solution": {
                "circuit": minizinc_circuit_to_python(lns_info["partial_solution"].fixed_next)
            },
        }

        return observation, reward, terminated, truncated, info

    def _observation(self, solution) -> dict:
        circuit = minizinc_circuit_to_python(solution.next)

        result = {
            "problem": {
                "node_positions": [{"x": position.x, "y": position.y} for position in self.problem.node_positions],
            },
            "solution": {
                "route": route_from_circuit(circuit),
                "circuit": circuit,
            },
            "how_many_times_node_chosen": self.action_counter.tolist()
        }
        return result

    def _reward(self, score, action: np.ndarray[int]):
        return score

    @staticmethod
    def preprocess(observation: dict, fully_connected: bool = False) -> pyg.data.Data:
        node_positions = observation['problem']['node_positions']
        how_many_times_node_chosen = observation['how_many_times_node_chosen']
        node_features = torch.tensor([
            [node['x'] / 1000, node['y'] / 1000, times] for node, times in zip(node_positions, how_many_times_node_chosen)
        ], dtype=torch.float)
        pos = node_features.clone()

        circuit = observation['solution']['circuit']

        route_edges = []
        for node, next_node in enumerate(circuit):
            # route_edges in two directions are added
            route_edges.append((node, next_node))
            route_edges.append((next_node, node))

        if not fully_connected:
            edge_index = torch.tensor(route_edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1] for _ in route_edges], dtype=torch.float)

            return pyg.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        else:
            all_edges = list(permutations(range(len(node_positions)), 2))

            route_edges_lookup = set(tuple(edge) for edge in route_edges)

            edge_attr = []
            for edge in all_edges:
                if edge in route_edges_lookup:
                    edge_attr.append([1])
                else:
                    edge_attr.append([0])

            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            return pyg.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


if __name__ == "__main__":
    problem_path = "data/generated/train/20_1000_0.json"
    init_model_path = "minizinc/tsp_init_circuit.mzn"
    repair_model_path = "minizinc/tsp_repair_circuit.mzn"
    solver_name = "gecode"

    env = TSPEnvironmentMultiBinary(problem_path, init_model_path, repair_model_path, solver_name)
    print(env.problem)

    print("==== Observation Space ====")
    print(env.observation_space)

    print("==== Action Space ====")
    print(env.action_space)

    obs, _ = env.reset()
    print(f"\nObservation:\n{obs}")
    print(f"Route:\n{obs['solution']['route']}")
    print(f"Circuit:\n{obs['solution']['circuit']}\n")
    graph = env.preprocess(obs)
    plt = draw_graph(graph)
    plt.show()

    action = env.action_space.sample_limited(k=4)
    print(f"A restricted sample action: {action}")
    obs, _, _, _, info = env.step(action)
    print(f"\nObservation:\n{obs}")
    print(f"Route:\n{obs['solution']['route']}")
    print(f"Circuit:\n{obs['solution']['circuit']}")
    print(f"Info:\n{info}")
    graph = env.preprocess(obs)
    plt = draw_graph(graph)
    plt.show()

    obs, _, _, _, info = env.step(action )
    print(f"\nObservation:\n{obs}")
