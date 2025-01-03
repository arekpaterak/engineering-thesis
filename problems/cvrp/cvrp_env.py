from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch_geometric as pyg

from general.lns_env import LNSEnvironment
from general.utils import MultiBinaryWithLimitedSampling, minizinc_list_to_python
from problems.cvrp.cvrp import CVRP
from problems.cvrp.cvrp_lns import CVRPSolver


class CVRPEnvironment(LNSEnvironment):

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
            problem_cls=CVRP,
            solver_cls=CVRPSolver,
            problem_instance_path=problem_instance_path,
            init_model_path=init_model_path,
            repair_model_path=repair_model_path,
            solver_name=solver_name,
            max_episode_length=max_episode_length,
            processes=processes,
        )

        num_nodes = self.problem.num_nodes
        self.fully_connected = fully_connected

        # ==== Observation Space ====
        # self.observation_space = gym.spaces.Dict({
        #     "problem": gym.spaces.Dict({
        #         "node_positions": gym.spaces.Box(
        #             low=0,
        #             high=1000,
        #             shape=(num_nodes, 2),
        #             dtype=int
        #         ),
        #         "demands": None,
        #         "capacity": None
        #     }),
        #     "solution": gym.spaces.Dict({
        #         "routes": None,
        #         "successor": None,
        #         "vehicle": None
        #     })
        # })

        # ==== Action Space ====
        self.action_space = MultiBinaryWithLimitedSampling(num_nodes)

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
        self.action_counter[action] += 1

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
                "successor": lns_info["partial_solution"].fixed_successor,
                "vehicle": lns_info["partial_solution"].fixed_vehicle,
            },
        }

        return observation, reward, terminated, truncated, info

    def _observation(self, solution) -> dict:
        successor = solution.successor
        routes = []

        N = self.problem.num_nodes
        for vehicle in range(N):
            route = [0]
            next_node = successor[N+vehicle]
            end_node = successor[N*2+vehicle]

            while next_node != end_node:
                route.append(next_node)
                next_node = successor[next_node-1]

            if len(route) > 2:
                routes.append(route)

        result = {
            "problem": {
                "node_positions": [{"x": position.x, "y": position.y} for position in self.problem.node_positions],
                "demands": self.problem.demands,
                "capacity": self.problem.capacity,
            },
            "solution": {
                "routes": routes,
                "successor": solution.successor,
                "vehicle": minizinc_list_to_python(solution.vehicle)[:],
            },
            "how_many_times_node_chosen": self.action_counter.tolist()
        }
        return result

    def _reward(self, score, action):
        return score

    @staticmethod
    def preprocess(observation: dict, fully_connected: bool = False) -> pyg.data.Data:
        node_positions = observation['problem']['node_positions']
        how_many_times_node_chosen = observation['how_many_times_node_chosen']
        demands = observation["problem"]["demands"]
        node_features = torch.tensor(
            [
                [node['x'] / 1000, node['y'] / 1000, times, demand / sum(demands)] for node, times, demand in
                zip(node_positions, how_many_times_node_chosen, demands)
            ], dtype=torch.float
        )
        pos = node_features.clone()

        return

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
    problem_path = "data/generated/train/XMLike20/XML20_1113_00.json"
    init_model_path = "minizinc/cvrp_init.mzn"
    repair_model_path = "minizinc/cvrp_repair.mzn"
    solver_name = "gecode"

    env = CVRPEnvironment(problem_path, init_model_path, repair_model_path, solver_name)
    print(env.problem)

    print("==== Observation Space ====")

    print("==== Action Space ====")
    print(env.action_space)

    obs, info = env.reset()
    print(f"\nObservation:\n{obs}")
    print(f"\nInfo:\n{info}")

    action = np.where(env.action_space.sample_limited(k=4) == 1.0)[0]
    print(f"A restricted sample action: {action}")
    obs, _, _, _, info = env.step(action)
    print(f"\nObservation:\n{obs}")
    print(f"Routes:")
    print("\n".join([str(route) for route in obs["solution"]["routes"]]))
