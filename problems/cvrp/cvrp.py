import json
from dataclasses import dataclass
from typing import Self

import numpy as np

from general.problem import Problem
from problems.tsp.tsp import Position


@dataclass
class CVRP(Problem):
    num_nodes: int
    node_positions: list[Position]
    distance_matrix: np.ndarray
    capacity: int
    demands: list[int]

    @classmethod
    def load_from_file(cls, problem_path: str) -> Self:
        with open(problem_path, "r") as f:
            data = json.load(f)

        num_nodes = data["N"]
        nodes_positions = [Position(position[0], position[1]) for position in data["NodePositions"]]
        distance_matrix = np.array(data["Distance"])

        capacity = data["Capacity"]
        demands = data["Demand"]

        return cls(num_nodes, nodes_positions, distance_matrix, capacity, demands)

    @classmethod
    def parse_from_vrp_file(cls, problem_path: str) -> Self:
        with open(problem_path, 'r') as f:
            lines = f.readlines()

        header_data = {}
        current_section = None
        node_positions = []
        demands = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse header information
            if ':' in line:
                key, value = map(str.strip, line.split(':'))
                header_data[key] = value
                continue

            if line in ['NODE_COORD_SECTION', 'DEMAND_SECTION']:
                current_section = line
                continue

            if line in ['DEPOT_SECTION']:
                break

            # Parse coordinate section
            if current_section == 'NODE_COORD_SECTION':
                node_id, x, y = map(int, line.split())
                node_positions.append(Position(x, y))
            # Parse demand section
            elif current_section == 'DEMAND_SECTION':
                node_id, demand = map(int, line.split())
                demands.append(demand)

        num_nodes = int(header_data['DIMENSION']) - 1
        capacity = int(header_data['CAPACITY'])

        # Compute distance matrix
        distance_matrix = np.zeros((num_nodes, num_nodes))
        for idx1, position1 in enumerate(node_positions):
            for idx2, position2 in enumerate(node_positions):
                distance_matrix[idx1, idx2] = position1.distanceTo(position2)

        return cls(
            num_nodes=num_nodes,
            node_positions=node_positions,
            distance_matrix=distance_matrix,
            capacity=capacity,
            demands=demands[1:]
        )

    @classmethod
    def generate(cls) -> Self:
        pass

    def save(self, path: str, use_ints: bool = False) -> None:
        distance_matrix = self.distance_matrix
        if use_ints:
            distance_matrix = distance_matrix.astype(int)

        data = {
            "N": self.num_nodes,
            "Capacity": self.capacity,
            "Demand": self.demands,
            "Distance": distance_matrix.tolist(),
            "NodePositions": [
                [position.x, position.y] for position in self.node_positions
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
