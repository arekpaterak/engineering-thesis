from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Self, Any

import numpy as np

from general.problem import Problem


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def distanceTo(self, other: Self) -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

@dataclass
class TravelingSalesmanProblem(Problem):
    num_nodes: int
    node_positions: list[Position]
    distance_matrix: np.ndarray
    config: dict[str, Any]

    @classmethod
    def load_from_file(cls, problem_path: str) -> Self:
        with open(problem_path, "r") as f:
            data = json.load(f)

        num_nodes = data["nodes"]
        nodes_positions = [Position(position[0], position[1]) for position in data["node_positions"]]
        distance_matrix = np.array(data["dist"])
        config = data["config"]

        return cls(num_nodes, nodes_positions, distance_matrix, config=config)

    @classmethod
    def generate(
        cls,
        num_nodes: int,
        max_coordinate: int
    ) -> Self:
        # TODO: Change to sampling from a unit square?

        config = {
            "max_coordinate": max_coordinate,
        }

        node_positions = set()
        for _ in range(num_nodes):
            x = np.random.randint(0, max_coordinate)
            y = np.random.randint(0, max_coordinate)
            position = Position(x, y)
            while position in node_positions:
                x = np.random.randint(0, max_coordinate)
                y = np.random.randint(0, max_coordinate)
                position = Position(x, y)
            node_positions.add(position)

        distance_matrix = np.zeros((num_nodes, num_nodes))
        for idx1, position1 in enumerate(node_positions):
            for idx2, position2 in enumerate(node_positions):
                distance_matrix[idx1, idx2] = position1.distanceTo(position2)

        return cls(num_nodes, list(node_positions), distance_matrix, config=config)

    def save(self, path: str, use_ints: bool = False) -> None:
        distance_matrix = self.distance_matrix
        if use_ints:
            distance_matrix = distance_matrix.astype(int)

        data = {
            "config": self.config,
            "nodes": self.num_nodes,
            "node_positions": [
                [position.x, position.y] for position in self.node_positions
            ],
            "dist": distance_matrix.tolist(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
