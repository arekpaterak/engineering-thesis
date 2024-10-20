from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Self

import numpy as np

from general.problem import Problem


@dataclass
class TravelingSalesmanProblem(Problem):
    num_nodes: int
    distance_matrix: np.ndarray

    @classmethod
    def load_from_file(cls, problem_path: str) -> Self:
        with open(problem_path, "r") as f:
            data = json.load(f)

        num_nodes = data["Nodes"]
        distance_matrix = np.array(data["Dist"])

        return cls(num_nodes, distance_matrix)

    @classmethod
    def generate(
        cls,
        num_nodes: int,
    ) -> Self:
        # TODO
        pass

    def save(self, path: str) -> None:
        data = {"Nodes": self.num_nodes, "Dist": self.distance_matrix}
        with open(path, "w") as f:
            json.dump(data, f)
