import json
from dataclasses import dataclass
from typing import Self

from general.problem import Problem

@dataclass
class CVRP(Problem):
    # TODO

    @classmethod
    def load_from_file(cls, problem_path: str) -> Self:
        with open(problem_path, "r") as f:
            data = json.load(f)

        # TODO

    def save(self, path: str) -> None:
        data = None  # TODO
        with open(path, "w") as f:
            json.dump(data, f)
