from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from minizinc import Model, Solver

from general.lns.solution import P, CPSolution

S = TypeVar("S", bound=CPSolution)


class CPSolver(ABC, Generic[S, P]):
    def __init__(
            self,
            problem_path: str,
            init_model_path: str,
            repair_model_path: str,
            solver_name: str = "gecode",
            processes: int = 1,
    ) -> None:
        super().__init__()

        # load minizinc solver
        self._solver = Solver.lookup(solver_name)
        self._processes = processes

        # load model finding the initial solution
        self._initial_model = Model()
        self._initial_model.add_file(init_model_path)
        self._initial_model.add_file(problem_path)

        # load model improving the solution
        self._repair_model= Model()
        self._repair_model.add_file(repair_model_path)
        self._repair_model.add_file(problem_path)

    @abstractmethod
    def find_initial_solution(self) -> S:
        """finds initial feasible solution and its objective value"""

    @abstractmethod
    def repair(self, partial_solution: P) -> S:
        """improves the given partial solution"""
