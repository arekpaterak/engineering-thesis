from __future__ import annotations

from dataclasses import dataclass, InitVar
from typing import Self

from PyQt5.QtCore import fixed
from minizinc import Instance

from general.lns.solution import CPPartialSolution, CPSolution
from general.lns.solver import CPSolver


@dataclass
class TSPPartialSolution(CPPartialSolution):
    solution: TSPSolution
    fixed_next: list[int]

    def to_output(self) -> str:
        return f"fixed_next = {self.fixed_next};"

    def fix_instance(self, instance: Instance):
        instance["fixed_next"] = self.fixed_next


@dataclass
class TSPSolution(CPSolution):
    next: list[int]
    total_distance: int

    # attribute required to be used by the MiniZinc library (stores objective in optimization problems)
    objective: int = 0
    # attribute required to be used by the MiniZinc library
    _output_item: InitVar[str] = None

    def to_output(self) -> str:
        return "\n".join([
            f"next = {self.next};",
            f"total_distance = {self.total_distance};"
        ])

    @property
    def objective_value(self) -> int:
        return self.total_distance

    @property
    def should_minimize(self) -> bool:
        return True

    def destroy(self, action: list[int]) -> TSPPartialSolution:
        fixed_next = self.next.copy()

        # remove all edges of the selected nodes; 1 subtracted from indices because of the MiniZinc representation
        for node, next_node in enumerate(fixed_next, start=1):
            if action[node-1] or action[next_node-1]:
                fixed_next[node-1] = 0
        return TSPPartialSolution(self, fixed_next)

    def score_against(self, other: Self) -> int:
        return other.objective_value - self.objective_value


class TSPSolver(CPSolver[TSPSolution, TSPPartialSolution]):
    def __init__(
            self,
            problem_path: str,
            init_model_path: str,
            repair_model_path: str,
            solver_name: str = "gecode",
            processes: int = 1
    ) -> None:
        super().__init__(
            problem_path,
            init_model_path,
            repair_model_path,
            solver_name,
            processes
        )
        self._initial_model.output_type = TSPSolution
        self._repair_model.output_type = TSPSolution
        self._initial_instance = Instance(self._solver, self._initial_model)
        self._repair_instance = Instance(self._solver, self._repair_model)

    def find_initial_solution(self) -> TSPSolution:
        return self._initial_instance.solve(processes = self._processes).solution

    def repair(self, partial_solution: TSPPartialSolution) -> TSPSolution:
        # the branch method creates a new "copy of the model instance"
        with self._repair_instance.branch() as opt:
            # then we set the initial_route
            partial_solution.fix_instance(opt)
            return opt.solve(processes = self._processes).solution
