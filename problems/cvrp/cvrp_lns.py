from __future__ import annotations

from dataclasses import dataclass, InitVar

import numpy as np
from minizinc import Instance

from general.lns.solution import CPPartialSolution, CPSolution
from general.lns.solver import CPSolver


@dataclass
class CVRPPartialSolution(CPPartialSolution):
    solution: CVRPSolution
    fixed_successor: list[int]
    fixed_vehicle: list[int]

    def to_output(self) -> str:
        return f"fixed_successor = {self.fixed_successor}; fixed_vehicle = {self.fixed_vehicle}"

    def fix_instance(self, instance: Instance):
        instance["fixed_successor"] = self.fixed_successor
        instance["fixed_vehicle"] = self.fixed_vehicle


@dataclass
class CVRPSolution(CPSolution):
    successor: list[int]
    vehicle: list[int]
    total_distance: int

    predecessor: list[int]
    load: list[int]
    arrivalTime: list[int]

    # attribute required to be used by the MiniZinc library (stores objective in optimization problems)
    objective: int = 0
    # attribute required to be used by the MiniZinc library
    _output_item: InitVar[str] = None

    def to_output(self) -> str:
        return "\n".join([
            f"successor = {self.successor};",
            f"vehicle = {self.vehicle};",
            f"total_distance = {self.total_distance};"
        ])

    @property
    def objective_value(self) -> int:
        return self.total_distance

    @property
    def should_minimize(self) -> bool:
        return True

    def destroy(self, action: np.ndarray[int]) -> CVRPPartialSolution:
        fixed_successor = self.successor.copy()
        fixed_vehicle = self.vehicle.copy()

        # remove all edges of the selected nodes; 1 subtracted from indices because of the MiniZinc representation
        for node, next_node in enumerate(fixed_successor, start=1):
            if node-1 in action or next_node-1 in action:
                fixed_successor[node-1] = 0
                fixed_vehicle[node-1] = 0

        return CVRPPartialSolution(self, fixed_successor, fixed_vehicle)


class CVRPSolver(CPSolver[CVRPSolution, CVRPPartialSolution]):
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
        self._initial_model.output_type = CVRPSolution
        self._repair_model.output_type = CVRPSolution
        self._initial_instance = Instance(self._solver, self._initial_model)
        self._repair_instance = Instance(self._solver, self._repair_model)

    def find_initial_solution(self) -> CVRPSolution:
        return self._initial_instance.solve(processes = self._processes).solution

    def repair(self, partial_solution: CVRPPartialSolution) -> CVRPSolution:
        # the branch method creates a new "copy of the model instance"
        with self._repair_instance.branch() as opt:
            # then we set the initial_route
            partial_solution.fix_instance(opt)
            return opt.solve(processes = self._processes).solution
