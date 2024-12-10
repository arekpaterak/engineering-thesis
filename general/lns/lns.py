from typing import Optional

from general.lns.solution import P
from general.lns.solver import CPSolver, S


class LNS:
    def __init__(
        self,
        solver: CPSolver[S, P],
    ) -> None:
        self.solver = solver
        self.best_solution: Optional[S] = None
        self.step_objective_value: Optional[float] = None

    def reset(self):
        """
        Returns:
            initial_solution
        """
        self.best_solution = self.solver.find_initial_solution()

        return self.best_solution

    def step(self, action):
        """
        Args:
            action

        Returns:
            best_solution after action
            score
            terminated
            truncated
        """
        partial_solution = self.best_solution.relax(action)
        new_solution = self.solver.repair(partial_solution)

        self.step_objective_value = new_solution.objective_value

        score = new_solution.score_against(self.best_solution)
        if new_solution.is_better_than(self.best_solution):
            self.best_solution = new_solution

        # TODO: Think about terminated and truncated.
        # Terminated could be added if the optimal solution objective is available.
        return self.best_solution, score, False, False
