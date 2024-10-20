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

    def reset(self):
        """
        Returns:
            initial_solution
        """
        self.best_solution = self.solver.find_initial_solution()

        return self.best_solution

    def step(
        self,
        action
    ):
        """
        Args:
            action

        Returns:
            current_solution
            score
            terminated
            truncated
        """
        partial_solution = self.best_solution.relax(action)
        new_solution = self.solver.repair(partial_solution)

        score = new_solution.score_against(self.best_solution)
        if new_solution.is_better_than(self.best_solution):
            self.best_solution = new_solution

        return self.best_solution, score, False, False