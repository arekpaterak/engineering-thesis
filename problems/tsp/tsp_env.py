from __future__ import annotations

from typing import Optional

from general.env import LNSEnvironment
from problems.tsp.tsp import TravelingSalesmanProblem
from problems.tsp.tsp_lns import TSPSolver


class TSPEnvironment(LNSEnvironment):
    def __init__(
        self,
        problem_path: str,
        init_model_path: str,
        repair_model_path: str,
        solver_name: str = "gecode",
        max_episode_length: Optional[int] = None,
    ):
        super().__init__(
            problem_cls=TravelingSalesmanProblem,
            solver_cls=TSPSolver,
            problem_path=problem_path,
            init_model_path=init_model_path,
            repair_model_path=repair_model_path,
            solver_name=solver_name,
            max_episode_length=max_episode_length
        )

    def _observation(self, solution):
        result = {
            "problem": {"node_positions": [{"x": position.x, "y": position.y} for position in self.problem.node_positions]},
            "solution": {"route": solution.route},
        }
        return result

    def _reward(self, score):
        # TODO
        return score


if __name__ == "__main__":
    problem_path = "data/eil10.json"
    init_model_path = "minizinc/tsp_init.mzn"
    repair_model_path = "minizinc/tsp_repair.mzn"
    solver_name = "coinbc"

    env = TSPEnvironment(problem_path, init_model_path, repair_model_path, solver_name)
    print(env.problem)

    obs, _ = env.reset()
    print(obs)

    obs = env.step([1, 2])[0]
    print(obs)
