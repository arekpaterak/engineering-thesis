from abc import abstractmethod, ABC
from typing import Optional, Any, Type

import gymnasium as gym

from general.lns.lns import LNS
from general.lns.solution import P
from general.lns.solver import CPSolver, S
from general.problem import Problem


class LNSEnvironment(ABC, gym.Env):
    def __init__(
        self,
        problem_cls: Type[Problem],
        solver_cls: Type[CPSolver[S, P]],
        problem_path: str,
        init_model_path: str,
        repair_model_path: str,
        solver_name: str = "gecode",
        max_episode_length: Optional[int] = None,
    ) -> None:
        solver = solver_cls(
            problem_path=problem_path,
            init_model_path=init_model_path,
            repair_model_path=repair_model_path,
            solver_name=solver_name,
        )
        self.lns = LNS(solver=solver)
        self.problem = problem_cls.load_from_file(problem_path)
        self.max_episode_length = max_episode_length
        self.episode_length = 0

    def reset(
        self,
        seed: Optional[int] = None,
    ):
        """
        Args:
            seed:

        Returns:
            observation
            info (dict)
        """
        super().reset(seed=seed)

        self.episode_length = 0

        solution = self.lns.reset()

        observation = self._observation(solution)
        info = dict()
        return observation, info

    def step(self, action):
        """
        Args:
            action

        Returns:
            observation
            reward
            terminated (bool)
            truncated (bool)
            info (dict)
        """
        solution, score, terminated, truncated = self.lns.step(action)

        self.episode_length += 1
        if self.max_episode_length:
            if self.episode_length >= self.max_episode_length:
                truncated = True

        observation = self._observation(solution)
        reward = self._reward(score)
        info = dict()
        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _observation(self, solution):
        pass

    @abstractmethod
    def _reward(self, score):
        pass
