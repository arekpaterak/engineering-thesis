from abc import abstractmethod, ABC
from typing import Optional, Any, Type, List, Self, Dict

import gymnasium as gym
import torch_geometric as pyg

from general.lns.lns import LNS
from general.lns.solution import P
from general.lns.solver import CPSolver, S
from general.problem import Problem


class LNSEnvironment(ABC, gym.Env):
    def __init__(
        self,
        problem_cls: Type[Problem],
        solver_cls: Type[CPSolver[S, P]],
        problem_instance_path: str,
        init_model_path: str,
        repair_model_path: str,
        solver_name: str = "gecode",
        max_episode_length: Optional[int] = None,
        processes: int = 1
    ) -> None:
        solver = solver_cls(
            problem_path=problem_instance_path,
            init_model_path=init_model_path,
            repair_model_path=repair_model_path,
            solver_name=solver_name,
            processes=processes
        )
        self.lns = LNS(solver=solver)
        self.problem = problem_cls.load_from_file(problem_instance_path)
        self.max_episode_length = max_episode_length
        self.episode_length = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed, options=options)

        self.episode_length = 0
        initial_solution = self.lns.reset()

        observation = self._observation(initial_solution)
        info = {
            "best_objective_value": self.lns.best_solution.objective_value,
        }
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
        reward = self._reward(score, action)

        info = {
            "best_objective_value": self.lns.best_solution.objective_value,
            "step_objective_value": self.lns.step_objective_value,
        }

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _observation(self, solution) -> dict:
        pass

    @abstractmethod
    def _reward(self, score, action):
        pass

    @classmethod
    def create_multiple(
        cls,
        problem_instances_paths: List[str],
        **kwargs
    ) -> List[Self]:
        envs = []
        for problem_instance_path in problem_instances_paths:
            env = cls(problem_instance_path=problem_instance_path, **kwargs)
            envs.append(env)
        return envs

    @staticmethod
    @abstractmethod
    def preprocess(observation: dict) -> pyg.data.Data:
        pass
