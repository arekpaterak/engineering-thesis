from abc import ABC, abstractmethod
from typing import Self


class Problem(ABC):
    @classmethod
    @abstractmethod
    def load_from_file(cls, problem_path: str) -> Self:
        pass

    @classmethod
    @abstractmethod
    def generate(cls, **kwargs) -> Self:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass
