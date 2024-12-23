from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Self, List

from minizinc import Instance


class CPPartialSolution(ABC):

    @abstractmethod
    def to_output(self) -> str:
        """just print the solution"""

    @abstractmethod
    def fix_instance(self, instance: Instance):
        """fixes fields in the solution"""


P = TypeVar("P", bound=CPPartialSolution)
class CPSolution(ABC, Generic[P]):

    @property
    @abstractmethod
    def objective_value(self):
        """should return objective value"""

    @abstractmethod
    def to_output(self) -> str:
        """just serialized solution to string"""

    @property
    @abstractmethod
    def should_minimize(self) -> bool:
        """returns True, if we minimize objective, otherwise False"""

    @abstractmethod
    def destroy(self, action) -> P:
        pass

    def is_better_than(self, other: Self) -> bool:
        if self.should_minimize:
            return self.objective_value < other.objective_value
        else:
            return self.objective_value > other.objective_value

    @abstractmethod
    def score_against(self, other: Self):
        pass
        # if self.should_minimize:
        #     return other.objective_value - self.objective_value
        # else:
        #     return self.objective_value - other.objective_value
