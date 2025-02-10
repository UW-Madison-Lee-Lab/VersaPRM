from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StepScore:
    step: str
    score: float


class PRM(ABC):
    '''An interface for Process Reward Models'''

    @abstractmethod
    def __call__(self, steps: list[str]) -> list[StepScore]:
        '''
        Args:
            steps (list[str]): A list of reasoning solutions.

        Returns:
            list[StepScore]: Step scores and corresponding cot solution.
        '''
        pass
