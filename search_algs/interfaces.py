from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore

DEBUG = True
LOGGING = True


@dataclass
class ReasoningNode:
    parent: Optional["ReasoningNode"]
    children: list["ReasoningNode"]
    current_ids: Tensor
    gen_state: Optional[tuple[tuple[Tensor, Tensor], ...]]
    prm_state: Optional[tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]]
    prm_scores: Optional[Tensor] = None
    score: float = 1.0
    n_visits: int = 1
    is_leaf: bool = False

    def free_state(self):
        self.gen_state = None
        self.prm_state = None
        self.prm_scores = None


class BaseGenerator(ABC):
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    @abstractmethod
    def encode(self, question: str) -> Tensor: ...

    @abstractmethod
    def decode(self, input_ids: Tensor) -> str: ...

    @abstractmethod
    def init_state(
        self,
        input_ids: Tensor,
    ) -> Optional[tuple[tuple[Tensor, Tensor], ...]]: ...

    @abstractmethod
    def filter_state(
        self,
        state: Optional[tuple[tuple[Tensor, Tensor], ...]],
        idxs: list[int],
    ) -> Optional[tuple[tuple[Tensor, Tensor], ...]]: ...

    @abstractmethod
    def inflate_state(
        self,
        state: Optional[tuple[tuple[Tensor, Tensor], ...]],
        n: int,
    ) -> Optional[tuple[tuple[Tensor, Tensor], ...]]: ...

    @abstractmethod
    def is_complete(self, input_ids: Tensor) -> Tensor: ...

    @abstractmethod
    def __call__(
        self,
        input_ids: Tensor,
        state: Optional[tuple[tuple[Tensor, Tensor], ...]] = None,
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]: ...


class BasePRM(ABC):
    @abstractmethod
    def init_state(
        self, question: str
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]: ...

    @abstractmethod
    def filter_state(
        self,
        state: tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]],
        idxs: list[int],
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]: ...

    @abstractmethod
    def inflate_state(
        self, state: tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]], n: int
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]: ...

    @abstractmethod
    def __call__(
        self,
        new_text: list[str],
        state: Optional[
            tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]
        ] = None,
    ) -> tuple[Tensor, tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]]: ...
