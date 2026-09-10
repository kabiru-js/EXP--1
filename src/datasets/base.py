"""Base dataset interface for trajectory research."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Sample:
    """A single evaluation sample."""

    sample_id: str
    dataset: str
    question: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetLoader(abc.ABC):
    """Abstract base for dataset loaders."""

    @abc.abstractmethod
    def load(self, split: str, num_samples: Optional[int] = None) -> list[Sample]:
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def extract_answer(self, raw_answer: str) -> str:
        """Extract the final answer from model output."""
        ...

    @abc.abstractmethod
    def check_correctness(self, predicted: str, ground_truth: str) -> bool:
        ...
