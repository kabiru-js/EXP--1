"""Model abstraction layer for trajectory research."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class TokenSignal:
    """Per-token measurement."""

    token_id: int
    token_text: str
    token_index: int
    probability: float
    log_probability: float
    entropy: float
    top_k_token_ids: list[int]
    top_k_token_texts: list[str]
    top_k_probabilities: list[float]
    probability_margin: float
    cumulative_log_prob: float
    logits: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None


@dataclass
class GenerationResult:
    """Result of a single generation with per-token instrumentation."""

    text: str
    token_ids: list[int]
    token_texts: list[str]
    signals: list[TokenSignal]
    model_id: str
    model_version: str
    available_signals: list[str]
    prompt: str
    prompt_token_count: int
    generation_tokens: int
    seed: int


class ModelBackend(abc.ABC):
    """Abstract model interface for trajectory-aware generation."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        seed: int = 42,
        num_return_sequences: int = 1,
    ) -> list[GenerationResult]:
        """Generate text and return instrumented results."""
        ...

    @abc.abstractmethod
    def model_id(self) -> str:
        ...

    @abc.abstractmethod
    def model_version(self) -> str:
        ...

    @abc.abstractmethod
    def available_signals(self) -> list[str]:
        """List of signal types this model backend supports."""
        ...

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[int]:
        ...

    @abc.abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        ...
