"""Model backend registry and factory."""

from __future__ import annotations

import logging
from typing import Optional

from src.models.base import ModelBackend

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type[ModelBackend]] = {}


def register_backend(name: str, cls: type[ModelBackend]) -> None:
    _BACKENDS[name] = cls


def get_backend(name: str, **kwargs) -> ModelBackend:
    if name == "huggingface":
        from src.models.huggingface import HuggingFaceBackend
        return HuggingFaceBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Available: {list(_BACKENDS.keys())}")


def available_backends() -> list[str]:
    return list(_BACKENDS.keys()) + ["huggingface"]
