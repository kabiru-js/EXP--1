"""Dataset registry and factory."""

from __future__ import annotations

from typing import Optional

from src.datasets.base import DatasetLoader, Sample
from src.datasets.gsm8k import GSM8KLoader
from src.datasets.synthetic import SyntheticArithmeticLoader, SyntheticSequenceLoader
from src.datasets.truthfulqa import TruthfulQALoader

_REGISTRY: dict[str, type[DatasetLoader]] = {
    "gsm8k": GSM8KLoader,
    "truthfulqa": TruthfulQALoader,
    "synthetic_arithmetic": SyntheticArithmeticLoader,
    "synthetic_sequences": SyntheticSequenceLoader,
}


def get_loader(name: str) -> DatasetLoader:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def load_dataset_by_name(
    name: str, split: str = "test", num_samples: Optional[int] = None
) -> list[Sample]:
    loader = get_loader(name)
    return loader.load(split=split, num_samples=num_samples)


def register_dataset(name: str, loader_cls: type[DatasetLoader]) -> None:
    _REGISTRY[name] = loader_cls


def available_datasets() -> list[str]:
    return list(_REGISTRY.keys())
