"""Configuration system for model trajectory research."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    split: str = "test"
    samples: int = 1000
    subset: Optional[str] = None
    cache_dir: str = "data/cache"


class ModelConfig(BaseModel):
    provider: str = "huggingface"
    name: str
    revision: Optional[str] = None
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = False


class GenerationConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 50
    seed: int = 42
    do_sample: bool = True
    num_return_sequences: int = 1


class FeatureConfig(BaseModel):
    token_probability: bool = True
    entropy: bool = True
    top_k: bool = True
    top_k_count: int = 10
    hidden_states: bool = False
    attention: bool = False
    logits: bool = True


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["auroc", "auprc", "accuracy", "calibration"]
    )
    temporal_fractions: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )


class ExperimentConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    prompt_template: str = "Solve step by step.\n\n{question}\n\nAnswer:"
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    baseline_models: list[str] = Field(default_factory=list)
    num_samples: Optional[int] = None

    def config_hash(self) -> str:
        raw = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        if "experiment" in data:
            data = data["experiment"]
        return cls(**data)


class StorageConfig(BaseModel):
    base_dir: Path = Path("experiments")
    trajectories_dir: Path = Path("trajectories")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("figures")


class ProjectConfig(BaseModel):
    experiment: ExperimentConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ProjectConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            experiment=ExperimentConfig(**data["experiment"]),
            storage=StorageConfig(**{k: Path(v) for k, v in data.get("storage", {}).items()}),
        )
