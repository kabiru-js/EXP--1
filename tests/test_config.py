"""Tests for experiment configuration."""

import tempfile
from pathlib import Path

import pytest

from src.config import (
    DatasetConfig,
    ExperimentConfig,
    FeatureConfig,
    GenerationConfig,
    ModelConfig,
)


class TestExperimentConfig:
    def test_creation(self):
        config = ExperimentConfig(
            id="TEST-001",
            name="test",
            dataset=DatasetConfig(name="gsm8k", samples=100),
            model=ModelConfig(name="test-model"),
            generation=GenerationConfig(seed=42),
        )
        assert config.id == "TEST-001"
        assert config.dataset.samples == 100

    def test_yaml_roundtrip(self):
        config = ExperimentConfig(
            id="TEST-002",
            name="test-roundtrip",
            dataset=DatasetConfig(name="gsm8k", samples=50),
            model=ModelConfig(name="test-model"),
            generation=GenerationConfig(seed=123),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            config.to_yaml(path)
            loaded = ExperimentConfig.from_yaml(path)
            assert loaded.id == "TEST-002"
            assert loaded.generation.seed == 123

    def test_config_hash(self):
        config = ExperimentConfig(
            id="TEST-003",
            name="test-hash",
            dataset=DatasetConfig(name="gsm8k"),
            model=ModelConfig(name="model"),
            generation=GenerationConfig(seed=42),
        )
        h = config.config_hash()
        assert len(h) == 12
        # Same config should produce same hash
        config2 = ExperimentConfig(
            id="TEST-003",
            name="test-hash",
            dataset=DatasetConfig(name="gsm8k"),
            model=ModelConfig(name="model"),
            generation=GenerationConfig(seed=42),
        )
        assert config2.config_hash() == h

    def test_default_features(self):
        config = ExperimentConfig(
            id="TEST-004",
            name="test-defaults",
            dataset=DatasetConfig(name="gsm8k"),
            model=ModelConfig(name="model"),
            generation=GenerationConfig(),
        )
        assert config.features.token_probability is True
        assert config.features.entropy is True
        assert config.features.top_k_count == 10
