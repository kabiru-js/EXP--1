"""Tests for trajectory storage."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.trajectories.storage import (
    Trajectory,
    TrajectoryStorage,
    TokenMetrics,
)


class TestTrajectoryStorage:
    def _make_trajectory(self, sample_id: str = "test_0") -> Trajectory:
        token_metrics = [
            TokenMetrics(
                token_id=i,
                token_text=f"tok_{i}",
                token_index=i,
                probability=np.random.uniform(0.1, 0.9),
                log_probability=np.random.uniform(-2.0, -0.1),
                entropy=np.random.uniform(0.5, 3.0),
                probability_margin=np.random.uniform(0.05, 0.5),
                cumulative_log_prob=-i * 0.3,
            )
            for i in range(10)
        ]
        return Trajectory(
            sample_id=sample_id,
            dataset="test_dataset",
            model_id="test_model",
            model_version="0.1",
            experiment_id="TEST-001",
            question="What is 2+2?",
            ground_truth="4",
            prompt="Solve: What is 2+2?",
            prompt_token_count=8,
            generated_text="4",
            extracted_answer="4",
            is_correct=True,
            generation_tokens=10,
            temperature=0.7,
            seed=42,
            prompt_template="{question}",
            token_metrics=token_metrics,
            available_signals=["probability", "entropy", "logits"],
        )

    def test_store_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TrajectoryStorage(tmpdir)
            traj = self._make_trajectory()
            storage.store(traj)

            loaded = storage.load_trajectory("test_0")
            assert loaded is not None
            assert loaded.sample_id == "test_0"
            assert loaded.is_correct is True
            assert len(loaded.token_metrics) == 10

    def test_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TrajectoryStorage(tmpdir)
            for i in range(5):
                storage.store(self._make_trajectory(f"test_{i}"))
            assert storage.count("TEST-001") == 5

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TrajectoryStorage(tmpdir)
            storage.store(self._make_trajectory("t1"))
            traj = self._make_trajectory("t2")
            traj.is_correct = False
            storage.store(traj)
            summary = storage.summary("TEST-001")
            assert summary["total"] == 2
            assert summary["correct"] == 1

    def test_load_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TrajectoryStorage(tmpdir)
            storage.store(self._make_trajectory())
            table = storage.load_parquet("test_0")
            assert table is not None
            assert len(table) == 10

    def test_load_all_token_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TrajectoryStorage(tmpdir)
            for i in range(3):
                storage.store(self._make_trajectory(f"t_{i}"))
            arrays = storage.load_all_token_arrays("TEST-001")
            assert len(arrays) == 3
            for sid, a in arrays.items():
                assert "probabilities" in a
                assert "entropies" in a
                assert len(a["probabilities"]) == 10
