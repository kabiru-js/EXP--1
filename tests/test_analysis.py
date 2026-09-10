"""Tests for analysis module."""

import numpy as np
import pytest

from src.analysis.divergence import (
    compute_temporal_prediction_performance,
    compute_trajectory_curves,
    find_divergence_point,
    normalize_trajectory_length,
)


class TestNormalize:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = normalize_trajectory_length(arr, 5)
        assert len(result) == 5
        assert result[0] == 1.0
        assert result[-1] == 3.0

    def test_empty(self):
        result = normalize_trajectory_length(np.array([]), 5)
        assert len(result) == 5
        assert all(v == 0.0 for v in result)

    def test_single(self):
        result = normalize_trajectory_length(np.array([5.0]), 5)
        assert all(v == 5.0 for v in result)


class TestDivergencePoint:
    def test_clear_divergence(self):
        rng = np.random.RandomState(42)
        corr = np.concatenate([
            rng.normal(1.0, 0.1, 50),
            rng.normal(1.0, 0.1, 50),
        ])
        incorr = np.concatenate([
            rng.normal(1.0, 0.1, 50),
            rng.normal(2.0, 0.1, 50),
        ])
        result = find_divergence_point(corr, incorr)
        assert result is not None
        assert result["divergence_index"] >= 45

    def test_no_divergence(self):
        rng = np.random.RandomState(42)
        corr = rng.normal(1.0, 0.1, 100)
        incorr = rng.normal(1.0, 0.1, 100)
        result = find_divergence_point(corr, incorr)
        assert result is None


class TestTrajectoryCurves:
    def test_basic(self):
        token_arrays = {
            "s1": {"entropies": np.ones(50), "probabilities": np.ones(50) * 0.8,
                   "margins": np.ones(50) * 0.3, "cumulative_log_probs": np.arange(50) * -0.2},
            "s2": {"entropies": np.ones(50) * 2, "probabilities": np.ones(50) * 0.4,
                   "margins": np.ones(50) * 0.1, "cumulative_log_probs": np.arange(50) * -0.5},
        }
        labels = {"s1": True, "s2": False}
        curves = compute_trajectory_curves(token_arrays, labels, n_points=20)
        assert "correct_entropy_mean" in curves
        assert "incorrect_entropy_mean" in curves
        assert len(curves["correct_entropy_mean"]) == 20


class TestTemporalPrediction:
    def test_basic(self):
        rng = np.random.RandomState(42)
        token_arrays = {}
        labels = {}
        for i in range(50):
            sid = f"s_{i}"
            is_correct = i < 25
            base = 1.0 if is_correct else 2.0
            token_arrays[sid] = {
                "entropies": rng.normal(base, 0.3, 50),
                "probabilities": rng.uniform(0.3, 0.8, 50),
                "margins": rng.uniform(0.1, 0.4, 50),
                "cumulative_log_probs": np.cumsum(rng.normal(-0.3, 0.1, 50)),
            }
            labels[sid] = is_correct

        result = compute_temporal_prediction_performance(
            token_arrays, labels, [0.2, 0.5, 0.8]
        )
        assert 0.2 in result
        assert 0.5 in result
        assert 0.8 in result
        for frac, metrics in result.items():
            assert "cohens_d" in metrics
            assert "p_value" in metrics
