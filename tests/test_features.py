"""Tests for feature extraction and evaluation."""

import numpy as np
import pytest

from src.features.extraction import (
    build_feature_matrix,
    extract_temporal_features,
    extract_trajectory_features,
)
from src.evaluation.baselines import (
    GradientBoostedBaseline,
    LogisticRegressionBaseline,
    compute_metrics,
)


class TestFeatureExtraction:
    def _make_token_arrays(self, n_tokens=20, correct=True):
        rng = np.random.RandomState(42 if correct else 99)
        base_entropy = 1.0 if correct else 1.5
        base_prob = 0.7 if correct else 0.5
        return {
            "probabilities": rng.uniform(base_prob - 0.1, base_prob + 0.1, n_tokens),
            "log_probabilities": rng.uniform(-0.5, -0.1, n_tokens),
            "entropies": rng.normal(base_entropy, 0.2, n_tokens),
            "margins": rng.uniform(0.1, 0.5, n_tokens),
            "cumulative_log_probs": np.cumsum(rng.uniform(-0.5, -0.1, n_tokens)),
            "token_indices": np.arange(n_tokens),
            "token_ids": rng.randint(0, 1000, n_tokens),
        }

    def test_extract_features(self):
        token_arrays = {
            "s1": self._make_token_arrays(correct=True),
            "s2": self._make_token_arrays(correct=False),
        }
        features = extract_trajectory_features(token_arrays)
        assert "s1" in features
        assert "s2" in features
        assert "mean_entropy" in features["s1"]
        assert "final_probability" in features["s1"]

    def test_build_feature_matrix(self):
        token_arrays = {
            "s1": self._make_token_arrays(correct=True),
            "s2": self._make_token_arrays(correct=False),
            "s3": self._make_token_arrays(correct=True),
        }
        labels = {"s1": True, "s2": False, "s3": True}
        features = extract_trajectory_features(token_arrays)
        X, y, names, sids = build_feature_matrix(features, labels)

        assert X.shape == (3, len(names))
        assert y.shape == (3,)
        assert sum(y) == 2
        assert len(sids) == 3

    def test_temporal_features(self):
        token_arrays = {"s1": self._make_token_arrays(n_tokens=100)}
        temporal = extract_temporal_features(token_arrays, [0.25, 0.5, 0.75])
        assert "s1" in temporal
        assert "frac_0.3" in temporal["s1"]
        assert "mean_entropy" in temporal["s1"]["frac_0.3"]


class TestBaselines:
    def _make_data(self, n=100):
        rng = np.random.RandomState(42)
        X = rng.randn(n, 5)
        y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
        return X, y, ["mean_entropy", "mean_probability", "mean_margin", "std_entropy", "final_probability"]

    def test_logistic_regression(self):
        X, y, names = self._make_data()
        model = LogisticRegressionBaseline()
        model.fit(X[:80], y[:80], names)
        probs = model.predict_proba(X[80:], names)
        assert len(probs) == 20
        assert all(0 <= p <= 1 for p in probs)

    def test_gradient_boosted(self):
        X, y, names = self._make_data()
        model = GradientBoostedBaseline()
        model.fit(X[:80], y[:80], names)
        probs = model.predict_proba(X[80:], names)
        assert len(probs) == 20
        assert all(0 <= p <= 1 for p in probs)

    def test_compute_metrics(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
        metrics = compute_metrics(y_true, y_prob)
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["auroc"] <= 1
