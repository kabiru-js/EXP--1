"""Baseline models for correctness prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class EvaluationResult:
    """Results from a baseline evaluation."""

    model_name: str
    auroc: float
    auprc: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    calibration_error: float
    feature_importances: Optional[dict[str, float]] = None
    details: dict[str, Any] = None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute standard classification metrics."""
    y_pred = (y_prob >= 0.5).astype(int)

    n_true = int(np.sum(y_true))
    n_total = len(y_true)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = 0.5

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = float(n_true / n_total) if n_total > 0 else 0.5

    # Simple calibration error
    bin_edges = np.linspace(0, 1, 11)
    cal_error = 0.0
    for i in range(len(bin_edges) - 1):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_true_mean = y_true[mask].mean()
            bin_prob_mean = y_prob[mask].mean()
            cal_error += mask.sum() / len(y_true) * abs(bin_true_mean - bin_prob_mean)
    metrics["calibration_error"] = float(cal_error)

    return metrics


class EntropyThresholdBaseline:
    """Baseline using entropy thresholding."""

    def __init__(self):
        self.threshold: float = 0.0
        self.name = "entropy_threshold"

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        entropy_idx = feature_names.index("mean_entropy")
        correct_entropies = X[y == 1, entropy_idx]
        incorrect_entropies = X[y == 0, entropy_idx]

        if len(correct_entropies) > 0 and len(incorrect_entropies) > 0:
            self.threshold = float(
                np.median(correct_entropies) + np.std(correct_entropies)
            )
        else:
            self.threshold = float(np.median(X[:, entropy_idx]))

    def predict_proba(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        entropy_idx = feature_names.index("mean_entropy")
        raw = 1.0 - (X[:, entropy_idx] / (self.threshold + 1e-10))
        return np.clip(raw, 0.0, 1.0)


class LogisticRegressionBaseline:
    """Logistic regression baseline with standardized features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.name = "logistic_regression"
        self._feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        coefs = self.model.coef_[0]
        return {name: float(abs(c)) for name, c in zip(self._feature_names, coefs)}


class GradientBoostedBaseline:
    """Gradient-boosted tree baseline."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        self.name = "gradient_boosted_tree"
        self._feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self._feature_names, importances)}


def run_baseline(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> EvaluationResult:
    """Run a baseline model and evaluate."""
    if name == "entropy_threshold":
        baseline = EntropyThresholdBaseline()
        baseline.fit(X_train, y_train, feature_names)
        y_prob = baseline.predict_proba(X_test, feature_names)
    elif name == "logistic_regression":
        baseline = LogisticRegressionBaseline()
        baseline.fit(X_train, y_train, feature_names)
        y_prob = baseline.predict_proba(X_test, feature_names)
    elif name == "gradient_boosted_tree":
        baseline = GradientBoostedBaseline()
        baseline.fit(X_train, y_train, feature_names)
        y_prob = baseline.predict_proba(X_test, feature_names)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    metrics = compute_metrics(y_test, y_prob)

    feature_importances = None
    if hasattr(baseline, "get_feature_importances"):
        feature_importances = baseline.get_feature_importances()

    return EvaluationResult(
        model_name=name,
        auroc=metrics["auroc"],
        auprc=metrics["auprc"],
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        brier_score=metrics["brier_score"],
        calibration_error=metrics["calibration_error"],
        feature_importances=feature_importances,
    )
