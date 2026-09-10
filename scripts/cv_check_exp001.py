"""5-fold stratified cross-validation sanity check for EXP-001 baselines.

Motivation: gradient_boosted_tree reached AUROC=1.0000 on a single
80/20 split. This script re-evaluates all baselines with proper
cross-validation to check whether the result reflects genuine separable
signal or over-fitting to the single split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.evaluation.baselines import (
    EntropyThresholdBaseline,
    GradientBoostedBaseline,
    LogisticRegressionBaseline,
    compute_metrics,
)
from src.features.extraction import build_feature_matrix, extract_trajectory_features
from src.trajectories.storage import TrajectoryStorage

BASELINE_CLASSES = {
    "entropy_threshold": EntropyThresholdBaseline,
    "logistic_regression": LogisticRegressionBaseline,
    "gradient_boosted_tree": GradientBoostedBaseline,
}


def main() -> None:
    exp = "EXP-001"
    exp_dir = Path("experiments") / exp

    storage = TrajectoryStorage(exp_dir)
    token_arrays = storage.load_all_token_arrays(exp)
    labels = storage.get_correctness_labels(exp)
    features = extract_trajectory_features(token_arrays)
    X, y, feature_names, sample_ids = build_feature_matrix(features, labels)

    if len(X) == 0:
        print("No valid feature matrix")
        sys.exit(1)

    print(f"Loaded {len(X)} samples, {X.shape[1]} features")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict] = {}

    for name, cls in BASELINE_CLASSES.items():
        per_fold = {"auroc": [], "auprc": [], "accuracy": [], "balanced_accuracy": []}
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            baseline = cls()
            baseline.fit(X_train, y_train, feature_names)
            y_prob = baseline.predict_proba(X_test, feature_names)
            metrics = compute_metrics(y_test, y_prob)

            per_fold["auroc"].append(metrics["auroc"])
            per_fold["auprc"].append(metrics["auprc"])
            per_fold["accuracy"].append(metrics["accuracy"])
            per_fold["balanced_accuracy"].append(metrics["balanced_accuracy"])

        summary = {}
        for metric, values in per_fold.items():
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_fold": [float(v) for v in values],
            }
        results[name] = summary
        print(
            f"  {name}: AUROC={summary['auroc']['mean']:.4f} +/- {summary['auroc']['std']:.4f}, "
            f"AUPRC={summary['auprc']['mean']:.4f} +/- {summary['auprc']['std']:.4f}"
        )

    out_path = exp_dir / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump({"experiment_id": exp, "n_splits": 5, "baselines": results}, f, indent=2)
    print(f"Saved cross-validation results to {out_path}")


if __name__ == "__main__":
    main()