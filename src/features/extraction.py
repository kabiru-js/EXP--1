"""Feature extraction from trajectories for ML-based analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.trajectories.storage import TrajectoryStorage


def extract_trajectory_features(
    token_arrays: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Extract aggregate features from token-level trajectory data."""
    features = {}

    for sample_id, arrays in token_arrays.items():
        probs = arrays["probabilities"]
        log_probs = arrays["log_probabilities"]
        entropies = arrays["entropies"]
        margins = arrays["margins"]
        cum_log_probs = arrays["cumulative_log_probs"]
        n = len(probs)

        if n == 0:
            continue

        features[sample_id] = {
            "length": n,
            # Entropy features
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "max_entropy": float(np.max(entropies)),
            "min_entropy": float(np.min(entropies)),
            "entropy_trend": float(np.polyfit(np.arange(n), entropies, 1)[0]) if n > 1 else 0.0,
            "entropy_last_quarter": float(np.mean(entropies[n * 3 // 4:])) if n >= 4 else float(np.mean(entropies)),
            "entropy_first_quarter": float(np.mean(entropies[:n // 4])) if n >= 4 else float(np.mean(entropies)),
            # Probability features
            "mean_probability": float(np.mean(probs)),
            "std_probability": float(np.std(probs)),
            "min_probability": float(np.min(probs)),
            "max_probability": float(np.max(probs)),
            "probability_trend": float(np.polyfit(np.arange(n), probs, 1)[0]) if n > 1 else 0.0,
            "probability_last_quarter": float(np.mean(probs[n * 3 // 4:])) if n >= 4 else float(np.mean(probs)),
            # Margin features
            "mean_margin": float(np.mean(margins)),
            "std_margin": float(np.std(margins)),
            "min_margin": float(np.min(margins)),
            "margin_trend": float(np.polyfit(np.arange(n), margins, 1)[0]) if n > 1 else 0.0,
            # Log probability features
            "total_log_prob": float(cum_log_probs[-1]),
            "mean_step_log_prob": float(np.mean(log_probs)),
            "std_step_log_prob": float(np.std(log_probs)),
            "min_step_log_prob": float(np.min(log_probs)),
            # Final token features
            "final_probability": float(probs[-1]),
            "final_log_probability": float(log_probs[-1]),
            "final_entropy": float(entropies[-1]),
            "final_margin": float(margins[-1]),
            # Early features (first 25%)
            "early_mean_entropy": float(np.mean(entropies[:max(1, n // 4)])),
            "early_mean_probability": float(np.mean(probs[:max(1, n // 4)])),
            "early_mean_margin": float(np.mean(margins[:max(1, n // 4)])),
            # Acceleration / second-order
            "entropy_acceleration": float(
                np.polyfit(np.arange(n), entropies, 2)[0]
            ) if n > 2 else 0.0,
        }

    return features


def extract_temporal_features(
    token_arrays: dict[str, dict[str, np.ndarray]],
    fractions: list[float],
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract features at specific fractions of generation for early prediction."""
    temporal = {}

    for sample_id, arrays in token_arrays.items():
        probs = arrays["probabilities"]
        log_probs = arrays["log_probabilities"]
        entropies = arrays["entropies"]
        margins = arrays["margins"]
        n = len(probs)

        temporal[sample_id] = {}
        for frac in fractions:
            cutoff = max(1, int(n * frac))
            t_entropies = entropies[:cutoff]
            t_probs = probs[:cutoff]
            t_margins = margins[:cutoff]
            t_log_probs = log_probs[:cutoff]

            temporal[sample_id][f"frac_{frac:.1f}"] = {
                "mean_entropy": float(np.mean(t_entropies)),
                "std_entropy": float(np.std(t_entropies)),
                "mean_probability": float(np.mean(t_probs)),
                "mean_margin": float(np.mean(t_margins)),
                "min_margin": float(np.min(t_margins)),
                "total_log_prob": float(np.sum(t_log_probs)),
                "length": cutoff,
            }

    return temporal


def build_feature_matrix(
    features: dict[str, Any],
    labels: dict[str, bool],
    exclude_keys: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build a feature matrix and label vector from extracted features.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label vector (n_samples,)
        feature_names: list of feature names
        sample_ids: list of sample IDs in order
    """
    exclude_keys = exclude_keys or set()
    sample_ids = sorted(set(features.keys()) & set(labels.keys()))

    if not sample_ids:
        return np.array([]), np.array([]), [], []

    # Get feature names from first sample
    first = features[sample_ids[0]]
    feature_names = [k for k in sorted(first.keys()) if k not in exclude_keys]

    X = np.array([[features[sid][fn] for fn in feature_names] for sid in sample_ids])
    y = np.array([labels[sid] for sid in sample_ids], dtype=int)

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    return X, y, feature_names, sample_ids
