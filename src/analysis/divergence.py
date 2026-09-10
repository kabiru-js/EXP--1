"""Divergence analysis: investigating temporal signals in trajectories."""

from __future__ import annotations

import numpy as np
from scipy import stats


def normalize_trajectory_length(
    values: np.ndarray, target_length: int = 100
) -> np.ndarray:
    """Normalize a trajectory to a fixed number of points via interpolation."""
    if len(values) == 0:
        return np.zeros(target_length)
    if len(values) == 1:
        return np.full(target_length, values[0])
    x_orig = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_orig, values)


def compute_trajectory_curves(
    token_arrays: dict[str, dict[str, np.ndarray]],
    labels: dict[str, bool],
    n_points: int = 100,
) -> dict[str, np.ndarray]:
    """Compute mean trajectories for correct vs incorrect groups."""
    correct_entropies = []
    incorrect_entropies = []
    correct_probs = []
    incorrect_probs = []
    correct_margins = []
    incorrect_margins = []
    correct_cumlog = []
    incorrect_cumlog = []

    for sample_id, arrays in token_arrays.items():
        is_correct = labels.get(sample_id, False)
        target = [
            (correct_entropies if is_correct else incorrect_entropies, arrays["entropies"]),
            (correct_probs if is_correct else incorrect_probs, arrays["probabilities"]),
            (correct_margins if is_correct else incorrect_margins, arrays["margins"]),
            (correct_cumlog if is_correct else incorrect_cumlog, arrays["cumulative_log_probs"]),
        ]
        for lst, arr in target:
            if len(arr) > 0:
                lst.append(normalize_trajectory_length(arr, n_points))

    result = {}
    for name, lst in [
        ("correct_entropy", correct_entropies),
        ("incorrect_entropy", incorrect_entropies),
        ("correct_probability", correct_probs),
        ("incorrect_probability", incorrect_probs),
        ("correct_margin", correct_margins),
        ("incorrect_margin", incorrect_margins),
        ("correct_cumlog", correct_cumlog),
        ("incorrect_cumlog", incorrect_cumlog),
    ]:
        if lst:
            result[f"{name}_mean"] = np.mean(lst, axis=0)
            result[f"{name}_std"] = np.std(lst, axis=0)
            result[f"{name}_n"] = np.array([len(lst)])
        else:
            result[f"{name}_mean"] = np.zeros(n_points)
            result[f"{name}_std"] = np.zeros(n_points)
            result[f"{name}_n"] = np.array([0])

    return result


def find_divergence_point(
    correct_trajectory: np.ndarray,
    incorrect_trajectory: np.ndarray,
    significance: float = 0.05,
    min_region: int = 5,
) -> dict | None:
    """Find earliest point where correct and incorrect trajectories diverge.

    Uses sliding window t-tests to identify the first region of significant
    difference.
    """
    n = min(len(correct_trajectory), len(incorrect_trajectory))
    if n < min_region:
        return None

    for start in range(0, n - min_region + 1):
        end = start + min_region
        t_stat, p_val = stats.ttest_ind(
            correct_trajectory[start:end],
            incorrect_trajectory[start:end],
        )
        if p_val < significance and abs(t_stat) > 1.0:
            return {
                "divergence_index": start,
                "divergence_fraction": start / n,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "window_size": min_region,
            }
    return None


def compute_temporal_prediction_performance(
    token_arrays: dict[str, dict[str, np.ndarray]],
    labels: dict[str, bool],
    fractions: list[float],
) -> dict[float, dict[str, float]]:
    """At each fraction of generation, compute mean entropy difference.

    This measures how early we can distinguish correct from incorrect.
    """
    results = {}

    for frac in fractions:
        correct_vals = []
        incorrect_vals = []

        for sample_id, arrays in token_arrays.items():
            n = len(arrays["entropies"])
            cutoff = max(1, int(n * frac))
            mean_ent = float(np.mean(arrays["entropies"][:cutoff]))

            if labels.get(sample_id, False):
                correct_vals.append(mean_ent)
            else:
                incorrect_vals.append(mean_ent)

        if correct_vals and incorrect_vals:
            t_stat, p_val = stats.ttest_ind(correct_vals, incorrect_vals)
            cohens_d = (
                (np.mean(correct_vals) - np.mean(incorrect_vals))
                / np.sqrt(
                    (np.std(correct_vals) ** 2 + np.std(incorrect_vals) ** 2) / 2
                )
                if (np.std(correct_vals) ** 2 + np.std(incorrect_vals) ** 2) > 0
                else 0.0
            )
            results[frac] = {
                "correct_mean_entropy": float(np.mean(correct_vals)),
                "incorrect_mean_entropy": float(np.mean(incorrect_vals)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(cohens_d),
                "correct_n": len(correct_vals),
                "incorrect_n": len(incorrect_vals),
            }
        else:
            results[frac] = {
                "correct_mean_entropy": 0.0,
                "incorrect_mean_entropy": 0.0,
                "t_statistic": 0.0,
                "p_value": 1.0,
                "cohens_d": 0.0,
                "correct_n": len(correct_vals),
                "incorrect_n": len(incorrect_vals),
            }

    return results
