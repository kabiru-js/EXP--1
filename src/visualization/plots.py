"""Publication-quality visualization for trajectory research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _setup_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


_setup_style()

COLORS = {
    "correct": "#2196F3",
    "incorrect": "#F44336",
    "baseline": "#9E9E9E",
    "highlight": "#FF9800",
    "primary": "#1976D2",
    "secondary": "#7B1FA2",
}


def plot_trajectory_comparison(
    curves: dict[str, np.ndarray],
    metric_name: str,
    save_path: Path | str,
    title: str | None = None,
) -> None:
    """Plot correct vs incorrect trajectories for a given metric."""
    fig, ax = plt.subplots()

    x = np.linspace(0, 100, len(curves.get(f"correct_{metric_name}_mean", [])))
    if len(x) > 0 and f"correct_{metric_name}_mean" in curves:
        mean_c = curves[f"correct_{metric_name}_mean"]
        std_c = curves[f"correct_{metric_name}_std"]
        ax.plot(x, mean_c, color=COLORS["correct"], label="Correct")
        ax.fill_between(x, mean_c - std_c, mean_c + std_c, color=COLORS["correct"], alpha=0.15)

    x = np.linspace(0, 100, len(curves.get(f"incorrect_{metric_name}_mean", [])))
    if len(x) > 0 and f"incorrect_{metric_name}_mean" in curves:
        mean_i = curves[f"incorrect_{metric_name}_mean"]
        std_i = curves[f"incorrect_{metric_name}_std"]
        ax.plot(x, mean_i, color=COLORS["incorrect"], label="Incorrect")
        ax.fill_between(x, mean_i - std_i, mean_i + std_i, color=COLORS["incorrect"], alpha=0.15)

    ax.set_xlabel("Generation progress (%)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title or f"{metric_name.replace('_', ' ').title()}: Correct vs Incorrect")
    ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_temporal_prediction(
    temporal_results: dict[float, dict[str, float]],
    save_path: Path | str,
) -> None:
    """Plot prediction quality as function of generation observed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    fractions = sorted(temporal_results.keys())
    cohens_ds = [temporal_results[f]["cohens_d"] for f in fractions]
    p_values = [temporal_results[f]["p_value"] for f in fractions]
    correct_means = [temporal_results[f]["correct_mean_entropy"] for f in fractions]
    incorrect_means = [temporal_results[f]["incorrect_mean_entropy"] for f in fractions]

    # Panel 1: Mean entropy over time
    ax1.plot([f * 100 for f in fractions], correct_means, "o-", color=COLORS["correct"], label="Correct")
    ax1.plot([f * 100 for f in fractions], incorrect_means, "s-", color=COLORS["incorrect"], label="Incorrect")
    ax1.set_xlabel("Generation observed (%)")
    ax1.set_ylabel("Mean entropy")
    ax1.set_title("Entropy by generation fraction")
    ax1.legend()

    # Panel 2: Effect size over time
    ax2.plot([f * 100 for f in fractions], cohens_ds, "D-", color=COLORS["primary"])
    ax2.axhline(y=0, color=COLORS["baseline"], linestyle="--", alpha=0.5)
    ax2.set_xlabel("Generation observed (%)")
    ax2.set_ylabel("Cohen's d")
    ax2.set_title("Effect size (correct vs incorrect)")

    fig.suptitle("Early Prediction: How early can we detect incorrect trajectories?", fontsize=14)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(
    results: list[dict[str, Any]],
    save_path: Path | str,
) -> None:
    """Plot baseline model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [r["model_name"] for r in results]
    aurocs = [r["auroc"] for r in results]
    auprcs = [r["auprc"] for r in results]
    accs = [r["accuracy"] for r in results]

    for ax, values, metric in [
        (axes[0], aurocs, "AUROC"),
        (axes[1], auprcs, "AUPRC"),
        (axes[2], accs, "Accuracy"),
    ]:
        bars = ax.bar(names, values, color=[COLORS["primary"] for _ in names])
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Baseline Model Comparison", fontsize=14)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_individual_trajectories(
    token_arrays: dict[str, dict[str, np.ndarray]],
    labels: dict[str, bool],
    sample_ids: list[str],
    save_path: Path | str,
) -> None:
    """Plot individual trajectory examples."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ["entropies", "probabilities", "margins"]
    titles = ["Entropy", "Probability", "Margin"]

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        for sid in sample_ids:
            if sid not in token_arrays:
                continue
            arr = token_arrays[sid][metric]
            color = COLORS["correct"] if labels.get(sid, False) else COLORS["incorrect"]
            style = "-" if labels.get(sid, False) else "--"
            ax.plot(arr, style, color=color, alpha=0.6, linewidth=1)
        ax.set_xlabel("Token index")
        ax.set_ylabel(title)
        ax.set_title(title)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["correct"], label="Correct"),
        Line2D([0], [0], color=COLORS["incorrect"], linestyle="--", label="Incorrect"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2)
    fig.suptitle("Individual Trajectory Examples", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_divergence_distribution(
    divergence_times: list[dict],
    save_path: Path | str,
) -> None:
    """Plot distribution of estimated divergence times."""
    fig, ax = plt.subplots()

    fractions = [d["divergence_fraction"] for d in divergence_times if d is not None]
    if fractions:
        ax.hist(fractions, bins=20, color=COLORS["primary"], alpha=0.7, edgecolor="white")
        ax.axvline(x=np.mean(fractions), color=COLORS["highlight"], linestyle="--",
                    label=f"Mean: {np.mean(fractions):.2f}")
        ax.axvline(x=np.median(fractions), color=COLORS["secondary"], linestyle="--",
                    label=f"Median: {np.median(fractions):.2f}")
        ax.legend()

    ax.set_xlabel("Estimated divergence fraction")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Estimated Divergence Points")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_feature_importances(
    importances: dict[str, float],
    save_path: Path | str,
    top_n: int = 15,
) -> None:
    """Plot feature importances from tree-based model."""
    fig, ax = plt.subplots()

    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_features][::-1]
    values = [f[1] for f in sorted_features][::-1]

    ax.barh(names, values, color=COLORS["primary"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_figure_data(data: dict, save_path: Path | str) -> None:
    """Save underlying figure data as JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v
    with open(save_path.with_suffix(".json"), "w") as f:
        json.dump(serializable, f, indent=2)
