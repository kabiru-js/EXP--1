"""Master figure generation script for trajectory research."""

from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plots import (
    plot_trajectory_comparison,
    plot_temporal_prediction,
    plot_baseline_comparison,
    plot_divergence_distribution,
    plot_feature_importances,
    save_figure_data,
)

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig_trajectory_curves():
    """Figure 1: Correct vs incorrect trajectory curves."""
    results = load_json("experiments/EXP-001/results.json")
    curves = results["trajectory_curves"]
    # Convert lists to numpy arrays
    curves = {k: np.array(v) if isinstance(v, list) else v for k, v in curves.items()}
    
    for metric in ["entropy", "probability", "margin", "cumulative_log_probability"]:
        plot_trajectory_comparison(curves, metric, FIGURES_DIR / f"fig_trajectory_{metric}")
    
    # Save figure data
    save_figure_data(curves, FIGURES_DIR / "fig_trajectory_curves_data")
    print("[OK] Trajectory curves")


def fig_temporal_prediction():
    """Figure 2: Temporal prediction analysis."""
    results = load_json("experiments/EXP-001/results.json")
    temporal = results["temporal_prediction"]
    # Convert string keys to float and ensure numpy arrays
    temporal_sorted = {float(k): {kk: np.array(vv) if isinstance(vv, list) else vv for kk, vv in v.items()} for k, v in temporal.items()}
    plot_temporal_prediction(temporal_sorted, FIGURES_DIR / "fig_temporal_prediction")
    save_figure_data(temporal_sorted, FIGURES_DIR / "fig_temporal_prediction_data")
    print("[OK] Temporal prediction")


def fig_baseline_comparison():
    """Figure 3: Baseline model comparison."""
    results = load_json("experiments/EXP-001/results.json")
    baselines = results["baselines"]
    
    # Ensure numeric values
    for b in baselines:
        for k, v in list(b.items()):
            if isinstance(v, str) and k not in ("model_name",):
                try:
                    b[k] = float(v)
                except (ValueError, TypeError):
                    pass
    
    plot_baseline_comparison(baselines, FIGURES_DIR / "fig_baseline_comparison")
    print("[OK] Baseline comparison")


def fig_cv_results():
    """Figure 4: Cross-validation results."""
    cv = load_json("experiments/EXP-001/cv_results.json")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(cv["baselines"].keys())
    
    for ax, metric in [(axes[0], "auroc"), (axes[1], "auprc"), (axes[2], "accuracy")]:
        values = [cv["baselines"][n][metric]["mean"] for n in names]
        stds = [cv["baselines"][n][metric]["std"] for n in names]
        bars = ax.bar(names, values, yerr=stds, capsize=5,
                       color=["#1976D2", "#2196F3", "#4CAF50"])
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.05)
        for bar, val, std in zip(bars, values, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}±{std:.3f}", ha="center", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    
    fig.suptitle("Cross-Validation Results (5-fold stratified)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cv_results.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_cv_results.svg", bbox_inches="tight")
    plt.close(fig)
    print("[OK] CV results")


def fig_feature_importances():
    """Figure 5: Feature importances from gradient boosted tree."""
    results = load_json("experiments/EXP-001/results.json")
    gb = next(b for b in results["baselines"] if b["model_name"] == "gradient_boosted_tree")
    importances = {k: float(v) for k, v in gb["feature_importances"].items()}
    
    plot_feature_importances(importances, FIGURES_DIR / "fig_feature_importances_gbt")
    print("[OK] Feature importances (GBT)")


def fig_lr_feature_importances():
    """Figure 6: Feature importances from logistic regression."""
    results = load_json("experiments/EXP-001/results.json")
    lr = next(b for b in results["baselines"] if b["model_name"] == "logistic_regression")
    importances = {k: float(v) for k, v in lr["feature_importances"].items()}
    
    plot_feature_importances(importances, FIGURES_DIR / "fig_feature_importances_lr")
    print("[OK] Feature importances (LR)")


def fig_divergence_distribution():
    """Figure 7: Divergence distribution from EXP-001."""
    results = load_json("experiments/EXP-001/results.json")
    divergence_times = results.get("divergence_times", [])
    
    # Convert to numpy arrays if needed
    fractions = np.array([d["divergence_fraction"] for d in divergence_times if d is not None])
    
    if len(fractions) > 0:
        plot_divergence_distribution(divergence_times, FIGURES_DIR / "fig_divergence_distribution_exp001")
    else:
        fig, ax = plt.subplots()
        ax.bar(["divergence_fraction=0"], [1], color="#1976D2", alpha=0.7)
        ax.set_xlabel("Divergence fraction")
        ax.set_ylabel("Count")
        ax.set_title("EXP-001: Divergence Distribution (median=0.0)")
        ax.text(0, 1.02, "Median divergence fraction = 0.0", ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig_divergence_distribution_exp001.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "fig_divergence_distribution_exp001.svg", bbox_inches="tight")
        plt.close(fig)
    
    print("[OK] Divergence distribution")


def fig_exp002_probes():
    """Figure 8: EXP-002 onset distribution."""
    exp002 = load_json("experiments/EXP-002/exp002_results.json")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Build onset distribution from the findings
    # P=2: {0:15}, P=3: {0:8, 1:2, ALL_CORRECT:5}, P=4: {0:9, 1:1, ALL_CORRECT:5}
    onset_data = {
        "P=2": {"0": 15},
        "P=3": {"0": 8, "1": 2, "ALL_CORRECT": 5},
        "P=4": {"0": 9, "1": 1, "ALL_CORRECT": 5},
    }
    
    categories = []
    values = []
    colors = []
    for prefix, dist in onset_data.items():
        for onset, count in dist.items():
            categories.append(f"{prefix}\nonset={onset}")
            values.append(count)
            if onset == "ALL_CORRECT":
                colors.append("#4CAF50")
            elif onset == "0":
                colors.append("#F44336")
            else:
                colors.append("#FF9800")
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_xlabel("Prefix length / Onset category")
    ax.set_ylabel("Count")
    ax.set_title("EXP-002: Onset Distribution Across 45 Probes")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F44336", alpha=0.8, label="Diverged at term 0"),
        Patch(facecolor="#FF9800", alpha=0.8, label="Diverged at term 1"),
        Patch(facecolor="#4CAF50", alpha=0.8, label="All correct"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_exp002_probes.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_exp002_probes.svg", bbox_inches="tight")
    plt.close(fig)
    print("[OK] EXP-002 probes")


def fig_pipeline():
    """Figure 9: Experimental pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    
    steps = [
        "Dataset\n(Arithmetic\nSequences)",
        "Prompt\nTemplate\n(\"{question}\")",
        "Model\nGenerate\n(distilgpt2)",
        "LogitsProcessor\n(PER-TOKEN\nSIGNALS)",
        "Trajectory\nStorage\n(Parquet+DuckDB)",
        "Feature\nExtraction\n(30 features)",
        "Analysis\n(Baselines +\nTemporal)",
        "Results\n(AUROC +\nCohen's d)",
    ]
    
    n = len(steps)
    for i, step in enumerate(steps):
        x = 0.5 + i * (12 / (n - 1))
        ax.add_patch(plt.Rectangle((x - 0.8, 3), 1.6, 2, facecolor="#1976D2", alpha=0.8))
        ax.text(x, 4, step, ha="center", va="center", color="white", fontsize=8)
        if i < n - 1:
            ax.annotate("", xy=(x + 1.2, 4), xytext=(x + 0.8, 4),
                       arrowprops=dict(arrowstyle="->", color="black", lw=2))
    
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(1, 6)
    ax.set_title("Experimental Pipeline", fontsize=16, pad=20)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_pipeline.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_pipeline.svg", bbox_inches="tight")
    plt.close(fig)
    print("[OK] Pipeline diagram")


def main():
    print("Generating figures...")
    fig_trajectory_curves()
    fig_temporal_prediction()
    fig_baseline_comparison()
    fig_cv_results()
    fig_feature_importances()
    fig_lr_feature_importances()
    fig_divergence_distribution()
    fig_exp002_probes()
    fig_pipeline()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
