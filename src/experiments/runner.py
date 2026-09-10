"""Experiment runner: orchestrates full experiment lifecycle."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.analysis.divergence import (
    compute_temporal_prediction_performance,
    compute_trajectory_curves,
    find_divergence_point,
    normalize_trajectory_length,
)
from src.config import ExperimentConfig
from src.evaluation.baselines import (
    GradientBoostedBaseline,
    LogisticRegressionBaseline,
    compute_metrics,
    run_baseline,
)
from src.features.extraction import (
    build_feature_matrix,
    extract_temporal_features,
    extract_trajectory_features,
)
from src.inference.pipeline import InferencePipeline
from src.reporting.report import ExperimentReport
from src.trajectories.storage import TrajectoryStorage
from src.visualization.plots import (
    plot_baseline_comparison,
    plot_divergence_distribution,
    plot_feature_importances,
    plot_individual_trajectories,
    plot_temporal_prediction,
    plot_trajectory_comparison,
    save_figure_data,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs a complete experiment from config to report."""

    def __init__(self, config: ExperimentConfig, base_dir: str | Path = "experiments"):
        self.config = config
        self.base_dir = Path(base_dir) / config.id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.storage = TrajectoryStorage(self.base_dir)
        self.figures_dir = self.base_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run(self, force: bool = False, resume: bool = False) -> dict[str, Any]:
        """Run the complete experiment pipeline.

        Args:
            force: Re-run everything, overwriting existing trajectories.
            resume: Skip samples that already have stored trajectories.
        """
        existing = self.storage.count(self.config.id)
        if existing > 0 and not force and not resume:
            logger.info(f"Experiment {self.config.id} already has {existing} trajectories. Use force=True to rerun.")
            return self.analyze()

        if force and existing > 0:
            logger.info(f"Force re-run requested: clearing {existing} existing trajectories")
            self.storage.clear(self.config.id)

        logger.info(f"Starting experiment {self.config.id}")

        # Step 1: Run inference
        t0 = time.time()
        pipeline = InferencePipeline(self.config, self.storage)
        trajectories = pipeline.run(
            num_samples=self.config.num_samples, resume=resume or not force
        )
        inference_time = time.time() - t0
        logger.info(f"Inference completed in {inference_time:.1f}s")

        # Step 2: Analyze
        results = self.analyze()
        results["inference_time_seconds"] = inference_time
        return results

    def analyze(self) -> dict[str, Any]:
        """Analyze stored trajectories for an experiment."""
        logger.info(f"Analyzing experiment {self.config.id}")

        # Load data
        token_arrays = self.storage.load_all_token_arrays(self.config.id)
        labels = self.storage.get_correctness_labels(self.config.id)
        summary = self.storage.summary(self.config.id)

        if not token_arrays:
            logger.warning("No trajectories found")
            return {"error": "No trajectories found"}

        # Step 1: Extract aggregate features
        features = extract_trajectory_features(token_arrays)
        X, y, feature_names, sample_ids = build_feature_matrix(features, labels)

        if len(X) == 0:
            return {"error": "No valid feature matrix"}

        results: dict[str, Any] = {
            "experiment_id": self.config.id,
            "summary": summary,
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }

        # Step 2: Baseline models
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, sample_ids, test_size=0.2, random_state=42, stratify=y if sum(y) > 1 and sum(y) < len(y) - 1 else None
        )

        baseline_results = []
        for baseline_name in self.config.baseline_models:
            try:
                result = run_baseline(
                    baseline_name, X_train, y_train, X_test, y_test, feature_names
                )
                baseline_results.append({
                    "model_name": result.model_name,
                    "auroc": result.auroc,
                    "auprc": result.auprc,
                    "accuracy": result.accuracy,
                    "balanced_accuracy": result.balanced_accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "brier_score": result.brier_score,
                    "calibration_error": result.calibration_error,
                    "feature_importances": result.feature_importances,
                })
                logger.info(
                    f"  {baseline_name}: AUROC={result.auroc:.4f}, "
                    f"AUPRC={result.auprc:.4f}, Acc={result.accuracy:.4f}"
                )
            except Exception as e:
                logger.error(f"Baseline {baseline_name} failed: {e}")
                baseline_results.append({"model_name": baseline_name, "error": str(e)})

        results["baselines"] = baseline_results

        # Step 3: Trajectory curves
        curves = compute_trajectory_curves(token_arrays, labels, n_points=100)
        results["trajectory_curves"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in curves.items()
        }

        # Step 4: Temporal prediction performance
        temporal = compute_temporal_prediction_performance(
            token_arrays, labels, self.config.evaluation.temporal_fractions
        )
        results["temporal_prediction"] = {
            str(k): v for k, v in temporal.items()
        }

        # Step 5: Divergence analysis
        correct_arrays = {sid: a for sid, a in token_arrays.items() if labels.get(sid, False)}
        incorrect_arrays = {sid: a for sid, a in token_arrays.items() if not labels.get(sid, False)}

        divergence_points = []
        for sid in list(correct_arrays.keys())[:50]:
            if sid in incorrect_arrays:
                continue
            corr_mean = normalize_trajectory_length(correct_arrays[sid]["entropies"], 100)
            for isid, iarr in incorrect_arrays.items():
                incorr_mean = normalize_trajectory_length(iarr["entropies"], 100)
                dp = find_divergence_point(corr_mean, incorr_mean)
                if dp:
                    dp["correct_sample"] = sid
                    dp["incorrect_sample"] = isid
                    divergence_points.append(dp)
                break  # One comparison per correct sample

        results["divergence_analysis"] = {
            "n_comparisons": len(divergence_points),
            "divergence_points": divergence_points[:100],
        }

        if divergence_points:
            frac = [d["divergence_fraction"] for d in divergence_points]
            results["divergence_analysis"]["mean_fraction"] = float(np.mean(frac))
            results["divergence_analysis"]["median_fraction"] = float(np.median(frac))

        # Step 6: Generate visualizations
        self._generate_figures(curves, temporal, baseline_results, token_arrays, labels, divergence_points)

        # Step 7: Save results JSON
        results_path = self.base_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Step 8: Generate report
        self._generate_report(results, summary)

        logger.info(f"Analysis complete for {self.config.id}")
        return results

    def _generate_figures(
        self,
        curves: dict,
        temporal: dict,
        baselines: list[dict],
        token_arrays: dict,
        labels: dict,
        divergence_points: list,
    ) -> None:
        fig_dir = self.figures_dir

        # Figure 1: Trajectory comparison
        for metric in ["entropy", "probability", "margin", "cumlog"]:
            plot_trajectory_comparison(
                curves, metric, fig_dir / f"fig_trajectory_{metric}",
                title=f"{metric.title()}: Correct vs Incorrect Trajectories"
            )

        # Figure 2: Temporal prediction
        temporal_float = {float(k): v for k, v in temporal.items()}
        plot_temporal_prediction(temporal_float, fig_dir / "fig_temporal_prediction")

        # Figure 3: Baseline comparison
        valid_baselines = [b for b in baselines if "error" not in b]
        if valid_baselines:
            plot_baseline_comparison(valid_baselines, fig_dir / "fig_baseline_comparison")

        # Figure 4: Individual trajectories
        sample_ids = list(token_arrays.keys())[:10]
        plot_individual_trajectories(token_arrays, labels, sample_ids, fig_dir / "fig_individual_trajectories")

        # Figure 5: Divergence distribution
        if divergence_points:
            plot_divergence_distribution(divergence_points, fig_dir / "fig_divergence_distribution")

        # Figure 6: Feature importances
        for b in valid_baselines:
            if "feature_importances" in b and b["feature_importances"]:
                plot_feature_importances(
                    b["feature_importances"],
                    fig_dir / f"fig_feature_importances_{b['model_name']}",
                )

        # Save figure data
        save_figure_data(curves, fig_dir / "data_trajectory_curves")
        save_figure_data({str(k): v for k, v in temporal.items()}, fig_dir / "data_temporal_prediction")

    def _generate_report(self, results: dict, summary: dict) -> None:
        report = ExperimentReport(self.config.id, "reports")

        report.set_metadata(
            hypothesis=(
                "Incorrect generations may exhibit measurable changes in their "
                "inference trajectory before the final answer is produced."
            ),
            setup={
                "dataset": self.config.dataset.name,
                "split": self.config.dataset.split,
                "samples": summary["total"],
                "model": self.config.model.name,
                "temperature": self.config.generation.temperature,
                "seed": self.config.generation.seed,
            },
            limitations=(
                "Single model (distilgpt2) and single synthetic arithmetic-sequence "
                "task; results may not generalize to larger models or real language "
                "tasks. Correctness is determined by the first integer emitted, and "
                "trajectories are capped at 16 tokens, so the 'early' signal at "
                "10-30% of generation substantially overlaps the token where the "
                "answer commits. High early-token entropy therefore largely reflects "
                "uncertainty about the answer itself rather than a distinct "
                "pre-error degradation phase. Findings do not establish causality."
            ),
            next_experiment=(
                "EXP-002: Cross-model generalization study. "
                "Train on Model A, test on Model B."
            ),
        )

        # Results section
        results_lines = [
            f"- **Total samples:** {summary['total']}",
            f"- **Correct:** {summary['correct']} ({summary['accuracy']:.1%})",
            f"- **Incorrect:** {summary['incorrect']}",
            "",
        ]

        for b in results.get("baselines", []):
            if "error" in b:
                results_lines.append(f"- **{b['model_name']}:** FAILED - {b['error']}")
            else:
                results_lines.append(
                    f"- **{b['model_name']}:** AUROC={b['auroc']:.4f}, "
                    f"AUPRC={b['auprc']:.4f}, Acc={b['accuracy']:.4f}, "
                    f"CalErr={b['calibration_error']:.4f}"
                )

        report.add_section("Results", "\n".join(results_lines))

        # Temporal analysis
        temporal = results.get("temporal_prediction", {})
        if temporal:
            temp_lines = ["| Fraction | Cohen's d | p-value | Correct Mean | Incorrect Mean |",
                          "|----------|-----------|---------|-------------|---------------|"]
            for frac_key in sorted(temporal.keys(), key=float):
                v = temporal[frac_key]
                temp_lines.append(
                    f"| {float(frac_key):.0%} | {v['cohens_d']:.3f} | {v['p_value']:.4f} | "
                    f"{v['correct_mean_entropy']:.3f} | {v['incorrect_mean_entropy']:.3f} |"
                )
            report.add_section("Temporal Analysis", "\n".join(temp_lines))

        # Interpretation
        interp_lines = []
        best_baseline = None
        valid = [b for b in results.get("baselines", []) if "error" not in b]
        if valid:
            best_baseline = max(valid, key=lambda b: b["auroc"])
            interp_lines.append(
                f"Best baseline model: **{best_baseline['model_name']}** with AUROC={best_baseline['auroc']:.4f}"
            )

            if best_baseline["auroc"] > 0.7:
                interp_lines.append(
                    "Conventional uncertainty features provide meaningful predictive signal. "
                    "The question becomes whether the temporal pattern adds information beyond "
                    "what aggregate features capture."
                )
            elif best_baseline["auroc"] > 0.55:
                interp_lines.append(
                    "Weak but non-trivial predictive signal exists in conventional features. "
                    "Temporal analysis may reveal whether early signals are informative."
                )
            else:
                interp_lines.append(
                    "Conventional features show little predictive power. This may indicate "
                    "that aggregate token-level uncertainty is insufficient, or that the "
                    "model's errors are not well-captured by these signals."
                )

        report.add_section("Interpretation", "\n".join(interp_lines) or "Pending analysis.")
        report.add_section("Method",
            "1. Run generation with token-level logging\n"
            "2. Extract aggregate trajectory features\n"
            "3. Train baseline classifiers\n"
            "4. Analyze temporal prediction at various generation fractions\n"
            "5. Compare correct vs incorrect trajectory distributions"
        )
        report.save()
