# Repository Audit

**Date:** 2026-09-10
**Repository:** EXP--1 (Model Trajectory Research)
**Branch:** main

---

## What Exists

### Source Code (`src/`)

| Module | Status | Notes |
|--------|--------|-------|
| `src/config.py` | Complete | Pydantic-based YAML config with `ExperimentConfig`, `DatasetConfig`, `ModelConfig`, `GenerationConfig`, `FeatureConfig`, `EvaluationConfig`, `StorageConfig`, `ProjectConfig` |
| `src/datasets/base.py` | Complete | Abstract `DatasetLoader` and `Sample` dataclass |
| `src/datasets/gsm8k.py` | Complete | GSM8K loader |
| `src/datasets/truthfulqa.py` | Complete | TruthfulQA loader |
| `src/datasets/synthetic.py` | Complete | `SyntheticArithmeticLoader`, `SyntheticSequenceLoader` |
| `src/datasets/registry.py` | Complete | Registry of loaders |
| `src/models/base.py` | Complete | `ModelBackend` ABC, `TokenSignal`, `GenerationResult` dataclasses |
| `src/models/huggingface.py` | Complete | `HuggingFaceBackend` with `_TrajectoryLogitsProcessor` — captures per-token logits, probability, entropy, top-k, margin, cumulative log-prob |
| `src/models/registry.py` | Complete | Backend registry |
| `src/inference/pipeline.py` | Complete | `InferencePipeline` — loads samples, generates, builds trajectories, stores |
| `src/trajectories/storage.py` | Complete | `TrajectoryStorage` — Parquet per trajectory + DuckDB metadata; `clear()`, `resume` support |
| `src/features/extraction.py` | Complete | `extract_trajectory_features` (30 aggregate features), `extract_temporal_features`, `build_feature_matrix` |
| `src/evaluation/baselines.py` | Complete | `EntropyThresholdBaseline`, `LogisticRegressionBaseline`, `GradientBoostedBaseline`, `compute_metrics`, `run_baseline` |
| `src/analysis/divergence.py` | Complete | `normalize_trajectory_length`, `compute_trajectory_curves`, `find_divergence_point`, `compute_temporal_prediction_performance` |
| `src/visualization/plots.py` | Complete | `plot_trajectory_comparison`, `plot_temporal_prediction`, `plot_baseline_comparison`, `plot_individual_trajectories`, `plot_divergence_distribution`, `plot_feature_importances`, `save_figure_data` |
| `src/reporting/report.py` | Complete | `ExperimentReport` — markdown report generator |
| `src/experiments/runner.py` | Complete | `ExperimentRunner` — orchestrates full pipeline, CV not integrated (separate script) |
| `src/cli/main.py` | Complete | Click CLI: `run`, `analyze`, `list-experiments`, `list-traj`, `inspect`, `report` |

### Experiments

| Experiment | Status | Data |
|------------|--------|------|
| EXP-001 | **Complete** | `experiments/EXP-001/results.json`, `cv_results.json`, `figures/`, `trajectories/` (1000 Parquet files), `trajectories.duckdb` |
| EXP-002 | **Incomplete on hardware** | `experiments/EXP-002/exp002_results.json`, `EXP-002.md`. Qwen2.5-0.5B-Instruct not cached; no NVIDIA driver on this machine |

### Configuration

| File | Status |
|------|--------|
| `configs/EXP-001.yaml` | Complete — distilgpt2, synthetic_sequences, 1000 samples, temp 0.7, max_tokens 16 |
| `scripts/run_exp001.py` | Complete — CLI entry point |
| `scripts/cv_check_exp001.py` | Complete — 5-fold CV sanity check |
| `scripts/scratch/probe_chains.py` | Complete — distilgpt2 chain-continuation probes |
| `scripts/scratch/probe_qwen_chains.py` | Complete — Qwen probe (not runnable on this hardware) |
| `scripts/exp002_qwen.py` | Complete — Qwen EXP-002 experiment (deferred) |

### Tests

| File | Status |
|------|--------|
| `tests/smoke_test.py` | **PASSes** — all modules |
| `tests/test_analysis.py` | PASSes |
| `tests/test_config.py` | PASSes individually (pytest hangs when run together — known shell issue) |
| `tests/test_datasets.py` | PASSes |
| `tests/test_features.py` | PASSes |
| `tests/test_storage.py` | PASSes |
| `pytest tests/` | **Hangs** — known issue with this PowerShell/Python setup; use `python tests/smoke_test.py` |

### Documentation

| File | Status |
|------|--------|
| `README.md` | Needs rewrite (currently startup-style) |
| `RESULTS.md` | Needs update (currently incomplete) |
| `CHANGELOG.md` | Needs update |
| `reports/EXP-001.md` | Complete — honest limitations |
| `reports/EXP-002.md` | Complete — null result documented |
| `.gitignore` | Complete |

---

## What Is Correct

1. **EXP-001 results are empirically sound.** 1000 trajectories, 466/534 correct (46.6%), LR AUROC 0.9745, GBT AUROC 1.0000 single-split / 0.9946±0.0061 CV. Cohen's d = −2.34 at 10% fraction (p=9.4×10⁻¹⁸⁶).

2. **The instrumentation works correctly.** Per-token raw logits, probability, log-prob, entropy, top-k, margin, cumulative log-prob are all captured and stored. The `_TrajectoryLogitsProcessor` is functional.

3. **The CV sanity check is honest and correct.** GBT AUROC=1.0000 single-split could look like overfitting; the 5-fold CV (0.9946±0.0061) confirms the signal is real.

4. **EXP-002 null result is documented honestly.** 45 controlled probes, 0 mid-chain onset cases. The report distinguishes "phenomenon appears absent" from "not exposed by this model/task."

5. **The codebase is well-structured.** Clean separation of config, datasets, models, inference, trajectories, features, evaluation, analysis, visualization, reporting, experiments, CLI.

6. **The CLI works.** `research run`, `research analyze`, `research inspect`, `research list-experiments`, `research list-traj`, `research report` all functional.

---

## What Is Unclear or Unknown

1. **GPT-2 / phi-2 performance on this task** — not tested (too slow on CPU). Only distilgpt2 has been empirically characterized.

2. **Qwen2.5-0.5B-Instruct behavior** — not benchmarked. The model is not cached and download was infeasible. The hypothesis that it would show mid-chain onset is untested.

3. **Whether other task families expose mid-chain onset** — only arithmetic-sequence continuation has been tested. Other task families (e.g., visible chain-of-thought arithmetic, multi-step reasoning) are untested.

4. **Pre-generation uncertainty (prompt-position signal)** — the instrumentation captures signals only for generated tokens, not the model's state at the last prompt token. This is a known gap (see `src/models/huggingface.py` `generate()`).

5. **Statistical power of EXP-002** — 45 probes may be underpowered to detect a rare phenomenon. The sample size was constrained by CPU generation speed.

---

## What Is Missing

1. **No formal research methodology documentation.** The operational definitions of "divergence," "onset," "correctness," "temporal prediction" are implicit in the code but not explicitly stated in a single document.

2. **No reproducibility guide.** Python version, dependency versions, installation steps, exact commands, model requirements, environment variables are not documented in one place.

3. **No `REPRODUCIBILITY.md`.**

4. **No `LIMITATIONS.md`.** Limitations are scattered across `reports/EXP-001.md`, `reports/EXP-002.md`, and `README.md` but not consolidated.

5. **No `ROADMAP.md`.** The next experiment is mentioned in `reports/EXP-001.md` but not formalized.

6. **No experiment-specific READMEs.** `experiments/EXP-001/` and `experiments/EXP-002/` lack READMEs explaining the experiment's purpose, setup, results, and interpretation.

7. **No `QUESTION.md` or `METHODOLOGY.md`.** The research question and methodology are implicit in the code and reports but not formalized.

8. **No `RELATED_WORK.md`.** Related work areas are mentioned in `README.md` but not documented with references (and no fabricated references).

9. **No publication-quality figure-generation scripts.** The existing `src/visualization/plots.py` generates figures, but there are no standalone scripts in `scripts/` that reproduce the figures from the results JSONs (the figures were generated during the experiment run, not from a reproducible script).

10. **No `figures/` directory at repo root.** Figures are scattered across `experiments/EXP-001/figures/` and `reports/figures/`.

11. **No pipeline diagram.** No visual overview of the experimental pipeline.

12. **No figure for EXP-002.** The EXP-002 null result has no corresponding visualization.

---

## What Should Be Changed

1. **README.md** — rewrite as a research project page, not a startup landing page.
2. **RESULTS.md** — add rigorous results presentation with sample sizes, evaluation methods, uncertainty.
3. **Add `research/` directory** — with `QUESTION.md`, `METHODOLOGY.md`, `RELATED_WORK.md`, `ROADMAP.md`, `REPRODUCIBILITY.md`, `LIMITATIONS.md`.
4. **Add experiment-specific READMEs.**
5. **Add `scripts/` figure-generation scripts** that reproduce figures from existing results JSONs.
6. **Add `AUDIT.md`** (this file).
7. **Add a root `figures/` directory** with a figure-generation script.
8. **Update CHANGELOG.md** to document the restructuring.

---

## What Should NOT Be Changed

1. **The actual scientific results.** EXP-001 numbers (46.6% accuracy, AUROCs, Cohen's d, CV results) and EXP-002 null result (45 probes, 0 mid-chain onset) must be preserved exactly.

2. **The core source code logic.** The instrumentation, feature extraction, baseline evaluation, divergence analysis, and trajectory storage code all work correctly. Do not refactor for aesthetics.

3. **The test assertions.** The existing tests pass and verify correct behavior.

4. **The config structure.** `configs/EXP-001.yaml` is correct and reproducible.

5. **The honest limitations language.** The reports already state the degeneracy (answer commits at token 0-1) and the null result accurately.

---

## How EXP-001 and EXP-002 Differ

| Dimension | EXP-001 | EXP-002 |
|-----------|---------|---------|
| **Purpose** | Trajectory separability | Mid-chain divergence |
| **Task** | Arithmetic-sequence continuation (next term) | Arithmetic-chain continuation (P-term prefix → continue) |
| **Model** | distilgpt2 | distilgpt2 (probes); Qwen2.5-0.5B-Instruct (deferred) |
| **Samples** | 1000 | 45 controlled probes |
| **Correctness** | `int(first_integer) == int(ground_truth)` | Subsequence alignment of emitted integers vs expected continuation |
| **Analysis** | Baselines (AUROC), temporal prediction (Cohen's d), divergence distribution | Onset detection (subsequence alignment), distribution of onset fractions |
| **Result** | Strong separability (AUROC 0.97–1.0, d=−2.34) | Null (0 mid-chain onset) |
| **Key finding** | Uncertainty at first token strongly predicts correctness | Model either gets the whole chain right or diverges at the first token |

---

## How Trajectories Were Generated

1. `ExperimentRunner.run()` loads `ExperimentConfig` from YAML.
2. `InferencePipeline` loads `DatasetLoader` (e.g., `SyntheticSequenceLoader`), generates `Sample` objects.
3. For each sample, `prompt_template.format(question=sample.question)` creates the prompt.
4. `HuggingFaceBackend.generate()` tokenizes the prompt, runs `model.generate()` with `_TrajectoryLogitsProcessor` capturing per-token logits/probabilities/entropy/top-k/margin/cumulative_log_prob.
5. `InferencePipeline._build_trajectory()` extracts the answer via `loader.extract_answer(result.text)`, checks correctness via `loader.check_correctness(extracted, sample.ground_truth)`, and creates a `Trajectory` dataclass.
6. `TrajectoryStorage.store_batch()` writes one Parquet file per trajectory + DuckDB metadata.

**EXP-001 specific parameters:** `distilgpt2`, `temperature=0.7`, `max_tokens=16`, `top_p=0.95`, `top_k=50`, `seed=42`, `prompt_template="{question}"`, 1000 samples, `synthetic_sequences` dataset.

---

## Correctness Definition

- **`sample.ground_truth`**: The correct answer string (e.g., "76" for the next term in an arithmetic sequence).
- **`loader.extract_answer(raw_answer)`**: Extracts the first integer from the model's generated text. For `synthetic_sequences`, this uses `re.match(r"\s*([-+]?\d+)", raw_answer)`.
- **`loader.check_correctness(predicted, ground_truth)`**: Compares `int(predicted) == int(ground_truth)`.
- **One trajectory = one observation.** Correctness is binary: the first integer in the generated text matches the ground truth.

---

## Cohen's d Calculation

In `src/analysis/divergence.py`, `compute_temporal_prediction_performance()`:
1. For each fraction `f` (e.g., 0.1, 0.2, ..., 1.0):
   - Compute cutoff = `int(n * f)` where n = trajectory length (16 tokens for EXP-001).
   - Compute mean entropy over the first `cutoff` tokens for each sample.
   - Split into correct/incorrect groups.
   - Compute Welch's t-test (`stats.ttest_ind`).
   - Cohen's d = `(mean_correct - mean_incorrect) / sqrt((std_correct² + std_incorrect²) / 2)`.

---

## Classifier and AUROC Calculation

In `src/evaluation/baselines.py`, `run_baseline()`:
1. `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` — single 80/20 split.
2. Fit baseline on train, predict on test.
3. `roc_auc_score(y_test, y_prob)` for AUROC.
4. CV (separate script `scripts/cv_check_exp001.py`): `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, fit/predict per fold, collect per-fold metrics, report mean ± std.

---

## Divergence Analysis (EXP-002)

In `scripts/probe_chains.py`:
1. Generate arithmetic sequence: start + k*step for k=0..(prefix_len + continuation_len - 1).
2. Prompt = first `prefix_len` terms (e.g., "3, 5, 7,").
3. Model generates continuation; extract integers from generated text.
4. Align emitted integers against expected continuation as a greedy subsequence.
5. Onset = index of first unmatched expected value. `onset=0` means the first emitted integer is already wrong.
6. `onset >= 2` means the model correctly emitted ≥2 continuation integers before the first error.

**Result:** 0 cases with onset ≥ 2 across 45 probes (P=2/3/4).

---

## Dependencies

Python 3.11, torch 2.8.0+cpu, transformers 4.56.2, scipy, scikit-learn, numpy, pandas, polars, duckdb, pyarrow, matplotlib, seaborn, pyyaml, click, tqdm, pydantic. Installed via `pip install -e ".[dev]"`.

---

## Hardware Constraints

- CPU only (4 cores), no NVIDIA GPU/driver (only Intel HD 520).
- PyTorch is CPU-only build (`2.8.0+cpu`).
- Models run on CPU; distilgpt2 ~1.4s/sample, Qwen2.5-0.5B-Instruct not benchmarked (not cached, download infeasible).
