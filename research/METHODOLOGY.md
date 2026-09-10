# Research Methodology

## Instrumentation

Generation is intercepted at the model level using a custom `LogitsProcessor` (`src/models/huggingface.py`, `_TrajectoryLogitsProcessor`).

At each generated token position `t`, the processor captures:

```
TokenSignal(
    token_id=int,
    token_text=str,
    probability=float,       # P(token_t | token_1...t-1)
    log_probability=float,   # log P(token_t | token_1...t-1)
    entropy=float,           # H(P(token_t | token_1...t-1))
    top_k_token_ids=list[int],
    top_k_token_texts=list[str],
    top_k_probabilities=list[float],
    probability_margin=float, # P(top-1) - P(top-2)
    cumulative_log_prob=float,
    logits=torch.Tensor,     # raw model logits at position t
)
```

The processor is attached to `model.generate()` via `LogitsProcessorList`. It records signals for **generated tokens only** (tokens after the prompt). Prompt-position signals are NOT captured — this is a known limitation (see `LIMITATIONS.md`).

## Recorded State

Each trajectory is stored as a `Trajectory` dataclass (`src/trajectories/storage.py`) containing:

- `sample_id`, `dataset`, `model_id`, `experiment_id`
- `question`, `ground_truth`, `prompt`
- `generated_text`, `extracted_answer`, `is_correct`
- `token_metrics: list[TokenMetrics]` — the per-token signals above
- `generation_tokens`, `temperature`, `seed`, `prompt_template`
- `metadata`

Trajectories are stored as **one Parquet file per trajectory** plus a **DuckDB metadata table**.

## Dataset and Prompts

EXP-001 uses the `synthetic_sequences` dataset (`src/datasets/synthetic.py`, `SyntheticSequenceLoader`):

- Arithmetic progressions: `start + k * step` for k = 0..6 (7 terms).
- Step distribution weighted: `[1,1,1,1,2,2,3,4,5,6]` (~40% step-1).
- Starts in `range(3, 60)` (multi-digit answers).
- Prompt = `"{question}"` where question = `", ".join(terms) + ","`.
- The model must continue the sequence; the first integer it emits is the answer.

**This prompt template contains NO instruction wrapper** — just the comma-terminated sequence. The model sees only the numbers.

## Correctness Definition

1. `extract_answer(raw_answer)`: Extract the first integer from the model's generated text using `re.match(r"\s*([-+]?\d+)", raw_answer)`.
2. `check_correctness(predicted, ground_truth)`: `int(predicted) == int(ground_truth)`.
3. Correctness is binary and determined by the first emitted integer.

**Important:** In this task, the answer commits at the first token. This is a structural property of the task, not a limitation of the model. See `LIMITATIONS.md`.

## Feature Extraction

`src/features/extraction.py` extracts 30 aggregate features per trajectory from the token-level signals:

- **Entropy features:** mean, std, max, min, trend (linear slope), first/last quarter means, acceleration (quadratic coefficient).
- **Probability features:** mean, std, max, min, trend, last quarter mean.
- **Margin features:** mean, std, min, trend.
- **Log-probability features:** total, mean, std, min step log-prob.
- **Final token features:** final probability, log-probability, entropy, margin.
- **Early features:** first-quarter mean entropy, probability, margin.

Feature names are listed in `experiments/EXP-001/results.json` → `feature_names`.

## Classification

**Train/test separation:** `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.

- Stratification ensures the class balance is preserved in both splits.
- The same split is used across all baselines for comparability.

**Baselines:**
1. `EntropyThresholdBaseline`: threshold = median(correct_entropies) + std(correct_entropies); predicts incorrect if mean_entropy > threshold.
2. `LogisticRegressionBaseline`: `StandardScaler` + `LogisticRegression(max_iter=1000, C=1.0, random_state=42)`.
3. `GradientBoostedBaseline`: `GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)`.

**Metrics:** AUROC, AUPRC, accuracy, balanced_accuracy, precision, recall, F1, Brier score, calibration_error.

**Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — separate script `scripts/cv_check_exp001.py`.

## Temporal Prediction

`src/analysis/divergence.py`, `compute_temporal_prediction_performance()`:

1. For each fraction `f ∈ {0.1, 0.2, ..., 1.0}`:
   - Compute cutoff = `int(n * f)` where n = trajectory length.
   - Compute mean entropy over the first `cutoff` tokens for each sample.
   - Split into correct/incorrect groups.
   - Compute Welch's t-test: `stats.ttest_ind(correct_vals, incorrect_vals)`.
   - Cohen's d = `(mean_correct - mean_incorrect) / sqrt((std_correct² + std_incorrect²) / 2)`.

**Cohen's d interpretation:** |d| > 0.8 is considered a large effect. The observed d = −2.34 at 10% fraction is very large. The negative sign indicates incorrect generations have higher entropy.

## Divergence Analysis (EXP-002)

`scripts/probe_chains.py`:

1. Generate arithmetic sequence: `start + k * step` for k = 0..(prefix_len + continuation_len - 1).
2. Prompt = first `prefix_len` terms.
3. Model generates continuation; extract integers from generated text.
4. **Onset detection:** Greedy subsequence alignment of emitted integers against expected continuation chain.
   - For each expected value in order, find its first occurrence in emitted integers at or after the position of the previous matched value.
   - Onset = index of the first unmatched expected value.
   - `onset = 0` → the first emitted integer is already wrong.
   - `onset >= 2` → the model correctly emitted ≥2 continuation integers before the first error.
5. The divergence distribution across all probes is summarized (mean, median fraction).

**Operational definition of "divergence":** The first point at which the model's emitted integer differs from the ground-truth continuation. This is a task-specific definition, not a general notion of "reasoning divergence."

## Evaluation Procedure Summary

| Step | Method |
|------|--------|
| Trajectory generation | `model.generate()` with `_TrajectoryLogitsProcessor` |
| Feature extraction | `extract_trajectory_features()` — 30 aggregate features |
| Classification | `train_test_split` (80/20, stratified) + 3 baselines |
| AUROC | `roc_auc_score(y_test, y_prob)` |
| Cross-validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Temporal prediction | Welch's t-test + Cohen's d at each generation fraction |
| Divergence detection | Greedy subsequence alignment |

---

## Limitations of the Methodology

1. **Prompt-position signals not captured.** The `_TrajectoryLogitsProcessor` records signals only for generated tokens, not the model's state at the last prompt token. A true precursor would require measuring uncertainty before generation starts.

2. **Correctness commits at the first token.** In the `synthetic_sequences` task, the first emitted integer IS the answer. So the "temporal" analysis at fraction 0.1 is measuring the first token — there is no "before."

3. **Single task family.** Only arithmetic-sequence continuation has been tested. Other task families may behave differently.

4. **Single model.** Only distilgpt2 has been empirically characterized. Other models may exhibit different behavior.

5. **Binary correctness.** The correctness definition is binary (first integer matches). Partial correctness (e.g., getting the first few terms right before diverging) is not captured by the baseline labels but IS captured by the onset analysis in EXP-002.

6. **No causal inference.** The experiment measures correlation between uncertainty signals and correctness. It does not establish that uncertainty causes or predicts errors in a causal sense.
