# Limitations

## Task-Related Limitations

### Answer Commits at Token 0

In the `synthetic_sequences` dataset, the prompt is a comma-terminated arithmetic sequence (e.g., "3, 5, 7,"). The model must continue the sequence, and the **first integer it emits is the answer**. This means:

- The "temporal" analysis at fraction 0.1 is measuring the **first token** — there is no "before."
- Any signal at fraction 0.1 is necessarily a signal about the first token, not a precursor to the first token.
- **This is a structural property of the task, not a limitation of the model or measurement.**

**Honest interpretation:** The strong signal at early fractions (Cohen's d = −2.34 at 10%) indicates that the model's uncertainty about the first token strongly predicts correctness. It does NOT prove that the model has a "precursor" signal before committing to the answer.

### Single Task Family

Only arithmetic-sequence continuation has been tested. Other task families (visible chain-of-thought arithmetic, multi-step reasoning, code generation) may behave differently. The phenomenon may be task-specific.

### Synthetic Data

The arithmetic-sequence dataset is synthetic. Real-world reasoning tasks (GSM8K, MATH) may have different characteristics: longer sequences, more ambiguity, different error modes.

## Instrumentation Limitations

### Prompt-Position Signals Not Captured

The `_TrajectoryLogitsProcessor` records signals only for generated tokens. The model's state at the last prompt token (before generation starts) is NOT measured. A true precursor signal would require measuring uncertainty at the prompt position.

### Binary Correctness

Correctness is binary: the first emitted integer matches the ground truth. Partial correctness (e.g., getting the first 3 terms right before diverging) is not captured by the baseline labels. EXP-002's onset analysis partially addresses this.

### No Hidden-State Analysis

Only output probabilities are measured. Internal representations, attention patterns, and hidden states are not analyzed. Mechanistic interpretability could reveal different insights.

## Model Limitations

### Single Model (EXP-001)

Only distilgpt2 (124M) has been empirically characterized. Results may not generalize to other models.

### No GPU

All experiments run on CPU. This limits:
- Model choices (only small models feasible)
- Sample sizes (generation is slow)
- The ability to benchmark larger models (Qwen2.5-0.5B-Instruct not cached)

## Statistical Limitations

### EXP-002 Sample Size

45 probes may be underpowered to detect a rare phenomenon. The null result (0 mid-chain onset) could be due to low sample size rather than the phenomenon being truly absent.

### Single Split for Baselines

The main AUROC results use a single 80/20 train/test split. The CV results (5-fold) are more robust, but the single-split results are the primary reported numbers.

### No Confidence Intervals for AUROC

AUROC values are reported without confidence intervals. Bootstrapped CIs would provide more complete uncertainty quantification.

## Interpretation Limitations

### Correlation ≠ Causation

The experiment measures correlation between uncertainty signals and correctness. It does not establish that uncertainty causes or predicts errors in a causal sense.

### No Process-Level Analysis

The experiment does not analyze the model's internal reasoning process. It only measures output probabilities. Any claims about "reasoning" or "self-awareness" are not supported by the data.

### Potential Confounding

The strong signal at early fractions could be confounded by task structure (answer commits at first token). EXP-003 (prompt-position uncertainty) is designed to address this.

## What We Did NOT Find

- **No mid-chain divergence** in 45 controlled probes (EXP-002).
- **No evidence** that the model exhibits different behavior at intermediate tokens before diverging.
- **No causal evidence** that uncertainty signals cause errors.

These null findings are documented honestly and are valid scientific results.

## Summary

| Limitation | Severity | Addressed By |
|------------|----------|-------------|
| Answer commits at token 0 | **High** | EXP-003 (prompt-position signals) |
| Single task family | Medium | EXP-004 (multiple tasks) |
| Prompt-position signals not captured | **High** | EXP-003 |
| No GPU / single model | Medium | EXP-005 (model comparison) |
| Small EXP-002 sample size | Medium | EXP-006 (larger samples) |
| Binary correctness | Low | EXP-002 onset analysis |
| No hidden-state analysis | Medium | Future work |
| Correlation ≠ causation | Low | Design limitation |
| No confidence intervals for AUROC | Low | Future work |
