# EXP-001: Baseline Trajectory Study

## Question

Can token-level generation trajectories of a language model contain measurable signals that indicate whether the final answer will be correct or incorrect?

## Setup

- **Model:** distilgpt2 (124M parameters)
- **Dataset:** synthetic_sequences (7-term arithmetic progressions)
- **Samples:** 1,000 (466 correct, 534 incorrect — 46.6% accuracy)
- **Generation parameters:** temperature=0.7, max_tokens=16, top_p=0.95, top_k=50, seed=42
- **Prompt template:** `"{question}"` (no instruction wrapper — just the comma-terminated sequence)
- **Instrumentation:** Per-token probability, entropy, margin, cumulative log-probability, top-k, logits

## What Was Measured

1. **Per-token signals:** At each generated token position, probability, entropy, margin, top-k probabilities, cumulative log-probability, and raw logits were recorded.
2. **Trajectory features:** 30 aggregate features were extracted per trajectory (mean, std, trend, acceleration of entropy/probability/margin/cumulative log-prob).
3. **Classification:** Logistic regression and gradient boosted tree classifiers were trained to predict correctness from trajectory features.
4. **Temporal prediction:** Cohen's d was computed at each generation fraction to measure how early correctness can be distinguished.
5. **Cross-validation:** 5-fold stratified CV was performed to verify the signal is not due to overfitting.

## Key Results

| Metric | Value |
|--------|-------|
| Accuracy | 46.6% (466/1000 correct) |
| LR AUROC (single split) | 0.9745 |
| GBT AUROC (single split) | 1.0000 |
| GBT AUROC (5-fold CV) | 0.9946 ± 0.0061 |
| Cohen's d at 10% fraction | −2.34 (p = 9.4×10⁻¹⁸⁶) |
| Divergence median fraction | 0.0 |

## Interpretation

Token-level uncertainty signals strongly separate correct from incorrect generations. The effect is very large (Cohen's d = −2.34 at 10% of tokens generated) and confirmed by cross-validation (AUROC 0.9946±0.0061).

**Critical caveat:** In this task, the answer commits at the first token. The signal at fraction 0.1 is measuring the first token's uncertainty, not a precursor to it. See `LIMITATIONS.md` for full discussion.

## Files

- `configs/EXP-001.yaml` — experiment configuration
- `experiments/EXP-001/results.json` — aggregated results
- `experiments/EXP-001/cv_results.json` — cross-validation results
- `experiments/EXP-001/trajectories/` — 1000 Parquet trajectory files
- `experiments/EXP-001/trajectories.duckdb` — metadata database
- `experiments/EXP-001/figures/` — visualization figures
- `reports/EXP-001.md` — detailed report

## How to Reproduce

```bash
python scripts/run_exp001.py
```

Or with custom parameters:
```bash
python scripts/run_exp001.py --samples 500 --model gpt2
```

## Next Experiment

EXP-003 will measure prompt-position uncertainty to determine whether the token-0 signal is a genuine precursor or a labeling artifact. See `research/ROADMAP.md`.
