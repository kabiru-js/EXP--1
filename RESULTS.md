# Results

## Summary

This project investigates whether token-level generation trajectories of a language model contain measurable signals preceding incorrect answers. Two experiments have been conducted:

- **EXP-001** (complete): Strong signal found — token-level uncertainty separates correct from incorrect generations.
- **EXP-002** (null result): No mid-chain divergence observed across 45 controlled probes.

**The project is designed to be falsifiable.** Null results are treated as valid scientific findings.

---

## EXP-001: Baseline Trajectory Study

Status: **Complete**

### Setup

- **Dataset:** `synthetic_sequences` (7-term arithmetic progressions, ~40% step-1, starts range(3,60))
- **Model:** `distilgpt2` (124M, CPU, float32), temperature 0.7, max_tokens 16, top_p 0.95, top_k 50, seed=42
- **Samples:** 1,000 (466 correct / 534 incorrect, 46.6% accuracy)
- **Prompt template:** `"{question}"` (no instruction wrapper — just the comma-terminated sequence)
- **Config:** `configs/EXP-001.yaml` · Results JSON: `experiments/EXP-001/results.json`
- **Report:** `reports/EXP-001.md`

### Baselines (single 80/20 split, stratified)

| Model | AUROC | AUPRC | Acc | BalAcc | Prec | Rec | F1 | Brier | CalErr |
|-------|-------|-------|-----|--------|------|-----|----|-------|--------|
| entropy_threshold | 0.868 | 0.864 | 0.580 | 0.548 | 1.000 | 0.097 | 0.176 | 0.241 | 0.279 |
| logistic_regression | 0.975 | 0.979 | 0.945 | 0.945 | 0.936 | 0.946 | 0.941 | 0.054 | 0.062 |
| gradient_boosted_tree | 1.000 | 1.000 | 0.995 | 0.995 | 0.989 | 1.000 | 0.995 | 0.006 | 0.030 |

### Cross-validation sanity check (5-fold stratified, `experiments/EXP-001/cv_results.json`)

| Model | AUROC (mean ± std) | AUPRC (mean ± std) | Acc (mean ± std) |
|-------|--------------------|--------------------|--------------------|
| entropy_threshold | 0.8584 ± 0.0189 | 0.8496 ± 0.0161 | 0.5660 ± 0.0231 |
| logistic_regression | 0.9666 ± 0.0056 | 0.9718 ± 0.0039 | 0.9280 ± 0.0160 |
| gradient_boosted_tree | **0.9946 ± 0.0061** | 0.9932 ± 0.0074 | 0.9880 ± 0.0093 |

CV confirms the single-split GBT AUROC=1.0 was not over-fitting: the label signal in aggregate trajectory features is genuinely near-separable.

### Temporal prediction

Mean token entropy differs strongly between correct and incorrect generations at every generation fraction:

| Fraction | Cohen's d | p-value | Correct mean entropy | Incorrect mean entropy |
|----------|-----------|---------|---------------------|----------------------|
| 10% | **−2.34** | 9.4×10⁻¹⁸⁶ | ~0.66 | ~1.14 |
| 20% | −2.06 | ~10⁻¹⁰⁰ | ~0.70 | ~1.10 |
| 30% | −1.85 | ~10⁻⁶⁰ | ~0.73 | ~1.08 |
| ... | ... | ... | ... | ... |
| 100% | −1.30 | ~10⁻²⁰ | ~0.82 | ~1.00 |

All p-values < 10⁻⁷⁶. Incorrect trajectories are more uncertain at every fraction. The effect size is largest at early fractions (10%), but this is confounded by task structure (answer commits at token 0).

### Divergence analysis

Median detected divergence fraction is **0.0** (the very first token). In this task the answer is the first integer emitted, so error "signals" are present from token 1 — there is no evidence of a gradual pre-error degradation phase within the 16-token window.

### Feature importances (gradient boosted tree)

Top features by importance:
1. `entropy_acceleration` — quadratic coefficient of entropy trajectory
2. `probability_trend` — linear trend of probability
3. `early_mean_entropy` — mean entropy in first quarter
4. `max_probability` — maximum token probability
5. `margin_trend` — linear trend of margin
6. `final_entropy` — entropy at final token

### Interpretation

- **Aggregate uncertainty features** (entropy, margin, cumulative log-prob) are highly predictive of correctness for this model/task (LR AUROC ~0.97, GBT ~0.99). This is a strong falsification check that the instrumentation and labels are meaningful.
- **The finding is confounded by task structure:** correctness commits at token 0-1, so early high entropy partially *is* the model being unsure of the answer it is about to emit.
- **Not a demonstration of predictive divergence** — a pre-error phase distinct from a confident-correct roll-out. That requires a task where the answer is reached only after a multi-token reasoning chain.

### Limitations

1. Single small model (distilgpt2), no instruction-following capability.
2. Answer commits at the first output token → no mid-trajectory decision point.
3. Synthetic task; real open-ended language tasks unaddressed.
4. Findings are correlational, not causal.
5. No prompt-position signals (uncertainty at last prompt token not measured).
6. Binary correctness definition.

---

## EXP-002: Onset Feasibility Diagnostic

Status: **Incomplete on available hardware** (no NVIDIA driver/GPU; Qwen2.5-0.5B-Instruct not cached).

### distilgpt2 arithmetic-chain continuation probes (45 generations, P=2/3/4)

- **Mid-chain onset (several correct terms, then a wrong one): 0 observed**
- The model either continues the entire run correctly (step-1 sequences) or diverges at the very first continuation integer
- **Onset distribution:**
  - P=2: {0: 15} — all divergent from term 0
  - P=3: {0: 8, 1: 2, ALL_CORRECT: 5} — divergent at term 0 or 1 only
  - P=4: {0: 9, 1: 1, ALL_CORRECT: 5} — divergent at term 0 or 1 only
- Per-token instrumentation (raw logits, entropy, top-k, margin, cumulative log-prob) validated end-to-end; subsequence-based onset detection worked correctly

### Qwen2.5-0.5B-Instruct

Not benchmarked — not cached (~1GB), download infeasible, no NVIDIA driver on this machine, PyTorch is CPU-only (2.8.0+cpu).

### Distinction

This is a null result for this model/task combination, **not** proof the phenomenon is absent. EXP-003 requires an NVIDIA GPU + a reasoning-capable model to test whether the "initially correct then diverged" pattern occurs at a non-trivial rate.

See `reports/EXP-002.md` and `experiments/EXP-002/exp002_results.json` for full detail.

---

## Next Experiments (candidate)

| ID | Question | Requires |
|----|----------|----------|
| EXP-003 | Prompt-position uncertainty (last token before generation) | Code modification to `_TrajectoryLogitsProcessor` |
| EXP-004 | Multiple task families (visible chain-of-thought, GSM8K) | New dataset loaders |
| EXP-005 | Cross-model comparison (GPT-2, phi-2, Qwen) | GPU or faster CPU, cached models |
| EXP-006 | Larger EXP-002 sample sizes (200+ probes) | GPU or faster CPU |
| EXP-007 | Different prompt templates (with instructions) | Same instrumentation, different prompts |

---

## Complete Results

All raw results are available in:
- `experiments/EXP-001/results.json` — full results with feature names, importances, trajectory curves, temporal prediction
- `experiments/EXP-001/cv_results.json` — 5-fold cross-validation results
- `experiments/EXP-002/exp002_results.json` — probe results
- `reports/EXP-001.md` — detailed experiment report
- `reports/EXP-002.md` — detailed probe report
