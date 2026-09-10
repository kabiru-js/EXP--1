# Results

## EXP-001: Baseline Trajectory Study

Status: Complete

- **Dataset:** `synthetic_sequences` (7-term arithmetic progressions, 40% step-1)
- **Model:** `distilgpt2` (CPU, float32), temperature 0.7, max_tokens 16
- **Samples:** 1000 (466 correct / 534 incorrect, 46.6% accuracy)
- **Config:** `configs/EXP-001.yaml` · Results JSON: `experiments/EXP-001/results.json`
- **Report:** `reports/EXP-001.md`

### Baselines (single 80/20 split)

| Model | AUROC | AUPRC | Acc | CalErr |
|-------|-------|-------|-----|--------|
| entropy_threshold | 0.8681 | 0.8636 | 0.5800 | 0.2794 |
| logistic_regression | 0.9745 | 0.9788 | 0.9450 | 0.0624 |
| gradient_boosted_tree | 1.0000 | 1.0000 | 0.9950 | 0.0303 |

### Cross-validation sanity check (5-fold stratified, `experiments/EXP-001/cv_results.json`)

| Model | AUROC (mean ± std) | AUPRC (mean ± std) |
|-------|--------------------|--------------------|
| entropy_threshold | 0.8584 ± 0.0189 | 0.8496 ± 0.0161 |
| logistic_regression | 0.9666 ± 0.0056 | 0.9718 ± 0.0039 |
| gradient_boosted_tree | 0.9946 ± 0.0061 | 0.9932 ± 0.0074 |

CV confirms the single-split AUROC=1.0 was not over-fitting: the label
signal in aggregate trajectory features is genuinely (near-)separable.

### Temporal prediction

Mean token entropy differs strongly between correct and incorrect generations
at every generation fraction (Cohen's d from -2.34 at 10% to -1.30 at 100%,
all p < 1e-76). Incorrect trajectories are more uncertain at every fraction.

### Divergence analysis

Median detected divergence fraction is **0.0** (the very first token). In this
task the answer is the first integer emitted, so error "signals" are present
from token 1 — there is no evidence of a gradual pre-error degradation phase
within the 16-token window.

### Interpretation

- Aggregate uncertainty features (entropy, margin, cumulative log-prob) are
  highly predictive of correctness for this model/task (LR AUROC ~0.97,
  GBT ~0.99). This is a strong falsification check that the instrumentation and
  labels are meaningful.
- The finding is **confounded by task structure**: correctness commits at
  token 0-1, so early high entropy partially *is* the model being unsure of the
  answer it is about to emit.
- Not a demonstration of predictive *divergence* — a pre-error phase distinct
  from a confident-correct roll-out. That requires a task where the answer is
  reached only after a multi-token reasoning chain.

### Limitations

1. Single small model (distilgpt2), no instruction-following capability.
2. Answer commits at the first output token → no mid-trajectory decision point.
3. Synthetic task; real open-ended language tasks unaddressed.
4. Findings are correlational, not causal.

## Next experiments (candidate)

- **EXP-002:** Longer trajectories with a held answer — e.g., linear
  recurrence/sequence prediction where the model must emit 8+ tokens before a
  final answer, isolating the pre-error roll-out phase.
- **EXP-003:** Cross-model generalization (train on Model A, test on Model B).
- **EXP-004:** Real reasoning tasks on a larger instruction-following model
  (requires GPU), e.g., GSM8K with visible chain-of-thought.