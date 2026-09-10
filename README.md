# Model Trajectory Research

**Does the token-level generation trajectory of a language model contain measurable signals preceding incorrect answers?**

This project is designed to **falsify** this hypothesis, not confirm it. Null results are treated as valid scientific findings.

---

## Research Question

> Can token-level generation trajectories of a language model contain measurable signals that indicate whether the final answer will be correct or incorrect?

**Null hypothesis:** Token-level trajectory signals (entropy, probability, margin, cumulative log-probability) are independent of whether the final generation is correct or incorrect.

---

## Key Findings

| Experiment | Result | Status |
|------------|--------|--------|
| **EXP-001** | Token-level uncertainty strongly separates correct from incorrect (LR AUROC=0.975, GBT AUROC=0.995±0.006 CV, Cohen's d=−2.34 at 10%) | **Complete** |
| **EXP-002** | No mid-chain divergence observed across 45 controlled probes (distilgpt2) | **Null result** |

**Critical caveat:** In the task used (arithmetic-sequence continuation), the answer commits at the first token. The token-0 signal is the model's uncertainty about the first token, not a precursor to it. See `research/LIMITATIONS.md`.

---

## Experiments

### EXP-001: Baseline Trajectory Study

- **Model:** distilgpt2 (124M), CPU
- **Dataset:** synthetic_sequences (7-term arithmetic progressions)
- **Samples:** 1,000 (466 correct / 534 incorrect)
- **Instrumentation:** Per-token probability, entropy, margin, cumulative log-probability, top-k, logits
- **Analysis:** 3 baseline classifiers, 5-fold CV, temporal prediction (Cohen's d at each generation fraction)
- **Results:** [See RESULTS.md](RESULTS.md)
- **Config:** `configs/EXP-001.yaml`
- **Data:** `experiments/EXP-001/`

### EXP-002: Mid-Chain Divergence Probes

- **Model:** distilgpt2 (CPU); Qwen2.5-0.5B-Instruct (deferred — no GPU)
- **Probes:** 45 controlled arithmetic-chain continuations (P=2/3/4)
- **Onset detection:** Greedy subsequence alignment of emitted integers vs expected continuation
- **Result:** **0 mid-chain onset cases** — the model either continues the whole chain correctly or diverges at the first token
- **Data:** `experiments/EXP-002/`

---

## Method

1. Load dataset (arithmetic-sequence continuation tasks)
2. Generate text with token-level instrumentation (`_TrajectoryLogitsProcessor` captures per-token probability, entropy, margin, top-k, cumulative log-prob, logits)
3. Extract the answer (first integer in generated text)
4. Label correctness (`int(predicted) == int(ground_truth)`)
5. Store trajectories (one Parquet file per trajectory + DuckDB metadata)
6. Extract 30 aggregate features per trajectory
7. Train baseline classifiers (entropy threshold, logistic regression, gradient boosted tree)
8. Compute AUROC, AUPRC, accuracy on held-out test set
9. Verify with 5-fold stratified cross-validation
10. Analyze temporal prediction (Cohen's d at each generation fraction)

See `research/METHODOLOGY.md` for complete methodology documentation.

---

## Repository Structure

```
src/                          # Core source code
  config.py                   # YAML-based experiment configuration
  datasets/                   # Dataset loaders (GSM8K, TruthfulQA, synthetic)
  models/                     # Model backends (HuggingFace)
  inference/                  # Generation pipeline with instrumentation
  trajectories/               # Parquet + DuckDB trajectory storage
  features/                   # Feature extraction (30 aggregate features)
  evaluation/                 # Baseline models and metrics
  analysis/                   # Divergence analysis and temporal prediction
  visualization/              # Publication-quality plotting
  reporting/                  # Experiment report generation
  experiments/                # Experiment runner orchestration
  cli/                        # Terminal interface
configs/                      # Experiment YAML configs
scripts/                      # Run scripts and probes
experiments/                  # Experiment data and results
  EXP-001/                    # Full results (1000 trajectories, figures, CV)
  EXP-002/                    # Probe results (45 probes)
research/                     # Research methodology docs
tests/                        # Test suite (smoke_test.py passes)
figures/                      # Publication-quality figures
reports/                      # Markdown reports
```

---

## Reproducing the Experiments

```bash
# Install
pip install -e ".[dev]"

# Run EXP-001 (full 1000 samples, ~23 min on CPU)
python scripts/run_exp001.py

# Quick test (100 samples)
python scripts/run_exp001.py --samples 100

# Analyze existing data (regenerate figures)
python scripts/run_exp001.py --analyze-only

# Cross-validation check
python scripts/cv_check_exp001.py

# Generate all figures from existing results
python scripts/make_figures.py

# Run tests
python tests/smoke_test.py
```

**Note:** `pytest tests/` hangs in PowerShell. Use `python tests/smoke_test.py` instead.

See `research/REPRODUCIBILITY.md` for complete reproducibility guide.

---

## Limitations

- **Answer commits at token 0-1** in the task used — the signal at early fractions is the first token's uncertainty, not a precursor
- **Single task family** (arithmetic-sequence continuation) — other task families may behave differently
- **Single model** (distilgpt2) — results may not generalize
- **No GPU** — all experiments run on CPU; larger models not benchmarked
- **Binary correctness** — partial correctness is not captured by baseline labels
- **No prompt-position signals** — uncertainty at the last prompt token is not measured
- **Correlation ≠ causation** — the experiment measures correlation, not causal relationships

See `research/LIMITATIONS.md` for the complete limitations list.

---

## Roadmap

| Experiment | Question | Status |
|------------|----------|--------|
| EXP-003 | What is the model's uncertainty at the last prompt token? | Not started |
| EXP-004 | Does the phenomenon generalize to other task families? | Not started |
| EXP-005 | Does the phenomenon appear in other models? | Blocked (no GPU) |
| EXP-006 | Is the EXP-002 null result due to low sample size? | Blocked (no GPU) |
| EXP-007 | Does prompt structure affect the phenomenon? | Not started |

See `research/ROADMAP.md` for details.

---

## Status

- EXP-001: **Complete** — documented in `reports/EXP-001.md`
- EXP-002: **Incomplete on hardware** — null result documented in `reports/EXP-002.md`
- Research restructure: **Complete** — `AUDIT.md`, `research/`, experiment READMEs
- Next: EXP-003 (prompt-position uncertainty) requires code modification

---

## What This Project Is NOT

- **Not a benchmark** — this is a focused investigation of a specific hypothesis
- **Not mechanistic interpretability** — no internal representation analysis
- **Not training/fine-tuning** — all experiments use pretrained models
- **Not prompt engineering** — the prompt template is minimal (`"{question}"`)
- **Not causal inference** — correlation between uncertainty and correctness is measured, not causation

---

## Related Areas

- Token-level uncertainty in LLM generation
- Detecting errors in model outputs (self-correction, abstention)
- Temporal analysis of reasoning trajectories
- Arithmetic reasoning in language models (GSM8K, MATH)
- Null result reporting in ML research
- Reproducibility in LLM experiments

See `research/RELATED_WORK.md` for a structured discussion (TODO reading list, no fabricated references).
