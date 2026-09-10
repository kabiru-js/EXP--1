# Model Trajectory Research

Investigating whether the inference trajectory of a language model contains measurable signals preceding incorrect answers.

## Question

Can we identify a measurable point during LLM generation where the model's trajectory begins to diverge toward an incorrect answer?

## Hypothesis

Incorrect generations may exhibit measurable changes in their inference trajectory before the final answer is produced.

**We do not assume this is true. The system is designed to falsify it.**

## Status

EXP-001 (baseline trajectory study) is complete: 1000 samples, results and
report available under `experiments/EXP-001/` and `reports/EXP-001.md`.
See `RESULTS.md` for the summary and honest interpretation.

## Method

1. Run reasoning questions through a language model with token-level instrumentation
2. Record per-token probability, entropy, log-probability, and top-k distributions at each generation step
3. Label each trajectory as correct or incorrect
4. Extract aggregate and temporal features from trajectories
5. Train baseline classifiers to predict correctness from trajectory features
6. Analyze temporal patterns: can correctness be predicted before generation completes?
7. Compare correct vs incorrect trajectory distributions

## Experiments

| ID | Name | Status | Dataset | Model | Samples |
|----|------|--------|---------|-------|---------|
| EXP-001 | Baseline trajectory study | Complete | synthetic_sequences | distilgpt2 | 1000 |

## Results

See `reports/` for experiment reports and `experiments/` for raw results.

## Architecture

```
src/
├── config.py              # YAML-based experiment configuration
├── datasets/              # Dataset loaders (GSM8K, TruthfulQA, synthetic)
├── models/                # Model backends (HuggingFace, vLLM)
├── inference/             # Generation pipeline with instrumentation
├── trajectories/          # Parquet + DuckDB trajectory storage
├── features/              # Feature extraction from trajectories
├── evaluation/            # Baseline models and metrics
├── analysis/              # Divergence analysis and temporal prediction
├── visualization/         # Publication-quality figures
├── reporting/             # Experiment report generation
├── experiments/           # Experiment runner orchestration
└── cli/                   # Terminal interface
```

## Reproduction

```bash
# Install
pip install -e ".[dev]"

# Run EXP-001 (full 1000 samples)
python scripts/run_exp001.py

# Quick test (100 samples)
python scripts/run_exp001.py --samples 100

# Analyze existing results
python scripts/run_exp001.py --analyze-only

# Run tests
pytest tests/ -v
```

## CLI

```bash
# Run experiment
research run configs/EXP-001.yaml

# Analyze
research analyze configs/EXP-001.yaml

# Inspect results
research inspect EXP-001

# List experiments
research list-experiments

# Generate report
research report configs/EXP-001.yaml
```

## Limitations

- Limited to models where token-level probabilities are accessible
- Answer extraction may not handle all answer formats
- Small-scale study; results may not generalize
- Does not establish causal relationships
- Single model, single dataset for EXP-001

## Open Questions

1. Do aggregate uncertainty features predict correctness better than chance?
2. Is there a temporal pattern that emerges before incorrect answers?
3. Does the signal generalize across model families?
4. Can the signal be used for intervention (early stopping, re-sampling)?

## Related Work

This project is informed by research in:
- Uncertainty estimation in language models
- Calibration and confidence
- Semantic entropy
- Hallucination detection
- Selective prediction
- Process supervision and verifier models
- Mechanistic interpretability

We explicitly distinguish what is already known from what this experiment tests. This project does not claim novelty.
