# EXP-002: Mid-Chain Divergence Probes

## Question

Does the model ever correctly emit multiple terms of an arithmetic chain before diverging? In other words, is there a "mid-chain onset" where the model gets the first N terms right and then makes an error?

## Setup

- **Model:** distilgpt2 (124M parameters)
- **Probes:** 45 controlled generations (15 per prefix length P ∈ {2, 3, 4})
- **Task:** Given a P-term prefix of an arithmetic sequence, generate the continuation
- **Onset detection:** Greedy subsequence alignment of emitted integers vs expected continuation chain

## What Was Measured

For each probe:
1. Generate arithmetic sequence with known start and step.
2. Prompt with first P terms (e.g., "3, 5, 7,").
3. Model generates continuation; extract integers from generated text.
4. Align emitted integers against expected continuation as a greedy subsequence.
5. Record the onset index (position of first mismatch).

## Key Results

- **Mid-chain onset (onset ≥ 2): 0 cases across 45 probes.**
- **Onset = 0:** The first emitted integer was already wrong.
- **Onset = 1:** The first emitted integer was correct, the second was wrong.
- **Distribution:** P=2 → {0:15}, P=3 → {0:8, 1:2, ALL_CORRECT:5}, P=4 → {0:9, 1:1, ALL_CORRECT:5}

## Interpretation

The model either continues the entire chain correctly or diverges at the first token. There is no evidence of "partial correctness followed by error."

**Critical caveat:** This null result could be due to:
1. The task structure (answer commits at first token).
2. Low statistical power (only 45 probes).
3. Model limitations (distilgpt2 may not have the capacity for partial reasoning).
4. The model may have different behavior with larger models or different prompts.

## Files

- `scripts/scratch/probe_chains.py` — probe generation script
- `experiments/EXP-002/exp002_results.json` — probe results
- `reports/EXP-002.md` — detailed report

## How to Reproduce

```bash
python scripts/scratch/probe_chains.py
```

## Qwen2.5-0.5B-Instruct

Qwen2.5-0.5B-Instruct was planned for EXP-002 but could not be benchmarked due to hardware constraints (no GPU, model not cached, 1GB download infeasible). See `scripts/exp002_qwen.py` for the planned experiment.

## Next Experiment

EXP-006 will run larger sample sizes to check if the null result is due to low statistical power. EXP-005 will test other models. See `research/ROADMAP.md`.
