# Research Question

## Central Question

Can token-level generation trajectories of a language model contain measurable signals that indicate the model is going to produce an incorrect answer?

## Refined Question

Given a language model generating a sequence token by token, can we — from the token-level probability, entropy, and margin signals — determine before the generation completes that the final output will be incorrect?

---

## Hypothesis

Incorrect generations may exhibit measurable changes in their inference trajectory before the final answer is produced.

**We do not assume this is true. The system is designed to falsify it.**

## Null Hypothesis

Token-level trajectory signals (entropy, probability, margin, cumulative log-probability) are independent of whether the final generation is correct or incorrect. Any apparent separability is due to chance or confounding by task structure (e.g., the answer commits at the first token).

## Experimental Unit

**One observation = one generated trajectory.** A trajectory is the sequence of tokens a model emits after being prompted with a question, together with the per-token probability, entropy, margin, and cumulative log-probability recorded at each step.

- **EXP-001:** 1,000 trajectories (1000 samples × 1 generation each).
- **EXP-002:** 45 controlled probes (15 per prefix length P ∈ {2, 3, 4}).

## Outcome

**Correct vs incorrect, binary.**

- The model generates text; the first integer in the generated text is extracted as the predicted answer.
- The predicted answer is compared to the ground-truth answer (the correct next term in the arithmetic sequence).
- `correct = (int(predicted) == int(ground_truth))`.

**This definition is operational and reproducible. It does not assume anything about the model's reasoning process — only about the final extracted answer.**

## Predictors (Token-Level Signals)

At each generated token position `t`, the following are recorded:

| Signal | Definition |
|--------|-----------|
| `token_text` | The decoded token string |
| `probability` | P(token_t \| token_1...t-1) |
| `log_probability` | log P(token_t \| token_1...t-1) |
| `entropy` | H(P(token_t \| token_1...t-1)) = −Σ p log p |
| `top_k_probabilities` | Probabilities of the top-k most likely tokens |
| `probability_margin` | P(top-1) − P(top-2) |
| `cumulative_log_probability` | Σ_{i=1}^{t} log P(token_i \| token_1...i-1) |
| `logits` | Raw model logits at position t |

From these, 30 aggregate features are extracted per trajectory (mean, std, min, max, trend, acceleration, quartile statistics, etc.) — see `src/features/extraction.py`.

## Target

The classifier predicts **correctness** (binary) from the trajectory features. The temporal analysis predicts **whether correctness can be determined at a given generation fraction** (e.g., at 10% of tokens generated).

---

## What Is Being Tested

The experiment does **not** test whether the model "reasons" or "knows" it's wrong. It tests a narrower, operational claim:

> *Do token-level probability/entropy signals at early positions in the generation correlate with whether the final answer is correct?*

If yes → trajectory signals are predictive. If the correlation is strongest at early positions → it may indicate a pre-error signal (but see limitations: the answer may commit at the first token).

---

## Related Questions (Not Addressed Here)

- Whether the model internally "decides" it's wrong before emitting the wrong token.
- Whether the model can self-correct mid-generation.
- Whether uncertainty signals correspond to genuine reasoning processes (vs. surface-level statistical patterns).

These questions require different methodologies (mechanistic interpretability, process supervision, larger models) and are not claimed here.

---

## Falsifiability

This research is designed to be falsifiable:

- If trajectory signals do **not** separate correct from incorrect → the hypothesis is rejected.
- If separation exists **only** at the first token → the signal is a labeling artifact, not a divergence precursor.
- If no mid-chain divergence is observed → the task structure may commit the answer too early.

All of these outcomes are documented in the repository. Null results are treated as valid scientific findings.
