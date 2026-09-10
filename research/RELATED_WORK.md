# Related Work

This section identifies the research areas most relevant to this project. **No fabricated citations are included.** References are listed only where they can be verified from the codebase and existing knowledge.

## Token-Level Uncertainty in LLM Generation

The project measures per-token entropy, probability, and margin during autoregressive generation. This is closely related to:

- **Calibration of language models.** Whether the predicted probabilities of language models match their empirical accuracy. This project uses entropy as a proxy for uncertainty.
- **Token-level confidence scores.** Work on extracting confidence estimates from transformer models at each generation step (e.g., using the probability of the chosen token).

## Detecting Errors in Model Outputs

The project asks whether a model's own uncertainty signals can flag incorrect answers before generation completes. Related areas include:

- **Self-correction in LLMs.** Methods where a model identifies and fixes its own errors during generation (e.g., draft-and-verify, self-refine).
- **Answer abstention.** Allowing models to refuse to answer when uncertain. This project's binary correctness definition is a simpler analog.
- **Process supervision.** Training models to provide intermediate feedback during reasoning (e.g., OpenAI's process reward models). This project does not use process supervision but could benefit from it.

## Temporal Analysis of Generation

The project analyzes how token-level signals evolve over the course of generation. Related work includes:

- **Trajectory analysis of reasoning.** Studying how hidden states or attention patterns change during chain-of-thought generation.
- **Early stopping for generation.** Determining when to stop generating based on accumulated evidence. This project's temporal analysis is a diagnostic analog.

## Arithmetic Reasoning in Language Models

The `synthetic_sequences` dataset is an arithmetic-sequence continuation task. Related work includes:

- **GSM8K (math word problems).** A dataset of grade-school math problems requiring multi-step reasoning. The repository includes a GSM8K loader (`src/datasets/gsm8k.py`) but has not yet run experiments on it.
- **Symbolic arithmetic.** Tasks requiring exact arithmetic operations (addition, subtraction, multiplication). The synthetic dataset used here is a simplified version.
- **Chain-of-thought prompting.** Prompting strategies that encourage step-by-step reasoning. This project does NOT use chain-of-thought prompts — the prompt is just the comma-terminated sequence.

## Null Results and Scientific Reporting

EXP-002 documents a null result (no mid-chain divergence). Related discussions include:

- **Publication bias toward positive results.** The scientific community often favors positive findings over null results. This project explicitly documents null results as valid findings.
- **Falsifiability.** The project is designed to be falsifiable — if signals do not separate correct from incorrect, the hypothesis is rejected. This is a core design principle.
- **Reproducibility crises in ML.** Many ML results are difficult to reproduce due to nondeterminism, hardware differences, or undocumented hyperparameters. This project prioritizes reproducibility (see `REPRODUCIBILITY.md`).

## What Is NOT Related

The following areas are NOT part of this project's scope, even though they may be superficially related:

- **Mechanistic interpretability.** This project does not study internal representations, attention patterns, or circuit analysis.
- **Training or fine-tuning.** No model training is performed. All experiments use pretrained models.
- **Prompt engineering.** The prompt template is minimal (`"{question}"`). No instruction tuning or prompt optimization is attempted.
- **Benchmarking.** This is not a benchmark. It is a focused investigation of a specific hypothesis.

## Reading List (TODO)

The following topics should be investigated to complete the related work section:

- [ ] Papers on entropy as a measure of uncertainty in autoregressive generation
- [ ] Papers on detecting errors in LLM outputs (self-correction, abstention)
- [ ] Papers on temporal analysis of reasoning trajectories
- [ ] Papers on calibration of language models
- [ ] GSM8K task literature and model performance
- [ ] Process reward models and process supervision
- [ ] Null result reporting in ML research
- [ ] Reproducibility in LLM experiments

**No citations are listed above because they have not been verified.** This section should be expanded with verified references before publication.
