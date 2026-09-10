# Research Roadmap

## Status: EXP-001 Complete, EXP-002 Incomplete on Hardware

---

## Completed Experiments

### EXP-001: Baseline Trajectory Study
- **Question:** Can token-level signals separate correct from incorrect generations?
- **Result:** Yes — strong separability (AUROC 0.97–1.0, Cohen's d = −2.34 at 10%).
- **Finding:** Uncertainty at the first token strongly predicts correctness.
- **Status:** Documented in `reports/EXP-001.md`, `results/EXP-001/`.

### EXP-002: Mid-Chain Divergence Probes
- **Question:** Does the model ever correctly emit multiple terms before diverging?
- **Result:** No — 45 controlled probes, 0 mid-chain onset cases (distilgpt2).
- **Finding:** The model either continues the whole chain correctly or diverges at the first token.
- **Status:** Documented in `reports/EXP-002.md`. Qwen2.5-0.5B-Instruct not benchmarked (no GPU, model not cached).

---

## Planned Experiments

### EXP-003: Prompt-Position Uncertainty (HIGH PRIORITY)
- **Question:** What is the model's uncertainty at the last prompt token (before generation starts)?
- **Rationale:** EXP-001 shows signal at token 0, but this may be a labeling artifact. EXP-003 would measure the model's state BEFORE generation to see if uncertainty is already elevated.
- **Method:** Modify `_TrajectoryLogitsProcessor` to also record signals at the last prompt token position. Use `model.input_ids` to get prompt-end hidden states.
- **Expected outcome:** If uncertainty at prompt-end is already elevated for incorrect cases → genuine precursor. If not → the token-0 signal is likely a labeling artifact.

### EXP-004: Multiple Task Families
- **Question:** Does the phenomenon generalize beyond arithmetic-sequence continuation?
- **Rationale:** EXP-001 and EXP-002 use a single task family. Other tasks may expose different behaviors.
- **Tasks to test:**
  - Visible chain-of-thought arithmetic (where intermediate steps are visible)
  - Multi-step reasoning tasks (e.g., GSM8K-style)
  - Tasks where the answer does NOT commit at the first token
- **Method:** Add new dataset loaders to `src/datasets/`. Use the same instrumentation.

### EXP-005: Model Comparison
- **Question:** Does the phenomenon appear in other models?
- **Models:**
  - GPT-2 (smaller, faster on CPU — may be feasible)
  - phi-2 (small, may be feasible)
  - Qwen2.5-0.5B-Instruct (requires GPU or longer download time)
  - Larger models if GPU becomes available
- **Method:** Same instrumentation, same task. Compare AUROC, Cohen's d, and onset distributions.

### EXP-006: Larger Sample Sizes for EXP-002
- **Question:** Is the null result in EXP-002 due to low statistical power?
- **Rationale:** 45 probes may be underpowered. A larger sample might reveal rare mid-chain divergence cases.
- **Method:** Run more probes (e.g., 200+) on a faster setup or with GPU acceleration.

### EXP-007: Different Prompt Templates
- **Question:** Does the prompt structure affect the phenomenon?
- **Rationale:** EXP-001 uses `"{question}"` (no instruction wrapper). Adding instructions might change behavior.
- **Templates to test:**
  - `"Answer: {question}"`
  - `"What is the next term? {question}"`
  - `"Continue: {question}"`
- **Method:** Same task, same model, different prompt templates. Compare AUROC and temporal predictions.

---

## Long-Term Goals

1. **Formalize the null hypothesis testing framework.** Present results with confidence intervals and effect sizes.
2. **Extend to visible chain-of-thought tasks** where the answer does not commit at the first token.
3. **Build a reusable benchmarking framework** for trajectory-based evaluation of LLMs.
4. **Publish findings** (pending verification and expanded experiments).

---

## Blockers

- **No GPU.** All large-model experiments (Qwen, GPT-2-large, etc.) are blocked.
- **Slow CPU generation.** distilgpt2 at ~1.4s/sample limits sample sizes.
- **Qwen model not cached.** Download was infeasible within session timeouts.
- **pytest hangs.** Testing is limited to `python tests/smoke_test.py`.

---

## Dependencies for Planned Experiments

| Experiment | Requires | Status |
|------------|----------|--------|
| EXP-003 | Code modification to `_TrajectoryLogitsProcessor` | Not started |
| EXP-004 | New dataset loaders | Not started |
| EXP-005 | GPU or faster CPU, cached models | Blocked (no GPU) |
| EXP-006 | GPU or faster CPU | Blocked (no GPU) |
| EXP-007 | Same instrumentation, different prompts | Not started |
