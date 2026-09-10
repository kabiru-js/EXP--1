"""EXP-002: CPU feasibility experiment with Qwen2.5-0.5B-Instruct.

Goal: determine whether Qwen2.5-0.5B-Instruct ever shows a trajectory that is
INITIALLY consistent with a correct step-by-step solution, then later diverges
toward an incorrect final answer. Preserve full per-token instrumentation.

Detection: for each arithmetic expression, the correct chain of intermediate
values is known (e.g. 5 + 3 - 2 -> [5, 8, 6]). We align emitted integers
against this expected chain as a subsequence. The first unmatched expected
value is the suspected onset.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from src.models.registry import get_backend

logging.basicConfig(level=logging.WARNING)

N_SAMPLES = 6
SEED = 2026
DEPTHS = [2]
OPS = ["+", "-"]
NUM_RANGE = (1, 20)

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nSolve the following step by step. "
    "Write each intermediate step and the final answer.\n"
    "Problem: {expr}<|im_end|>\n<|im_start|>assistant\n"
)


def generate_problem(rng: random.Random) -> tuple[str, int, list[str], list[str]]:
    depth = rng.choice(DEPTHS)
    nums = [rng.randint(*NUM_RANGE) for _ in range(depth + 1)]
    ops = [rng.choice(OPS) for _ in range(depth)]

    parts = [str(nums[0])]
    running = nums[0]
    chain_values = [running]
    for op, num in zip(ops, nums[1:]):
        parts.append(f" {op} {num}")
        running = running + num if op == "+" else running - num
        chain_values.append(running)
    expr = "".join(parts)
    answer = str(running)
    ops_str = ops
    return expr, running, chain_values, ops_str


def emitted_integers(text: str) -> list[int]:
    return [int(x) for x in __import__("re").findall(r"[-+]?\d+", text)]


def detect_onset(emitted: list[int], expected_chain: list[int]) -> tuple[int, str]:
    """Greedy subsequence alignment of expected_chain in emitted.

    Returns (onset_index, explanation) where onset_index is the index in
    expected_chain of the first value NOT found as a subsequence.
    """
    pos = 0
    matched = 0
    for ev in expected_chain:
        while pos < len(emitted) and emitted[pos] != ev:
            pos += 1
        if pos < len(emitted):
            matched += 1
            pos += 1
        else:
            break
    if matched == len(expected_chain):
        return len(expected_chain), "all expected values matched as subsequence"
    # Find the first unmatched expected value
    # Report what was expected but not found
    return (
        matched,
        f"expected value {expected_chain[matched]} not found after matching {matched} prefix",
    )


def main() -> None:
    backend = get_backend(
        "huggingface", model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="float32", device_map="cpu",
    )

    rng = random.Random(SEED)
    samples = []
    t_start = time.time()
    gen_times = []

    for i in range(N_SAMPLES):
        expr, answer, chain_values, ops_str = generate_problem(rng)
        question = CHAT_TEMPLATE.format(expr=expr)

        t0 = time.time()
        result = backend.generate(
            prompt=question,
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            seed=SEED + i,
        )[0]
        gen_times.append(time.time() - t0)

        emitted = emitted_integers(result.text)
        onset_idx, onset_explanation = detect_onset(emitted, chain_values)
        last_int = emitted[-1] if emitted else None
        final_correct = last_int == answer if last_int is not None else False
        fully_correct = onset_idx == len(chain_values) and final_correct

        # Determine divergence category
        # initially_correct_then_diverged: matched >= 1 prefix value AND final wrong
        initially_correct_then_diverged = (
            onset_idx >= 1 and not final_correct
        )

        # Build token signal records
        token_signals = []
        for s in result.signals:
            token_signals.append({
                "token_text": s.token_text,
                "token_index": s.token_index,
                "probability": s.probability,
                "log_probability": s.log_probability,
                "entropy": round(s.entropy, 4),
                "top_k_token_texts": s.top_k_token_texts[:5],
                "top_k_probabilities": [round(p, 4) for p in s.top_k_probabilities[:5]],
                "probability_margin": round(s.probability_margin, 4),
                "cumulative_log_prob": round(s.cumulative_log_prob, 4),
                "logits_shape": list(s.logits.shape) if s.logits is not None else None,
            })

        samples.append({
            "index": i,
            "expr": expr,
            "ground_truth": str(answer),
            "chain_values": chain_values,
            "question": question,
            "generated_text": result.text,
            "emitted_integers": emitted,
            "last_integer": last_int,
            "final_correct": final_correct,
            "fully_correct": fully_correct,
            "onset_idx": onset_idx,
            "onset_explanation": onset_explanation,
            "initially_correct_then_diverged": initially_correct_then_diverged,
            "gen_time_s": round(gen_times[-1], 2),
            "token_count": len(result.token_ids),
            "token_signals": token_signals,
            "raw_excerpt": result.text[:240].replace("\n", "\\n"),
        })

        print(f"[{i}] expr={expr} answer={answer} emitted={emitted} "
              f"onset={onset_idx} final_ok={final_correct} div={initially_correct_then_diverged} "
              f"time={gen_times[-1]:.1f}s")

    total_time = time.time() - t_start
    n_gen = len(gen_times)
    total_toks = sum(s["token_count"] for s in samples)
    n_incorrect = sum(1 for s in samples if not s["final_correct"])
    n_mid_diverge = sum(1 for s in samples if s["initially_correct_then_diverged"])
    n_onset_ge1 = sum(1 for s in samples if s["onset_idx"] >= 1 and not s["fully_correct"])

    summary = {
        "experiment_id": "EXP-002",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "cpu",
        "n_samples_requested": N_SAMPLES,
        "n_valid_generations": n_gen,
        "total_runtime_s": round(total_time, 1),
        "mean_gen_time_s": round(np.mean(gen_times), 2) if gen_times else None,
        "samples_per_sec": round(n_gen / total_time, 3) if total_time else None,
        "total_tokens": total_toks,
        "tokens_per_sec": round(total_toks / total_time, 1) if total_time else None,
        "n_incorrect_final": n_incorrect,
        "n_incorrect_rate": round(n_incorrect / n_gen, 3) if n_gen else None,
        "n_onset_at_least_1": n_onset_ge1,
        "n_initially_correct_then_diverged": n_mid_diverge,
        "n_fully_correct": sum(1 for s in samples if s["fully_correct"]),
    }

    out_dir = Path("experiments") / "EXP-002"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exp002_results.json", "w") as f:
        json.dump({"summary": summary, "samples": samples}, f, indent=2)
    print(f"\nSaved to {out_dir / 'exp002_results.json'}")
    print(json.dumps(summary, indent=2))

    # Generate EXP-002 report
    report_path = out_dir / "EXP-002.md"
    with open(report_path, "w") as f:
        f.write("# Experiment Report: EXP-002\n\n")
        f.write(f"**Model:** {summary['model']} on {summary['device']}\n\n")
        f.write("## Question\n\n")
        f.write("Does the model's trajectory appear **initially consistent with a "
                "correct step-by-step solution**, then later transition toward an "
                "incorrect answer?\n\n")
        f.write("## Summary\n\n")
        for k in ["total_runtime_s", "samples_per_sec", "tokens_per_sec",
                  "n_valid_generations", "n_incorrect_final",
                  "n_initially_correct_then_diverged", "n_fully_correct"]:
            f.write(f"- **{k}:** {summary[k]}\n")
        f.write("\n")
        f.write("## Findings\n\n")
        if n_mid_diverge == 0:
            f.write("**No cases of 'initially correct then diverged' were observed "
                    "in this sample.** Every trajectory either was fully correct "
                    "from the start or diverged at the very first expected value.\n\n")
            f.write("### Distinction\n\n"
                    "This result distinguishes:\n\n"
                    "- **'the phenomenon appears absent'** for this model/task "
                    "combination at this sample size, from\n"
                    "- **'this model/task combination did not expose the "
                    "phenomenon.'**\n\n"
                    "The latter would require a different task family (e.g. longer "
                    "chains, multi-digit arithmetic, or tasks where the answer is "
                    "a multi-token object produced late). The current evidence is "
                    "insufficient to conclude the phenomenon is absent in general.\n\n")
        else:
            f.write(f"**{n_mid_diverge} cases** of initially-correct-then-diverged "
                    "trajectories were observed. Representative examples:\n\n")
            for s in samples:
                if s["initially_correct_then_diverged"]:
                    f.write(f"- `{s['expr']}` = {s['ground_truth']}; "
                            f"emitted integers={s['emitted_integers']}; "
                            f"onset at chain index {s['onset_idx']} "
                            f"({s['onset_explanation']})\n")
                    f.write(f"  `{s['raw_excerpt']}`\n\n")
        f.write("## Representative Trajectories\n\n")
        for s in samples[:5]:
            f.write(f"### `{s['expr']}` = {s['ground_truth']}\n\n")
            f.write(f"- Correct: {s['fully_correct']}\n")
            f.write(f"- Emitted integers: {s['emitted_integers']}\n")
            f.write(f"- Onset: {s['onset_idx']} ({s['onset_explanation']})\n")
            f.write(f"- Time: {s['gen_time_s']}s, tokens: {s['token_count']}\n")
            f.write(f"- Raw: `{s['raw_excerpt']}`\n\n")
        f.write("## Conclusion\n\n")
        f.write("EXP-002 establishes the per-token instrumentation and the "
                "onset-detection methodology. The key open question — whether "
                "the 'initially correct then diverged' phenomenon occurs at a "
                "non-trivial rate — requires either a larger sample, a different "
                "task family, or a more capable model. See `exp002_results.json` "
                "for full token-level signals of all generations.\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()