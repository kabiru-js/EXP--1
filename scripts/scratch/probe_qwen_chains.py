"""Probe Qwen2.5-0.5B-Instruct on chain-continuation and CoT tasks.

Goal: find a task regime where the model produces several CORRECT
continuation terms before diverging (mid-chain onset), so EXP-002 can
analyze pre-error precursors. Also measures CPU throughput.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter

from src.models.registry import get_backend

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
)


def emitted_integers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"[-+]?\d+", text)]


def run_chain_probe(
    backend, question_template: str, steps: list[int], prefix_len: int,
    n_continuation: int, n: int, seed: int, tag: str,
) -> dict:
    rng = random.Random(seed)
    onset_counter: Counter[int] = Counter()
    correct = 0
    examples = []
    gen_times = []

    for i in range(n):
        step = rng.choice(steps)
        start = rng.randint(3, 80)
        total = prefix_len + n_continuation
        terms = [start + k * step for k in range(total)]
        prefix = ", ".join(str(t) for t in terms[:prefix_len])
        expected = terms[prefix_len:]
        question = question_template.format(prefix=prefix)

        t0 = time.time()
        result = backend.generate(
            prompt=CHAT_TEMPLATE.format(question=question),
            max_tokens=96,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            seed=seed,
        )[0]
        gen_times.append(time.time() - t0)

        emitted = emitted_integers(result.text)
        onset = None
        for idx, (em, ex) in enumerate(zip(emitted, expected)):
            if em != ex:
                onset = idx
                break
        if onset is None:
            if len(emitted) >= len(expected):
                correct += 1
                onset_counter[999] += 1
            else:
                onset = len(emitted)
                onset_counter[onset] += 1
        else:
            onset_counter[onset] += 1

        if len(examples) < 3:
            examples.append(
                {
                    "step": step,
                    "prefix": prefix,
                    "expected": expected,
                    "emitted": emitted,
                    "onset": onset,
                    "raw": result.text[:160].replace("\n", "\\n"),
                }
            )

    mean_time = sum(gen_times) / len(gen_times)
    return {
        "tag": tag,
        "correct": correct,
        "samples": n,
        "mean_s_per_sample": round(mean_time, 2),
        "onset_distribution": dict(sorted((k, v) for k, v in onset_counter.items())),
        "examples": examples,
    }


def main() -> None:
    backend = get_backend(
        "huggingface", model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="float32", device_map="cpu",
    )

    chain_q = "Continue this number sequence. List the next 7 terms: {prefix}, ..."
    tasks = [
        ("chain-easy", chain_q, [1, 1, 1, 1, 2, 2, 3, 4, 5, 6], 4, 7),
        ("chain-hard", chain_q, [5, 6, 7, 8, 9, 10, 11, 12], 4, 7),
        ("arith-step", "What is {expr}? Show your step-by-step work and give the final answer.", [1], 0, 0),
    ]

    for tag, q, steps, plen, ncont in tasks:
        if tag == "arith-step":
            rng = random.Random(7)
            results = []
            times = []
            for i in range(8):
                depth = rng.choice([2, 3])
                nums = [rng.randint(1, 25) for _ in range(depth + 1)]
                ops = [rng.choice(["+", "-"]) for _ in range(depth)]
                parts = [str(nums[0])]
                running = nums[0]
                for op, num in zip(ops, nums[1:]):
                    parts.append(f" {op} {num}")
                    running = running + num if op == "+" else running - num
                expr = "".join(parts)
                t0 = time.time()
                result = backend.generate(
                    prompt=CHAT_TEMPLATE.format(question=q.format(expr=expr)),
                    max_tokens=128, temperature=0.7, top_p=0.95, top_k=50, seed=7,
                )[0]
                times.append(time.time() - t0)
                last = emitted_integers(result.text)
                gt = str(running)
                results.append(
                    {"expr": expr, "gt": gt, "last_int": last[-1] if last else None,
                     "correct": last[-1] == int(gt) if last else False,
                     "raw": result.text[:220].replace("\n", "\\n")}
                )
            print(f"\n=== {tag} ===")
            print(f"  mean {sum(times)/len(times):.2f}s/sample")
            for r in results:
                print(f"  {r['expr']} | gt={r['gt']} last={r['last_int']} correct={r['correct']}")
                print(f"    raw: {r['raw']}")
        else:
            res = run_chain_probe(backend, q, steps, plen, ncont, n=8, seed=11, tag=tag)
            print(f"\n=== {tag} ===")
            print(f"  correct {res['correct']}/{res['samples']}, {res['mean_s_per_sample']}s/sample")
            print(f"  onset: { {('ALL_CORRECT' if k == 999 else k): v for k, v in res['onset_distribution'].items()} }")
            for ex in res["examples"]:
                print(f"    step={ex['step']} prefix={ex['prefix']!r} exp={ex['expected']} emit={ex['emitted']} onset={ex['onset']}")
                print(f"      raw: {ex['raw']}")


if __name__ == "__main__":
    main()