"""Probe distilgpt2's arithmetic-chain continuation behavior.

For a prefix of P terms of an arithmetic progression, ask the model to
produce the continuation, then locate the first WRONG term. We need
"onset" (first divergent term) to land mid-chain (index >= 2) often
enough for a pre-error precursor analysis to be feasible.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from src.models.registry import get_backend

STEP_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
N_CONTINUATION_TERMS = 7


def emitted_integers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"[-+]?\d+", text)]


def run_probe(prefix_len: int, n: int) -> dict:
    backend = get_backend(
        "huggingface", model_name="distilgpt2", dtype="float32", device_map="cpu"
    )
    rng = random.Random(42 + prefix_len * 7)

    onset_counter: Counter[int] = Counter()
    correct = 0
    examples = []

    for i in range(n):
        step = rng.choice(STEP_WEIGHTS)
        start = rng.randint(3, 60)
        total_terms = prefix_len + N_CONTINUATION_TERMS
        terms = [start + k * step for k in range(total_terms)]

        prompt = ", ".join(str(t) for t in terms[:prefix_len]) + ","
        expected = terms[prefix_len:]

        result = backend.generate(
            prompt=prompt,
            max_tokens=20,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            seed=42,
        )[0]

        emitted = emitted_integers(result.text)
        onset = None
        for idx, (em, ex) in enumerate(zip(emitted, expected)):
            if em != ex:
                onset = idx
                break
        if onset is None:
            if len(emitted) >= len(expected):
                correct += 1
                onset_counter[999] += 1  # marker: all terms correct
            else:
                onset = len(emitted)  # ran out of tokens before divergence
                onset_counter[onset] += 1
        else:
            onset_counter[onset] += 1

        if len(examples) < 3:
            examples.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "emitted": emitted,
                    "onset": onset,
                }
            )

    total_marked = sum(onset_counter.values())
    return {
        "prefix_len": prefix_len,
        "correct": correct,
        "samples": n,
        "onset_distribution": dict(
            sorted((k, v) for k, v in onset_counter.items())
        ),
        "examples": examples,
    }


def main() -> None:
    for p in [2, 3, 4]:
        res = run_probe(p, n=15)
        print(f"\n=== Prefix length P={p} ===")
        print(f"  Fully correct: {res['correct']}/{res['samples']}")
        print(f"  Onset distribution: { {('ALL_CORRECT' if k == 999 else k): v for k, v in res['onset_distribution'].items()} }")
        for ex in res["examples"]:
            print(f"    prompt={ex['prompt']!r} expected={ex['expected']} emitted={ex['emitted']} onset_ok={ex['onset'] is None or ex['onset']>=2}")


if __name__ == "__main__":
    main()