"""Synthetic reasoning datasets for trajectory research."""

from __future__ import annotations

import random
import re
from typing import Optional

from src.datasets.base import DatasetLoader, Sample


def _generate_arithmetic_chain(seed: int, num_samples: int) -> list[Sample]:
    """Generate small-number arithmetic problems a small model can partially solve.

    Uses only + and - with small numbers (1-20) so that standard arithmetic
    evaluation is unambiguous (left-to-right equals PEMDAS for these ops).
    """
    rng = random.Random(seed)
    samples = []
    for i in range(num_samples):
        depth = rng.choice([1, 1, 2, 2, 3])
        numbers = [rng.randint(1, 20) for _ in range(depth + 1)]
        ops = [rng.choice(["+", "-"]) for _ in range(depth)]

        expression_parts = [str(numbers[0])]
        result = numbers[0]
        for op, num in zip(ops, numbers[1:]):
            expression_parts.append(f" {op} {num}")
            if op == "+":
                result += num
            else:
                result -= num

        expression = "".join(expression_parts)
        question = f"What is {expression}?"
        ground_truth = str(result)

        chain = []
        running = numbers[0]
        chain.append(f"{running}")
        for op, num in zip(ops, numbers[1:]):
            if op == "+":
                running += num
            else:
                running -= num
            chain.append(f"{op} {num} = {running}")

        samples.append(
            Sample(
                sample_id=f"synth_arith_{i}",
                dataset="synthetic_arithmetic",
                question=question,
                ground_truth=ground_truth,
                metadata={"chain": ", ".join(chain), "depth": depth},
            )
        )
    return samples


class SyntheticArithmeticLoader(DatasetLoader):
    """Loader for synthetic multi-step arithmetic problems."""

    def name(self) -> str:
        return "synthetic_arithmetic"

    def load(self, split: str = "test", num_samples: Optional[int] = None) -> list[Sample]:
        n = num_samples or 200
        seed = 12345 if split == "train" else 99999
        return _generate_arithmetic_chain(seed, n)

    def extract_answer(self, raw_answer: str) -> str:
        import re
        numbers = re.findall(r"[-+]?\d*\.?\d+", raw_answer)
        if numbers:
            return numbers[-1]
        return raw_answer.strip()

    def check_correctness(self, predicted: str, ground_truth: str) -> bool:
        try:
            return abs(float(predicted) - float(ground_truth)) < 1e-4
        except ValueError:
            return predicted.strip() == ground_truth.strip()


def _generate_sequences(seed: int, num_samples: int) -> list[Sample]:
    """Generate arithmetic-sequence continuation problems.

    Each item shows 7 terms of an arithmetic progression and asks for the
    next term. The step distribution is weighted toward consecutive runs
    (step=1) so that a small base model produces a roughly balanced mix of
    correct and incorrect answers. All sequences use starts in a range that
    produces multi-digit answers, giving multi-token answer trajectories.
    """
    rng = random.Random(seed)
    samples = []
    for i in range(num_samples):
        # Weighted step distribution: ~45% consecutive, rest larger steps.
        step = rng.choice([1, 1, 1, 1, 2, 2, 3, 4, 5, 6])
        start = rng.randint(3, 60)
        n_terms = 7
        terms = [start + k * step for k in range(n_terms)]
        next_term = start + n_terms * step
        prompt = ", ".join(str(t) for t in terms) + ","
        samples.append(
            Sample(
                sample_id=f"synth_seq_{i}",
                dataset="synthetic_sequences",
                question=prompt,
                ground_truth=str(next_term),
                metadata={"step": step, "start": start, "n_terms": n_terms},
            )
        )
    return samples


class SyntheticSequenceLoader(DatasetLoader):
    """Loader for arithmetic-sequence continuation problems."""

    def name(self) -> str:
        return "synthetic_sequences"

    def load(self, split: str = "test", num_samples: Optional[int] = None) -> list[Sample]:
        n = num_samples or 500
        seed = 12345 if split == "train" else 54321
        return _generate_sequences(seed, n)

    def extract_answer(self, raw_answer: str) -> str:
        m = re.match(r"\s*([-+]?\d+)", raw_answer)
        if m:
            return m.group(1)
        numbers = re.findall(r"[-+]?\d+", raw_answer)
        return numbers[0] if numbers else raw_answer.strip()

    def check_correctness(self, predicted: str, ground_truth: str) -> bool:
        try:
            return int(predicted) == int(ground_truth)
        except ValueError:
            return predicted.strip() == ground_truth.strip()
