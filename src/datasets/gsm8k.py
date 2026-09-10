"""GSM8K dataset loader for trajectory research."""

from __future__ import annotations

import re
from typing import Optional

from datasets import load_dataset

from src.datasets.base import DatasetLoader, Sample


def _extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from GSM8K output (#### format)."""
    match = re.search(r"####\s*(.+?)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1]
    return text.strip()


def _normalize_number(s: str) -> float | None:
    """Normalize a numeric string to a float for comparison."""
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


class GSM8KLoader(DatasetLoader):
    """Loader for GSM8K grade-school math problems."""

    def name(self) -> str:
        return "gsm8k"

    def load(self, split: str = "test", num_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("gsm8k", "main", split=split, cache_dir="data/cache")
        if num_samples is not None:
            ds = ds.select(range(min(num_samples, len(ds))))
        samples = []
        for i, row in enumerate(ds):
            samples.append(
                Sample(
                    sample_id=f"gsm8k_{split}_{i}",
                    dataset="gsm8k",
                    question=row["question"],
                    ground_truth=row["answer"],
                    metadata={"cot": row["answer"]},
                )
            )
        return samples

    def extract_answer(self, raw_answer: str) -> str:
        return _extract_gsm8k_answer(raw_answer)

    def check_correctness(self, predicted: str, ground_truth: str) -> bool:
        pred_num = _normalize_number(predicted)
        gt_num = _normalize_number(ground_truth)
        if pred_num is not None and gt_num is not None:
            return abs(pred_num - gt_num) < 1e-4
        return predicted.strip().lower() == ground_truth.strip().lower()
