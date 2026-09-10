"""TruthfulQA dataset loader for trajectory research."""

from __future__ import annotations

import re
from typing import Optional

from datasets import load_dataset

from src.datasets.base import DatasetLoader, Sample


class TruthfulQALoader(DatasetLoader):
    """Loader for TruthfulQA generation task."""

    def name(self) -> str:
        return "truthfulqa"

    def load(self, split: str = "validation", num_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("truthfulqa", "generation", split=split, cache_dir="data/cache")
        if num_samples is not None:
            ds = ds.select(range(min(num_samples, len(ds))))
        samples = []
        for i, row in enumerate(ds):
            best_answer = row["best_answer"] if "best_answer" in row else ""
            samples.append(
                Sample(
                    sample_id=f"tqa_{split}_{i}",
                    dataset="truthfulqa",
                    question=row["question"],
                    ground_truth=best_answer,
                    metadata={
                        "correct_answers": row.get("correct_answers", []),
                        "incorrect_answers": row.get("incorrect_answers", []),
                        "category": row.get("category", ""),
                    },
                )
            )
        return samples

    def extract_answer(self, raw_answer: str) -> str:
        return raw_answer.strip().split("\n")[0][:200]

    def check_correctness(self, predicted: str, ground_truth: str) -> bool:
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()
        if gt_lower in pred_lower:
            return True
        gt_words = set(gt_lower.split())
        pred_words = set(pred_lower.split())
        if gt_words and len(gt_words & pred_words) / len(gt_words) > 0.7:
            return True
        return False
