"""Inference pipeline: connects datasets, models, and trajectory storage."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from src.config import ExperimentConfig
from src.datasets.base import DatasetLoader, Sample
from src.datasets.registry import get_loader
from src.models.base import GenerationResult, ModelBackend
from src.models.registry import get_backend
from src.trajectories.storage import Trajectory, TrajectoryStorage, TokenMetrics

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Orchestrates generation and trajectory collection."""

    def __init__(self, config: ExperimentConfig, storage: TrajectoryStorage):
        self.config = config
        self.storage = storage
        self.loader: DatasetLoader = get_loader(config.dataset.name)
        self.backend: ModelBackend = get_backend(
            config.model.provider,
            model_name=config.model.name,
            dtype=config.model.dtype,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
        )

    def run(
        self,
        num_samples: Optional[int] = None,
        prompt_template: Optional[str] = None,
        resume: bool = False,
    ) -> list[Trajectory]:
        """Run inference and store trajectories.

        Args:
            num_samples: Number of samples (defaults to config).
            prompt_template: Prompt format string (defaults to config).
            resume: If True, skip sample IDs already present in storage.
        """
        n = num_samples or self.config.dataset.samples
        samples = self.loader.load(
            split=self.config.dataset.split,
            num_samples=n,
        )

        if resume:
            existing = {
                r["sample_id"] for r in self.storage.get_experiment_trajectories(self.config.id)
            }
            samples = [s for s in samples if s.sample_id not in existing]
            logger.info(f"Resume: {len(existing)} already stored, generated {len(samples)}")

        prompt_template = prompt_template or self.config.prompt_template
        logger.info(
            f"Running {len(samples)} samples through {self.backend.model_id()}"
        )

        trajectories = []
        gen = self.config.generation

        for sample in tqdm(samples, desc="Generating"):
            prompt = prompt_template.format(question=sample.question)
            try:
                results = self.backend.generate(
                    prompt=prompt,
                    max_tokens=gen.max_tokens,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    top_k=gen.top_k,
                    seed=gen.seed,
                    num_return_sequences=gen.num_return_sequences,
                )
            except Exception as e:
                logger.error(f"Error generating for {sample.sample_id}: {e}")
                continue

            for result in results:
                trajectory = self._build_trajectory(
                    sample=sample,
                    result=result,
                    prompt=prompt,
                    prompt_template=prompt_template,
                )
                trajectories.append(trajectory)

        self.storage.store_batch(trajectories)
        logger.info(
            f"Stored {len(trajectories)} trajectories "
            f"({sum(1 for t in trajectories if t.is_correct)} correct, "
            f"{sum(1 for t in trajectories if not t.is_correct)} incorrect)"
        )
        return trajectories

    def _build_trajectory(
        self,
        sample: Sample,
        result: GenerationResult,
        prompt: str,
        prompt_template: str,
    ) -> Trajectory:
        extracted = self.loader.extract_answer(result.text)
        is_correct = self.loader.check_correctness(extracted, sample.ground_truth)

        token_metrics = []
        for signal in result.signals:
            token_metrics.append(
                TokenMetrics(
                    token_id=signal.token_id,
                    token_text=signal.token_text,
                    token_index=signal.token_index,
                    probability=signal.probability,
                    log_probability=signal.log_probability,
                    entropy=signal.entropy,
                    top_k_token_ids=signal.top_k_token_ids,
                    top_k_token_texts=signal.top_k_token_texts,
                    top_k_probabilities=signal.top_k_probabilities,
                    probability_margin=signal.probability_margin,
                    cumulative_log_prob=signal.cumulative_log_prob,
                    has_logits=signal.logits is not None,
                    has_hidden_state=signal.hidden_state is not None,
                    has_attention=signal.attention is not None,
                )
            )

        return Trajectory(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            model_id=result.model_id,
            model_version=result.model_version,
            experiment_id=self.config.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            prompt=prompt,
            prompt_token_count=result.prompt_token_count,
            generated_text=result.text,
            extracted_answer=extracted,
            is_correct=is_correct,
            generation_tokens=result.generation_tokens,
            temperature=self.config.generation.temperature,
            seed=self.config.generation.seed,
            prompt_template=prompt_template,
            token_metrics=token_metrics,
            available_signals=result.available_signals,
            metadata=sample.metadata,
        )
