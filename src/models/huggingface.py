"""HuggingFace Transformers model backend."""

from __future__ import annotations

import logging
import os
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from src.models.base import GenerationResult, ModelBackend, TokenSignal

logger = logging.getLogger(__name__)


class _TrajectoryLogitsProcessor(LogitsProcessor):
    """Collects logits at each step for trajectory analysis."""

    def __init__(self):
        self.all_logits: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.all_logits.append(scores.detach().cpu())
        return scores


class HuggingFaceBackend(ModelBackend):
    """HuggingFace Transformers backend with token-level instrumentation."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ):
        self._model_name = model_name
        logger.info(f"Loading model: {model_name}")
        torch.set_num_threads(max(1, os.cpu_count() or 2))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = getattr(torch, dtype, torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.model.eval()
        logger.info(f"Model loaded: {model_name}")

    def model_id(self) -> str:
        return f"huggingface/{self._model_name}"

    def model_version(self) -> str:
        return self._model_name

    def available_signals(self) -> list[str]:
        return [
            "token_id",
            "token_text",
            "probability",
            "log_probability",
            "entropy",
            "top_k",
            "probability_margin",
            "cumulative_log_prob",
            "logits",
        ]

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        seed: int = 42,
        num_return_sequences: int = 1,
    ) -> list[GenerationResult]:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = input_ids.shape[1]

        logits_processor = _TrajectoryLogitsProcessor()
        processor_list = LogitsProcessorList([logits_processor])

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            num_return_sequences=num_return_sequences,
            logits_processor=processor_list,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            output = self.model.generate(input_ids, **gen_kwargs)

        results = []
        for seq_idx in range(output.shape[0]):
            generated_ids = output[seq_idx, prompt_len:].tolist()
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_text = self.tokenizer.decode(output[seq_idx], skip_special_tokens=True)

            signals = self._build_signals(
                generated_ids, logits_processor.all_logits, top_k_count=10
            )

            token_texts = [self.tokenizer.decode([tid]) for tid in generated_ids]

            results.append(
                GenerationResult(
                    text=generated_text,
                    token_ids=generated_ids,
                    token_texts=token_texts,
                    signals=signals,
                    model_id=self.model_id(),
                    model_version=self.model_version(),
                    available_signals=self.available_signals(),
                    prompt=prompt,
                    prompt_token_count=prompt_len,
                    generation_tokens=len(generated_ids),
                    seed=seed,
                )
            )

        return results

    def _build_signals(
        self,
        token_ids: list[int],
        all_logits: list[torch.Tensor],
        top_k_count: int = 10,
    ) -> list[TokenSignal]:
        signals = []
        cumulative_log_prob = 0.0

        for t, token_id in enumerate(token_ids):
            if t >= len(all_logits):
                break

            logits = all_logits[t].float()
            if logits.dim() > 1:
                logits = logits[0]

            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            token_log_prob = log_probs[token_id].item()
            token_prob = probs[token_id].item()
            cumulative_log_prob += token_log_prob

            entropy = -(probs * log_probs).sum().item()

            top_k_probs, top_k_ids = torch.topk(probs, min(top_k_count, probs.shape[-1]))
            top_k_texts = [
                self.tokenizer.decode([tid.item()]) for tid in top_k_ids
            ]

            top1_prob = top_k_probs[0].item() if len(top_k_probs) > 0 else 0.0
            top2_prob = top_k_probs[1].item() if len(top_k_probs) > 1 else 0.0
            margin = top1_prob - top2_prob

            signals.append(
                TokenSignal(
                    token_id=token_id,
                    token_text=self.tokenizer.decode([token_id]),
                    token_index=t,
                    probability=token_prob,
                    log_probability=token_log_prob,
                    entropy=entropy,
                    top_k_token_ids=top_k_ids.tolist(),
                    top_k_token_texts=top_k_texts,
                    top_k_probabilities=top_k_probs.tolist(),
                    probability_margin=margin,
                    cumulative_log_prob=cumulative_log_prob,
                    logits=logits.cpu(),
                )
            )

        return signals
