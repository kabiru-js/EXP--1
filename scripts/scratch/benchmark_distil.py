"""Benchmark small model on easier synthetic arithmetic with fast settings."""
import sys, time
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.WARNING)

import torch
torch.set_num_threads(4)

from src.models.registry import get_backend
from src.datasets.synthetic import SyntheticArithmeticLoader

backend = get_backend("huggingface", model_name="distilgpt2", dtype="float32", device_map="cpu")

loader = SyntheticArithmeticLoader()
samples = loader.load(num_samples=12)

prompt_template = "Q: {question}\nA:"

n_correct = 0
t0 = time.time()
for i, s in enumerate(samples):
    prompt = prompt_template.format(question=s.question)
    t1 = time.time()
    results = backend.generate(prompt, max_tokens=32, temperature=0.7, top_p=0.95, top_k=50, seed=42)
    dt = time.time() - t1
    text = results[0].text
    extracted = loader.extract_answer(text)
    correct = loader.check_correctness(extracted, s.ground_truth)
    n_correct += correct
    print(f"[{i}] ({dt:.1f}s) {'OK ' if correct else 'XX '} q={s.question!r} gt={s.ground_truth} ex={extracted!r}")
    print(f"      out={text[:80]!r}")

elapsed = time.time() - t0
print(f"\nAccuracy: {n_correct}/{len(samples)} = {n_correct/len(samples):.1%}")
print(f"Per-sample: {elapsed/len(samples):.1f}s  Total: {elapsed:.1f}s")