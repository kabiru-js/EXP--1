"""Probe distilgpt2 with completion-style tasks (best format for base models)."""
import sys
import time
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.ERROR)
import torch
torch.set_num_threads(4)

from src.models.registry import get_backend

backend = get_backend("huggingface", model_name="distilgpt2", dtype="float32", device_map="cpu")

probes = [
    ("cloze", "7 + 8 =", "15"),
    ("cloze", "10 - 3 =", "7"),
    ("cloze", "5 * 9 =", "45"),
    ("cloze", "12 + 13 =", "25"),
    ("cloze", "2 + 2 = 4, 3 + 3 = 6, 4 + 4 =", "8"),
    ("cloze", "1, 2, 3, 4, 5,", "6"),
    ("cap", "The capital of France is", "Paris"),
    ("cap", "The capital of Japan is", "Tokyo"),
    ("fact", "The chemical symbol for gold is", "Au"),
    ("fact", "Water boils at", "100"),
    ("yes", "Is the sky blue? Yes or No?", "Yes"),
    ("tf", "A square has 5 sides. True or False?", "False"),
]

for fmt, prompt, gt in probes:
    t1 = time.time()
    results = backend.generate(prompt, max_tokens=12, temperature=0.0, top_p=1.0, top_k=50, seed=42)
    dt = time.time() - t1
    out = results[0].text.strip().split("\n")[0]
    ok = gt.lower() in out.lower()
    print("[{}] {} gt={!r:8} out={!r} ({:.1f}s)".format(
        fmt, "OK " if ok else "XX ", gt, out[:40], dt))

t0 = time.time()
for _ in range(3):
    backend.generate("12 + 13 =", max_tokens=24, temperature=0.0, top_p=1.0, top_k=50, seed=1)
print("distilgpt2 24 tokens: {:.1f}s each".format((time.time() - t0) / 3))