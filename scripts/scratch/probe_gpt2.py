"""Probe gpt2 with few-shot Q/A and factual tasks."""
import sys
import time
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.ERROR)
import torch
torch.set_num_threads(4)

from src.models.registry import get_backend

backend = get_backend("huggingface", model_name="gpt2", dtype="float32", device_map="cpu")

FEWSHOT = (
    "Q: What is 3 + 4?\nA: 7\n\n"
    "Q: What is 12 - 5?\nA: 7\n\n"
    "Q: What is 6 * 7?\nA: 42\n\n"
)

probes = [
    ("arith", FEWSHOT + "Q: What is 7 + 8?\nA:", "15"),
    ("arith", FEWSHOT + "Q: What is 20 - 6?\nA:", "14"),
    ("arith", FEWSHOT + "Q: What is 5 * 9?\nA:", "45"),
    ("arith", FEWSHOT + "Q: What is 16 + 14?\nA:", "30"),
    ("cap", "Q: What is the capital of France?\nA:", "Paris"),
    ("cap", "Q: What is the capital of Japan?\nA:", "Tokyo"),
    ("cap", "Q: What is the capital of Germany?\nA:", "Berlin"),
    ("fact", "The capital of France is ", "Paris"),
    ("fact", "The chemical symbol for gold is ", "Au"),
    ("fact", "Water boils at ", "100"),
]

for fmt, prompt, gt in probes:
    t1 = time.time()
    results = backend.generate(prompt, max_tokens=12, temperature=0.0, top_p=1.0, top_k=50, seed=42)
    dt = time.time() - t1
    out = results[0].text.strip().split("\n")[0]
    ok = gt.lower() in out.lower()
    print("[{}] {} gt={!r} out={!r} ({:.1f}s)".format(
        fmt, "OK " if ok else "XX ", gt, out[:45], dt))

t0 = time.time()
for _ in range(3):
    backend.generate(FEWSHOT + "Q: What is 100 - 33?\nA:", max_tokens=32, temperature=0.0, top_p=1.0, top_k=50, seed=1)
print("gpt2 few-shot 32 tokens: {:.1f}s each".format((time.time() - t0) / 3))