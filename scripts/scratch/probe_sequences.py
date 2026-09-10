"""Probe distilgpt2 accuracy on arithmetic sequence completion."""
import sys
import time
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.ERROR)
import torch
torch.set_num_threads(4)

from src.models.registry import get_backend

backend = get_backend("huggingface", model_name="distilgpt2", dtype="float32", device_map="cpu")

seqs = []
for step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for start in [1, 2, 3, 5, 10, 20]:
        terms = [start + i * step for i in range(4)]
        nxt = terms[-1] + step
        seqs.append((", ".join(str(t) for t in terms) + ",", str(nxt)))

n_correct = 0
total = 0
t0 = time.time()
for prompt_prefix, gt in seqs:
    prompt = prompt_prefix
    t1 = time.time()
    results = backend.generate(prompt, max_tokens=8, temperature=0.0, top_p=1.0, top_k=50, seed=42)
    dt = time.time() - t1
    out = results[0].text.strip().split(",")[0].strip()
    total += 1
    ok = out == gt
    n_correct += ok
    print("[{}] {} seq={!r:24} next={!r} out={!r} ({:.1f}s)".format(
        "OK " if ok else "XX ", fmt if False else "", prompt, gt, out, dt))

elapsed = time.time() - t0
print("\nAccuracy: {}/{} = {:.1%}".format(n_correct, total, n_correct / total))
print("Total time: {:.1f}s ({:.2f}s/sample)".format(elapsed, elapsed / total))