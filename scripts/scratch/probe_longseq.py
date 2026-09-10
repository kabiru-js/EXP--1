"""Probe distilgpt2 with longer sequences (more context terms)."""
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
    for start in [1, 2, 3, 5]:
        n_terms = 7
        terms = [start + i * step for i in range(n_terms)]
        nxt = terms[-1] + step
        seqs.append((", ".join(str(t) for t in terms) + ",", str(nxt)))

n_correct = 0
total = 0
from collections import Counter
tokens_by_correct = Counter()
t0 = time.time()
for prompt, gt in seqs:
    results = backend.generate(prompt, max_tokens=16, temperature=0.0, top_p=1.0, top_k=50, seed=42)
    out = results[0].text.strip().split(",")[0].strip()
    total += 1
    ok = out == gt
    n_correct += ok
    tokens_by_correct[ok] += len(results[0].signals)
    print("[{}] step={:>2} start={:>2} seq={!r:30} gt={!r:4} out={!r:4} ({:>2} tokens)".format(
        "OK " if ok else "XX ", prompt.split(",")[1].strip() if len(prompt.split(","))>1 else "?", 
        prompt.split(",")[0].strip(), prompt, gt, out, len(results[0].signals)))

elapsed = time.time() - t0
print("\nAccuracy: {}/{} = {:.1%}".format(n_correct, total, n_correct / total))
print("Total time: {:.1f}s ({:.3f}s/sample)".format(elapsed, elapsed / total))
print("Tokens per trajectory (correct): {}, (incorrect): {}".format(
    tokens_by_correct.get(True, 0), tokens_by_correct.get(False, 0)))