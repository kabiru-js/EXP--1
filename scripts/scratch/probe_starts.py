"""Probe consecutive runs with large starts (multi-digit answers) and odd starting values."""
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
# Consecutive runs with various starts (correct answers expected)
for start in [1, 2, 3, 5, 7, 9, 11, 13, 15, 20, 21, 25, 30, 33, 40, 41, 50, 51, 60, 70, 80, 90, 99, 100, 101]:
    terms = list(range(start, start + 7))
    nxt = start + 7
    seqs.append((", ".join(str(t) for t in terms) + ",", str(nxt)))

n_correct = 0
total = 0
t0 = time.time()
for prompt, gt in seqs:
    results = backend.generate(prompt, max_tokens=16, temperature=0.0, top_p=1.0, top_k=50, seed=42)
    out = results[0].text.strip().split(",")[0].strip()
    total += 1
    ok = out == gt
    n_correct += ok
    ans_tokens = len(gt)
    print("[{}] seq={!r:40} gt={!r:4} out={!r:4} ({} ans tokens, {} gen tokens)".format(
        "OK " if ok else "XX ", prompt[:-1], gt, out, ans_tokens, len(results[0].signals)))

elapsed = time.time() - t0
print("\nConsecutive-start accuracy: {}/{} = {:.1%}".format(n_correct, total, n_correct / total))
print("Time: {:.1f}s ({:.3f}s/sample)".format(elapsed, elapsed / total))