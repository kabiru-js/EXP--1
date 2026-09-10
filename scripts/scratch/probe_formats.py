"""Probe distilgpt2 with different task formats."""
import sys, time
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.ERROR)

import torch
torch.set_num_threads(4)

from src.models.registry import get_backend

backend = get_backend("huggingface", model_name="distilgpt2", dtype="float32", device_map="cpu")

probes = [
    ("truefalse", "7 + 8 = 15. True or False?", "True"),
    ("truefalse", "10 - 3 = 6. True or False?", "False"),
    ("truefalse", "12 / 4 = 3. True or False?", "True"),
    ("truefalse", "5 * 5 = 24. True or False?", "False"),
    ("mc", "What is 7 + 8?\n(a) 13\n(b) 15\n(c) 17\n(d) 19", "b"),
    ("mc", "What is 10 - 3?\n(a) 5\n(b) 6\n(c) 7\n(d) 8", "c"),
    ("factual", "The capital of France is Paris. True or False?", "True"),
    ("factual", "The capital of Spain is Paris. True or False?", "False"),
    ("yesno", "Is 23 a prime number? Yes or No?", "Yes"),
    ("yesno", "Is 24 a prime number? Yes or No?", "No"),
    ("vowel", "Does the word 'apple' start with a vowel? Yes or No?", "Yes"),
    ("vowel", "Does the word 'cat' start with a vowel? Yes or No?", "No"),
]

for fmt, q, gt in probes:
    prompt = q
    t1 = time.time()
    results = backend.generate(prompt, max_tokens=8, temperature=0.0, top_p=1.0, top_k=50, seed=42)
    dt = time.time() - t1
    out = results[0].text.strip()
    ok = gt.lower() in out.lower()
    print(f"[{fmt}] {'OK ' if ok else 'XX '} gt={gt!r:7} out={out[:40]!r} ({dt:.1f}s)")

print("\nNote: greedy decoding for probes")