"""Probe natural-language sequence prompts to push the numeric answer later in trajectory."""
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
    ("1, 3, 5, 7, 9. The next number is", "11"),
    ("1, 4, 7, 10, 13. The next number is", "16"),
    ("The pattern is 2, 4, 6, 8, 10, so the next number is", "12"),
    ("Counting by twos: 3, 5, 7, 9, 11, ... the next term is", "13"),
    ("1, 3, 5, 7, 9, 11, 13, 15, 17. We add 2 each time, so the next number is", "19"),
    ("A pattern of even numbers: 2, 4, 6, 8, 10, 12, ... the next even number is", "14"),
    ("5, 10, 15, 20, 25, 30. Adding five each time, the next number is", "35"),
    ("10, 20, 30, 40, 50, 60, ... continuing the pattern gives", "70"),
]

for prompt, gt in probes:
    results = backend.generate(prompt, max_tokens=24, temperature=0.0, top_p=1.0, top_k=50, seed=42)
    text = results[0].text
    out = text.strip()
    ok = gt in out
    n = len(results[0].signals)
    # find where the number appears
    num_pos = out.find(gt)
    tok_texts = results[0].token_texts
    print("[{}] gt={!r:4} ({} tok) out={!r}".format("OK " if ok else "XX ", gt, n, out[:100]))
    if num_pos >= 0:
        # count tokens before answer
        before = out[:num_pos]
        print("      num appears at char {}, tokens before approx {}".format(num_pos, len(before)//4))

print("\nDone")

# Timing check for 24 tokens
t0 = time.time()
for _ in range(5):
    backend.generate("1, 3, 5, 7, 9. The next number is", max_tokens=24, temperature=0.0, top_p=1.0, top_k=50, seed=1)
print("{:.1f}s each for 24 tokens".format((time.time() - t0) / 5))