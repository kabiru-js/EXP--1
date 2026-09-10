"""Debug raw HF generation vs my backend."""
import sys
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.ERROR)
import torch
torch.set_num_threads(4)

model_name = "distilgpt2"
tok = __import__("transformers").AutoTokenizer.from_pretrained(model_name)
model = __import__("transformers").AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

print("pad:", tok.pad_token, "eos:", tok.eos_token, "bos:", tok.bos_token)
ids = tok.encode("7 + 8 = 15. True or False?")
print("encoded:", ids)

# Direct greedy generation without custom processor
inputs = tok("7 + 8 = 15. True or False?", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print("Direct greedy:", repr(tok.decode(out[0], skip_special_tokens=True)))

# Direct sampling
out2 = model.generate(**inputs, max_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.95)
print("Direct sample:", repr(tok.decode(out2[0], skip_special_tokens=True)))

# Now test my backend
from src.models.registry import get_backend
backend = get_backend("huggingface", model_name="distilgpt2", dtype="float32", device_map="cpu")
r = backend.generate("7 + 8 = 15. True or False?", max_tokens=8, temperature=0.0, top_p=1.0, top_k=50, seed=42)
print("My backend greedy:", repr(r[0].text), "token_ids:", r[0].token_ids)
print("signals:", len(r[0].signals), "token_texts:", r[0].token_texts)