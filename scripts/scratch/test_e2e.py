"""Quick end-to-end test with reduced parameters for CPU."""

import sys
import logging
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from src.config import ExperimentConfig
from src.experiments.runner import ExperimentRunner

config = ExperimentConfig.from_yaml("configs/EXP-001.yaml")
config.num_samples = 5
config.generation.max_tokens = 128  # Much shorter for CPU test

runner = ExperimentRunner(config)
results = runner.run(force=True)

print("\n" + "=" * 60)
summary = results.get("summary", {})
print(f"Total: {summary.get('total', 0)}")
print(f"Correct: {summary.get('correct', 0)}")
print(f"Accuracy: {summary.get('accuracy', 0):.1%}")

if "baselines" in results:
    for b in results["baselines"]:
        if "error" not in b:
            print(f"  {b['model_name']}: AUROC={b['auroc']:.4f}")

if "temporal_prediction" in results:
    print("\nTemporal prediction (Cohen's d):")
    for frac in sorted(results["temporal_prediction"].keys(), key=float):
        v = results["temporal_prediction"][frac]
        print(f"  {float(frac):>5.0%}: d={v['cohens_d']:.3f}, p={v['p_value']:.4f}")

print("=" * 60)
print("PIPELINE END-TO-END TEST PASSED")
