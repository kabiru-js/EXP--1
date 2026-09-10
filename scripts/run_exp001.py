"""
run_exp001.py - Run EXP-001: Baseline Trajectory Study

This script runs the complete EXP-001 experiment end-to-end.
Usage:
    python scripts/run_exp001.py
    python scripts/run_exp001.py --samples 100  # Quick test run
    python scripts/run_exp001.py --model gpt2   # Use a different model
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run EXP-001")
    parser.add_argument("--config", default="configs/EXP-001.yaml", help="Config file path")
    parser.add_argument("--samples", type=int, default=None, help="Override sample count")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    parser.add_argument("--resume", action="store_true", help="Resume (skip existing samples)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ExperimentConfig.from_yaml(Path(args.config))

    if args.samples is not None:
        config.num_samples = args.samples
    if args.model is not None:
        config.model.name = args.model

    runner = ExperimentRunner(config)

    if args.analyze_only:
        results = runner.analyze()
    elif args.resume:
        results = runner.run(resume=True)
    else:
        results = runner.run(force=args.force)

    print("\n" + "=" * 60)
    print(f"Experiment {config.id}: {config.name}")
    print("=" * 60)

    summary = results.get("summary", {})
    print(f"Samples: {summary.get('total', 0)}")
    print(f"Correct: {summary.get('correct', 0)}")
    print(f"Incorrect: {summary.get('incorrect', 0)}")
    print(f"Accuracy: {summary.get('accuracy', 0):.1%}")

    if "baselines" in results:
        print("\nBaseline Results:")
        for b in results["baselines"]:
            if "error" in b:
                print(f"  {b['model_name']}: FAILED - {b['error']}")
            else:
                print(
                    f"  {b['model_name']}: "
                    f"AUROC={b['auroc']:.4f}, "
                    f"AUPRC={b['auprc']:.4f}, "
                    f"Acc={b['accuracy']:.4f}"
                )

    if "temporal_prediction" in results:
        print("\nTemporal Prediction (Cohen's d):")
        for frac in sorted(results["temporal_prediction"].keys(), key=float):
            v = results["temporal_prediction"][frac]
            print(f"  {float(frac):>5.0%}: d={v['cohens_d']:.3f}, p={v['p_value']:.4f}")

    print(f"\nResults saved to: experiments/{config.id}/")
    print(f"Report saved to: reports/{config.id}.md")


if __name__ == "__main__":
    main()
