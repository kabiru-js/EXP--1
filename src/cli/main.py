"""CLI for model trajectory research."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from src.config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)


@click.group()
def cli() -> None:
    """Model Trajectory Research CLI."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Re-run even if trajectories exist")
@click.option("--samples", type=int, default=None, help="Override sample count")
def run(config_path: str, force: bool, samples: int | None) -> None:
    """Run an experiment from a YAML config."""
    from src.experiments.runner import ExperimentRunner

    config = ExperimentConfig.from_yaml(Path(config_path))
    if samples is not None:
        config.num_samples = samples

    runner = ExperimentRunner(config)
    results = runner.run(force=force)

    click.echo(f"\nExperiment {config.id} complete.")
    summary = results.get("summary", {})
    click.echo(f"  Total: {summary.get('total', 0)}")
    click.echo(f"  Correct: {summary.get('correct', 0)}")
    click.echo(f"  Accuracy: {summary.get('accuracy', 0):.1%}")

    for b in results.get("baselines", []):
        if "error" not in b:
            click.echo(f"  {b['model_name']}: AUROC={b['auroc']:.4f}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def analyze(config_path: str) -> None:
    """Analyze trajectories for an experiment."""
    from src.experiments.runner import ExperimentRunner

    config = ExperimentConfig.from_yaml(Path(config_path))
    runner = ExperimentRunner(config)
    results = runner.analyze()

    click.echo(json.dumps(results.get("summary", {}), indent=2))


@cli.command()
@click.argument("experiment_id")
def list_traj(experiment_id: str) -> None:
    """List trajectories for an experiment."""
    from src.trajectories.storage import TrajectoryStorage

    storage = TrajectoryStorage(Path("experiments") / experiment_id)
    trajectories = storage.get_experiment_trajectories(experiment_id)
    click.echo(f"Found {len(trajectories)} trajectories")
    for t in trajectories[:10]:
        status = "CORRECT" if t["is_correct"] else "INCORRECT"
        click.echo(f"  [{status}] {t['sample_id']}: {t['generation_tokens']} tokens")
    if len(trajectories) > 10:
        click.echo(f"  ... and {len(trajectories) - 10} more")


@cli.command()
@click.argument("experiment_id")
def inspect(experiment_id: str) -> None:
    """Inspect experiment results."""
    results_path = Path("experiments") / experiment_id / "results.json"
    if not results_path.exists():
        click.echo(f"No results found for {experiment_id}")
        return

    with open(results_path) as f:
        results = json.load(f)

    click.echo(f"Experiment: {experiment_id}")
    click.echo(f"Summary: {json.dumps(results.get('summary', {}), indent=2)}")

    if "baselines" in results:
        click.echo("\nBaselines:")
        for b in results["baselines"]:
            if "error" in b:
                click.echo(f"  {b['model_name']}: FAILED")
            else:
                click.echo(
                    f"  {b['model_name']}: AUROC={b['auroc']:.4f}, "
                    f"AUPRC={b['auprc']:.4f}"
                )


@cli.command()
def list_experiments() -> None:
    """List all experiments."""
    exp_dir = Path("experiments")
    if not exp_dir.exists():
        click.echo("No experiments directory found")
        return

    for d in sorted(exp_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            results_file = d / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)
                summary = results.get("summary", {})
                click.echo(
                    f"  {d.name}: {summary.get('total', 0)} samples, "
                    f"accuracy={summary.get('accuracy', 0):.1%}"
                )
            else:
                click.echo(f"  {d.name}: (no results)")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def report(config_path: str) -> None:
    """Generate report for an experiment."""
    from src.experiments.runner import ExperimentRunner

    config = ExperimentConfig.from_yaml(Path(config_path))
    runner = ExperimentRunner(config)
    results_path = Path("experiments") / config.id / "results.json"

    if not results_path.exists():
        click.echo("Run the experiment first")
        return

    with open(results_path) as f:
        results = json.load(f)

    summary = runner.storage.summary(config.id)
    runner._generate_report(results, summary)
    click.echo(f"Report saved to reports/{config.id}.md")


if __name__ == "__main__":
    cli()
