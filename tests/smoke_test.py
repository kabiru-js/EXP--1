"""Quick smoke test for all modules."""

import sys
sys.path.insert(0, ".")

import numpy as np

# Test config
from src.config import ExperimentConfig, DatasetConfig, ModelConfig, GenerationConfig
config = ExperimentConfig(
    id="TEST", name="test",
    dataset=DatasetConfig(name="gsm8k", samples=10),
    model=ModelConfig(name="model"),
    generation=GenerationConfig(seed=42),
)
print("[PASS] Config")

# Test datasets
from src.datasets.gsm8k import GSM8KLoader
loader = GSM8KLoader()
assert loader.extract_answer("#### 42") == "42"
assert loader.check_correctness("42", "42")
print("[PASS] GSM8K")

from src.datasets.synthetic import SyntheticArithmeticLoader
sloader = SyntheticArithmeticLoader()
samples = sloader.load(num_samples=5)
assert len(samples) == 5
print("[PASS] Synthetic")

# Test features
from src.features.extraction import extract_trajectory_features, build_feature_matrix
rng = np.random.RandomState(42)
token_arrays = {}
labels = {}
for i in range(20):
    sid = f"s{i}"
    is_correct = i < 10
    token_arrays[sid] = {
        "probabilities": rng.uniform(0.3, 0.8, 30),
        "log_probabilities": rng.uniform(-1.0, -0.1, 30),
        "entropies": rng.normal(1.0 if is_correct else 2.0, 0.3, 30),
        "margins": rng.uniform(0.1, 0.5, 30),
        "cumulative_log_probs": np.cumsum(rng.normal(-0.3, 0.1, 30)),
        "token_indices": np.arange(30),
        "token_ids": rng.randint(0, 1000, 30),
    }
    labels[sid] = is_correct

features = extract_trajectory_features(token_arrays)
X, y, names, sids = build_feature_matrix(features, labels)
assert X.shape == (20, len(names))
print("[PASS] Features")

# Test baselines
from src.evaluation.baselines import LogisticRegressionBaseline, GradientBoostedBaseline, compute_metrics
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LogisticRegressionBaseline()
lr.fit(X_tr, y_tr, names)
probs = lr.predict_proba(X_te, names)
metrics = compute_metrics(y_te, probs)
assert 0 <= metrics["auroc"] <= 1
print(f"[PASS] Logistic Regression AUROC={metrics['auroc']:.3f}")

gb = GradientBoostedBaseline()
gb.fit(X_tr, y_tr, names)
probs = gb.predict_proba(X_te, names)
metrics = compute_metrics(y_te, probs)
print(f"[PASS] Gradient Boosted AUROC={metrics['auroc']:.3f}")

# Test analysis
from src.analysis.divergence import compute_trajectory_curves, compute_temporal_prediction_performance
curves = compute_trajectory_curves(token_arrays, labels)
assert "correct_entropy_mean" in curves
print("[PASS] Trajectory curves")

temporal = compute_temporal_prediction_performance(token_arrays, labels, [0.3, 0.6, 0.9])
assert 0.3 in temporal
print(f"[PASS] Temporal prediction d@60%={temporal[0.6]['cohens_d']:.3f}")

# Test storage
import tempfile
from src.trajectories.storage import Trajectory, TrajectoryStorage, TokenMetrics
with tempfile.TemporaryDirectory() as tmpdir:
    storage = TrajectoryStorage(tmpdir)
    metrics_list = [
        TokenMetrics(token_id=i, token_text=f"t{i}", token_index=i,
                     probability=0.5, log_probability=-0.7, entropy=1.5,
                     probability_margin=0.3, cumulative_log_prob=-0.7*i)
        for i in range(5)
    ]
    traj = Trajectory(
        sample_id="test", dataset="test", model_id="m", model_version="0.1",
        experiment_id="EXP-001", question="Q", ground_truth="A",
        prompt="P", prompt_token_count=3, generated_text="A",
        extracted_answer="A", is_correct=True, generation_tokens=5,
        temperature=0.7, seed=42, prompt_template="{q}",
        token_metrics=metrics_list, available_signals=["p"]
    )
    storage.store(traj)
    assert storage.count("EXP-001") == 1
    loaded = storage.load_trajectory("test")
    assert loaded is not None
print("[PASS] Storage")

# Test visualization (just imports and basic call)
from src.visualization.plots import plot_trajectory_comparison
import matplotlib
matplotlib.use("Agg")
with tempfile.TemporaryDirectory() as tmpdir:
    plot_trajectory_comparison(curves, "entropy", f"{tmpdir}/test_fig")
print("[PASS] Visualization")

# Test reporting
from src.reporting.report import ExperimentReport
with tempfile.TemporaryDirectory() as tmpdir:
    report = ExperimentReport("TEST-001", tmpdir)
    report.set_metadata(hypothesis="Test hypothesis")
    report.add_section("Results", "Some results")
    path = report.save()
    assert path.exists()
print("[PASS] Reporting")

print("\n" + "=" * 50)
print("ALL MODULE TESTS PASSED")
print("=" * 50)
