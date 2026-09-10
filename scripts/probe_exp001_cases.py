"""Find the most informative individual EXP-001 trajectories."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from src.trajectories.storage import TrajectoryStorage
from src.features.extraction import extract_trajectory_features, build_feature_matrix

storage = TrajectoryStorage(Path("experiments/EXP-001"))
trajs = storage.get_experiment_trajectories("EXP-001")
n_correct = sum(1 for t in trajs if t["is_correct"])
n_incorrect = sum(1 for t in trajs if not t["is_correct"])
print(f"Total: {len(trajs)} | Correct: {n_correct}, Incorrect: {n_incorrect}")

token_arrays = storage.load_all_token_arrays("EXP-001")
features = extract_trajectory_features(token_arrays)
labels = storage.get_correctness_labels("EXP-001")
X, y, feature_names, sample_ids = build_feature_matrix(features, labels)

def toks(sid, n=6):
    t = next(x for x in trajs if x["sample_id"] == sid)
    return t["generated_text"][:60]

print("\n=== CORRECT, HIGHEST MEAN ENTROPY (surprisingly uncertain but correct) ===")
cf = [(sid, features[sid]["mean_entropy"], features[sid]["final_entropy"], features[sid]["final_probability"], features[sid]["probability_trend"]) for sid in sample_ids if labels[sid]]
cf.sort(key=lambda x: -x[1])
for sid, mean_ent, fin_ent, fin_prob, pt in cf[:5]:
    t = next(x for x in trajs if x["sample_id"] == sid)
    print(f"  {sid}: mean_ent={mean_ent:.3f} fin_ent={fin_ent:.3f} fin_prob={fin_prob:.3f} prob_trend={pt:.4f}")
    print(f"    gt={t['ground_truth']} extracted={t['extracted_answer']} prompt={t['question'][:45]}")
    print(f"    toks: {toks(sid)}")

print("\n=== INCORRECT, LOWEST MEAN ENTROPY (confident but wrong) ===")
if_ = [(sid, features[sid]["mean_entropy"], features[sid]["final_entropy"], features[sid]["final_probability"], features[sid]["probability_trend"]) for sid in sample_ids if not labels[sid]]
if_.sort(key=lambda x: x[1])
for sid, mean_ent, fin_ent, fin_prob, pt in if_[:5]:
    t = next(x for x in trajs if x["sample_id"] == sid)
    print(f"  {sid}: mean_ent={mean_ent:.3f} fin_ent={fin_ent:.3f} fin_prob={fin_prob:.3f} prob_trend={pt:.4f}")
    print(f"    gt={t['ground_truth']} extracted={t['extracted_answer']} prompt={t['question'][:45]}")
    print(f"    toks: {toks(sid)}")

print("\n=== HIGHEST FIRST-TOKEN ENTROPY ===")
first_ent = [(sid, token_arrays[sid]["entropies"][0], labels[sid]) for sid in sample_ids]
first_ent.sort(key=lambda x: -x[1])
print("Top 3 highest first-token entropy:")
for sid, ent, lab in first_ent[:3]:
    t = next(x for x in trajs if x["sample_id"] == sid)
    print(f"  {sid}: first_ent={ent:.3f} correct={lab} gt={t['ground_truth']} extracted={t['extracted_answer']}")
    print(f"    toks: {toks(sid, 3)}")

print("\n=== LOWEST FIRST-TOKEN ENTROPY ===")
first_ent.sort(key=lambda x: x[1])
print("Top 3 lowest first-token entropy:")
for sid, ent, lab in first_ent[:3]:
    t = next(x for x in trajs if x["sample_id"] == sid)
    print(f"  {sid}: first_ent={ent:.3f} correct={lab} gt={t['ground_truth']} extracted={t['extracted_answer']}")
    print(f"    toks: {toks(sid, 3)}")

print("\n=== CONFIDENT-WRONG: lowest entropy but wrong ===")
sid, mean_ent, fin_ent, fin_prob, pt = if_[0]
t = next(x for x in trajs if x["sample_id"] == sid)
arr = token_arrays[sid]
print(f"  {sid}: mean_ent={mean_ent:.3f} fin_ent={fin_ent:.3f} fin_prob={fin_prob:.3f} prob_trend={pt:.4f}")
print(f"    gt={t['ground_truth']} extracted={t['extracted_answer']} prompt={t['question']}")
print(f"    generated_text: {t['generated_text'][:120]}")
print(f"    first token prob={arr['probabilities'][0]:.4f} entropy={arr['entropies'][0]:.4f} margin={arr['margins'][0]:.4f}")
print(f"    last token prob={arr['probabilities'][-1]:.4f} entropy={arr['entropies'][-1]:.4f} margin={arr['margins'][-1]:.4f}")

print("\n=== BEST UNCERTAINTY CONTRAST ===")
best_cor = cf[0]
best_inc = if_[0]
print(f"  Correct: {best_cor[0]} mean_ent={best_cor[1]:.3f} fin_prob={best_cor[3]:.3f}")
print(f"  Incorrect: {best_inc[0]} mean_ent={best_inc[1]:.3f} fin_prob={best_inc[3]:.3f}")

print("\n=== GBT TOP 10 FEATURES ===")
r = json.load(open("experiments/EXP-001/results.json"))
gb = [b for b in r["baselines"] if b["model_name"] == "gradient_boosted_tree"][0]
fi = sorted(gb["feature_importances"].items(), key=lambda x: -x[1])[:10]
print("Top 10 GBT features:")
for name, imp in fi:
    print(f"  {name}: {imp:.4f}")
