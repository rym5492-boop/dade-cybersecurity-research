"""
Sensitivity Analysis (Supervisor Report Section 3.10 / Experiment E7).
Tests robustness of DADE's Priority Score and resulting decisions by:
  1. Perturbing weights by +/-10% and +/-20% around the equal-weight baseline
  2. Perturbing the High/Medium decision thresholds by +/-10%
  3. Reporting how often the final priority LEVEL (not just the score) changes
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# Load the same n=200 ablation sample for consistency
df = pd.read_csv("ablation_sample.csv")
n = len(df)

mls = df["MLS"].values
threat_severity = df["Threat_Severity"].values
# Reconstruct business criteria is not saved individually; regenerate with same seed logic
# For consistency, we regenerate using the same random calls as ablation_study.py
np.random.seed(42)
_ = np.random.choice(82332, size=200, replace=False)  # reproduce the same draw sequence
asset_criticality = np.random.uniform(0.2, 1.0, size=n)
business_impact = np.random.uniform(0.2, 1.0, size=n)
historical_incidents = np.random.uniform(0.0, 1.0, size=n)
rto_urgency = np.random.uniform(0.2, 1.0, size=n)
rpo_urgency = np.random.uniform(0.2, 1.0, size=n)

criteria = np.column_stack([mls, threat_severity, asset_criticality, business_impact,
                             historical_incidents, rto_urgency, rpo_urgency])
# columns: MLS, TS, AC, BI, HI, RTO, RPO

def compute_scores(weights):
    return criteria @ weights

def map_level(score, high_thresh=0.66, med_thresh=0.33):
    return np.where(score >= high_thresh, "High", np.where(score >= med_thresh, "Medium", "Low"))

# ---------- Baseline: equal weights ----------
baseline_weights = np.array([1/7]*7)
baseline_scores = compute_scores(baseline_weights)
baseline_levels = map_level(baseline_scores)

# ---------- 1. Weight perturbation: +/-10% and +/-20% on EACH criterion individually ----------
perturbation_results = []
for pct in [0.10, 0.20]:
    for crit_idx, crit_name in enumerate(["MLS", "TS", "AC", "BI", "HI", "RTO", "RPO"]):
        for direction in [1, -1]:
            w = baseline_weights.copy()
            delta = direction * pct * w[crit_idx]
            w[crit_idx] += delta
            # renormalize so weights still sum to 1
            w = w / w.sum()
            scores = compute_scores(w)
            levels = map_level(scores)
            pct_changed = np.mean(levels != baseline_levels) * 100
            perturbation_results.append({
                "criterion": crit_name,
                "perturbation": f"{'+' if direction > 0 else '-'}{int(pct*100)}%",
                "pct_decisions_changed": round(pct_changed, 2),
            })

pert_df = pd.DataFrame(perturbation_results)
print("="*70)
print("WEIGHT SENSITIVITY: % of priority-level decisions that CHANGE")
print("="*70)
print(pert_df.to_string(index=False))
pert_df.to_csv("sensitivity_weights.csv", index=False)

avg_change_10 = pert_df[pert_df["perturbation"].str.contains("10%")]["pct_decisions_changed"].mean()
avg_change_20 = pert_df[pert_df["perturbation"].str.contains("20%")]["pct_decisions_changed"].mean()
print(f"\nAverage decision change rate at +/-10% weight perturbation: {avg_change_10:.2f}%")
print(f"Average decision change rate at +/-20% weight perturbation: {avg_change_20:.2f}%")

# ---------- 2. Threshold perturbation ----------
threshold_results = []
for thresh_pct in [-0.10, -0.05, 0.05, 0.10]:
    high_t = 0.66 * (1 + thresh_pct)
    med_t = 0.33 * (1 + thresh_pct)
    levels = map_level(baseline_scores, high_thresh=high_t, med_thresh=med_t)
    pct_changed = np.mean(levels != baseline_levels) * 100
    threshold_results.append({
        "threshold_perturbation": f"{'+' if thresh_pct > 0 else ''}{int(thresh_pct*100)}%",
        "high_threshold": round(high_t, 3),
        "medium_threshold": round(med_t, 3),
        "pct_decisions_changed": round(pct_changed, 2),
    })

thresh_df = pd.DataFrame(threshold_results)
print("\n" + "="*70)
print("THRESHOLD SENSITIVITY: % of priority-level decisions that CHANGE")
print("="*70)
print(thresh_df.to_string(index=False))
thresh_df.to_csv("sensitivity_thresholds.csv", index=False)

summary = {
    "avg_change_rate_10pct_weight_perturbation": round(avg_change_10, 2),
    "avg_change_rate_20pct_weight_perturbation": round(avg_change_20, 2),
    "max_change_rate_single_criterion_20pct": round(pert_df[pert_df["perturbation"].str.contains("20%")]["pct_decisions_changed"].max(), 2),
    "most_sensitive_criterion": pert_df.loc[pert_df["pct_decisions_changed"].idxmax(), "criterion"],
    "threshold_sensitivity": threshold_results,
}
with open("sensitivity_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved sensitivity_weights.csv, sensitivity_thresholds.csv, sensitivity_summary.json")
