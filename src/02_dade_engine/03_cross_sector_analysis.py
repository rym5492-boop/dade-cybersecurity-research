"""
Cross-Sector Adaptivity Test (Supervisor Report Section 3.11 / Experiment E8).
Applies DADE to the SAME set of technical incidents under two different
illustrative sector weight profiles (Healthcare vs. Financial Services),
demonstrating that the SAME incident can receive a different priority and
action depending on organizational context -- supporting the "adaptive
across domains" claim.

DISCLOSURE: weight profiles below are illustrative, informed by commonly
cited sector priorities in business-continuity literature (not derived from
real expert elicitation), consistent with the placeholder status of all
business-context values used elsewhere in the preliminary study.
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

# Reuse the SAME n=200 sample and seed logic as the ablation study for consistency
df = pd.read_csv("/home/claude/collected_results/ablation_sample.csv")
n = len(df)
mls = df["MLS"].values
threat_severity = df["Threat_Severity"].values

np.random.seed(42)
_ = np.random.choice(82332, size=200, replace=False)  # reproduce the same draw sequence
asset_criticality = np.random.uniform(0.2, 1.0, size=n)
business_impact = np.random.uniform(0.2, 1.0, size=n)
historical_incidents = np.random.uniform(0.0, 1.0, size=n)
rto_urgency = np.random.uniform(0.2, 1.0, size=n)
rpo_urgency = np.random.uniform(0.2, 1.0, size=n)

criteria = np.column_stack([mls, threat_severity, asset_criticality, business_impact,
                             historical_incidents, rto_urgency, rpo_urgency])
# columns order: MLS, TS, AC, BI, HI, RTO, RPO

# ---------- Two illustrative sector weight profiles (sum to 1.0) ----------
# Order: MLS, TS, AC, BI, HI, RTO, RPO
weights_healthcare = np.array([0.10, 0.10, 0.15, 0.25, 0.05, 0.25, 0.10])
weights_finance    = np.array([0.10, 0.10, 0.15, 0.15, 0.20, 0.10, 0.20])
weights_equal      = np.array([1/7]*7)

assert abs(weights_healthcare.sum() - 1.0) < 1e-9
assert abs(weights_finance.sum() - 1.0) < 1e-9

def compute_scores(weights):
    return criteria @ weights

def map_level(score, high_thresh=0.66, med_thresh=0.33):
    return np.where(score >= high_thresh, "High", np.where(score >= med_thresh, "Medium", "Low"))

scores_hc = compute_scores(weights_healthcare)
scores_fin = compute_scores(weights_finance)
scores_eq = compute_scores(weights_equal)

levels_hc = map_level(scores_hc)
levels_fin = map_level(scores_fin)
levels_eq = map_level(scores_eq)

# How many incidents get a DIFFERENT priority level depending on sector?
pct_diff_hc_fin = np.mean(levels_hc != levels_fin) * 100
pct_diff_hc_eq = np.mean(levels_hc != levels_eq) * 100
pct_diff_fin_eq = np.mean(levels_fin != levels_eq) * 100

print("="*70)
print("CROSS-SECTOR ADAPTIVITY TEST (n=200 identical technical incidents)")
print("="*70)
print(f"\nWeight profiles (MLS, TS, AC, BI, HI, RTO, RPO):")
print(f"  Healthcare: {weights_healthcare}")
print(f"  Finance:    {weights_finance}")
print(f"  Equal:      {np.round(weights_equal,3)}")

print(f"\n% of incidents receiving a DIFFERENT priority level:")
print(f"  Healthcare vs Finance: {pct_diff_hc_fin:.1f}%")
print(f"  Healthcare vs Equal-weight: {pct_diff_hc_eq:.1f}%")
print(f"  Finance vs Equal-weight: {pct_diff_fin_eq:.1f}%")

print(f"\nPriority level distribution:")
for name, levels in [("Healthcare", levels_hc), ("Finance", levels_fin), ("Equal-weight", levels_eq)]:
    unique, counts = np.unique(levels, return_counts=True)
    print(f"  {name}: {dict(zip(unique, counts))}")

# Show a few illustrative examples where sector matters most
diff_idx = np.where(levels_hc != levels_fin)[0][:5]
examples = []
for i in diff_idx:
    examples.append({
        "instance": int(i)+1,
        "MLS": round(float(mls[i]), 3),
        "business_impact": round(float(business_impact[i]), 3),
        "historical_incidents": round(float(historical_incidents[i]), 3),
        "rto_urgency": round(float(rto_urgency[i]), 3),
        "rpo_urgency": round(float(rpo_urgency[i]), 3),
        "healthcare_score": round(float(scores_hc[i]), 3),
        "healthcare_level": levels_hc[i],
        "finance_score": round(float(scores_fin[i]), 3),
        "finance_level": levels_fin[i],
    })

results = {
    "weights": {
        "healthcare": weights_healthcare.tolist(),
        "finance": weights_finance.tolist(),
        "equal": weights_equal.tolist(),
    },
    "pct_decisions_differing": {
        "healthcare_vs_finance": round(pct_diff_hc_fin, 1),
        "healthcare_vs_equal": round(pct_diff_hc_eq, 1),
        "finance_vs_equal": round(pct_diff_fin_eq, 1),
    },
    "illustrative_examples": examples,
}
with open("/home/claude/collected_results/cross_sector_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved cross_sector_results.json")
