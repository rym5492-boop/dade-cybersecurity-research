"""
Ablation Study (Supervisor Report Section 3.9 / Experiment E5).
Compares:
  (A) ML-only decision (threshold on MLS alone)
  (B) Business-context-only decision (threshold on business criteria alone, no ML)
  (C) Equal-weight DADE (current illustrative approach: MLS + business context, equal weights)
  (D) Expert-weighted DADE -- DEFERRED until real AHP expert elicitation is completed

This uses the UNSW-NB15 test set predictions from the Random Forest model
(the best-performing model from Priority 2 experiments) combined with the
same style of illustrative business-context values used in Section 5.2,
extended to a larger, more defensible sample (n=200 instead of 12) to allow
meaningful agreement/accuracy comparisons across ablation conditions.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ---------- 1. Load data + retrain best model (Random Forest) ----------
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

categorical_cols = ["proto", "service", "state"]
drop_cols = ["id", "attack_cat", "label"]

combined = pd.concat([train_df[categorical_cols], test_df[categorical_cols]], axis=0)
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(combined[col].astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["label"]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df["label"].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
mls_all = rf.predict_proba(X_test_scaled)[:, 1]  # MLS for every test instance

# Attack category severity taxonomy (same convention as Section 5.2, but using
# UNSW-NB15's OWN official attack_cat labels directly -- more defensible than
# a manually-mapped taxonomy since it is the dataset's ground truth category)
SEVERITY_MAP = {
    "Normal": 0.00,
    "Fuzzers": 0.35,
    "Reconnaissance": 0.40,
    "Analysis": 0.55,
    "DoS": 0.70,
    "Backdoor": 0.85,
    "Exploits": 0.80,
    "Shellcode": 0.90,
    "Worms": 0.95,
    "Generic": 0.60,
}
attack_cat_all = test_df["attack_cat"].values
threat_severity_all = np.array([SEVERITY_MAP.get(c, 0.6) for c in attack_cat_all])

# ---------- 2. Sample n=200 instances for the ablation comparison ----------
n = 200
sample_idx = np.random.choice(len(y_test), size=n, replace=False)

mls = mls_all[sample_idx]
threat_severity = threat_severity_all[sample_idx]
true_label = y_test[sample_idx]

# Illustrative business-context values (same convention as Section 5.2)
asset_criticality = np.random.uniform(0.2, 1.0, size=n)
business_impact = np.random.uniform(0.2, 1.0, size=n)
historical_incidents = np.random.uniform(0.0, 1.0, size=n)
rto_urgency = np.random.uniform(0.2, 1.0, size=n)
rpo_urgency = np.random.uniform(0.2, 1.0, size=n)

def to_binary_decision(score, threshold=0.5):
    return (score >= threshold).astype(int)

# (A) ML-only: decision based purely on MLS
decision_A = to_binary_decision(mls)

# (B) Business-context-only: decision based purely on averaged business criteria (no ML at all)
business_only_score = (asset_criticality + business_impact + historical_incidents + rto_urgency + rpo_urgency) / 5
decision_B = to_binary_decision(business_only_score)

# (C) Equal-weight DADE: all 7 criteria (MLS, Threat Severity, 5 business criteria), equal weights
w = 1/7
dade_score = (
    w * mls + w * threat_severity + w * asset_criticality + w * business_impact +
    w * historical_incidents + w * rto_urgency + w * rpo_urgency
)
decision_C = to_binary_decision(dade_score)

# ---------- 3. Evaluate each condition against ground truth (true_label) ----------
results = {}
for name, decision, score in [
    ("A_ML_only", decision_A, mls),
    ("B_BusinessContext_only", decision_B, business_only_score),
    ("C_EqualWeight_DADE", decision_C, dade_score),
]:
    acc = accuracy_score(true_label, decision)
    f1 = f1_score(true_label, decision)
    kappa = cohen_kappa_score(true_label, decision)
    results[name] = {
        "accuracy_vs_ground_truth": round(acc, 4),
        "f1_vs_ground_truth": round(f1, 4),
        "cohen_kappa_vs_ground_truth": round(kappa, 4),
    }

print("="*70)
print("ABLATION STUDY RESULTS (n=200 test instances, UNSW-NB15)")
print("="*70)
for name, r in results.items():
    print(f"\n{name}:")
    for k, v in r.items():
        print(f"  {k}: {v}")

# Agreement between conditions (how often do they agree with each other?)
print("\n--- Pairwise agreement between decision conditions ---")
agree_AB = np.mean(decision_A == decision_B)
agree_AC = np.mean(decision_A == decision_C)
agree_BC = np.mean(decision_B == decision_C)
print(f"A (ML-only) vs B (Business-only) agreement: {agree_AB:.4f}")
print(f"A (ML-only) vs C (Equal-weight DADE) agreement: {agree_AC:.4f}")
print(f"B (Business-only) vs C (Equal-weight DADE) agreement: {agree_BC:.4f}")

results["pairwise_agreement"] = {
    "A_vs_B": round(agree_AB, 4),
    "A_vs_C": round(agree_AC, 4),
    "B_vs_C": round(agree_BC, 4),
}

with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save the raw sample for transparency/reproducibility
ablation_df = pd.DataFrame({
    "true_label": true_label,
    "attack_cat": attack_cat_all[sample_idx],
    "MLS": np.round(mls, 3),
    "Threat_Severity": np.round(threat_severity, 3),
    "Business_Only_Score": np.round(business_only_score, 3),
    "DADE_Score": np.round(dade_score, 3),
    "Decision_A_ML_only": decision_A,
    "Decision_B_Business_only": decision_B,
    "Decision_C_DADE": decision_C,
})
ablation_df.to_csv("ablation_sample.csv", index=False)
print("\nSaved ablation_results.json and ablation_sample.csv")
