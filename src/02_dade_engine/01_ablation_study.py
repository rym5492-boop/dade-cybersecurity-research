"""
Ablation Study (Supervisor Report Section 3.9 / Experiment E5).
Compares:
  (A) ML-only decision (threshold on MLS alone)
  (B) Business-context-only decision (threshold on business criteria alone, no ML)
  (C) Equal-weight DADE (illustrative baseline approach: MLS + business context, equal weights)
  (D) Expert-weighted DADE (real AHP group weights from Section 4.3 / 00_ahp_expert_weights.py,
      applied to the same seven criteria in place of the equal-weight baseline)

This uses the UNSW-NB15 test set predictions from the Random Forest model
(the best-performing model from Priority 2 experiments) combined with the
same style of illustrative business-context values used in Section 5.2,
extended to a larger, more defensible sample (n=200 instead of 12) to allow
meaningful agreement/accuracy comparisons across ablation conditions.

Requires results/ahp_weights_results.json to already exist
(run 00_ahp_expert_weights.py first).

Run this script from the REPOSITORY ROOT (not from inside src/02_dade_engine),
with UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv placed directly in
the repository root, e.g.:
    python src/02_dade_engine/01_ablation_study.py
All paths below (data/ahp weights in results/, and this script's own output)
are resolved relative to the repository root regardless of the working
directory the script is launched from, so results/ always ends up in the
actual results/ folder.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 0. Load real AHP expert group weights (Section 4.3) ----------
with open(RESULTS_DIR / "ahp_weights_results.json") as f:
    _ahp = json.load(f)
AHP_WEIGHTS = _ahp["group"]["weights"]  # {"TS":..,"AC":..,"BI":..,"HI":..,"RTO":..,"RPO":..,"MLS":..}
_ahp_sum = sum(AHP_WEIGHTS.values())
AHP_WEIGHTS = {k: v / _ahp_sum for k, v in AHP_WEIGHTS.items()}  # renormalize rounded weights to sum to 1

def _find_unsw_file(filename):
    """Look for the UNSW-NB15 CSV in the repo root, data/, or the current
    working directory (in that order), so this script works whether it is
    launched from the repository root or from src/02_dade_engine/."""
    for candidate in (REPO_ROOT / filename, REPO_ROOT / "data" / filename, Path(filename)):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Place UNSW_NB15_training-set.csv and "
        f"UNSW_NB15_testing-set.csv in the repository root or in data/."
    )


# ---------- 1. Load data + retrain best model (Random Forest) ----------
train_df = pd.read_csv(_find_unsw_file("UNSW_NB15_training-set.csv"))
test_df = pd.read_csv(_find_unsw_file("UNSW_NB15_testing-set.csv"))

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
w = 1 / 7
dade_score = (
    w * mls + w * threat_severity + w * asset_criticality + w * business_impact +
    w * historical_incidents + w * rto_urgency + w * rpo_urgency
)
decision_C = to_binary_decision(dade_score)

# (D) Expert-weighted DADE: same seven criteria, real AHP group weights (Section 4.3)
# instead of the equal-weight (1/7) baseline used in (C).
dade_score_ahp = (
    AHP_WEIGHTS["MLS"] * mls + AHP_WEIGHTS["TS"] * threat_severity +
    AHP_WEIGHTS["AC"] * asset_criticality + AHP_WEIGHTS["BI"] * business_impact +
    AHP_WEIGHTS["HI"] * historical_incidents + AHP_WEIGHTS["RTO"] * rto_urgency +
    AHP_WEIGHTS["RPO"] * rpo_urgency
)
decision_D = to_binary_decision(dade_score_ahp)

# ---------- 3. Evaluate each condition against ground truth (true_label) ----------
results = {"ahp_weights_used": AHP_WEIGHTS}
for name, decision, score in [
    ("A_ML_only", decision_A, mls),
    ("B_BusinessContext_only", decision_B, business_only_score),
    ("C_EqualWeight_DADE", decision_C, dade_score),
    ("D_ExpertWeighted_DADE", decision_D, dade_score_ahp),
]:
    acc = accuracy_score(true_label, decision)
    f1 = f1_score(true_label, decision)
    kappa = cohen_kappa_score(true_label, decision)
    results[name] = {
        "accuracy_vs_ground_truth": round(acc, 4),
        "f1_vs_ground_truth": round(f1, 4),
        "cohen_kappa_vs_ground_truth": round(kappa, 4),
    }

print("=" * 70)
print("ABLATION STUDY RESULTS (n=200 test instances, UNSW-NB15)")
print("=" * 70)
for name, r in results.items():
    if name == "ahp_weights_used":
        continue
    print(f"\n{name}:")
    for k, v in r.items():
        print(f"  {k}: {v}")

# Agreement between conditions (how often do they agree with each other?)
print("\n--- Pairwise agreement between decision conditions ---")
agree_AB = np.mean(decision_A == decision_B)
agree_AC = np.mean(decision_A == decision_C)
agree_BC = np.mean(decision_B == decision_C)
agree_AD = np.mean(decision_A == decision_D)
agree_CD = np.mean(decision_C == decision_D)
agree_BD = np.mean(decision_B == decision_D)
print(f"A (ML-only) vs B (Business-only) agreement: {agree_AB:.4f}")
print(f"A (ML-only) vs C (Equal-weight DADE) agreement: {agree_AC:.4f}")
print(f"B (Business-only) vs C (Equal-weight DADE) agreement: {agree_BC:.4f}")
print(f"A (ML-only) vs D (Expert-weighted DADE) agreement: {agree_AD:.4f}")
print(f"B (Business-only) vs D (Expert-weighted DADE) agreement: {agree_BD:.4f}")
print(f"C (Equal-weight DADE) vs D (Expert-weighted DADE) agreement: {agree_CD:.4f}")

results["pairwise_agreement"] = {
    "A_vs_B": round(agree_AB, 4),
    "A_vs_C": round(agree_AC, 4),
    "B_vs_C": round(agree_BC, 4),
    "A_vs_D": round(agree_AD, 4),
    "B_vs_D": round(agree_BD, 4),
    "C_vs_D": round(agree_CD, 4),
}

with open(RESULTS_DIR / "ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save the raw sample for transparency/reproducibility
ablation_df = pd.DataFrame({
    "true_label": true_label,
    "attack_cat": attack_cat_all[sample_idx],
    "MLS": np.round(mls, 3),
    "Threat_Severity": np.round(threat_severity, 3),
    "Asset_Criticality": np.round(asset_criticality, 3),
    "Business_Impact": np.round(business_impact, 3),
    "Historical_Incidents": np.round(historical_incidents, 3),
    "RTO_Urgency": np.round(rto_urgency, 3),
    "RPO_Urgency": np.round(rpo_urgency, 3),
    "Business_Only_Score": np.round(business_only_score, 3),
    "DADE_Score_EqualWeight": np.round(dade_score, 3),
    "DADE_Score_ExpertWeighted": np.round(dade_score_ahp, 3),
    "Decision_A_ML_only": decision_A,
    "Decision_B_Business_only": decision_B,
    "Decision_C_DADE_EqualWeight": decision_C,
    "Decision_D_DADE_ExpertWeighted": decision_D,
})
ablation_df.to_csv(RESULTS_DIR / "ablation_sample.csv", index=False)
print(f"\nSaved {RESULTS_DIR / 'ablation_results.json'} and {RESULTS_DIR / 'ablation_sample.csv'}")
