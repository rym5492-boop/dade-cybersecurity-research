"""
R04 - Leakage audit of the paper's ablation (Section 5.5, Figure 10; supervisor Critical 1).

Step 1 reproduces the original ablation exactly (same model, seed, 200-row sample and random
business-context values) and checks the published numbers (A 0.830, C 0.815, D 0.920).
Step 2 changes ONE thing: Threat Severity.
    oracle    = severity of the TRUE test attack_cat (the original, not available at decision time)
    predicted = severity of the category predicted by R03 (trained on the training file only)
    none      = Threat Severity removed and the remaining weights renormalised
Everything else (MLS, random context, thresholds) is held fixed, so differences are attributable to
leakage alone. The audit is also repeated on the full official test set (n = 82,332).

IMPORTANT: agreement with attack/non-attack labels is a DETECTION-level check, used here only to
quantify leakage. It is not evidence about continuity-priority decision quality (supervisor
Critical 1, second bullet); that is evaluated separately against expert priority labels.
Requires: results/ahp_weights_results.json, results/revision/r03_test_predicted_severity.npz
Output : results/revision/r04_ablation_leakage_audit.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import OUT_DIR, REPO_ROOT, SEVERITY_MAP, bootstrap_ci, find_file, save_json

np.random.seed(42)  # identical RNG stream to 01_ablation_study.py
with open(REPO_ROOT / "results" / "ahp_weights_results.json") as f:
    W = json.load(f)["group"]["weights"]
s = sum(W.values())
W = {k: v / s for k, v in W.items()}

# ---- replicate the original model and sample exactly ----
train_df = pd.read_csv(find_file("UNSW_NB15_training-set.csv"))
test_df = pd.read_csv(find_file("UNSW_NB15_testing-set.csv"))
cat_cols, drop_cols = ["proto", "service", "state"], ["id", "attack_cat", "label"]
combined = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0)
for col in cat_cols:
    le = LabelEncoder().fit(combined[col].astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
X_tr, X_te = train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols)
sc = StandardScaler()
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(sc.fit_transform(X_tr), train_df["label"])
mls_all = rf.predict_proba(sc.transform(X_te))[:, 1]
y_all = test_df["label"].values
ts_oracle_all = np.array([SEVERITY_MAP.get(c, 0.6) for c in test_df["attack_cat"].values])

n = 200
idx = np.random.choice(len(y_all), size=n, replace=False)
ctx = {k: np.random.uniform(lo, 1.0, size=n) for k, lo in
       [("AC", 0.2), ("BI", 0.2), ("HI", 0.0), ("RTO", 0.2), ("RPO", 0.2)]}

pred = np.load(OUT_DIR / "r03_test_predicted_severity.npz", allow_pickle=True)
assert (pred["true_cat"] == test_df["attack_cat"].astype(str).str.strip().values).all()
ts_pred_all = pred["ts_pred"]


def scores(mls, ts, c, variant):
    """Return equal-weight (C) and AHP-weighted (D) DADE scores for one TS variant."""
    if variant == "none":
        keys = ["MLS", "AC", "BI", "HI", "RTO", "RPO"]
        vals = {"MLS": mls, **c}
    else:
        keys = ["MLS", "TS", "AC", "BI", "HI", "RTO", "RPO"]
        vals = {"MLS": mls, "TS": ts, **c}
    eq = sum(vals[k] for k in keys) / len(keys)
    wsum = sum(W[k] for k in keys)
    ahp = sum(W[k] / wsum * vals[k] for k in keys)
    return eq, ahp


def evaluate(y, mls, ts_by_variant, c, with_ci):
    res = {"A_ML_only": {"accuracy": round(accuracy_score(y, mls >= .5), 4),
                         "kappa": round(cohen_kappa_score(y, (mls >= .5).astype(int)), 4)}}
    for v, ts in ts_by_variant.items():
        eq, ahp = scores(mls, ts, c, v)
        for name, sc_ in [("C_equal_weight", eq), ("D_AHP_weight", ahp)]:
            d = (sc_ >= .5).astype(int)
            r = {"accuracy": round(accuracy_score(y, d), 4), "kappa": round(cohen_kappa_score(y, d), 4)}
            if with_ci:
                r["accuracy_95ci"] = bootstrap_ci(lambda a, b: (a == b).mean(), y, d, n_boot=2000)
            res[f"{name}__TS_{v}"] = r
    return res


out = {"note": "Detection-level leakage audit only; not a continuity-priority validation.",
       "weights_used": {k: round(v, 4) for k, v in W.items()}}
out["paper_sample_n200"] = evaluate(
    y_all[idx], mls_all[idx],
    {"oracle": ts_oracle_all[idx], "predicted": ts_pred_all[idx], "none": None}, ctx, with_ci=True)
check = out["paper_sample_n200"]
out["reproduces_published_numbers"] = bool(check["A_ML_only"]["accuracy"] == 0.83 and
                                           check["C_equal_weight__TS_oracle"]["accuracy"] == 0.815 and
                                           check["D_AHP_weight__TS_oracle"]["accuracy"] == 0.92)

rng = np.random.default_rng(42)
N = len(y_all)
ctx_full = {k: rng.uniform(lo, 1.0, size=N) for k, lo in
            [("AC", 0.2), ("BI", 0.2), ("HI", 0.0), ("RTO", 0.2), ("RPO", 0.2)]}
out["full_test_n82332"] = evaluate(
    y_all, mls_all, {"oracle": ts_oracle_all, "predicted": ts_pred_all, "none": None}, ctx_full, with_ci=False)

save_json(out, "r04_ablation_leakage_audit.json")
print("reproduces published:", out["reproduces_published_numbers"])
for part in ["paper_sample_n200", "full_test_n82332"]:
    print(f"\n{part}")
    for k, v in out[part].items():
        print(f"  {k:32s} acc {v['accuracy']:.4f}  kappa {v['kappa']:.4f}  {v.get('accuracy_95ci', '')}")
