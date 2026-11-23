"""Independent recomputation of the main decision-layer and base-rate results (verification of 23 November 2026).

Written from the definitions in the paper (Sections 3.2, 4.3, 4.6 and 4.7), not from the analysis scripts in
src/04_revision, and reading only raw inputs: data/scenario_labels.csv, data/external_panel_labels.csv,
data/external_reference_labels.csv, data/ahp_expert_judgments_anonymized.json, the scenario inputs
(results/revision/r10_decisions_stage1.csv: normalized criteria and stored levels), the isotonic test scores
(results/revision/r17_test_isotonic_scores.npz), the split manifest and the UNSW-NB15 files.
Output: results/verification/recompute_decision_layer.json, compared with the stored results at the end.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "revision"
OUT = ROOT / "results" / "verification"
OUT.mkdir(parents=True, exist_ok=True)
L = {"Low": 0, "Medium": 1, "High": 2, "L": 0, "M": 1, "H": 2}
C = np.array([[0, 1, 2], [2, 0, 1], [6, 3, 0]])                       # Eq. (6): rows rater, columns method
SAATY = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}

# ---------------------------------------------------------------- AHP weights (Eq. 5, AIJ)
ahp = json.load(open(ROOT / "data" / "ahp_expert_judgments_anonymized.json"))
crit = ahp["criteria_order"]; n = len(crit)
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]


def matrix(resp):
    A = np.ones((n, n))
    for (i, j), v in zip(pairs, resp):
        s = SAATY[abs(v - 5)]
        A[i, j], A[j, i] = (s, 1 / s) if v < 5 else ((1 / s, s) if v > 5 else (1, 1))
    return A


M = {e: matrix(r) for e, r in ahp["pairwise_responses"].items()}
EXPERTS = sorted(M)


def weights(experts):
    G = np.exp(np.mean([np.log(M[e]) for e in experts], axis=0))      # aggregation of individual judgments
    w = np.exp(np.mean(np.log(G), axis=1))                              # row geometric mean
    return dict(zip(crit, w / w.sum()))


W_GROUP = weights(EXPERTS)

# ---------------------------------------------------------------- rules (Eqs. 1-2, baselines B1-B5)
sc = list(csv.DictReader(open(RES / "r10_decisions_stage1.csv", encoding="utf-8-sig")))
S = {r["scenario_id"]: r for r in sc}
x = lambda r, c: float(r[c])
level = lambda s: 2 if s >= 0.66 else (0 if s < 0.33 else 1)
impact = lambda r: (lambda v: 3 if v >= 4 - 1e-9 else (2 if v >= 3 - 1e-9 else 1))(max(x(r, "AC"), x(r, "BI")) * 4 + 1)
cell = lambda p: 2 if p >= 6 else (1 if p >= 3 else 0)
RULES = {
    "DADE_AHP": lambda r, w: level(sum(w[c] * x(r, c) for c in crit)),
    "B1_ML_only": lambda r, w: level(x(r, "MLS")),
    "B2_equal_weight": lambda r, w: level(sum(x(r, c) for c in crit) / 7),
    "B3_BIA_only": lambda r, w: level(np.mean([x(r, c) for c in ["AC", "BI", "HI", "RTO", "RPO"]])),
    "B4_severity_x_impact": lambda r, w: cell((3 if x(r, "TS") >= 0.70 - 1e-9 else (2 if x(r, "TS") >= 0.40 - 1e-9 else 1)) * impact(r)),
    "B5_ML_x_impact": lambda r, w: cell((3 if x(r, "MLS") >= 0.66 else (2 if x(r, "MLS") >= 0.33 else 1)) * impact(r)),
}


def metrics(yp):
    y = np.array([a for a, _ in yp]); p = np.array([b for _, b in yp])
    O = np.zeros((3, 3))
    for a, b in zip(y, p):
        O[a, b] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    kap = lambda W: float(1 - (W * O).sum() / (W * E).sum())
    f1 = [0 if ((p == k) & (y == k)).sum() == 0 else 2 * ((p == k) & (y == k)).sum() / ((p == k).sum() + (y == k).sum()) for k in range(3)]
    return {"exact": float((y == p).mean()), "macro_f1": float(np.mean(f1)),
            "quadratic_kappa": kap(np.array([[(i - j) ** 2 for j in range(3)] for i in range(3)])),
            "linear_kappa": kap(np.array([[abs(i - j) for j in range(3)] for i in range(3)])),
            "high_recall": float(((p == 2) & (y == 2)).sum() / (y == 2).sum()), "escalation_rate": float((p == 2).mean()),
            "under_prioritisation": float((p < y).mean()), "over_escalation": float((p > y).mean()),
            "mean_cost": float(np.mean([C[a, b] for a, b in zip(y, p)]))}


out = {"group_weights": W_GROUP}
labels = list(csv.DictReader(open(ROOT / "data" / "scenario_labels.csv", encoding="utf-8-sig")))
out["table6_expert_labels_leave_one_expert_out"] = {
    m: metrics([(L[r["label"]], f(S[r["scenario_id"]], weights([e for e in EXPERTS if e != r["expert_id"]]) if m == "DADE_AHP" else None))
                for r in labels]) for m, f in RULES.items()}
col = {m: m for m in RULES}
out["stored_scenario_levels_reproduced"] = all(RULES[m](r, W_GROUP) == L[r[m]] for r in sc for m in RULES)

votes = defaultdict(list)
for r in labels:
    votes[r["scenario_id"]].append(L[r["label"]])
cons = {s: sorted(v)[2] for s, v in votes.items()}                     # median of four, ties to the higher level
out["consensus_counts_LMH"] = [sum(v == k for v in cons.values()) for k in range(3)]
out["table7_confusion_vs_consensus"] = {}
for m, f in RULES.items():
    T = np.zeros((3, 3), int)
    for s, c in cons.items():
        T[c, f(S[s], W_GROUP)] += 1
    out["table7_confusion_vs_consensus"][m] = T.tolist()

ref = {r["scenario_id"]: L[r["external_reference"]] for r in csv.DictReader(open(ROOT / "data" / "external_reference_labels.csv", encoding="utf-8-sig"))}
ext = list(csv.DictReader(open(ROOT / "data" / "external_panel_labels.csv", encoding="utf-8-sig")))
out["section57_external_reference"] = {m: metrics([(ref[s], f(S[s], W_GROUP)) for s in ref]) for m, f in RULES.items()}
out["table9_external_individual_fixed"] = {m: metrics([(L[r[s]], f(S[s], W_GROUP)) for r in ext for s in ref]) for m, f in RULES.items()}
out["panels"] = {"agree": sum(cons[s] == ref[s] for s in ref), "experts_higher": sum(cons[s] > ref[s] for s in ref),
                 "practitioners_higher": sum(cons[s] < ref[s] for s in ref),
                 "quadratic_kappa": metrics([(cons[s], ref[s]) for s in ref])["quadratic_kappa"]}

# ---------------------------------------------------------------- base rate (Section 4.3, Table 10)
z = np.load(RES / "r17_test_isotonic_scores.npz", allow_pickle=True)
p, y = z["p"].astype(float), z["y"].astype(int)
pi = 0.6806
t_eq4 = lambda pp: pi * (1 - pp) / (pi * (1 - pp) + pp * (1 - pi))
out["table10_benign_counts"] = {}
for name, t in [("BR1", 0.5), ("BR2", t_eq4(0.01)), ("BR3_mean_em_prior", t_eq4(0.0219)), ("BR5", 0.9123)]:
    fp = int(((p >= t) & (y == 0)).sum())
    out["table10_benign_counts"][name] = {"threshold": t, "FP": fp, "TN": int((y == 0).sum() - fp),
                                          "expected_TP_of_374": 374 * float((p[y == 1] >= t).mean())}
pc = 0.01 * (1 - pi) * p / (0.01 * (1 - pi) * p + pi * 0.99 * (1 - p))
out["eq4_equivalence_flows_differing"] = int(((pc >= 0.5) != (p >= t_eq4(0.01))).sum())

# ---------------------------------------------------------------- data separation (Section 4.1)
tr = pd.read_csv(ROOT / "data" / "UNSW_NB15_training-set.csv"); te = pd.read_csv(ROOT / "data" / "UNSW_NB15_testing-set.csv")
sp = pd.read_csv(RES / "splits" / "unsw_fit_val_cal_ids.csv")
m = tr.merge(sp, on="id")
out["split_manifest"] = {"rows": len(tr), "rows_in_manifest": len(m), "ids_unique": bool(sp["id"].is_unique),
                         "counts": sp["split"].value_counts().to_dict(),
                         "attack_share": m.groupby("split")["label"].mean().round(6).to_dict()}
scen = pd.read_csv(RES / "r09_scenarios.csv")
SEV = {"Normal": 0, "Fuzzers": 0.35, "Reconnaissance": 0.40, "Analysis": 0.55, "Generic": 0.60, "DoS": 0.70,
       "Exploits": 0.80, "Backdoor": 0.85, "Shellcode": 0.90, "Worms": 0.95}
out["ts_from_predicted_category_all_30"] = bool(((scen["shown_category"].map(SEV) - scen["TS_pred"]).abs() < 1e-9).all())
feat = [c for c in tr.columns if c not in ("id", "label", "attack_cat")]
key = lambda d: d[feat].astype(str).agg("|".join, axis=1)
tr["k"] = key(tr).values; rows = te.loc[scen["test_row_index"]]; rk = key(rows).values
dup = []
for sid, k in zip(scen["scenario_id"], rk):
    hit = tr[tr["k"] == k]
    if len(hit):
        dup.append({"scenario": sid, "training_copies": len(hit), "splits": sp[sp["id"].isin(hit["id"])]["split"].tolist(),
                    "training_labels": hit["attack_cat"].tolist()})
out["scenario_flows_equal_to_training_rows"] = dup


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


json.dump(clean(out), open(OUT / "recompute_decision_layer.json", "w"), indent=1)

# ---------------------------------------------------------------- comparison with the stored results
st = json.load(open(RES / "r10_stage2.json"))["primary_leave_one_expert_out"]
diffs = []
for mth, v in out["table6_expert_labels_leave_one_expert_out"].items():
    for k in ("exact", "macro_f1", "quadratic_kappa", "high_recall", "mean_cost"):
        sk = {"exact": "exact_agreement", "macro_f1": "macro_f1"}.get(k, k)
        if sk in st[mth] and abs(st[mth][sk] - v[k]) > 5e-5:
            diffs.append((mth, k, st[mth][sk], v[k]))
print("Table 6 versus r10_stage2.json (stored to 4 decimals):", "no difference" if not diffs else diffs)
print("stored scenario levels reproduced:", out["stored_scenario_levels_reproduced"])
print("written", OUT / "recompute_decision_layer.json")
