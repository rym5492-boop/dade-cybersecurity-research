"""
R14 - Checks declared before the final run, kept out of r10 (r10 is not edited after the labels arrive).

1. AHP consistency of every expert before and after the October 2026 revisions (Section 4.6, Table 10).
2. Sensitivity declared on 20 Oct 2026: the primary leave-one-expert-out analysis with E3's AHP answers
   removed from every weight set (E3's scenario labels are kept), and the 30 group-weight decisions with and
   without E3.
3. External panel, item 3 of PREREG_ADDENDUM_external_panel.md: DADE (AHP weights of all six panel
   experts; the external raters gave no weights) against the external reference label, by linear-weighted
   kappa and mean misprioritization cost; quadratic-weighted kappa is added for comparison with the
   primary analysis (not in the addendum). Baselines B1-B5 are scored the same way for context.
4. Exploratory: agreement between the panel's consensus label (r10 rule) and the external reference.

Inputs : data/ahp_expert_judgments_anonymized.json, data/scenario_labels.csv,
         data/external_reference_labels.csv, results/revision/r10_decisions_stage1.csv
Output : results/revision/r14_final_checks.json
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import OUT_DIR, REPO_ROOT, save_json

CRIT = ["TS", "AC", "BI", "HI", "RTO", "RPO", "MLS"]
LEVELS = ["Low", "Medium", "High"]
rank = {l: i for i, l in enumerate(LEVELS)}
COST = np.array([[0, 1, 2], [2, 0, 1], [6, 3, 0]])          # same matrix as r10
PAIRS = list(itertools.combinations(range(7), 2))
SAATY = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
METHODS = ["DADE_AHP", "B1_ML_only", "B2_equal_weight", "B3_BIA_only", "B4_severity_x_impact", "B5_ML_x_impact"]

AHP = json.load(open(REPO_ROOT / "data" / "ahp_expert_judgments_anonymized.json", encoding="utf-8"))


def to_matrix(resp):
    A = np.ones((7, 7))
    for (i, j), v in zip(PAIRS, resp):
        s = SAATY[abs(v - 5)]
        if v < 5:
            A[i, j], A[j, i] = s, 1 / s
        elif v > 5:
            A[i, j], A[j, i] = 1 / s, s
    return A


def gm(A):
    w = np.exp(np.log(A).mean(axis=1))
    return w / w.sum()


def cr(A):
    w = gm(A)
    return ((np.mean((A @ w) / w) - 7) / 6) / 1.32


def group_weights(experts, src="pairwise_responses"):
    G = np.exp(np.mean([np.log(to_matrix(AHP[src][e])) for e in experts], axis=0))
    return gm(G)


def level(score):
    return np.where(score >= 0.66, "High", np.where(score < 0.33, "Low", "Medium"))


def scores(y, p):
    yi = np.array([rank[v] for v in y]); pi = np.array([rank[v] for v in p])
    hi = yi == 2
    return {"n": int(len(y)), "exact_agreement": round(float(np.mean(yi == pi)), 4),
            "linear_kappa": round(float(cohen_kappa_score(yi, pi, labels=[0, 1, 2], weights="linear")), 4),
            "quadratic_kappa": round(float(cohen_kappa_score(yi, pi, labels=[0, 1, 2], weights="quadratic")), 4),
            "high_recall": round(float(np.mean(pi[hi] == 2)), 4) if hi.any() else None,
            "under_prioritisation": round(float(np.mean(pi < yi)), 4),
            "over_escalation": round(float(np.mean(pi > yi)), 4),
            "mean_cost": round(float(np.mean(COST[yi, pi])), 4)}


out = {}

# 1. consistency before and after revision
out["individual_CR"] = {e: {"september": round(cr(to_matrix(AHP["pairwise_responses_original_sept2026"][e])), 4),
                            "final": round(cr(to_matrix(AHP["pairwise_responses"][e])), 4)}
                        for e in sorted(AHP["pairwise_responses"])}

# 2. leave-one-expert-out without E3's AHP answers
S = pd.read_csv(OUT_DIR / "r09_scenarios.csv", encoding="utf-8-sig").set_index("scenario_id")
S["TS"] = S["TS_pred"]
L = pd.read_csv(REPO_ROOT / "data" / "scenario_labels.csv", encoding="utf-8-sig")
ALL = sorted(AHP["pairwise_responses"])
y, p_with, p_without = [], [], []
for e, d in L.groupby("expert_id"):
    X = S.loc[d["scenario_id"], CRIT].values
    w_with = group_weights([x for x in ALL if x != e])
    w_without = group_weights([x for x in ALL if x not in (e, "E3")])
    y += list(d["label"]); p_with += list(level(X @ w_with)); p_without += list(level(X @ w_without))
out["loeo_DADE_with_E3_ahp"] = scores(y, p_with)
out["loeo_DADE_without_E3_ahp"] = scores(y, p_without)
out["loeo_decisions_differing"] = int(sum(a != b for a, b in zip(p_with, p_without)))
X_all = S[CRIT].values
g_with, g_without = level(X_all @ group_weights(ALL)), level(X_all @ group_weights([x for x in ALL if x != "E3"]))
out["group_weight_decisions_differing_without_E3"] = {s: {"with_E3": w, "without_E3": v}
                                                      for s, w, v in zip(S.index, g_with, g_without) if w != v}

# 3. external panel: DADE and baselines against the external reference
D = pd.read_csv(OUT_DIR / "r10_decisions_stage1.csv", encoding="utf-8-sig").set_index("scenario_id")
R = pd.read_csv(REPO_ROOT / "data" / "external_reference_labels.csv").set_index("scenario_id")
ref = R.loc[D.index, "external_reference"].values
out["external_reference_vs_methods"] = {m: scores(ref, D[m].values) for m in METHODS}

# 4. exploratory: panel consensus (median of four, ties to the more cautious level, as in r10) vs external
def consensus(v):
    r = sorted(rank[x] for x in v); m = len(r)
    return LEVELS[r[m // 2] if m % 2 else max(r[m // 2 - 1], r[m // 2])]
cons = L.groupby("scenario_id")["label"].apply(consensus).loc[D.index].values
out["exploratory_panel_consensus_vs_external_reference"] = scores(ref, cons)

save_json(out, "r14_final_checks.json")
print(json.dumps(out, indent=1))
