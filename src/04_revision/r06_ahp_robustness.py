"""
R06 - AHP reliability and weight uncertainty (supervisor review, Critical 4 and 3.3).

Uses only data/ahp_expert_judgments_anonymized.json (same input as 00_ahp_expert_weights.py).
Reports:
  * individual CR with two standard estimators of lambda_max (row-geometric-mean, as in the paper,
    and principal eigenvalue), plus a straight-lining flag (all 21 answers = "equal");
  * group weights under five aggregation choices: AIJ (paper), AIP, AIJ excluding the straight-liner,
    AIJ consistent experts only (CR <= 0.10, straight-liner excluded), AIJ excluding the most
    inconsistent expert;
  * expert-bootstrap 95% intervals for the AIJ weights and the probability of each rank position;
  * the projected effect of the targeted revisions sent to E1, E2 and E6 if the suggested values
    are accepted (sensitivity only; replaced by real answers once they arrive).
Output: results/revision/r06_ahp_robustness.json
"""
import itertools

import numpy as np

from common import REPO_ROOT, SEED, save_json
import json

with open(REPO_ROOT / "data" / "ahp_expert_judgments_anonymized.json", encoding="utf-8") as f:
    D = json.load(f)
C = D["criteria_order"]
n = len(C)
PAIRS = list(itertools.combinations(range(n), 2))
SAATY = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
RI7 = 1.32


def to_matrix(resp):
    A = np.ones((n, n))
    for (i, j), v in zip(PAIRS, resp):
        s = SAATY[abs(v - 5)]
        if v < 5:
            A[i, j], A[j, i] = s, 1 / s
        elif v > 5:
            A[i, j], A[j, i] = 1 / s, s
    return A


def gm_weights(A):
    w = np.exp(np.log(A).mean(axis=1))
    return w / w.sum()


def cr_gm(A):
    w = gm_weights(A)
    lam = np.mean((A @ w) / w)
    return ((lam - n) / (n - 1)) / RI7


def cr_eig(A):
    lam = np.max(np.linalg.eigvals(A).real)
    return ((lam - n) / (n - 1)) / RI7


def aij(mats):
    return np.exp(np.mean([np.log(M) for M in mats], axis=0))


def aip(mats):
    W = np.array([gm_weights(M) for M in mats])
    w = np.exp(np.log(W).mean(axis=0))
    return w / w.sum()


resp = D["pairwise_responses"]
mats = {e: to_matrix(r) for e, r in resp.items()}
out = {"individual": {}}
for e, r in resp.items():
    A = mats[e]
    out["individual"][e] = {"CR_geometric_mean": round(cr_gm(A), 4), "CR_eigenvalue": round(cr_eig(A), 4),
                            "straight_lined_all_equal": bool(all(v == 5 for v in r)),
                            "consistent_CR_le_0.10": bool(cr_gm(A) <= 0.10)}
straight = [e for e, v in out["individual"].items() if v["straight_lined_all_equal"]]
consistent = [e for e, v in out["individual"].items() if v["consistent_CR_le_0.10"] and e not in straight]
worst = max(out["individual"], key=lambda e: out["individual"][e]["CR_geometric_mean"])


def summarise(w, A=None):
    d = {"weights": {c: round(float(x), 3) for c, x in zip(C, w)},
         "ranking": [C[i] for i in np.argsort(-w)]}
    if A is not None:
        d["group_CR"] = round(cr_gm(A), 4)
    return d


variants = {
    "AIJ_all_six (paper)": list(mats),
    f"AIJ_excluding_straight_liner {straight}": [e for e in mats if e not in straight],
    f"AIJ_consistent_only {consistent}": consistent,
    f"AIJ_excluding_most_inconsistent [{worst}]": [e for e in mats if e != worst],
}
out["aggregation_variants"] = {}
for name, experts in variants.items():
    A = aij([mats[e] for e in experts])
    out["aggregation_variants"][name] = summarise(gm_weights(A), A)
out["aggregation_variants"]["AIP_all_six"] = summarise(aip(list(mats.values())))

# Expert bootstrap (resample experts with replacement)
rng = np.random.default_rng(SEED)
keys = list(mats)
B = 5000
W = np.empty((B, n))
for b in range(B):
    pick = rng.choice(keys, size=len(keys), replace=True)
    W[b] = gm_weights(aij([mats[k] for k in pick]))
ranks = np.argsort(np.argsort(-W, axis=1), axis=1)  # 0 = most important
out["expert_bootstrap_AIJ"] = {
    "B": B,
    "weight_95ci": {c: [round(float(np.quantile(W[:, i], .025)), 3), round(float(np.quantile(W[:, i], .975)), 3)]
                    for i, c in enumerate(C)},
    "probability_rank_1": {c: round(float((ranks[:, i] == 0).mean()), 3) for i, c in enumerate(C)},
    "probability_in_top_3": {c: round(float((ranks[:, i] <= 2).mean()), 3) for i, c in enumerate(C)},
}

# Projected effect of the targeted revision forms (if the suggested consistent values are accepted)
suggested = {"E1": {7: 5, 18: 5, 17: 5, 2: 4, 1: 4, 12: 5},
             "E2": {7: 5, 18: 5, 2: 5},
             "E6": {19: 5, 15: 5, 17: 6}}
proj = {}
for e, changes in suggested.items():
    r = list(resp[e])
    for q, v in changes.items():
        r[q - 1] = v
    proj[e] = {"CR_before": round(cr_gm(mats[e]), 4), "CR_if_suggestions_accepted": round(cr_gm(to_matrix(r)), 4),
               "questions_sent": sorted(changes)}
out["projected_targeted_revision_effect"] = proj

save_json(out, "r06_ahp_robustness.json")
print(json.dumps({k: out["individual"][k] for k in out["individual"]}, indent=0))
for k, v in out["aggregation_variants"].items():
    print(f"{k:45s} {v.get('group_CR', '-')!s:7s} {v['ranking']}  {v['weights']}")
print("bootstrap CI:", out["expert_bootstrap_AIJ"]["weight_95ci"])
print("P(rank1):", out["expert_bootstrap_AIJ"]["probability_rank_1"])
print("projected:", proj)
