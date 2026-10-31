"""
R25 - Sensitivity of the decision-layer comparison (exploratory; written on 31 October 2026 after a further
review of the manuscript, after all labels were known).

  A. Cost matrix. The primary cost matrix (r10, fixed in the protocol) charges under-prioritisation two to
     three times over-escalation. Here the same leave-one-expert-out decisions are re-scored with
       - symmetric linear costs |i - j|,
       - symmetric quadratic costs (i - j)^2,
       - under-prioritisation k times over-escalation, linear in the number of levels, k = 1.0 to 4.0,
     and the paired difference DADE minus each baseline is bootstrapped over scenarios (2,000 resamples).
  B. Two-way bootstrap. The intervals of r10 and r19 resample scenarios only. Here scenarios AND experts are
     resampled (2,000 resamples) for DADE minus B2 and DADE minus B5 (quadratic kappa and mean cost).
  C. Base rate at the decision layer. The scenario MLS is the isotonic score under the 68% attack share of the
     training data. It is corrected with Eq. (3) of the paper (Saerens prior shift) to deployment priors of
     50%, 10%, 5% and 1%, and every rule is re-applied with the group weights and the fixed cut-offs. Only the
     number of changed decisions and the number of High decisions are reported: the labels refer to the MLS
     that the raters saw, so costs under corrected inputs would not measure performance.
Outputs: results/revision/r25_decision_sensitivity.json
"""
import numpy as np
import pandas as pd

from common import SEED, prior_shift, save_json
from decision_common import (ALL_EXPERTS, LEVELS, RANK, W_AHP, all_methods, load_labels, load_scenarios,
                             loeo_frame, rnd, scores)

S = load_scenarios()
L = load_labels()
P = loeo_frame(L, S)
methods = [c for c in P.columns if c not in ("expert_id", "scenario_id", "label")]
PI_TRAIN = 0.6806
N_BOOT = 2000
out = {"n_labels": int(len(P)), "n_scenarios": int(P["scenario_id"].nunique()), "n_boot": N_BOOT,
       "status": "exploratory, post hoc (31 October 2026)"}


def cost_matrix(kind, k=1.0):
    C = np.zeros((3, 3))
    for i in range(3):           # i = rater's label (Low, Medium, High)
        for j in range(3):       # j = method's level
            d = abs(i - j)
            if kind == "symmetric_linear":
                C[i, j] = d
            elif kind == "symmetric_quadratic":
                C[i, j] = d ** 2
            elif kind == "under_k":
                C[i, j] = d * (k if j < i else 1.0)
    return C


PROTOCOL = np.array([[0, 1, 2], [2, 0, 1], [6, 3, 0]], dtype=float)
yi = np.array([RANK[v] for v in P["label"]])
pred = {m: np.array([RANK[v] for v in P[m]]) for m in methods}
groups = [np.flatnonzero(P["scenario_id"].values == s) for s in P["scenario_id"].unique()]


def mean_cost(C, idx=None):
    idx = np.arange(len(yi)) if idx is None else idx
    return {m: float(np.mean(C[yi[idx], pred[m][idx]])) for m in methods}


def boot_diff(C, base):
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        c = mean_cost(C, idx)
        d.append(c["DADE_AHP"] - c[base])
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


variants = {"protocol": PROTOCOL, "symmetric_linear": cost_matrix("symmetric_linear"),
            "symmetric_quadratic": cost_matrix("symmetric_quadratic")}
for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    variants[f"under_x{k}"] = cost_matrix("under_k", k)
A = {}
for name, C in variants.items():
    mc = mean_cost(C)
    A[name] = {"matrix_rows_label_LMH_cols_method_LMH": C.tolist(),
               "mean_cost": mc,
               "DADE_minus_B5": {"diff": mc["DADE_AHP"] - mc["B5_ML_x_impact"], "ci95": boot_diff(C, "B5_ML_x_impact")},
               "DADE_minus_B2": {"diff": mc["DADE_AHP"] - mc["B2_equal_weight"], "ci95": boot_diff(C, "B2_equal_weight")},
               "DADE_minus_B1": {"diff": mc["DADE_AHP"] - mc["B1_ML_only"], "ci95": boot_diff(C, "B1_ML_only")},
               "cheapest": min(mc, key=mc.get)}
out["A_cost_matrix_sensitivity"] = A

# ---------------------------------------------------------------- B. two-way bootstrap
rng = np.random.default_rng(SEED)
scen_ids = P["scenario_id"].unique()
two = {b: {"quadratic_kappa": [], "mean_cost": []} for b in ["B2_equal_weight", "B5_ML_x_impact"]}
by_es = {(e, s): i for i, (e, s) in enumerate(zip(P["expert_id"], P["scenario_id"]))}
for _ in range(N_BOOT):
    es = rng.choice(ALL_EXPERTS, len(ALL_EXPERTS), replace=True)
    ss = rng.choice(scen_ids, len(scen_ids), replace=True)
    idx = [by_es[(e, s)] for e in es for s in ss if (e, s) in by_es]
    if len(idx) < 10:
        continue
    y = P["label"].values[idx]
    sd = scores(y, P["DADE_AHP"].values[idx])
    for b in two:
        sb = scores(y, P[b].values[idx])
        for k in two[b]:
            two[b][k].append(sd[k] - sb[k])
out["B_two_way_bootstrap_DADE_minus"] = {
    b: {k: {"mean": round(float(np.mean(v)), 4),
            "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
            "n_resamples": len(v)} for k, v in d.items()} for b, d in two.items()}

# ---------------------------------------------------------------- C. base rate at the decision layer
base = all_methods(S, W_AHP)
C_ = {"pi_train": PI_TRAIN, "uncorrected_high_decisions": {m: int(np.sum(base[m] == "High")) for m in base}}
for pi_new in [0.5, 0.1, 0.05, 0.01]:
    Sc = S.copy()
    Sc["MLS"] = prior_shift(S["MLS"].values, PI_TRAIN, pi_new)
    new = all_methods(Sc, W_AHP)
    C_[f"prior_{pi_new}"] = {
        "decisions_changed_of_30": {m: int(np.sum(new[m] != base[m])) for m in base},
        "high_decisions": {m: int(np.sum(new[m] == "High")) for m in base},
        "scenario_mls_range": [round(float(Sc["MLS"].min()), 4), round(float(Sc["MLS"].max()), 4)]}
out["C_base_rate_at_decision_layer"] = C_
save_json(rnd(out), "r25_decision_sensitivity.json")
for name, a in A.items():
    print(f"{name:20s} DADE {a['mean_cost']['DADE_AHP']:.3f}  B5 {a['mean_cost']['B5_ML_x_impact']:.3f}  "
          f"diff {a['DADE_minus_B5']['diff']:+.3f} {a['DADE_minus_B5']['ci95']}  cheapest {a['cheapest']}")
print(out["B_two_way_bootstrap_DADE_minus"])
for k, v in C_.items():
    print(k, v)
