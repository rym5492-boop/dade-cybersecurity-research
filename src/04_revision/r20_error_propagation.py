"""
R20 - Uncertainty and error propagation from the detection layer and the BIA record into the decision
(supervisor review of 30 October 2026, Experiments: "an uncertainty/error propagation experiment").

Each of the 30 scenarios shows one UNSW-NB15 test flow. The detector inputs of DADE (MLS and the severity of
the predicted category) are random in the sense that another flow of the same true traffic type would have
given other values. Three Monte Carlo experiments (2,000 draws each) replace the inputs of every scenario:
  D  detector: a flow of the same true category is drawn from the UNSW-NB15 test file and its isotonic HGB
     score (r17) and predicted category (r03) replace the scenario's MLS and TS together;
  B  BIA record: asset criticality and business impact move one rating up or down with probability 0.25
     each, the incident count by one with probability 0.25 each way, and RTO / RPO one band in the scale of
     Equation (4.3) with probability 0.25 each way;
  DB both together.
For DADE and each baseline: share of decisions that differ from the decision on the shown inputs, per-scenario
probability of a different level, and mean cost and High recall against the 120 expert labels
(leave-one-expert-out weights), as mean and 95% range over the draws.
Outputs: results/revision/r20_error_propagation.json
"""
import numpy as np
import pandas as pd

from common import OUT_DIR, SEED, SEVERITY_MAP, save_json
from decision_common import ALL_EXPERTS, W_AHP, all_methods, group_weights, load_labels, load_scenarios, rnd, scores

S = load_scenarios()
L = load_labels()
z = np.load(OUT_DIR / "r17_test_isotonic_scores.npz", allow_pickle=True)
c = np.load(OUT_DIR / "r03_test_predicted_severity.npz", allow_pickle=True)
assert np.array_equal(z["attack_cat"], c["true_cat"].astype(str))
P_ISO, TRUE_CAT, PRED_CAT = z["p"], z["true_cat"] if "true_cat" in z.files else z["attack_cat"], c["pred_cat"].astype(str)
TS_OF = {k: v for k, v in SEVERITY_MAP.items()}
pool = {cat: np.flatnonzero(TRUE_CAT == cat) for cat in np.unique(TRUE_CAT)}
UTIL = [1.0, 0.75, 0.5, 0.25, 0.0]          # Equation (4.3) bands: <=1, <=4, <=24, <=72, >72 hours
rng = np.random.default_rng(SEED)
N = 2000
W_LOO = {e: group_weights([x for x in ALL_EXPERTS if x != e]) for e in ALL_EXPERTS}
base = all_methods(S, W_AHP)
methods = list(base)


def perturb_detector(df):
    d = df.copy()
    for i, cat in enumerate(d["true_category"].fillna("Normal")):
        j = rng.choice(pool[cat])
        d.loc[i, "MLS"] = P_ISO[j]
        d.loc[i, "TS"] = TS_OF.get(PRED_CAT[j], 0.0)
    return d


def step(v, grid):
    k = int(np.argmin(np.abs(np.array(grid) - v)))
    r = rng.random()
    if r < 0.25:
        k = max(k - 1, 0)
    elif r < 0.5:
        k = min(k + 1, len(grid) - 1)
    return grid[k]


def perturb_bia(df):
    d = df.copy()
    for i in range(len(d)):
        for col, raw in (("AC", "AC_1to5"), ("BI", "BI_1to5")):
            v = int(np.clip(d.loc[i, raw] + rng.choice([-1, 0, 0, 1]), 1, 5))
            d.loc[i, raw] = v
            d.loc[i, col] = (v - 1) / 4
        n = max(int(round(d.loc[i, "HI"] * 5)) + int(rng.choice([-1, 0, 0, 1])), 0)
        d.loc[i, "HI"] = min(n, 5) / 5
        d.loc[i, "RTO"] = step(d.loc[i, "RTO"], UTIL)
        d.loc[i, "RPO"] = step(d.loc[i, "RPO"], UTIL)
    return d


def label_scores(df):
    res = {}
    rows = []
    for e in ALL_EXPERTS:
        te = L[L["expert_id"] == e]
        sub = df.set_index("scenario_id").loc[te["scenario_id"]].reset_index()
        meth = all_methods(sub, W_LOO[e])
        rows.append(pd.DataFrame({"label": te["label"].values, **meth}))
    F = pd.concat(rows, ignore_index=True)
    for m in methods:
        sc = scores(F["label"], F[m])
        res[m] = (sc["mean_cost"], sc["high_recall"])
    return res


out = {"n_draws": N, "shown_inputs": {m: dict(zip(["mean_cost", "high_recall"], v)) for m, v in label_scores(S).items()}}
for name, fn in [("D_detector", perturb_detector), ("B_bia", perturb_bia),
                 ("DB_both", lambda d: perturb_bia(perturb_detector(d)))]:
    changed = {m: np.zeros(30) for m in methods}
    share, costs, recs = {m: [] for m in methods}, {m: [] for m in methods}, {m: [] for m in methods}
    for _ in range(N):
        d = fn(S)
        meth = all_methods(d, W_AHP)
        for m in methods:
            diff = meth[m] != base[m]
            changed[m] += diff
            share[m].append(diff.mean())
        for m, (cst, rec) in label_scores(d).items():
            costs[m].append(cst); recs[m].append(rec)
    out[name] = {m: {"share_of_decisions_changed": {"mean": float(np.mean(share[m])), "p2.5": float(np.percentile(share[m], 2.5)),
                                                     "p97.5": float(np.percentile(share[m], 97.5))},
                     "mean_cost_vs_labels": {"mean": float(np.mean(costs[m])), "p2.5": float(np.percentile(costs[m], 2.5)),
                                             "p97.5": float(np.percentile(costs[m], 97.5))},
                     "high_recall_vs_labels": {"mean": float(np.mean(recs[m])), "p2.5": float(np.percentile(recs[m], 2.5)),
                                               "p97.5": float(np.percentile(recs[m], 97.5))},
                     "scenarios_changed_in_over_half_of_draws": int(np.sum(changed[m] / N > 0.5)),
                     "per_scenario_probability_of_change": dict(zip(S["scenario_id"], np.round(changed[m] / N, 3)))}
                 for m in methods}
    print(name, {m: (round(np.mean(share[m]), 3), round(np.mean(costs[m]), 3)) for m in methods})
save_json(rnd(out), "r20_error_propagation.json")
