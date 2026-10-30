"""
R18 - Threshold selection evaluated on held-out scenarios (supervisor review of 30 October 2026, P0-5 and
"minimum additional evidence" item 2).

The fixed cut-offs 0.66 / 0.33 under-prioritise, and r10 tuned cut-offs on only one form of ten scenarios.
Here every threshold is chosen without the scenarios it is evaluated on:
  Policies for the rules that use cut-offs (DADE, B1, B2, B3, and the likelihood bands of B5):
    FIXED       0.66 / 0.33, set before any label existed;
    COST        (high, low) minimising the mean misprioritisation cost on the training scenarios;
    RECALL90    the highest High cut-off whose training High-priority recall is >= 0.90, then the low
                cut-off minimising cost (a cost-aware rule that protects against missed High incidents);
    NESTED      the policy among FIXED / COST / RECALL90 with the lowest inner cross-validated cost on the
                training scenarios (inner leave-one-group-out), refitted on all training scenarios.
  B4 (severity x impact) has no cut-off on a score and is reported as is.
Schemes (all labels of a scenario stay together; DADE weights leave the labelling expert out; thresholds are
tuned on the training scenarios WITHOUT the labels of the expert whose labels are scored):
  S1 leave-one-form-out (train 20 scenarios, test 10; 3 folds);
  S2 grouped 5-fold cross-validation over the 30 scenarios, repeated 100 times with different partitions.
Independent raters: thresholds LOCKED from all 120 expert labels (policy COST and RECALL90) are written to
results/revision/r18_locked_thresholds.json first and only then scored against the 120 labels of the four
external practitioners (same scenarios, raters who took no part in the weights or the tuning).
Trade-off curves: High cut-off swept from 0.30 to 0.90 for DADE, B1, B2 and B3 (low cut-off 0.33, or 0.02
below the High cut-off), with High recall, escalation rate and mean cost against the leave-one-expert-out labels.
Outputs: results/revision/r18_threshold_study.json, r18_locked_thresholds.json, r18_tradeoff_curves.csv
"""
import json

import numpy as np
import pandas as pd

from common import OUT_DIR, SEED, save_json
from decision_common import (ALL_EXPERTS, BIA, COST, HIGH_T, LEVELS, LOW_T, RANK, W_AHP, W_EQ, dade_score,
                             group_weights, impact_band, level, load_external, load_labels, load_scenarios,
                             matrix_rule, rnd, scores, severity_impact_matrix)

S = load_scenarios().set_index("scenario_id")
L = load_labels()
X = load_external()
GRID = [(round(h, 2), round(l, 2)) for h in np.arange(0.30, 0.91, 0.02) for l in np.arange(0.10, 0.61, 0.02) if l < h - 1e-9]
TUNABLE = ["DADE_AHP", "B1_ML_only", "B2_equal_weight", "B3_BIA_only", "B5_ML_x_impact"]
KEYS = ["mean_cost", "quadratic_kappa", "high_recall", "escalation_rate", "under_prioritisation", "over_escalation"]


def raw_score(method, sub, w):
    if method == "DADE_AHP":
        return dade_score(sub, w)
    if method == "B1_ML_only" or method == "B5_ML_x_impact":
        return sub["MLS"].values.astype(float)
    if method == "B2_equal_weight":
        return dade_score(sub, W_EQ)
    if method == "B3_BIA_only":
        return sub[BIA].mean(axis=1).values.astype(float)
    raise ValueError(method)


def decide(method, sub, w, high, low):
    s = raw_score(method, sub, w)
    if method == "B5_ML_x_impact":
        like = np.where(s >= high, 3, np.where(s >= low, 2, 1))
        return matrix_rule(like, impact_band(sub))
    return level(s, high, low)


def cost_of(labels, preds):
    return float(np.mean(COST[[RANK[v] for v in labels], [RANK[v] for v in preds]]))


GH = np.array([t[0] for t in GRID]); GL = np.array([t[1] for t in GRID])
CLOSE = np.abs(GH - HIGH_T) + np.abs(GL - LOW_T)


def grid_levels(method, s, band):
    """Level index (0 Low, 1 Medium, 2 High) for every grid pair (rows) and case (columns)."""
    hi = s[None, :] >= GH[:, None]
    mid = s[None, :] >= GL[:, None]
    if band is None:
        return np.where(hi, 2, np.where(mid, 1, 0))
    like = np.where(hi, 3, np.where(mid, 2, 1)) * band[None, :]
    return np.where(like >= 6, 2, np.where(like >= 3, 1, 0))


def tune(method, frame, policy):
    """frame: labels of the TRAINING scenarios (weights in frame.attrs['w'])."""
    if policy == "FIXED":
        return HIGH_T, LOW_T
    sub = S.loc[frame["scenario_id"]]
    s = raw_score(method, sub, frame.attrs["w"])
    yi = np.array([RANK[v] for v in frame["label"].values])
    band = impact_band(sub) if method == "B5_ML_x_impact" else None
    lv = grid_levels(method, s, band)
    cost = COST[yi[None, :], lv].mean(axis=1)
    order = np.lexsort((CLOSE, np.round(cost, 10)))
    if policy == "COST":
        return GRID[order[0]]
    if policy == "RECALL90":
        hi = yi == 2
        if not hi.any():
            return GRID[order[0]]
        rec = (lv[:, hi] == 2).mean(axis=1)
        ok = np.flatnonzero(rec >= 0.90)
        if ok.size == 0:
            return GRID[order[0]]
        best_h = GH[ok].max()
        cand = ok[GH[ok] == best_h]
        return GRID[cand[np.lexsort((CLOSE[cand], np.round(cost[cand], 10)))[0]]]
    raise ValueError(policy)


def train_frame(train_scen, exclude_expert, w):
    f = L[L["scenario_id"].isin(train_scen) & (L["expert_id"] != exclude_expert)].copy()
    f.attrs["w"] = w
    return f


def nested_choice(method, train_scen, exclude_expert, w):
    """Inner leave-one-group-out over the training scenarios (groups = forms when possible, else 3 random groups)."""
    tr = list(train_scen)
    groups = sorted({s[0] for s in tr})
    if len(groups) < 2:
        rng_ = np.random.default_rng(len(tr))
        perm = rng_.permutation(tr)
        folds = [list(perm[i::3]) for i in range(3)]
    else:
        folds = [[s for s in tr if s[0] == g] for g in groups]
    inner = {}
    for pol in ["FIXED", "COST", "RECALL90"]:
        cs = []
        for k, te in enumerate(folds):
            trn = [s for s in tr if s not in te]
            t = tune(method, train_frame(trn, exclude_expert, w), pol)
            f = L[L["scenario_id"].isin(te) & (L["expert_id"] != exclude_expert)]
            cs.append(cost_of(f["label"], decide(method, S.loc[f["scenario_id"]], w, *t)))
        inner[pol] = float(np.mean(cs))
    best = min(inner, key=lambda p: (round(inner[p], 10), ["FIXED", "COST", "RECALL90"].index(p)))
    return best, tune(method, train_frame(train_scen, exclude_expert, w), best)


POLICIES = ["FIXED", "COST", "RECALL90", "NESTED"]
W_LOO = {e: group_weights([x for x in ALL_EXPERTS if x != e]) for e in ALL_EXPERTS}


def evaluate_split(train_scen, test_scen, collect):
    for e in ALL_EXPERTS:
        te = L[L["scenario_id"].isin(test_scen) & (L["expert_id"] == e)]
        if te.empty:
            continue
        w = W_LOO[e]
        sub = S.loc[te["scenario_id"]]
        b4 = severity_impact_matrix(sub)
        collect.setdefault(("B4_severity_x_impact", "FIXED"), []).append((te["scenario_id"].values, te["label"].values, b4, None))
        for m in TUNABLE:
            for pol in POLICIES:
                if pol == "NESTED":
                    chosen, t = nested_choice(m, train_scen, e, w)
                else:
                    chosen, t = pol, tune(m, train_frame(train_scen, e, w), pol)
                collect.setdefault((m, pol), []).append((te["scenario_id"].values, te["label"].values,
                                                         decide(m, sub, w, *t), (chosen, t)))


def summarise(collect):
    res = {}
    for (m, pol), parts in collect.items():
        y = np.concatenate([p[1] for p in parts]); yhat = np.concatenate([p[2] for p in parts])
        sc = scores(y, yhat)
        th = [p[3][1] for p in parts if p[3] is not None]
        res.setdefault(m, {})[pol] = {k: (None if sc[k] is None else round(sc[k], 4)) for k in KEYS}
        if th:
            res[m][pol]["thresholds_chosen"] = {"high_median": float(np.median([t[0] for t in th])),
                                                "high_range": [float(min(t[0] for t in th)), float(max(t[0] for t in th))],
                                                "low_median": float(np.median([t[1] for t in th])),
                                                "low_range": [float(min(t[1] for t in th)), float(max(t[1] for t in th))]}
            if pol == "NESTED":
                ch = [p[3][0] for p in parts]
                res[m][pol]["policy_chosen_share"] = {k: round(ch.count(k) / len(ch), 3) for k in ["FIXED", "COST", "RECALL90"]}
    return res


scen = sorted(S.index)
out = {"grid": {"high": [0.30, 0.90, 0.02], "low": [0.10, 0.60, 0.02]}, "cost_matrix": COST.tolist()}

# S1 leave-one-form-out
c1 = {}
for f in "ABC":
    evaluate_split([s for s in scen if s[0] != f], [s for s in scen if s[0] == f], c1)
out["S1_leave_one_form_out"] = summarise(c1)

# S2 repeated grouped 5-fold over scenarios
rng = np.random.default_rng(SEED)
rep_results = []
for r in range(100):
    perm = rng.permutation(scen)
    c2 = {}
    for k in range(5):
        test_scen = list(perm[k::5])
        evaluate_split([s for s in scen if s not in test_scen], test_scen, c2)
    rep_results.append(summarise(c2))
agg = {}
for m in rep_results[0]:
    agg[m] = {}
    for pol in rep_results[0][m]:
        agg[m][pol] = {}
        for k in KEYS:
            v = [rr[m][pol][k] for rr in rep_results if rr[m][pol][k] is not None]
            agg[m][pol][k] = {"mean": round(float(np.mean(v)), 4),
                              "p2.5": round(float(np.percentile(v, 2.5)), 4), "p97.5": round(float(np.percentile(v, 97.5)), 4)}
        if "thresholds_chosen" in rep_results[0][m][pol]:
            agg[m][pol]["high_median_range_over_repeats"] = [
                float(min(rr[m][pol]["thresholds_chosen"]["high_median"] for rr in rep_results)),
                float(max(rr[m][pol]["thresholds_chosen"]["high_median"] for rr in rep_results))]
        if pol == "NESTED":
            agg[m][pol]["policy_chosen_share_mean"] = {
                k: round(float(np.mean([rr[m][pol]["policy_chosen_share"][k] for rr in rep_results])), 3)
                for k in ["FIXED", "COST", "RECALL90"]}
out["S2_repeated_grouped_5fold_100x"] = agg

# Locked thresholds from all expert labels, then the external practitioners.
locked = {}
for m in TUNABLE:
    w = W_AHP
    f = L.copy(); f.attrs["w"] = w
    locked[m] = {pol: list(tune(m, f, pol)) for pol in ["FIXED", "COST", "RECALL90"]}
with open(OUT_DIR / "r18_locked_thresholds.json", "w", encoding="utf-8") as fh:
    json.dump({"note": "Thresholds chosen on the 120 expert labels with the group weights, written before the "
                       "external practitioners' labels are scored.", "thresholds": locked}, fh, indent=2)
ext = {}
subx = S.loc[X["scenario_id"]]
for m in TUNABLE:
    ext[m] = {}
    for pol, t in locked[m].items():
        ext[m][pol] = {k: (None if v is None else round(v, 4)) for k, v in scores(X["label"].values, decide(m, subx, W_AHP, *t)).items() if k in KEYS}
        ext[m][pol]["thresholds"] = t
ext["B4_severity_x_impact"] = {"FIXED": {k: round(v, 4) for k, v in scores(X["label"].values, severity_impact_matrix(subx)).items() if k in KEYS}}
out["external_practitioners_with_locked_thresholds"] = ext

# Trade-off curves (leave-one-expert-out labels).
rows = []
for m in ["DADE_AHP", "B1_ML_only", "B2_equal_weight", "B3_BIA_only"]:
    for h in np.round(np.arange(0.30, 0.91, 0.02), 2):
        lo = min(LOW_T, round(h - 0.02, 2))
        ys, ps = [], []
        for e in ALL_EXPERTS:
            te = L[L["expert_id"] == e]
            ys.append(te["label"].values); ps.append(decide(m, S.loc[te["scenario_id"]], W_LOO[e], h, lo))
        sc = scores(np.concatenate(ys), np.concatenate(ps))
        rows.append({"method": m, "high_cutoff": h, "low_cutoff": lo, "high_recall": round(sc["high_recall"], 4),
                     "escalation_rate": round(sc["escalation_rate"], 4), "mean_cost": round(sc["mean_cost"], 4),
                     "quadratic_kappa": round(sc["quadratic_kappa"], 4)})
pd.DataFrame(rows).to_csv(OUT_DIR / "r18_tradeoff_curves.csv", index=False)
save_json(rnd(out), "r18_threshold_study.json")
for m in ["DADE_AHP", "B5_ML_x_impact", "B1_ML_only", "B2_equal_weight"]:
    print(m, {p: (agg[m][p]["mean_cost"]["mean"], agg[m][p]["high_recall"]["mean"]) for p in agg[m]})
print("B4", agg["B4_severity_x_impact"]["FIXED"]["mean_cost"]["mean"])
print("external", {m: {p: ext[m][p]["mean_cost"] for p in ext[m]} for m in ext})
