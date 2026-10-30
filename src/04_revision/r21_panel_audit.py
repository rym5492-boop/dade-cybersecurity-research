"""
R21 - Audit of the label reference: rater variability, scenario-level uncertainty, consensus rules, panel
differences and the effective sample size (supervisor review of 30 October 2026, P0-4 and "minimum
additional evidence" item 5).

The 120 expert labels are repeated ratings of 30 scenarios (four experts per scenario), not 120 independent
cases, and the 120 external labels rate the same 30 scenarios again. This script reports:
  1. each rater: label distribution, agreement (quadratic kappa) with the consensus of the OTHER raters of the
     same panel on the same scenarios, and with DADE (leave-one-expert-out for the experts) and B5;
  2. pairwise rater agreement within each panel;
  3. per scenario: votes, consensus margin, the probability that the consensus label changes when the votes are
     resampled (bootstrap of raters within the scenario), and every scenario with a split vote;
  4. sensitivity of DADE and B5 to the consensus rule (tie up, tie down, rounded mean, plurality);
  5. expert versus practitioner consensus: cross-tabulation, quadratic kappa, and a sign test on which panel
     rated each scenario higher;
  6. intra-scenario correlation of DADE's exact agreement and cost, the design effect and the effective number
     of independent labels.
Outputs: results/revision/r21_panel_audit.json
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

from common import SEED, save_json
from decision_common import (ALL_EXPERTS, COST, LEVELS, RANK, W_AHP, all_methods, consensus, group_weights,
                             load_external, load_labels, load_scenarios, loeo_frame, rnd, scores)

S = load_scenarios()
Si = S.set_index("scenario_id")
L = load_labels()
X = load_external()
M_ALL = all_methods(S, W_AHP)
DADE_OF = dict(zip(S["scenario_id"], M_ALL["DADE_AHP"]))
B5_OF = dict(zip(S["scenario_id"], M_ALL["B5_ML_x_impact"]))
rng = np.random.default_rng(SEED)


def qk(a, b):
    return float(cohen_kappa_score([RANK[x] for x in a], [RANK[x] for x in b], labels=[0, 1, 2], weights="quadratic"))


def rater_table(D, loeo):
    res = {}
    for r, d in D.groupby("expert_id"):
        others = D[(D["expert_id"] != r) & D["scenario_id"].isin(d["scenario_id"])]
        oc = others.groupby("scenario_id")["label"].apply(consensus)
        lab = d.set_index("scenario_id")["label"]
        if loeo:
            w = group_weights([x for x in ALL_EXPERTS if x != r])
            dd = all_methods(Si.loc[lab.index].reset_index(), w)["DADE_AHP"]
        else:
            dd = [DADE_OF[s] for s in lab.index]
        res[r] = {"n": int(len(d)), "label_distribution": {l: int((d["label"] == l).sum()) for l in LEVELS},
                  "kappa_with_consensus_of_others": round(qk(lab.values, oc.loc[lab.index].values), 4),
                  "kappa_with_DADE": round(qk(lab.values, dd), 4),
                  "kappa_with_B5": round(qk(lab.values, [B5_OF[s] for s in lab.index]), 4),
                  "mean_cost_DADE": round(scores(lab.values, dd)["mean_cost"], 4)}
    return res


def pairwise(D):
    out = {}
    for a, b in itertools.combinations(sorted(D["expert_id"].unique()), 2):
        da, db = D[D["expert_id"] == a].set_index("scenario_id")["label"], D[D["expert_id"] == b].set_index("scenario_id")["label"]
        common = da.index.intersection(db.index)
        if len(common) >= 5:
            out[f"{a}-{b}"] = {"n_shared": int(len(common)), "quadratic_kappa": round(qk(da.loc[common], db.loc[common]), 4),
                               "exact_agreement": round(float(np.mean(da.loc[common] == db.loc[common])), 4)}
    return out


def scenario_table(D):
    res = {}
    for s, d in D.groupby("scenario_id"):
        v = list(d["label"])
        cons = consensus(v)
        boot = [consensus(list(rng.choice(v, len(v), replace=True))) for _ in range(2000)]
        counts = {l: v.count(l) for l in LEVELS}
        res[s] = {"votes_LMH": [counts["Low"], counts["Medium"], counts["High"]], "consensus": cons,
                  "consensus_share": round(counts[cons] / len(v), 3),
                  "p_consensus_changes_under_rater_bootstrap": round(float(np.mean([b != cons for b in boot])), 3),
                  "split": len([c for c in counts.values() if c > 0]) > 1,
                  "DADE": DADE_OF[s], "B5": B5_OF[s], "profile": Si.loc[s, "profile"],
                  "true_category": Si.loc[s, "true_category"], "shown_category": Si.loc[s, "shown_category"],
                  "MLS": round(float(Si.loc[s, "MLS"]), 3)}
    return res


def rules(v):
    r = sorted(RANK[x] for x in v)
    m = len(r)
    up = r[m // 2] if m % 2 else max(r[m // 2 - 1], r[m // 2])
    down = r[m // 2] if m % 2 else min(r[m // 2 - 1], r[m // 2])
    mean = int(np.floor(np.mean(r) + 0.5))
    cnt = np.bincount(r, minlength=3)
    plural = int(np.flatnonzero(cnt == cnt.max()).max())       # ties to the higher level
    return {"median_tie_up_used": LEVELS[up], "median_tie_down": LEVELS[down], "rounded_mean": LEVELS[mean],
            "plurality_tie_up": LEVELS[plural]}


def consensus_rule_sensitivity(D):
    out = {}
    per = {s: rules(list(d["label"])) for s, d in D.groupby("scenario_id")}
    for rule in next(iter(per.values())):
        y = [per[s][rule] for s in sorted(per)]
        out[rule] = {"label_distribution": {l: y.count(l) for l in LEVELS},
                     "DADE": {k: round(v, 4) for k, v in scores(y, [DADE_OF[s] for s in sorted(per)]).items()
                              if k in ("quadratic_kappa", "mean_cost", "high_recall")},
                     "B5": {k: round(v, 4) for k, v in scores(y, [B5_OF[s] for s in sorted(per)]).items()
                            if k in ("quadratic_kappa", "mean_cost", "high_recall")}}
    return out


def icc1(values, groups):
    df = pd.DataFrame({"v": values, "g": groups})
    k = df.groupby("g").size().mean()
    grand = df["v"].mean()
    msb = (df.groupby("g")["v"].apply(lambda x: len(x) * (x.mean() - grand) ** 2).sum()) / (df["g"].nunique() - 1)
    msw = (df.groupby("g")["v"].apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()) / (len(df) - df["g"].nunique())
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else float("nan"), float(k)


out = {"experts": {"raters": rater_table(L, True), "pairwise": pairwise(L), "scenarios": scenario_table(L),
                   "consensus_rule_sensitivity": consensus_rule_sensitivity(L)},
       "practitioners": {"raters": rater_table(X, False), "pairwise": pairwise(X), "scenarios": scenario_table(X),
                         "consensus_rule_sensitivity": consensus_rule_sensitivity(X)}}
for k in ("experts", "practitioners"):
    sc = out[k]["scenarios"]
    out[k]["summary"] = {"scenarios_with_split_vote": int(sum(v["split"] for v in sc.values())),
                         "scenarios_unanimous": int(sum(not v["split"] for v in sc.values())),
                         "scenarios_where_consensus_changes_in_over_20pct_of_rater_bootstraps":
                             int(sum(v["p_consensus_changes_under_rater_bootstrap"] > 0.2 for v in sc.values()))}

ce = L.groupby("scenario_id")["label"].apply(consensus)
cx = X.groupby("scenario_id")["label"].apply(consensus).loc[ce.index]
higher_e = int(sum(RANK[a] > RANK[b] for a, b in zip(ce, cx)))
higher_x = int(sum(RANK[a] < RANK[b] for a, b in zip(ce, cx)))
out["panel_difference"] = {
    "crosstab_rows_expert_cols_practitioner_LMH": [[int(sum((ce == a) & (cx == b))) for b in LEVELS] for a in LEVELS],
    "quadratic_kappa_between_consensus_labels": round(qk(ce.values, cx.values), 4),
    "exact_agreement": round(float(np.mean(ce.values == cx.values)), 4),
    "scenarios_experts_higher": higher_e, "scenarios_practitioners_higher": higher_x,
    "sign_test_p_two_sided": round(float(binomtest(higher_e, higher_e + higher_x).pvalue), 4) if higher_e + higher_x else None}

P = loeo_frame(L, S)
P["exact"] = (P["label"] == P["DADE_AHP"]).astype(float)
P["cost"] = [COST[RANK[a], RANK[b]] for a, b in zip(P["label"], P["DADE_AHP"])]
eff = {}
for v in ("exact", "cost"):
    rho, k = icc1(P[v].values, P["scenario_id"].values)
    deff = 1 + (k - 1) * max(rho, 0)
    eff[v] = {"icc1_within_scenario": round(rho, 4), "labels_per_scenario": k, "design_effect": round(deff, 3),
              "effective_n_of_120": round(120 / deff, 1)}
out["effective_sample_size_DADE"] = eff
save_json(rnd(out), "r21_panel_audit.json")
print(out["experts"]["summary"], out["practitioners"]["summary"])
print(out["panel_difference"])
print(eff)
