"""
R19 - Decision-layer ablations and paired, scenario-clustered comparisons (supervisor review of 30 October
2026: "minimum additional evidence" items 3 and 4; Experiments: scenario-level confusion matrices, High recall
and cost by criticality, ablation of MLS, TS and each BIA criterion, removal of AHP / replacement by equal
weights, expert-weight aggregation).

All analyses use the 120 expert labels with leave-one-expert-out weights (the primary protocol of r10) and
the pre-fixed cut-offs 0.66 / 0.33, so no result here depends on tuning to the labels.
  A. Paired comparisons of DADE with B1-B5: mean difference and 95% cluster-bootstrap interval (2,000
     resamples of scenarios) for quadratic kappa, mean cost, High recall, escalation rate, under-prioritisation
     and over-escalation; 3x3 confusion matrices against the labels and against the consensus label;
     High recall, cost and escalation by business criticality of the profile, with intervals.
  B. Ablations of DADE: each of the seven criteria removed (its weight set to zero, the others rescaled),
     the five BIA criteria removed together, the two detector inputs removed together, and the AHP weights
     replaced by equal weights; each compared with full DADE (paired, clustered).
  C. Aggregation of the expert weights: AIJ (used), AIP, September answers, without E3, and DADE with the
     weights of each single expert; decisions changed on the 30 scenarios and agreement with the labels.
Outputs: results/revision/r19_ablation_and_paired.json
"""
import numpy as np
import pandas as pd

from common import SEED, save_json
from decision_common import (AHP, ALL_EXPERTS, CRIT, LEVELS, W_AHP, W_EQ, aip_weights, all_methods, cluster_bootstrap_diff,
                             cluster_bootstrap_metric, confusion, consensus, dade, group_weights, load_labels,
                             load_scenarios, loeo_frame, rnd, scores)

S = load_scenarios()
L = load_labels()
KEYS = ["quadratic_kappa", "mean_cost", "high_recall", "escalation_rate", "under_prioritisation", "over_escalation"]
P = loeo_frame(L, S)
methods = [c for c in P.columns if c not in ("expert_id", "scenario_id", "label")]
out = {"n_labels": int(len(P)), "n_scenarios": int(P["scenario_id"].nunique()), "n_boot": 2000}

# ---------------------------------------------------------------- A. paired comparisons
out["A_metrics_with_ci"] = {m: cluster_bootstrap_metric(P, m, KEYS) for m in methods}
out["A_paired_DADE_minus_baseline"] = {b: cluster_bootstrap_diff(P, "DADE_AHP", b, KEYS) for b in methods if b != "DADE_AHP"}
out["A_confusion_vs_labels_rows_label_cols_method_LMH"] = {m: confusion(P["label"], P[m]) for m in methods}
C = L.groupby("scenario_id")["label"].apply(consensus).rename("label").reset_index()
Cs = S.set_index("scenario_id").loc[C["scenario_id"]].reset_index()
for m, v in all_methods(Cs, W_AHP).items():
    C[m] = v
out["A_confusion_vs_consensus_rows_label_cols_method_LMH"] = {m: confusion(C["label"], C[m]) for m in methods}
crit = S.set_index("scenario_id")["profile_level"]
P["criticality"] = P["scenario_id"].map(crit)
out["A_by_criticality"] = {}
for g, d in P.groupby("criticality"):
    d = d.reset_index(drop=True)
    out["A_by_criticality"][g] = {"n_labels": int(len(d)), "n_scenarios": int(d["scenario_id"].nunique()),
                                  "label_distribution": d["label"].value_counts().to_dict(),
                                  "methods": {m: cluster_bootstrap_metric(d, m, ["high_recall", "mean_cost", "escalation_rate",
                                                                                  "under_prioritisation"]) for m in methods}}

# ---------------------------------------------------------------- B. ablations
def renorm(w, drop):
    w = w.copy()
    for c in drop:
        w[CRIT.index(c)] = 0.0
    return w / w.sum()


variants = {f"without_{c}": [c] for c in CRIT}
variants["without_all_BIA"] = ["AC", "BI", "HI", "RTO", "RPO"]
variants["without_detector_inputs_MLS_TS"] = ["MLS", "TS"]
rows = []
Si = S.set_index("scenario_id")
for e, d in L.groupby("expert_id"):
    w_full = group_weights([x for x in ALL_EXPERTS if x != e])
    sub = Si.loc[d["scenario_id"]].reset_index()
    r = {"expert_id": e, "scenario_id": d["scenario_id"].values, "label": d["label"].values,
         "full": dade(sub, w_full), "equal_weights": dade(sub, W_EQ)}
    for k, drop in variants.items():
        r[k] = dade(sub, renorm(w_full, drop))
    rows.append(pd.DataFrame(r))
A = pd.concat(rows, ignore_index=True)
abl = {}
for k in ["equal_weights"] + list(variants):
    abl[k] = {"metrics": {m: round(v, 4) for m, v in scores(A["label"], A[k]).items() if m in KEYS},
              "decisions_changed_vs_full": round(float(np.mean(A[k] != A["full"])), 4),
              "paired_full_minus_variant": cluster_bootstrap_diff(A, "full", k, ["quadratic_kappa", "mean_cost", "high_recall"])}
abl["full"] = {"metrics": {m: round(v, 4) for m, v in scores(A["label"], A["full"]).items() if m in KEYS}}
out["B_ablations"] = abl

# ---------------------------------------------------------------- C. aggregation of the expert weights
base = dade(S, W_AHP)
agg = {"AIJ_all_six_used": W_AHP, "AIP_all_six": aip_weights(ALL_EXPERTS),
       "AIJ_september_answers": group_weights(ALL_EXPERTS, AHP["pairwise_responses_original_sept2026"]),
       "AIJ_without_E3": group_weights([e for e in ALL_EXPERTS if e != "E3"]),
       "equal_weights": W_EQ}
for e in ALL_EXPERTS:
    agg[f"single_expert_{e}"] = group_weights([e])
res = {}
for k, w in agg.items():
    lv = dade(S, w)
    Pk = L.copy()
    Pk["m"] = Pk["scenario_id"].map(dict(zip(S["scenario_id"], lv)))
    res[k] = {"weights": dict(zip(CRIT, np.round(w, 4))), "scenarios_changed_vs_used": int(np.sum(lv != base)),
              "level_counts": {l: int(np.sum(lv == l)) for l in LEVELS},
              "vs_120_labels": {m: round(v, 4) for m, v in scores(Pk["label"], Pk["m"]).items() if m in KEYS}}
out["C_weight_aggregation"] = res
save_json(rnd(out), "r19_ablation_and_paired.json")
for k in ["full", "equal_weights"] + list(variants):
    print(f"{k:32s}", abl[k]["metrics"]["quadratic_kappa"], abl[k]["metrics"]["mean_cost"], abl[k]["metrics"]["high_recall"],
          abl[k].get("decisions_changed_vs_full"))
print({k: v["scenarios_changed_vs_used"] for k, v in res.items()})
