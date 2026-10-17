"""
R10 - Decision-quality evaluation of DADE against expert priority labels
(supervisor review: Critical 4 and 5, 3.2 "no evidence that DADE decisions are better").

This file is written and committed BEFORE any expert label exists, so the analysis plan is fixed in
advance (pre-registration in the repository history). It runs in two stages:

Stage 1 - no labels needed (runs now)
  * Priority levels for the 30 scenarios of r09 from DADE and five baselines:
      DADE-AHP : Equation (1), AHP group weights, predicted Threat Severity, thresholds 0.66 / 0.33
      B1 ML-only        : detection probability alone, same thresholds
      B2 Equal-weight   : Equation (1) with 1/7 weights (transparent unweighted score)
      B3 BIA-only       : mean of the five business criteria (AC, BI, HI, RTO, RPO)
      B4 Severity x impact : 3x3 matrix, severity band of the predicted category x impact band
      B5 ML x impact    : 3x3 matrix, calibrated detection probability band x impact band
      (matrices: product >= 6 High, 3-4 Medium, <= 2 Low, ISO/IEC 27005-style qualitative)
  * Checks of DADE itself: monotonicity (raising any criterion never lowers the level),
    stability of the 30 decisions under +/-20% weight perturbation and +/-0.05 threshold shifts.
  * A labels template: data/scenario_labels_TEMPLATE.csv (6 experts x 2 forms x 10 scenarios).

Stage 2 - runs automatically when data/scenario_labels.csv exists
  * Inter-rater reliability: Krippendorff's alpha (ordinal) with bootstrap CI; per-form alpha.
  * Primary analysis (leakage-free, leave-one-expert-out): for each expert e, DADE weights are
    recomputed from the AHP judgments of the other five experts and compared with e's own labels.
  * Secondary analysis: consensus label per scenario (median; a tie between two levels is resolved
    upwards, i.e. the more cautious level), compared with every method.
  * Metrics: exact agreement, macro-F1, quadratic-weighted kappa, High-priority recall,
    under-prioritisation rate, over-escalation rate, mean asymmetric cost (matrix below).
  * Paired bootstrap over scenarios: DADE minus each baseline, 95% CI.
  * Threshold tuning check: thresholds chosen on one form, evaluated on the other two (3 folds).
The primary claim of the paper is decided by the leave-one-expert-out comparison only.

Outputs: results/revision/r10_decisions_stage1.csv, r10_stage1.json, and (stage 2) r10_stage2.json
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from common import OUT_DIR, REPO_ROOT, SEED, save_json

CRIT = ["TS", "AC", "BI", "HI", "RTO", "RPO", "MLS"]
LEVELS = ["Low", "Medium", "High"]
HIGH_T, LOW_T = 0.66, 0.33
AR2EN = {"مرتفع": "High", "متوسط": "Medium", "منخفض": "Low", "H": "High", "M": "Medium", "L": "Low"}
# Asymmetric cost, rows = expert level, columns = method level (Low, Medium, High).
# Under-prioritising a High incident is the most expensive error; over-escalation costs analyst time.
COST = np.array([[0, 1, 2],
                 [2, 0, 1],
                 [6, 3, 0]])
FORM_ASSIGNMENT = {"E1": "AB", "E2": "BC", "E3": "CA", "E4": "AB", "E5": "BC", "E6": "CA"}
N_BOOT = 2000

# ---------------------------------------------------------------- AHP (same method as the paper / r06)
with open(REPO_ROOT / "data" / "ahp_expert_judgments_anonymized.json", encoding="utf-8") as f:
    AHP = json.load(f)
assert AHP["criteria_order"] == CRIT
PAIRS = list(itertools.combinations(range(7), 2))
SAATY = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}


def to_matrix(resp):
    A = np.ones((7, 7))
    for (i, j), v in zip(PAIRS, resp):
        s = SAATY[abs(v - 5)]
        if v < 5:
            A[i, j], A[j, i] = s, 1 / s
        elif v > 5:
            A[i, j], A[j, i] = 1 / s, s
    return A


def group_weights(experts):
    G = np.exp(np.mean([np.log(to_matrix(AHP["pairwise_responses"][e])) for e in experts], axis=0))
    w = np.exp(np.log(G).mean(axis=1))
    return w / w.sum()


ALL_EXPERTS = sorted(AHP["pairwise_responses"])
W_AHP = group_weights(ALL_EXPERTS)
W_EQ = np.full(7, 1 / 7)


# ---------------------------------------------------------------- decision rules
def level(score, high=HIGH_T, low=LOW_T):
    return np.where(score >= high, "High", np.where(score < low, "Low", "Medium"))


def dade(X, w, high=HIGH_T, low=LOW_T):
    return level(X[CRIT].values @ w, high, low)


def impact_band(df):
    imp_raw = np.maximum(df["AC_1to5"], df["BI_1to5"])
    return np.where(imp_raw >= 4, 3, np.where(imp_raw >= 3, 2, 1))


def matrix_rule(row_band, col_band):
    """3x3 qualitative matrix (ISO/IEC 27005 style): product >= 6 High, 3-4 Medium, <= 2 Low."""
    prod = row_band * col_band
    return np.where(prod >= 6, "High", np.where(prod >= 3, "Medium", "Low"))


def severity_impact_matrix(df):
    # Review Critical 5 (i): documented severity x impact. Severity band from the PREDICTED category:
    # >= 0.70 (DoS, Exploits, Backdoor, Shellcode, Worms) high; >= 0.40 (Reconnaissance, Analysis,
    # Generic) medium; otherwise (Normal, Fuzzers) low.
    sev = np.where(df["TS"] >= 0.70, 3, np.where(df["TS"] >= 0.40, 2, 1))
    return matrix_rule(sev, impact_band(df))


def ml_impact_rule(df):
    # Review Critical 5 (iv): calibrated ML score + simple impact rule (likelihood x impact).
    like = np.where(df["MLS"] >= 0.66, 3, np.where(df["MLS"] >= 0.33, 2, 1))
    return matrix_rule(like, impact_band(df))


def all_methods(df, w_dade):
    return {"DADE_AHP": dade(df, w_dade),
            "B1_ML_only": level(df["MLS"].values),
            "B2_equal_weight": dade(df, W_EQ),                                          # Critical 5 (iii)
            "B3_BIA_only": level(df[["AC", "BI", "HI", "RTO", "RPO"]].mean(axis=1).values),  # Critical 5 (ii)
            "B4_severity_x_impact": severity_impact_matrix(df),                         # Critical 5 (i)
            "B5_ML_x_impact": ml_impact_rule(df)}                                       # Critical 5 (iv)


# ---------------------------------------------------------------- stage 1
S = pd.read_csv(OUT_DIR / "r09_scenarios.csv", encoding="utf-8-sig").sort_values("scenario_id").reset_index(drop=True)
S["TS"] = S["TS_pred"]   # severity of the PREDICTED category (no ground truth in the decision engine)
assert len(S) == 30 and S[CRIT].min().min() >= 0 and S[CRIT].max().max() <= 1
M = all_methods(S, W_AHP)
dec = S[["scenario_id", "form", "true_category", "shown_category", "profile"] + CRIT].copy()
dec["DADE_score"] = (S[CRIT].values @ W_AHP).round(3)
for k, v in M.items():
    dec[k] = v
dec.to_csv(OUT_DIR / "r10_decisions_stage1.csv", index=False, encoding="utf-8-sig")

rng = np.random.default_rng(SEED)
# Monotonicity: raise each criterion of each scenario by 0.1 (capped at 1) and check the level never drops.
rank = {l: i for i, l in enumerate(LEVELS)}
viol = 0
for c in CRIT:
    S2 = S.copy()
    S2[c] = np.minimum(S2[c] + 0.1, 1.0)
    viol += int(sum(rank[b] < rank[a] for a, b in zip(dade(S, W_AHP), dade(S2, W_AHP))))
# Weight perturbation: multiply each weight by U(0.8, 1.2), renormalise.
base = dade(S, W_AHP)
changed = []
for _ in range(N_BOOT):
    w = W_AHP * rng.uniform(0.8, 1.2, 7)
    changed.append(np.mean(dade(S, w / w.sum()) != base))
thr = {f"high{h:+.2f}_low{l:+.2f}": int(np.sum(dade(S, W_AHP, HIGH_T + h, LOW_T + l) != base))
       for h in (-0.05, 0, 0.05) for l in (-0.05, 0, 0.05) if (h, l) != (0, 0)}

# Severity-table dependence: (a) TS of the TRUE category instead of the predicted one (oracle, for
# diagnosis only); (b) each category's severity perturbed by U(-0.10, +0.10), clipped to [0, 1].
SEV = {"Normal": 0.0, "Fuzzers": .35, "Reconnaissance": .40, "Analysis": .55, "DoS": .70, "Backdoor": .85,
       "Exploits": .80, "Shellcode": .90, "Worms": .95, "Generic": .60}
assert all(abs(SEV[c] - t) < 1e-9 for c, t in zip(S["shown_category"].fillna("Normal"), S["TS"])
           if c in SEV), "severity table differs from r09"
S_or = S.copy()
S_or["TS"] = S["true_category"].map(SEV)
oracle_changed = int(np.sum(dade(S_or, W_AHP) != base))
sev_changed = []
for _ in range(N_BOOT):
    pert = {c: float(np.clip(v + rng.uniform(-0.1, 0.1), 0, 1)) if c != "Normal" else 0.0 for c, v in SEV.items()}
    S_p = S.copy()
    S_p["TS"] = S["shown_category"].map(pert).fillna(0.0)
    sev_changed.append(np.mean(dade(S_p, W_AHP) != base))

# Counterfactual BIA: score every alert under each of the six business profiles.
PROF = S.drop_duplicates("profile").set_index("profile")[["AC", "BI", "HI", "RTO", "RPO", "profile_level"]].sort_index()
cf = []
for _, r in S.iterrows():
    levels = []
    for p, q in PROF.iterrows():
        x = r.copy()
        x[["AC", "BI", "HI", "RTO", "RPO"]] = q[["AC", "BI", "HI", "RTO", "RPO"]].values
        levels.append(dade(x.to_frame().T.astype({c: float for c in CRIT}), W_AHP)[0])
    cf.append(dict(zip(PROF.index, levels)))
CF = pd.DataFrame(cf, index=S["scenario_id"])
hi_prof = [p for p in PROF.index if PROF.loc[p, "profile_level"] == "high"]
lo_prof = [p for p in PROF.index if PROF.loc[p, "profile_level"] == "low"]
cf_summary = {
    "alerts_whose_level_differs_between_profiles": int((CF.nunique(axis=1) > 1).sum()),
    "alerts_spanning_all_three_levels": int((CF.nunique(axis=1) == 3).sum()),
    "alerts_lower_under_high_criticality_than_low_criticality": int(sum(
        min(rank[CF.loc[s, p]] for p in hi_prof) < max(rank[CF.loc[s, p]] for p in lo_prof) for s in CF.index)),
}

stage1 = {
    "ahp_group_weights": dict(zip(CRIT, W_AHP.round(4))),
    "ts_true_category_instead_of_predicted_scenarios_changed": oracle_changed,
    "severity_table_perturbation_10pts": {"mean_share_changed": round(float(np.mean(sev_changed)), 4),
                                          "p95_share_changed": round(float(np.percentile(sev_changed, 95)), 4)},
    "counterfactual_bia_profiles": cf_summary,
    "level_counts": {k: {l: int(np.sum(v == l)) for l in LEVELS} for k, v in M.items()},
    "pairwise_agreement_between_methods": {f"{a} vs {b}": round(float(np.mean(M[a] == M[b])), 3)
                                           for a, b in itertools.combinations(M, 2)},
    "dade_monotonicity_violations": viol,
    "dade_weight_perturbation_20pct": {"mean_share_changed": round(float(np.mean(changed)), 4),
                                       "p95_share_changed": round(float(np.percentile(changed, 95)), 4)},
    "dade_threshold_shift_scenarios_changed": thr,
    "cost_matrix_rows_expert_cols_method_LMH": COST.tolist(),
    "form_assignment": FORM_ASSIGNMENT,
}
save_json(stage1, "r10_stage1.json")

tmpl = [{"expert_id": e, "form": f, "scenario_id": s, "label": ""}
        for e, forms in FORM_ASSIGNMENT.items() for f in forms
        for s in S.loc[S["form"] == f, "scenario_id"]]
pd.DataFrame(tmpl).to_csv(REPO_ROOT / "data" / "scenario_labels_TEMPLATE.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(stage1["level_counts"]).T)


# ---------------------------------------------------------------- stage 2 helpers
def kripp_alpha_ordinal(units):
    """units: list of lists of integer ratings (0..k-1) per scenario; ordinal metric."""
    k = len(LEVELS)
    o = np.zeros((k, k))
    for r in units:
        m = len(r)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            o[r[a], r[b]] += 1 / (m - 1)
    nc = o.sum(axis=1)
    N = nc.sum()
    d = np.zeros((k, k))
    for c in range(k):
        for kk in range(k):
            lo, hi = min(c, kk), max(c, kk)
            d[c, kk] = (nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2
    Do = (o * d).sum() / N
    De = (np.outer(nc, nc) * d).sum() / (N * (N - 1))
    return 1 - Do / De if De > 0 else float("nan")


def scores(y, p):
    yi = np.array([rank[v] for v in y])
    pi = np.array([rank[v] for v in p])
    hi = yi == 2
    return {"n": int(len(y)),
            "exact_agreement": float(np.mean(yi == pi)),
            "macro_f1": float(f1_score(yi, pi, labels=[0, 1, 2], average="macro", zero_division=0)),
            "quadratic_kappa": float(cohen_kappa_score(yi, pi, labels=[0, 1, 2], weights="quadratic")),
            "high_recall": float(np.mean(pi[hi] == 2)) if hi.any() else None,
            "under_prioritisation": float(np.mean(pi < yi)),
            "over_escalation": float(np.mean(pi > yi)),
            "mean_cost": float(np.mean(COST[yi, pi]))}


BOOT_KEYS = ("quadratic_kappa", "mean_cost", "high_recall", "under_prioritisation")


def paired_bootstrap(frame, method, base, cluster="scenario_id"):
    """Cluster bootstrap over scenarios (all labels of a resampled scenario move together)."""
    groups = [np.flatnonzero(frame[cluster].values == s) for s in frame[cluster].unique()]
    y, a, b = frame["label"].values, frame[method].values, frame[base].values
    diffs = {k: [] for k in BOOT_KEYS}
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sa, sb = scores(y[idx], a[idx]), scores(y[idx], b[idx])
        for k in BOOT_KEYS:
            if sa[k] is not None and sb[k] is not None:
                diffs[k].append(sa[k] - sb[k])
    return {k: {"diff": round(float(np.mean(v)), 4),
                "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
            for k, v in diffs.items()}


def consensus(labels):
    r = sorted(rank[v] for v in labels)
    m = len(r)
    med = r[m // 2] if m % 2 else max(r[m // 2 - 1], r[m // 2])   # tie -> more cautious level
    return LEVELS[med]


def evaluate(L):
    out = {"n_labels": int(len(L)), "experts": sorted(L["expert_id"].unique()),
           "label_distribution": L["label"].value_counts().to_dict()}
    units = [[rank[v] for v in d["label"]] for _, d in L.groupby("scenario_id")]
    a = kripp_alpha_ordinal(units)
    boot = [kripp_alpha_ordinal([units[i] for i in rng.integers(0, len(units), len(units))]) for _ in range(N_BOOT)]
    out["krippendorff_alpha_ordinal"] = {"alpha": round(a, 4), "ci95": [round(float(np.nanpercentile(boot, 2.5)), 4),
                                                                         round(float(np.nanpercentile(boot, 97.5)), 4)]}
    out["alpha_per_form"] = {f: round(kripp_alpha_ordinal([[rank[v] for v in d["label"]]
                                                           for _, d in L[L["form"] == f].groupby("scenario_id")]), 4)
                             for f in sorted(L["form"].unique())}

    # Primary: leave-one-expert-out DADE weights vs that expert's own labels.
    rows = []
    for e, d in L.groupby("expert_id"):
        others = [x for x in ALL_EXPERTS if x != e]
        sub = S.set_index("scenario_id").loc[d["scenario_id"]].reset_index()
        meth = all_methods(sub, group_weights(others))
        rows.append(pd.DataFrame({"expert_id": e, "scenario_id": d["scenario_id"].values,
                                  "label": d["label"].values, **meth}))
    P = pd.concat(rows, ignore_index=True)
    methods = [c for c in P.columns if c not in ("expert_id", "scenario_id", "label")]
    out["primary_leave_one_expert_out"] = {m: {k: (round(v, 4) if isinstance(v, float) else v)
                                               for k, v in scores(P["label"], P[m]).items()} for m in methods}
    out["primary_paired_bootstrap_DADE_minus_baseline"] = {
        b: paired_bootstrap(P, "DADE_AHP", b) for b in methods if b != "DADE_AHP"}

    # Breakdown by business criticality of the profile and by true traffic type (review 3.2:
    # "performance by sector and attack type"). Exact agreement and mean cost per group.
    info = S.set_index("scenario_id")[["profile_level", "profile", "true_category"]]
    P = P.join(info, on="scenario_id")

    def by(col):
        return {str(g): {m: {"n": int(len(d)), "exact_agreement": round(float(np.mean(d["label"] == d[m])), 4),
                             "mean_cost": round(scores(d["label"], d[m])["mean_cost"], 4)} for m in methods}
                for g, d in P.groupby(col)}
    out["primary_by_profile_criticality"] = by("profile_level")
    out["primary_by_profile"] = by("profile")
    out["primary_by_true_traffic_type"] = by("true_category")

    # Secondary: consensus label per scenario.
    C = L.groupby("scenario_id")["label"].apply(consensus).rename("label").reset_index()
    sub = S.set_index("scenario_id").loc[C["scenario_id"]].reset_index()
    for m, v in all_methods(sub, W_AHP).items():
        C[m] = v
    out["secondary_consensus"] = {m: {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in scores(C["label"], C[m]).items()} for m in methods}
    # Failure cases: every scenario where DADE differs from the expert consensus, with the full vote.
    votes = {s: {l: int((d == l).sum()) for l in LEVELS} for s, d in L.groupby("scenario_id")["label"]}
    fail = C[C["label"] != C["DADE_AHP"]].join(S.set_index("scenario_id")[
        ["true_category", "shown_category", "profile", "MLS"]], on="scenario_id")
    out["dade_failure_cases"] = [{"scenario_id": r.scenario_id, "consensus": r.label, "dade": r.DADE_AHP,
                                  "votes": votes[r.scenario_id], "true_category": r.true_category,
                                  "shown_category": r.shown_category, "profile": r.profile,
                                  "MLS": round(float(r.MLS), 3),
                                  "dade_score": round(float(S.set_index("scenario_id").loc[r.scenario_id, CRIT].values
                                                            @ W_AHP), 3)}
                                 for r in fail.itertuples()]

    # Threshold tuning: choose (high, low) on one form, test on the other two.
    grid = [(h, l) for h in np.arange(0.40, 0.91, 0.02) for l in np.arange(0.10, 0.61, 0.02) if l < h]
    sc_ = S.set_index("scenario_id")
    folds = {}
    for f in sorted(C["scenario_id"].str[0].unique()):
        tr, te = C[C["scenario_id"].str[0] == f], C[C["scenario_id"].str[0] != f]
        best = min(grid, key=lambda t: np.mean(COST[[rank[v] for v in tr["label"]],
                                                    [rank[v] for v in dade(sc_.loc[tr["scenario_id"]], W_AHP, *t)]]))
        folds[f] = {"thresholds": [round(best[0], 2), round(best[1], 2)],
                    "test_mean_cost_tuned": round(scores(te["label"], dade(sc_.loc[te["scenario_id"]], W_AHP, *best))["mean_cost"], 4),
                    "test_mean_cost_fixed": round(scores(te["label"], te["DADE_AHP"])["mean_cost"], 4)}
    out["threshold_tuning_by_form"] = folds
    return out


lab_path = REPO_ROOT / "data" / "scenario_labels.csv"
if lab_path.exists():
    L = pd.read_csv(lab_path, encoding="utf-8-sig").dropna(subset=["label"])
    L["label"] = L["label"].astype(str).str.strip().replace(AR2EN)
    assert set(L["label"]) <= set(LEVELS), set(L["label"])
    assert set(L["scenario_id"]) <= set(S["scenario_id"])
    save_json(evaluate(L), "r10_stage2.json")
else:
    print("Stage 2 skipped: data/scenario_labels.csv not found (expert labels not collected yet).")
