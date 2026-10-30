"""
Shared decision-layer code for the second revision (r18-r22).

The functions reproduce the definitions of r10_decision_evaluation.py exactly (AHP aggregation, DADE,
baselines B1-B5, metrics, cost matrix, consensus rule, Krippendorff's alpha). r10 itself is left
unchanged because it is the pre-registered analysis; r14 checks that both give the same decisions.
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from common import OUT_DIR, REPO_ROOT, SEED

CRIT = ["TS", "AC", "BI", "HI", "RTO", "RPO", "MLS"]
BIA = ["AC", "BI", "HI", "RTO", "RPO"]
LEVELS = ["Low", "Medium", "High"]
RANK = {l: i for i, l in enumerate(LEVELS)}
HIGH_T, LOW_T = 0.66, 0.33
COST = np.array([[0, 1, 2],
                 [2, 0, 1],
                 [6, 3, 0]])
AR2EN = {"مرتفع": "High", "متوسط": "Medium", "منخفض": "Low", "H": "High", "M": "Medium", "L": "Low"}
METHOD_NAMES = {"DADE_AHP": "DADE (AHP weights)", "B1_ML_only": "B1 detection score alone",
                "B2_equal_weight": "B2 equal weights", "B3_BIA_only": "B3 BIA only",
                "B4_severity_x_impact": "B4 severity x impact", "B5_ML_x_impact": "B5 detection score x impact"}

with open(REPO_ROOT / "data" / "ahp_expert_judgments_anonymized.json", encoding="utf-8") as f:
    AHP = json.load(f)
assert AHP["criteria_order"] == CRIT
PAIRS = list(itertools.combinations(range(7), 2))
SAATY = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
ALL_EXPERTS = sorted(AHP["pairwise_responses"])


def to_matrix(resp):
    A = np.ones((7, 7))
    for (i, j), v in zip(PAIRS, resp):
        s = SAATY[abs(v - 5)]
        if v < 5:
            A[i, j], A[j, i] = s, 1 / s
        elif v > 5:
            A[i, j], A[j, i] = 1 / s, s
    return A


def group_weights(experts, responses=None):
    responses = responses or AHP["pairwise_responses"]
    G = np.exp(np.mean([np.log(to_matrix(responses[e])) for e in experts], axis=0))
    w = np.exp(np.log(G).mean(axis=1))
    return w / w.sum()


def aip_weights(experts, responses=None):
    """Aggregation of individual priorities: geometric mean of each expert's row-geometric-mean weights."""
    responses = responses or AHP["pairwise_responses"]
    ws = []
    for e in experts:
        A = to_matrix(responses[e])
        w = np.exp(np.log(A).mean(axis=1))
        ws.append(w / w.sum())
    g = np.exp(np.mean(np.log(ws), axis=0))
    return g / g.sum()


W_AHP = group_weights(ALL_EXPERTS)
W_EQ = np.full(7, 1 / 7)


def level(score, high=HIGH_T, low=LOW_T):
    return np.where(score >= high, "High", np.where(score < low, "Low", "Medium"))


def dade_score(X, w):
    return X[CRIT].values.astype(float) @ w


def dade(X, w, high=HIGH_T, low=LOW_T):
    return level(dade_score(X, w), high, low)


def impact_band(df):
    imp_raw = np.maximum(df["AC_1to5"], df["BI_1to5"])
    return np.where(imp_raw >= 4, 3, np.where(imp_raw >= 3, 2, 1))


def matrix_rule(row_band, col_band):
    prod = row_band * col_band
    return np.where(prod >= 6, "High", np.where(prod >= 3, "Medium", "Low"))


def severity_impact_matrix(df):
    sev = np.where(df["TS"] >= 0.70, 3, np.where(df["TS"] >= 0.40, 2, 1))
    return matrix_rule(sev, impact_band(df))


def ml_impact_rule(df, high=0.66, low=0.33):
    like = np.where(df["MLS"] >= high, 3, np.where(df["MLS"] >= low, 2, 1))
    return matrix_rule(like, impact_band(df))


def all_methods(df, w_dade, high=HIGH_T, low=LOW_T):
    return {"DADE_AHP": dade(df, w_dade, high, low),
            "B1_ML_only": level(df["MLS"].values, high, low),
            "B2_equal_weight": dade(df, W_EQ, high, low),
            "B3_BIA_only": level(df[BIA].mean(axis=1).values, high, low),
            "B4_severity_x_impact": severity_impact_matrix(df),
            "B5_ML_x_impact": ml_impact_rule(df)}


def load_scenarios():
    S = pd.read_csv(OUT_DIR / "r09_scenarios.csv", encoding="utf-8-sig").sort_values("scenario_id").reset_index(drop=True)
    S["TS"] = S["TS_pred"]
    assert len(S) == 30
    return S


def load_labels():
    L = pd.read_csv(REPO_ROOT / "data" / "scenario_labels.csv", encoding="utf-8-sig").dropna(subset=["label"])
    L["label"] = L["label"].astype(str).str.strip().replace(AR2EN)
    assert set(L["label"]) <= set(LEVELS) and len(L) == 120
    return L


def load_external():
    X = pd.read_csv(REPO_ROOT / "data" / "external_panel_labels.csv", encoding="utf-8-sig")
    X = X[X["consent"].astype(str).str.lower().str.startswith("y")]
    sc = [c for c in X.columns if len(c) == 3 and c[0] in "ABC" and c[1:].isdigit()]
    long = X.melt(id_vars=["rater_id"], value_vars=sc, var_name="scenario_id", value_name="label")
    long["label"] = long["label"].astype(str).str.strip().replace(AR2EN)
    assert set(long["label"]) <= set(LEVELS)
    return long.rename(columns={"rater_id": "expert_id"})


def scores(y, p):
    yi = np.array([RANK[v] for v in y])
    pi = np.array([RANK[v] for v in p])
    hi = yi == 2
    return {"n": int(len(y)),
            "exact_agreement": float(np.mean(yi == pi)),
            "macro_f1": float(f1_score(yi, pi, labels=[0, 1, 2], average="macro", zero_division=0)),
            "quadratic_kappa": float(cohen_kappa_score(yi, pi, labels=[0, 1, 2], weights="quadratic")),
            "high_recall": float(np.mean(pi[hi] == 2)) if hi.any() else None,
            "high_precision": float(np.mean(yi[pi == 2] == 2)) if (pi == 2).any() else None,
            "escalation_rate": float(np.mean(pi == 2)),
            "under_prioritisation": float(np.mean(pi < yi)),
            "over_escalation": float(np.mean(pi > yi)),
            "mean_cost": float(np.mean(COST[yi, pi]))}


def confusion(y, p):
    m = np.zeros((3, 3), dtype=int)
    for a, b in zip(y, p):
        m[RANK[a], RANK[b]] += 1
    return m.tolist()   # rows = reference label (L, M, H), columns = method output (L, M, H)


def consensus(labels):
    r = sorted(RANK[v] for v in labels)
    m = len(r)
    med = r[m // 2] if m % 2 else max(r[m // 2 - 1], r[m // 2])
    return LEVELS[med]


def kripp_alpha_ordinal(units):
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


def loeo_frame(L, S, weight_fn=group_weights, scen_transform=None, high=HIGH_T, low=LOW_T):
    """Leave-one-expert-out frame: each expert's labels next to every method, DADE weights from the other experts."""
    rows = []
    Si = S.set_index("scenario_id")
    for e, d in L.groupby("expert_id"):
        others = [x for x in ALL_EXPERTS if x != e]
        sub = Si.loc[d["scenario_id"]].reset_index()
        if scen_transform is not None:
            sub = scen_transform(sub)
        meth = all_methods(sub, weight_fn(others), high, low)
        rows.append(pd.DataFrame({"expert_id": e, "scenario_id": d["scenario_id"].values,
                                  "label": d["label"].values, **meth}))
    return pd.concat(rows, ignore_index=True)


def cluster_bootstrap_diff(frame, a_col, b_col, keys, n_boot=2000, seed=SEED, cluster="scenario_id"):
    """Paired bootstrap over scenarios (all labels of a resampled scenario move together)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(frame[cluster].values == s) for s in frame[cluster].unique()]
    y, a, b = frame["label"].values, frame[a_col].values, frame[b_col].values
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sa, sb = scores(y[idx], a[idx]), scores(y[idx], b[idx])
        for k in keys:
            if sa[k] is not None and sb[k] is not None:
                diffs[k].append(sa[k] - sb[k])
    return {k: {"diff": round(float(np.mean(v)), 4),
                "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
            for k, v in diffs.items()}


def cluster_bootstrap_metric(frame, col, keys, n_boot=2000, seed=SEED, cluster="scenario_id"):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(frame[cluster].values == s) for s in frame[cluster].unique()]
    y, a = frame["label"].values, frame[col].values
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        s = scores(y[idx], a[idx])
        for k in keys:
            if s[k] is not None:
                vals[k].append(s[k])
    full = scores(y, a)
    return {k: {"value": None if full[k] is None else round(full[k], 4),
                "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
            for k, v in vals.items()}


def rnd(d, n=4):
    if isinstance(d, dict):
        return {k: rnd(v, n) for k, v in d.items()}
    if isinstance(d, list):
        return [rnd(v, n) for v in d]
    if isinstance(d, (float, np.floating)):
        return round(float(d), n)
    if isinstance(d, np.integer):
        return int(d)
    return d
