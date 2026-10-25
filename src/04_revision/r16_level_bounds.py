"""r16 - Lowest and highest attainable DADE level for each hypothetical BIA profile.

Analytic check, no new experiment: with the business context of a profile fixed, the DPS
(Equation 1 of the paper, Equation 4.1 of the thesis) is lowest when MLS = 0 and TS = 0, and
highest when MLS = 1 and TS = 0.95 (the largest value in the severity table). The script reports
both bounds and the levels they give under the fixed cut-offs 0.66 / 0.33, with the AHP group
weights of all six experts and with equal weights.
"""
import itertools
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "revision" / "r16_level_bounds.json"
HIGH_T, LOW_T = 0.66, 0.33
TS_MAX = 0.95
# Profiles P1-P6 as in r09 / Table 12 of the paper: (AC = BI on 1-5, RTO hours, RPO hours, incidents in 12 months)
PROFILES = {"P1": (5, 1, 0.25, 1), "P2": (5, 2, 0.25, 2), "P3": (4, 4, 4, 1),
            "P4": (3, 24, 24, 0), "P5": (2, 72, 24, 0), "P6": (1, 168, 168, 3)}

with open(REPO_ROOT / "data" / "ahp_expert_judgments_anonymized.json", encoding="utf-8") as f:
    AHP = json.load(f)
CRIT = AHP["criteria_order"]
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


def urgency(h):
    return 1.0 if h <= 1 else 0.75 if h <= 4 else 0.5 if h <= 24 else 0.25 if h <= 72 else 0.0


def level(s):
    return "High" if s >= HIGH_T else ("Low" if s < LOW_T else "Medium")


def bounds(w):
    W = dict(zip(CRIT, w))
    out = {}
    for p, (c, rto, rpo, hi) in PROFILES.items():
        ctx = (W["AC"] * (c - 1) / 4 + W["BI"] * (c - 1) / 4 + W["HI"] * min(hi, 5) / 5
               + W["RTO"] * urgency(rto) + W["RPO"] * urgency(rpo))
        top = ctx + W["MLS"] * 1.0 + W["TS"] * TS_MAX
        out[p] = {"min_dps": round(ctx, 4), "min_level": level(ctx),
                  "max_dps": round(top, 4), "max_level": level(top)}
    return out


res = {"weights_ahp_all_six": dict(zip(CRIT, [round(float(x), 4) for x in group_weights(sorted(AHP["pairwise_responses"]))])),
       "ahp_all_six": bounds(group_weights(sorted(AHP["pairwise_responses"]))),
       "equal_weights": bounds(np.full(7, 1 / 7)),
       "cutoffs": {"high": HIGH_T, "low": LOW_T}, "ts_max": TS_MAX}
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
