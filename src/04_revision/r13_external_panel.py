"""
R13 - Independent external panel (secondary analysis, PREREG_ADDENDUM_external_panel.md).

Input : data/external_panel_labels.csv (4 raters x 30 scenarios, H/M/L; no names).
Output: results/revision/r13_external_panel.json and data/external_reference_labels.csv

Pre-declared analyses (addendum items 1-2):
  1. Krippendorff's alpha (ordinal) across the external raters, with a scenario bootstrap CI.
  2. External reference label per scenario by majority vote. Ties: a 2-2 split is resolved to the higher
     priority. A 2-1-1 split has no strict majority. The addendum does not name this case, so the
     plurality label (the one chosen by 2 of 4) is used and the case is counted and reported.
Item 3 (DADE vs the external reference) needs the final AHP weights and is run by r10 once they are fixed.

Sensitivity (NOT pre-declared, reported as such): r10's median rule (ties to the more cautious level),
alpha leaving one rater out, per-rater label distributions, pairwise agreement.
"""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import REPO_ROOT, SEED, save_json

LEVELS = ["Low", "Medium", "High"]
CODE = {"L": 0, "M": 1, "H": 2}
N_BOOT = 2000
rng = np.random.default_rng(SEED)

df = pd.read_csv(REPO_ROOT / "data" / "external_panel_labels.csv")
SCEN = [f"{g}{i:02d}" for g in "ABC" for i in range(1, 11)]
assert set(SCEN) <= set(df.columns) and len(df) == df["rater_id"].nunique()
assert (df["consent"] == "yes").all(), "non-consented response present"
R = df.set_index("rater_id")[SCEN].replace(CODE).astype(int)   # raters x scenarios
assert R.notna().all().all(), "incomplete response present"


def kripp_alpha_ordinal(units, k=3):
    o = np.zeros((k, k))
    for r in units:
        m = len(r)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            o[r[a], r[b]] += 1 / (m - 1)
    nc = o.sum(axis=1); N = nc.sum()
    d = np.zeros((k, k))
    for c in range(k):
        for kk in range(k):
            lo, hi = min(c, kk), max(c, kk)
            d[c, kk] = (nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2
    Do = (o * d).sum() / N
    De = (np.outer(nc, nc) * d).sum() / (N * (N - 1))
    return 1 - Do / De if De > 0 else float("nan")


def alpha_with_ci(mat):
    units = [list(mat[s]) for s in mat.columns]
    a = kripp_alpha_ordinal(units)
    boot = [kripp_alpha_ordinal([units[i] for i in rng.integers(0, len(units), len(units))]) for _ in range(N_BOOT)]
    return {"alpha": round(a, 4), "ci95": [round(float(np.nanpercentile(boot, 2.5)), 4),
                                           round(float(np.nanpercentile(boot, 97.5)), 4)]}


def prereg_label(votes):
    c = Counter(votes).most_common()
    top = [lvl for lvl, n in c if n == c[0][1]]
    if len(top) == 1:
        pattern = "strict_majority" if c[0][1] > len(votes) / 2 else "plurality_2-1-1"
        return top[0], pattern
    if len(top) == len(set(votes)) == 3:          # all differ (only possible with 3 raters)
        return 1, "all_differ->Medium"
    return max(top), "tie->higher"


def median_label(votes):                          # r10 rule
    r = sorted(votes); m = len(r)
    return r[m // 2] if m % 2 else max(r[m // 2 - 1], r[m // 2])


out = {"n_raters": int(len(R)), "n_scenarios": len(SCEN),
       "rater_profile": df[["rater_id", "field", "experience", "sector"]].to_dict("records")}
out["alpha_all_raters"] = alpha_with_ci(R)

ref, pattern, med = {}, {}, {}
for s in SCEN:
    v = list(R[s]); lab, pat = prereg_label(v)
    ref[s], pattern[s], med[s] = lab, pat, median_label(v)
out["reference_label_distribution"] = {LEVELS[k]: v for k, v in sorted(Counter(ref.values()).items())}
out["vote_patterns"] = dict(Counter(pattern.values()))
out["plurality_2-1-1_scenarios"] = [s for s in SCEN if pattern[s] == "plurality_2-1-1"]
out["tie_2-2_scenarios"] = [s for s in SCEN if pattern[s] == "tie->higher"]
out["unanimous_scenarios"] = int(sum(len(set(R[s])) == 1 for s in SCEN))

# ---- sensitivity (not pre-declared)
sens = {}
sens["median_rule_differs_from_prereg_rule"] = [s for s in SCEN if med[s] != ref[s]]
sens["alpha_leave_one_rater_out"] = {x: round(kripp_alpha_ordinal([list(R.drop(index=x)[s]) for s in SCEN]), 4)
                                     for x in R.index}
sens["rater_label_distribution"] = {x: {LEVELS[k]: int((R.loc[x] == k).sum()) for k in range(3)} for x in R.index}
sens["pairwise"] = {f"{a}-{b}": {"exact": round(float((R.loc[a] == R.loc[b]).mean()), 3),
                                 "quadratic_kappa": round(float(cohen_kappa_score(R.loc[a], R.loc[b], labels=[0, 1, 2],
                                                                                  weights="quadratic")), 3)}
                    for a, b in itertools.combinations(R.index, 2)}
out["sensitivity_not_preregistered"] = sens

pd.DataFrame({"scenario_id": SCEN, "external_reference": [LEVELS[ref[s]] for s in SCEN],
              "vote_pattern": [pattern[s] for s in SCEN],
              "votes_LMH": ["/".join(str(int((R[s] == k).sum())) for k in range(3)) for s in SCEN]}
             ).to_csv(REPO_ROOT / "data" / "external_reference_labels.csv", index=False)
save_json(out, "r13_external_panel.json")
print(out["alpha_all_raters"], out["vote_patterns"], out["reference_label_distribution"])
print("2-1-1:", out["plurality_2-1-1_scenarios"]); print("2-2:", out["tie_2-2_scenarios"])
print("unanimous:", out["unanimous_scenarios"]); print(sens)
