"""Fewest changes that would have made E5's September AHP answers consistent (CR <= 0.10).

Added on 4 November 2026 to check the statement in thesis Section 3.4 about E5. Uses the September answers in
data/ahp_expert_judgments_anonymized.json and the scale, matrix and geometric-mean CR of r06.

Lower bound: for every set of k pairs, the smallest CR reachable when those k entries may take ANY value in
[1/9, 9] is found by convex minimization. The geometric-mean lambda is a sum of exponentials of affine functions of
the log entries, so the problem is convex and its minimum is global. If that minimum exceeds 0.10 for every set of
k pairs, no k changes on the 1-9 scale can reach 0.10 either.
Upper bound: one combination of changes on the form's 1-9 scale, found by a search on 4 November 2026, is checked.
Writes results/revision/ahp_min_changes_E5.json. Runs in about two minutes on one core.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "data" / "ahp_expert_judgments_anonymized.json").read_text("utf-8"))
n = 7
PAIRS = list(itertools.combinations(range(n), 2))
SAATY = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}   # as in r06
RI7 = 1.32
I = np.array([p[0] for p in PAIRS])
J = np.array([p[1] for p in PAIRS])
LMAX = np.log(9.0)


def log_value(v):
    s = SAATY[abs(v - 5)]
    return np.log(s) if v < 5 else (-np.log(s) if v > 5 else 0.0)


def cr_gm_from_logs(y):
    """Geometric-mean CR of r06, from the 21 upper-triangle log entries."""
    x = np.zeros((n, n)); x[I, J] = y; x[J, I] = -y
    m = x.mean(1)
    lam = np.exp(x + m[None, :] - m[:, None]).sum() / n
    return ((lam - n) / (n - 1)) / RI7


def f_and_grad(z, sub, y0):
    y = y0.copy(); y[sub] = z
    x = np.zeros((n, n)); x[I, J] = y; x[J, I] = -y
    m = x.mean(1)
    T = np.exp(x + m[None, :] - m[:, None]) / n
    g = T + (T.sum(0)[:, None] - T.sum(1)[:, None]) / n
    return T.sum(), (g[I, J] - g[J, I])[sub]


def min_cr_continuous(sub, y0):
    sub = np.array(sub)
    r = minimize(f_and_grad, y0[sub], args=(sub, y0), jac=True, method="L-BFGS-B",
                 bounds=[(-LMAX, LMAX)] * len(sub), options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 500})
    return ((r.fun - n) / (n - 1)) / RI7


sept = D["pairwise_responses_original_sept2026"]["E5"]
y0 = np.array([log_value(v) for v in sept])
out = {"expert": "E5", "estimator": "geometric-mean CR, as in r06 and the thesis", "threshold": 0.10,
       "CR_september": round(cr_gm_from_logs(y0), 4)}

# Lower bound: every set of 7 of the 21 pairs (116,280 sets); fewer changes are covered, since a set of 7 may keep
# some entries at their September values.
k = 7
best = (np.inf, None)
for sub in itertools.combinations(range(21), k):
    c = min_cr_continuous(sub, y0)
    if c < best[0]:
        best = (c, sub)
out["lower_bound"] = {"k": k, "sets_checked": sum(1 for _ in itertools.combinations(range(21), k)),
                      "smallest_CR_reachable_with_k_changes_any_value_in_1/9_to_9": round(float(best[0]), 4),
                      "pairs_of_that_set (question numbers)": [q + 1 for q in best[1]],
                      "k_changes_are_enough": bool(best[0] <= 0.10)}

# Upper bound: eight changes on the 1-9 scale (question number: September answer -> changed answer)
eight = {3: 2, 7: 6, 10: 5, 13: 5, 15: 1, 18: 5, 20: 1, 21: 4}
r = list(sept)
changes = {}
for q, v in eight.items():
    changes[str(q)] = [r[q - 1], v]
    r[q - 1] = v
cr8 = cr_gm_from_logs(np.array([log_value(v) for v in r]))
out["eight_changes"] = {"question: [september, changed]": changes, "CR_after": round(cr8, 4),
                        "consistent": bool(cr8 <= 0.10)}
out["fewest_changes_needed"] = 8 if (not out["lower_bound"]["k_changes_are_enough"] and cr8 <= 0.10) else None
dst = ROOT / "results" / "revision" / "ahp_min_changes_E5.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
