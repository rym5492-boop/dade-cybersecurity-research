"""
AHP expert weighting for DADE (paper Section 4.3; supervisor review Section 3.4 / Experiment E4).

Input : data/ahp_expert_judgments_anonymized.json (six experts, 21 pairwise comparisons each)
Method: 1) convert each expert's slider responses to a 7x7 reciprocal Saaty matrix;
        2) individual weights by the row geometric-mean method and Saaty's Consistency Ratio
           (CR = CI / RI, RI(7) = 1.32; acceptance threshold CR <= 0.10, fixed before analysis);
        3) Aggregation of Individual Judgments (AIJ): element-wise geometric mean of the six
           matrices, then weights and CR of the group matrix;
        4) robustness: leave-one-expert-out re-aggregation; inter-expert agreement on the
           three validation scenarios (Fleiss' kappa).
Output: results/ahp_weights_results.json
Expected: group CR = 0.0495; TS 0.262, BI 0.211, AC 0.189, MLS 0.131, RTO 0.083, RPO 0.077, HI 0.048
"""
import itertools, json
import numpy as np

with open("data/ahp_expert_judgments_anonymized.json", encoding="utf-8") as f:
    D = json.load(f)
C = D["criteria_order"]; n = len(C)
SAATY = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def to_matrix(responses):
    A = np.ones((n, n))
    for (i, j), v in zip(itertools.combinations(range(n), 2), responses):
        s = SAATY[abs(v - 5)]
        if v < 5:
            A[i, j], A[j, i] = s, 1 / s
        elif v > 5:
            A[i, j], A[j, i] = 1 / s, s
    return A


def weights_and_cr(A):
    w = np.exp(np.log(A).mean(axis=1)); w /= w.sum()
    lam = np.mean((A @ w) / w)
    return w, ((lam - n) / (n - 1)) / RI[n]


def aggregate(mats):
    return np.exp(np.mean([np.log(M) for M in mats], axis=0))


mats = {e: to_matrix(r) for e, r in D["pairwise_responses"].items()}
out = {"individual": {}, "leave_one_out": {}}
for e, M in mats.items():
    w, cr = weights_and_cr(M)
    out["individual"][e] = {"CR": round(cr, 4), "consistent": bool(cr <= 0.10),
                            "weights": dict(zip(C, np.round(w, 4).tolist()))}
    print(f"{e}: CR = {cr:.3f} {'(consistent)' if cr <= 0.10 else ''}")

wg, crg = weights_and_cr(aggregate(mats.values()))
out["group"] = {"CR": round(crg, 4), "weights": dict(zip(C, np.round(wg, 3).tolist())),
                "ranking": [C[i] for i in np.argsort(-wg)]}
print(f"\nGroup (AIJ) CR = {crg:.4f}")
for i in np.argsort(-wg):
    print(f"  {C[i]:>4}: {wg[i]:.3f}")

for e in mats:
    w, cr = weights_and_cr(aggregate([M for k, M in mats.items() if k != e]))
    out["leave_one_out"][e] = {"CR": round(cr, 4), "ranking": [C[i] for i in np.argsort(-w)],
                               "max_abs_weight_change": round(float(np.abs(w - wg).max()), 4)}

cats = ["High", "Medium", "Low"]; votes = list(D["scenario_votes"].values()); r = len(votes[0])
M = np.array([[v.count(c) for c in cats] for v in votes])
P = ((M ** 2).sum(1) - r) / (r * (r - 1)); pj = M.sum(0) / M.sum(); Pe = (pj ** 2).sum()
out["scenario_agreement"] = {"per_scenario_P": np.round(P, 3).tolist(),
                             "fleiss_kappa": round(float((P.mean() - Pe) / (1 - Pe)), 3)}
print("\nLeave-one-out rankings:", {k: v["ranking"][:4] for k, v in out["leave_one_out"].items()})
print("Fleiss' kappa on scenarios:", out["scenario_agreement"]["fleiss_kappa"])

with open("results/ahp_weights_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved results/ahp_weights_results.json")
