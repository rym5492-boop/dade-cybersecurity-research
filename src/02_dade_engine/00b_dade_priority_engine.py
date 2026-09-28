"""
DADE scoring engine (paper Section 3.3-3.4, Equation (1)).

    PS = w1*TS + w2*AC + w3*BI + w4*HI + w5*RTO + w6*RPO + w7*MLS

All seven criteria are normalized to [0, 1] (1 = most urgent). Priority levels use the
thresholds of Section 5.6: High if PS >= 0.66, Low if PS < 0.33, otherwise Medium.
Each level maps to a continuity-oriented action (Section 3.4).

Weight profiles:
  - "equal": 1/7 for every criterion (Sections 5.2, 5.4-5.6)
  - "ahp"  : expert-derived AIJ group weights (Section 4.3), read from
             results/ahp_weights_results.json produced by 00_ahp_expert_weights.py

Tables 2 and 4 of the paper are illustrative applications of this engine to example
inputs; this module provides the engine so that any input vector can be scored
reproducibly. Example (inputs below are generic placeholders, not the paper's rows):
    python 00b_dade_priority_engine.py
"""
import json
import os

CRITERIA = ["TS", "AC", "BI", "HI", "RTO", "RPO", "MLS"]
HIGH_T, LOW_T = 0.66, 0.33
ACTIONS = {"High": "Incident response (activate continuity plan)",
           "Medium": "Escalate / monitor",
           "Low": "Routine logging / monitoring"}


def load_weights(profile="ahp", path="results/ahp_weights_results.json"):
    if profile == "equal":
        return {c: 1 / 7 for c in CRITERIA}
    with open(path) as f:
        w = json.load(f)["group"]["weights"]
    s = sum(w[c] for c in CRITERIA)            # renormalize rounded weights to sum to 1
    return {c: w[c] / s for c in CRITERIA}


def priority_score(x, weights):
    for c in CRITERIA:
        if not 0.0 <= x[c] <= 1.0:
            raise ValueError(f"{c} must be normalized to [0, 1], got {x[c]}")
    return sum(weights[c] * x[c] for c in CRITERIA)


def priority_level(ps, high=HIGH_T, low=LOW_T):
    return "High" if ps >= high else ("Low" if ps < low else "Medium")


def decide(x, weights):
    ps = priority_score(x, weights)
    lvl = priority_level(ps)
    return {"priority_score": round(ps, 3), "level": lvl, "action": ACTIONS[lvl]}


if __name__ == "__main__":
    example = {"TS": 0.70, "AC": 0.50, "BI": 0.50, "HI": 0.00, "RTO": 0.50, "RPO": 0.50, "MLS": 0.90}
    for prof in ("equal", "ahp"):
        if prof == "ahp" and not os.path.exists("results/ahp_weights_results.json"):
            print("Run 00_ahp_expert_weights.py first to create the AHP weights."); continue
        print(prof, decide(example, load_weights(prof)))
