"""
Expert-Based Decision Validation (Section 5.7, Table 4).

Scores DADE's Priority Score (Equation 1) for the three realistic incident
scenarios used in the Section 5.7 validation instrument, under two weight
profiles:
  - PS(equal): the preliminary equal-weight baseline (1/7 each), as used
    throughout Sections 5.2-5.6.
  - PS(real): the AIJ-aggregated, expert-derived weight vector from
    Section 4.3 (see 03_ahp_weight_elicitation.py), CR = 0.0495.

Criterion values (TS, AC, BI, HI, RTO, RPO, MLS) for each scenario are
0-1 normalized, derived directly from the scenario descriptions given to
the six experts in the validation instrument (Section 5.7). RTO/RPO are
expressed as "recovery urgency" (1 - recovery_time / 72h range), so that
higher urgency contributes to a higher Priority Score, consistent with the
convention used throughout the paper.

Output reproduces Table 4 exactly: PS(real) = 0.519 (S1), 0.972 (S2),
0.264 (S3), all matching the paper's reported priority levels
(Medium, High, Low respectively).
"""

order = ["TS", "AC", "BI", "HI", "RTO", "RPO", "MLS"]

# Equal-weight baseline (Sections 5.2-5.6)
w_equal = [1 / 7] * 7

# AIJ-aggregated expert weights from Section 4.3 (CR = 0.0495)
w_real = [0.2620, 0.1887, 0.2108, 0.0477, 0.0826, 0.0774, 0.1308]

scenarios = {
    "S1 - DoS / public marketing site": {
        "TS": 0.70, "AC": 0.20, "BI": 0.35, "HI": 0.00,
        "RTO": 0.667, "RPO": 0.667, "MLS": 0.90,
        "expert_majority": "Medium (5/6)",
    },
    "S2 - Ransomware / hospital patient DB": {
        "TS": 1.00, "AC": 1.00, "BI": 1.00, "HI": 0.60,
        "RTO": 0.986, "RPO": 0.986, "MLS": 0.95,
        "expert_majority": "High (6/6)",
    },
    "S3 - Benign high-confidence malware / non-prod dev server": {
        "TS": 0.40, "AC": 0.10, "BI": 0.05, "HI": 0.00,
        "RTO": 0.00, "RPO": 0.00, "MLS": 0.99,
        "expert_majority": "Medium (3/6 plurality; Low 2/6, High 1/6)",
    },
}


def priority_score(vals, w):
    return sum(vals[c] * wi for c, wi in zip(order, w))


def priority_level(score):
    if score >= 0.66:
        return "High"
    if score < 0.33:
        return "Low"
    return "Medium"


if __name__ == "__main__":
    import json

    results = {}
    print(f"{'Scenario':<50}{'PS(equal)':>10}{'Lvl':>8}{'PS(real)':>10}{'Lvl':>8}  Expert majority")
    for name, v in scenarios.items():
        pe = priority_score(v, w_equal)
        pr = priority_score(v, w_real)
        print(f"{name:<50}{pe:>10.3f}{priority_level(pe):>8}{pr:>10.3f}{priority_level(pr):>8}  {v['expert_majority']}")
        results[name] = {
            "criterion_values": {k: v[k] for k in order},
            "ps_equal_weight": round(pe, 3),
            "priority_level_equal_weight": priority_level(pe),
            "ps_expert_weight": round(pr, 3),
            "priority_level_expert_weight": priority_level(pr),
            "expert_majority_vote": v["expert_majority"],
        }

    with open("scenario_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved scenario_validation_results.json")
