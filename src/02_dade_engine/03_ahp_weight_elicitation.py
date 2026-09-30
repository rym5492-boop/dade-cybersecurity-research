"""
Expert-Based AHP Weighting (Section 4.3).

Reproduces the Analytic Hierarchy Process (AHP) weight computation
described in Section 4.3: six subject-matter experts each completed a
7x7 pairwise-comparison instrument (C(7,2) = 21 pairwise judgments per
expert, Saaty's 1-9 scale) over the seven DADE criteria (Threat
Severity, Asset Criticality, Business Impact, Historical Incidents,
Recovery Time Objective, Recovery Point Objective, Machine Learning
Score). Individual judgment matrices are aggregated via the
element-wise geometric mean across all 6 experts (Aggregation of
Individual Judgments, AIJ [28]), and the resulting group matrix's
priority weights and Consistency Ratio (CR) are computed.

Data privacy: expert responses were collected anonymously through the
project's pairwise-comparison instrument (a Tally form). Each expert
below is identified only by an anonymous "Expert N" label and an
opaque, non-identifying question ID from the form; no names,
affiliations, emails or other identifying information are stored
here or anywhere in this script, consistent with the paper's
Informed Consent Statement.

Running this script reproduces the paper's reported values exactly:
  CR = 0.0495 (consistent, <= 0.10)
  Threat Severity (TS)          = 26.2%
  Business Impact (BI)          = 21.1%
  Asset Criticality (AC)        = 18.9%
  Machine Learning Score (MLS)  = 13.1%
  Recovery Time Objective (RTO) =  8.3%
  Recovery Point Objective (RPO)=  7.7%
  Historical Incidents (HI)     =  4.8%
"""

import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["TS", "AC", "BI", "HI", "RTO", "RPO", "MLS"]
N = 7

# The 21 unique (i, j) criterion pairs compared under Saaty's C(7,2) scheme.
PAIR_SEQUENCE = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6),
    (4, 5), (4, 6),
    (5, 6),
]

# Opaque form-field identifiers (Tally question IDs) - not identifying.
QUESTION_IDS = [
    "9Eze8p", "e7dbV0", "WMv4Lj", "aD8WRv", "6OXqgP", "7V8QyP",
    "b80aog", "AqVkE0", "BQPvr5", "k7VL21", "vOQRlv",
    "KJqX9A", "LJQ8ej", "p7lgQq", "1J1jLQ",
    "MJ90Wp", "JJNyQ4", "g7RBEl",
    "yEWOBB", "XMKjpj",
    "8P09XO",
]

# Anonymized: each expert identified only as "Expert N". Raw pairwise
# answers (1-9 Likert-style responses from the instrument, where 5 is
# "equal importance") are exactly as submitted.
EXPERTS = [
    {"id": "Expert 1", "answers": {"9Eze8p": 2, "e7dbV0": 2, "WMv4Lj": 5, "aD8WRv": 4, "6OXqgP": 4, "7V8QyP": 4, "b80aog": 2, "AqVkE0": 5, "BQPvr5": 4, "k7VL21": 4, "vOQRlv": 4, "KJqX9A": 3, "LJQ8ej": 3, "p7lgQq": 4, "1J1jLQ": 4, "MJ90Wp": 4, "JJNyQ4": 7, "g7RBEl": 7, "yEWOBB": 5, "XMKjpj": 5, "8P09XO": 5}},
    {"id": "Expert 2", "answers": {"9Eze8p": 5, "e7dbV0": 4, "WMv4Lj": 2, "aD8WRv": 1, "6OXqgP": 2, "7V8QyP": 2, "b80aog": 7, "AqVkE0": 3, "BQPvr5": 3, "k7VL21": 2, "vOQRlv": 1, "KJqX9A": 2, "LJQ8ej": 2, "p7lgQq": 2, "1J1jLQ": 2, "MJ90Wp": 5, "JJNyQ4": 5, "g7RBEl": 7, "yEWOBB": 5, "XMKjpj": 5, "8P09XO": 5}},
    {"id": "Expert 3", "answers": {"9Eze8p": 5, "e7dbV0": 5, "WMv4Lj": 5, "aD8WRv": 5, "6OXqgP": 5, "7V8QyP": 5, "b80aog": 5, "AqVkE0": 5, "BQPvr5": 5, "k7VL21": 5, "vOQRlv": 5, "KJqX9A": 5, "LJQ8ej": 5, "p7lgQq": 5, "1J1jLQ": 5, "MJ90Wp": 5, "JJNyQ4": 5, "g7RBEl": 5, "yEWOBB": 5, "XMKjpj": 5, "8P09XO": 5}},
    {"id": "Expert 4", "answers": {"9Eze8p": 6, "e7dbV0": 6, "WMv4Lj": 3, "aD8WRv": 4, "6OXqgP": 3, "7V8QyP": 4, "b80aog": 6, "AqVkE0": 3, "BQPvr5": 4, "k7VL21": 3, "vOQRlv": 4, "KJqX9A": 2, "LJQ8ej": 3, "p7lgQq": 2, "1J1jLQ": 4, "MJ90Wp": 6, "JJNyQ4": 6, "g7RBEl": 7, "yEWOBB": 4, "XMKjpj": 6, "8P09XO": 7}},
    {"id": "Expert 5", "answers": {"9Eze8p": 5, "e7dbV0": 5, "WMv4Lj": 5, "aD8WRv": 5, "6OXqgP": 5, "7V8QyP": 3, "b80aog": 3, "AqVkE0": 2, "BQPvr5": 8, "k7VL21": 8, "vOQRlv": 3, "KJqX9A": 2, "LJQ8ej": 2, "p7lgQq": 2, "1J1jLQ": 8, "MJ90Wp": 8, "JJNyQ4": 8, "g7RBEl": 8, "yEWOBB": 2, "XMKjpj": 8, "8P09XO": 8}},
    {"id": "Expert 6", "answers": {"9Eze8p": 2, "e7dbV0": 2, "WMv4Lj": 1, "aD8WRv": 3, "6OXqgP": 2, "7V8QyP": 4, "b80aog": 5, "AqVkE0": 3, "BQPvr5": 3, "k7VL21": 4, "vOQRlv": 5, "KJqX9A": 4, "LJQ8ej": 5, "p7lgQq": 3, "1J1jLQ": 4, "MJ90Wp": 7, "JJNyQ4": 7, "g7RBEl": 8, "yEWOBB": 8, "XMKjpj": 7, "8P09XO": 8}},
]

RANDOM_INDEX_N7 = 1.32  # Saaty's Random Index for n=7


def to_saaty_intensity(v):
    """Map a 1-9 instrument response (5 = equal importance) to a Saaty
    pairwise-comparison intensity: values below 5 favor the first
    criterion of the pair (returned directly as 1..9), values above 5
    favor the second criterion (returned as the reciprocal)."""
    dist = abs(v - 5)
    intensity = 2 * dist + 1
    if v == 5:
        return 1.0
    elif v < 5:
        return intensity
    else:
        return 1.0 / intensity


def build_pairwise_matrix(answers):
    A = np.ones((N, N))
    for qid, (i, j) in zip(QUESTION_IDS, PAIR_SEQUENCE):
        val = to_saaty_intensity(answers[qid])
        A[i, j] = val
        A[j, i] = 1.0 / val
    return A


def ahp_weights_and_consistency(A):
    """Eigenvector-approximation AHP weights (normalized geometric mean
    of rows) plus the Saaty Consistency Ratio."""
    gm = np.prod(A, axis=1) ** (1.0 / N)
    w = gm / gm.sum()
    Aw = A.dot(w)
    lambda_max = (Aw / w).mean()
    CI = (lambda_max - N) / (N - 1)
    CR = CI / RANDOM_INDEX_N7
    return w, lambda_max, CI, CR


def main():
    individual = {}
    for expert in EXPERTS:
        A = build_pairwise_matrix(expert["answers"])
        w, lmax, ci, cr = ahp_weights_and_consistency(A)
        individual[expert["id"]] = {
            "weights": dict(zip(ORDER, w.round(4).tolist())),
            "lambda_max": round(float(lmax), 4),
            "CI": round(float(ci), 4),
            "CR": round(float(cr), 4),
            "consistent": bool(cr <= 0.10),
        }
        print(f"{expert['id']}: CR={cr:.4f} "
              f"({'consistent' if cr <= 0.10 else 'NOT consistent'})")

    n_consistent = sum(1 for v in individual.values() if v["consistent"])
    print(f"\n{n_consistent} of {len(EXPERTS)} individual matrices met CR <= 0.10 "
          f"(paper reports 2 of 6).")

    # Aggregation of Individual Judgments (AIJ): element-wise geometric
    # mean of all 6 experts' 7x7 matrices, per Section 4.3 / [28].
    stacked = np.stack([build_pairwise_matrix(e["answers"]) for e in EXPERTS], axis=0)
    agg_matrix = np.exp(np.mean(np.log(stacked), axis=0))
    w, lmax, ci, cr = ahp_weights_and_consistency(agg_matrix)

    print("\n" + "=" * 70)
    print("Aggregated Group Judgment Matrix (AIJ, geometric mean across 6 experts)")
    print("=" * 70)
    np.set_printoptions(precision=3, suppress=True)
    print("Criteria order:", ORDER)
    print(agg_matrix)
    print(f"\nlambda_max = {lmax:.4f}   CI = {ci:.4f}   CR = {cr:.4f} "
          f"({'CONSISTENT, <=0.10' if cr <= 0.10 else 'NOT within 0.10'})")

    print("\nFinal Group AHP Weights (AIJ method, all 6 experts):")
    weights = {}
    for name, wi in zip(ORDER, w):
        weights[name] = round(float(wi), 4)
        print(f"  {name}: {wi*100:.2f}%")

    output = {
        "individual_experts": individual,
        "group_aggregation_method": "AIJ (element-wise geometric mean of 6 pairwise-comparison matrices)",
        "group_weights": weights,
        "group_lambda_max": round(float(lmax), 4),
        "group_CI": round(float(ci), 4),
        "group_CR": round(float(cr), 4),
        "group_consistent": bool(cr <= 0.10),
    }
    with open(RESULTS_DIR / "ahp_weights_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {RESULTS_DIR / 'ahp_weights_results.json'}")


if __name__ == "__main__":
    main()
