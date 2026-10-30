"""
R17 - Base-rate (prevalence) correction with explicit thresholds and confusion counts
(supervisor review of 30 October 2026, P0-2 and "minimum additional evidence" item 1).

r02 reported, for isotonic HGB on UNSW-NB15 resampled to 1% attacks, precision at the 0.5 threshold of
3.4% before and 86.6% after Bayes prior correction. Prior correction is a monotone transformation of the
score, so it cannot change the ranking (ROC-AUC, PR-AUC). The precision gain comes only from re-applying
the fixed 0.5 threshold to the corrected probabilities, which is the same as a much higher threshold on the
uncorrected score. This script makes that explicit and separates what used test information from what did not.

Procedure (identical to r02 for the detector and the resampling):
  * HGB fitted on FIT, isotonic calibration fitted on CAL (disjoint 70/15/15 split of the training file).
  * Test file resampled to 1%, 5%, 10% and 50% attacks: all 37,000 benign flows plus attacks drawn without
    replacement, ten draws per prevalence. The random draws reproduce r02 exactly (same generator, same
    order: the Random Forest draws of r02 are consumed first and discarded).
Decision policies compared on each draw (precision, recall, FPR and TP/FP/FN/TN counts):
  P1 uncorrected score, threshold 0.5;
  P2 prior correction with the TRUE test prevalence (oracle, as in r02), threshold 0.5, with the equivalent
     threshold on the uncorrected score;
  P3 prior correction with a prevalence ESTIMATED from the unlabelled test scores by EM
     (Saerens, Latinne and Decaestecker, 2002): uses no test labels;
  P4 prior correction with a prevalence five times too high and five times too low (misspecification);
  P5 threshold chosen on the validation split for FPR <= 1% (no test information).
Outputs: results/revision/r17_base_rate_thresholds.json, r17_test_isotonic_scores.npz
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from common import (OUT_DIR, SEED, feature_cols, load_unsw, prior_shift, save_json, split_fit_val_cal,
                    threshold_at_fpr, tree_preprocessor)

train, test = load_unsw()
cols = feature_cols(train)
fit, val, cal = split_fit_val_cal(train)
y_test = test["label"].values
pi_cal = float(cal["label"].mean())

base = Pipeline([("prep", tree_preprocessor(cols)),
                 ("clf", HistGradientBoostingClassifier(random_state=SEED))]).fit(fit[cols], fit["label"])
iso = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(cal[cols], cal["label"])
p_test = iso.predict_proba(test[cols])[:, 1]
p_val = iso.predict_proba(val[cols])[:, 1]
np.savez_compressed(OUT_DIR / "r17_test_isotonic_scores.npz", p=p_test, y=y_test,
                    attack_cat=test["attack_cat"].values.astype(str))
thr_val_fpr1 = threshold_at_fpr(val["label"].values, p_val, 0.01)


def em_prior(p, pi_train, iters=1000, tol=1e-8):
    """EM estimate of the deployment prevalence from unlabelled calibrated scores (Saerens et al., 2002)."""
    pi = pi_train
    for _ in range(iters):
        q = prior_shift(p, pi_train, pi)
        new = float(np.mean(q))
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def equivalent_raw_threshold(pi_train, pi_new):
    """Uncorrected score t at which the corrected probability equals 0.5."""
    odds = ((1 - pi_new) / (1 - pi_train)) * (pi_train / pi_new)
    return odds / (1 + odds)


def counts(y, yhat):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn),
            "fpr": fp / (fp + tn), "alerts_per_1000_flows": 1000 * (tp + fp) / len(y)}


rng = np.random.default_rng(SEED)
benign_idx = np.where(y_test == 0)[0]
attack_idx = np.where(y_test == 1)[0]
# r02 draws ten samples per prevalence for the Random Forest first, then for HGB, from one generator.
for target in [0.01, 0.05, 0.10, 0.50]:
    for _ in range(10):
        rng.choice(attack_idx, size=int(round(target / (1 - target) * len(benign_idx))), replace=False)

POLICIES = ["P1_uncorrected_0.5", "P2_oracle_prior_0.5", "P3_em_prior_0.5", "P4a_prior_x5_0.5",
            "P4b_prior_div5_0.5", "P5_validation_fpr1pct"]
out = {"detector": "HistGradientBoosting (FIT) + isotonic calibration (CAL)",
       "calibration_split_prevalence": round(pi_cal, 4), "n_benign_in_every_draw": int(len(benign_idx)),
       "validation_threshold_fpr1pct": round(thr_val_fpr1, 4),
       "ranking_unchanged_by_correction": {}, "prevalence": {}}
for target in [0.01, 0.05, 0.10, 0.50]:
    n_att = int(round(target / (1 - target) * len(benign_idx)))
    per = {k: [] for k in POLICIES}
    em_est, auc_raw, auc_cor = [], [], []
    for _ in range(10):
        sel = np.concatenate([benign_idx, rng.choice(attack_idx, size=n_att, replace=False)])
        y, p = y_test[sel], p_test[sel]
        pc = prior_shift(p, pi_cal, target)
        pi_em = em_prior(p, pi_cal)
        em_est.append(pi_em)
        auc_raw.append((roc_auc_score(y, p), average_precision_score(y, p)))
        auc_cor.append((roc_auc_score(y, pc), average_precision_score(y, pc)))
        per["P1_uncorrected_0.5"].append(counts(y, p >= 0.5))
        per["P2_oracle_prior_0.5"].append(counts(y, pc >= 0.5))
        per["P3_em_prior_0.5"].append(counts(y, prior_shift(p, pi_cal, pi_em) >= 0.5))
        per["P4a_prior_x5_0.5"].append(counts(y, prior_shift(p, pi_cal, min(5 * target, 0.99)) >= 0.5))
        per["P4b_prior_div5_0.5"].append(counts(y, prior_shift(p, pi_cal, target / 5) >= 0.5))
        per["P5_validation_fpr1pct"].append(counts(y, p >= thr_val_fpr1))
        # P2 is identical to thresholding the uncorrected score at the equivalent threshold.
        t_eq = equivalent_raw_threshold(pi_cal, target)
        assert np.array_equal(pc >= 0.5, p >= t_eq - 1e-12) or np.sum((pc >= 0.5) != (p >= t_eq)) <= 2
    summ = {}
    for k, rows in per.items():
        summ[k] = {m: {"mean": round(float(np.mean([r[m] for r in rows])), 4),
                       "min": round(float(np.min([r[m] for r in rows])), 4),
                       "max": round(float(np.max([r[m] for r in rows])), 4)}
                   for m in ["precision", "recall", "fpr", "alerts_per_1000_flows", "tp", "fp", "fn", "tn"]}
    out["prevalence"][f"{target:.2f}"] = {
        "n_attacks_per_draw": n_att, "n_flows_per_draw": n_att + len(benign_idx), "draws": 10,
        "equivalent_uncorrected_threshold_of_P2": round(equivalent_raw_threshold(pi_cal, target), 5),
        "em_estimated_prevalence": {"mean": round(float(np.mean(em_est)), 4), "min": round(float(np.min(em_est)), 4),
                                    "max": round(float(np.max(em_est)), 4)},
        "policies": summ,
        "first_draw_counts": {k: {m: rows[0][m] for m in ["tp", "fp", "fn", "tn"]} for k, rows in per.items()},
    }
    out["ranking_unchanged_by_correction"][f"{target:.2f}"] = {
        "roc_auc_raw_mean": round(float(np.mean([a for a, _ in auc_raw])), 4),
        "roc_auc_corrected_mean": round(float(np.mean([a for a, _ in auc_cor])), 4),
        "pr_auc_raw_mean": round(float(np.mean([b for _, b in auc_raw])), 4),
        "pr_auc_corrected_mean": round(float(np.mean([b for _, b in auc_cor])), 4)}
    print(f"{target:.2f}", {k: (summ[k]["precision"]["mean"], summ[k]["recall"]["mean"]) for k in POLICIES},
          "EM prior", round(float(np.mean(em_est)), 4))
save_json(out, "r17_base_rate_thresholds.json")
