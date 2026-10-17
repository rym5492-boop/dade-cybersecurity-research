"""
R02 - Probability calibration with disjoint splits and prevalence shift (supervisor review 3.1).

The original script (src/01_predictive_layer/04_probability_calibration.py) fitted Platt scaling
with 3-fold CV on the training file and evaluated on the official test file, whose distribution
differs; Brier/ECE got worse. Here:
  * the base model is fitted on FIT, the calibrator (sigmoid and isotonic) on the disjoint CAL
    split, and evaluation is on the official TEST file only;
  * ECE (15 equal-width bins) is reported with a 95% bootstrap interval, plus Brier and log loss;
  * a reliability table (bin counts included) is saved for the figure;
  * prevalence shift: the test file (55% attacks) is re-sampled to 1%, 5%, 10% and 50% attacks and
    calibration is compared with and without Bayes prior correction. Precision at the 0.5
    threshold is reported, because it drives the false-escalation burden in deployment.
Output: results/revision/r02_calibration.json
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline

from common import (SEED, bootstrap_ci, ece, feature_cols, load_unsw, prior_shift, save_json,
                    split_fit_val_cal, tree_preprocessor)

train, test = load_unsw()
cols = feature_cols(train)
fit, val, cal = split_fit_val_cal(train)
y_test = test["label"].values
pi_cal = float(cal["label"].mean())


def reliability(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": [round(edges[b], 3), round(edges[b + 1], 3)], "count": int(m.sum()),
                         "mean_predicted": round(float(p[m].mean()), 4),
                         "observed_attack_rate": round(float(y[m].mean()), 4)})
    return rows


def calib_metrics(y, p, ci=True):
    d = {"brier": round(brier_score_loss(y, p), 4), "ece_15bins": round(ece(y, p), 4),
         "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4)}
    if ci:
        d["ece_95ci"] = bootstrap_ci(lambda a, b: ece(a, b), y, p, n_boot=300)
    return d


out = {"calibration_split_prevalence": round(pi_cal, 4), "test_prevalence": round(float(y_test.mean()), 4),
       "models": {}}
rng = np.random.default_rng(SEED)
benign_idx = np.where(y_test == 0)[0]
attack_idx = np.where(y_test == 1)[0]

for name, clf in [("RandomForest", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
                  ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=SEED))]:
    base = Pipeline([("prep", tree_preprocessor(cols)), ("clf", clf)]).fit(fit[cols], fit["label"])
    probs = {"uncalibrated": base.predict_proba(test[cols])[:, 1]}
    for method in ["sigmoid", "isotonic"]:
        c = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(cal[cols], cal["label"])
        probs[method] = c.predict_proba(test[cols])[:, 1]
    res = {k: calib_metrics(y_test, p) for k, p in probs.items()}
    res["reliability_table"] = {k: reliability(y_test, p) for k, p in probs.items()}

    shift = {}
    for target in [0.01, 0.05, 0.10, 0.50]:
        reps = []
        for _ in range(10):
            n_att = int(round(target / (1 - target) * len(benign_idx)))
            sel = np.concatenate([benign_idx, rng.choice(attack_idx, size=n_att, replace=False)])
            y_s = y_test[sel]
            row = {}
            for k in ["uncalibrated", "isotonic"]:
                p_s = probs[k][sel]
                p_c = prior_shift(p_s, pi_cal, target)
                row[k] = {"ece_raw": ece(y_s, p_s), "ece_prior_corrected": ece(y_s, p_c),
                          "precision_at_0.5_raw": precision_score(y_s, p_s >= .5, zero_division=0),
                          "recall_at_0.5_raw": recall_score(y_s, p_s >= .5, zero_division=0),
                          "precision_at_0.5_prior_corrected": precision_score(y_s, p_c >= .5, zero_division=0),
                          "recall_at_0.5_prior_corrected": recall_score(y_s, p_c >= .5, zero_division=0)}
            reps.append(row)
        shift[f"attack_prevalence_{target}"] = {
            k: {m: round(float(np.mean([r[k][m] for r in reps])), 4) for m in reps[0][k]}
            for k in reps[0]}
    res["prevalence_shift"] = shift
    out["models"][name] = res
    print(name, {k: res[k] for k in probs})
    for t, v in shift.items():
        print("  ", t, "isotonic precision raw", v["isotonic"]["precision_at_0.5_raw"],
              "ECE raw", v["isotonic"]["ece_raw"], "-> corrected", v["isotonic"]["ece_prior_corrected"])

save_json(out, "r02_calibration.json")
