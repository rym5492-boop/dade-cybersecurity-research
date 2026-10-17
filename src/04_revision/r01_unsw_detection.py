"""
R01 - Leakage-safe binary detection on UNSW-NB15 (supervisor review 3.1).

Fixes relative to src/01_predictive_layer/01_unsw_nb15_experiment.py:
  * encoders/scalers fitted inside a Pipeline on the fitting split only (the original fitted
    LabelEncoders on train+test combined);
  * decision thresholds chosen on a validation split carved from the official training file;
  * metrics include prevalence, majority baseline, balanced accuracy, macro-F1, MCC, FPR and
    absolute confusion counts; per-attack-category recall is reported;
  * a 5-fold cross-validation on the official training file is reported next to the official
    test result so the CV-to-test gap is measured with the same leakage-free pipeline.
Outputs: results/revision/r01_unsw_detection.json, results/revision/r01_test_scores.npz
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from common import (OUT_DIR, SEED, detection_metrics, feature_cols, linear_preprocessor,
                    load_unsw, save_json, split_fit_val_cal, threshold_at_fpr, threshold_max_f1,
                    tree_preprocessor)

train, test = load_unsw()
cols = feature_cols(train)
fit, val, cal = split_fit_val_cal(train)
print(f"FIT {len(fit)}  VAL {len(val)}  CAL {len(cal)}  TEST {len(test)}")


def make(name):
    if name == "LogisticRegression":
        return Pipeline([("prep", linear_preprocessor(cols)),
                         ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    if name == "RandomForest":
        return Pipeline([("prep", tree_preprocessor(cols)),
                         ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1))])
    return Pipeline([("prep", tree_preprocessor(cols)),
                     ("clf", HistGradientBoostingClassifier(random_state=SEED))])


out = {"splits": {"fit": len(fit), "val": len(val), "cal": len(cal), "test": len(test),
                  "train_prevalence": round(float(train["label"].mean()), 4),
                  "test_prevalence": round(float(test["label"].mean()), 4)},
       "models": {}}
scores = {}
for name in ["LogisticRegression", "RandomForest", "HistGradientBoosting"]:
    t0 = time.time()
    m = make(name).fit(fit[cols], fit["label"])
    p_val = m.predict_proba(val[cols])[:, 1]
    p_test = m.predict_proba(test[cols])[:, 1]
    thr = {"default_0.5": 0.5,
           "val_max_f1": threshold_max_f1(val["label"], p_val),
           "val_fpr_1pct": threshold_at_fpr(val["label"], p_val, 0.01)}
    res = {"validation_at_0.5": detection_metrics(val["label"], p_val, 0.5),
           "test": {k: detection_metrics(test["label"], p_test, t) for k, t in thr.items()}}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_acc = cross_val_score(make(name), train[cols], train["label"], cv=cv, scoring="accuracy", n_jobs=1)
    res["cv5_on_official_train_accuracy"] = {"mean": round(float(cv_acc.mean()), 4),
                                             "sd": round(float(cv_acc.std()), 4)}
    res["cv_minus_test_accuracy_gap_pp"] = round(100 * (cv_acc.mean() - res["test"]["default_0.5"]["accuracy"]), 2)
    yhat = (p_test >= 0.5).astype(int)
    res["test_recall_by_attack_category_at_0.5"] = {
        c: round(float(yhat[(test["attack_cat"] == c).values].mean()), 4)
        for c in sorted(test["attack_cat"].unique()) if c != "Normal"}
    res["test_false_positive_rate_on_Normal_at_0.5"] = round(float(yhat[(test["attack_cat"] == "Normal").values].mean()), 4)
    res["seconds"] = round(time.time() - t0, 1)
    out["models"][name] = res
    scores[name] = p_test
    d = res["test"]["default_0.5"]
    print(f"{name:22s} acc {d['accuracy']:.4f}  bal {d['balanced_accuracy']:.4f}  "
          f"FPR {d['false_positive_rate']:.4f}  recall {d['attack_recall']:.4f}  "
          f"CV {res['cv5_on_official_train_accuracy']['mean']:.4f}  ({res['seconds']}s)")

np.savez_compressed(OUT_DIR / "r01_test_scores.npz", y=test["label"].values,
                    attack_cat=test["attack_cat"].values, **scores)
save_json(out, "r01_unsw_detection.json")
