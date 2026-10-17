"""
R03 - Predicted Threat Severity (supervisor review, Critical 1).

The original ablation (src/02_dade_engine/01_ablation_study.py, lines 106-107) set Threat Severity
from the TEST SET's ground-truth attack_cat, which is not available at decision time. Here severity
comes from a multiclass attack-category classifier trained on the official training file only;
its predicted category is mapped to severity with the SAME table the paper used (SEVERITY_MAP).
We report how wrong the predicted severity is, so the paper can state the cost of removing leakage.
Outputs: results/revision/r03_attack_category.json, results/revision/r03_test_predicted_severity.npz
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from common import (OUT_DIR, SEED, SEVERITY_MAP, feature_cols, load_unsw, save_json,
                    split_fit_val_cal, tree_preprocessor)

train, test = load_unsw()
cols = feature_cols(train)
fit, val, cal = split_fit_val_cal(train)

model = Pipeline([("prep", tree_preprocessor(cols)),
                  ("clf", HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced",
                                                         max_iter=300))])
model.fit(fit[cols], fit["attack_cat"])

labels = sorted(train["attack_cat"].unique())
out = {}
for split_name, df in [("validation", val), ("test", test)]:
    pred = model.predict(df[cols])
    true = df["attack_cat"].values
    ts_true = np.array([SEVERITY_MAP[c] for c in true])
    ts_pred = np.array([SEVERITY_MAP[c] for c in pred])
    err = np.abs(ts_pred - ts_true)
    rep = classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)
    out[split_name] = {
        "n": int(len(df)),
        "category_accuracy": round(float((pred == true).mean()), 4),
        "category_balanced_accuracy": round(balanced_accuracy_score(true, pred), 4),
        "category_macro_f1": round(f1_score(true, pred, average="macro"), 4),
        "per_category_recall": {c: round(rep[c]["recall"], 4) for c in labels},
        "per_category_support": {c: int(rep[c]["support"]) for c in labels},
        "severity_mae": round(float(err.mean()), 4),
        "share_severity_exact": round(float((err < 1e-9).mean()), 4),
        "share_severity_error_gt_0.10": round(float((err > 0.10).mean()), 4),
        "share_severity_error_gt_0.30": round(float((err > 0.30).mean()), 4),
        "attack_given_severity_0 (missed as Normal)": round(float(((ts_pred == 0) & (ts_true > 0)).sum() / max((ts_true > 0).sum(), 1)), 4),
        "normal_given_severity_gt_0 (false severity)": round(float(((ts_pred > 0) & (ts_true == 0)).sum() / max((ts_true == 0).sum(), 1)), 4),
    }
    if split_name == "test":
        out["test"]["confusion_matrix"] = {"labels": labels,
                                           "matrix": confusion_matrix(true, pred, labels=labels).tolist()}
        np.savez_compressed(OUT_DIR / "r03_test_predicted_severity.npz",
                            pred_cat=pred, true_cat=true, ts_pred=ts_pred, ts_true=ts_true)
    print(f"{split_name}: acc {out[split_name]['category_accuracy']}  macroF1 {out[split_name]['category_macro_f1']}  "
          f"severity MAE {out[split_name]['severity_mae']}  exact {out[split_name]['share_severity_exact']}")

save_json(out, "r03_attack_category.json")
