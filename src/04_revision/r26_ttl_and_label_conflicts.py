"""
R26 - Two known artefacts of UNSW-NB15 checked against the detector (exploratory; 31 October 2026).

  A. TTL features. sttl, dttl and ct_state_ttl reflect the testbed (attack and normal hosts sit at different
     hop distances) and are widely suspected of shortcut learning. Two checks with the r01 pipeline, splits and
     seed: (i) HGB trained without the three TTL features, scored on the official test file at 0.5 and at the
     VAL thresholds; (ii) a depth-4 decision tree on the three TTL features alone.
  B. Label conflicts between the files. Test rows whose feature values equal a training row (r11) are checked
     for the label that row carries in the training file: benign test rows that are attacks in training cannot
     be classified correctly by any model fitted on that file.
Outputs: results/revision/r26_ttl_and_label_conflicts.json
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import (SEED, detection_metrics, feature_cols, load_unsw, save_json, split_fit_val_cal,
                    threshold_at_fpr, threshold_max_f1, tree_preprocessor)

TTL = ["sttl", "dttl", "ct_state_ttl"]
train, test = load_unsw()
cols = feature_cols(train)
fit, val, cal = split_fit_val_cal(train)
out = {"status": "exploratory, post hoc (31 October 2026)", "ttl_features": TTL}


def evaluate(model, use):
    m = model.fit(fit[use], fit["label"])
    p_val, p_test = m.predict_proba(val[use])[:, 1], m.predict_proba(test[use])[:, 1]
    thr = {"default_0.5": 0.5, "val_max_f1": threshold_max_f1(val["label"], p_val),
           "val_fpr_1pct": threshold_at_fpr(val["label"], p_val, 0.01)}
    return {"test_roc_auc": round(float(roc_auc_score(test["label"], p_test)), 4),
            "test": {k: detection_metrics(test["label"], p_test, t) for k, t in thr.items()}}


no_ttl = [c for c in cols if c not in TTL]
out["A_hgb_all_features"] = evaluate(Pipeline([("prep", tree_preprocessor(cols)),
                                               ("clf", HistGradientBoostingClassifier(random_state=SEED))]), cols)
out["A_hgb_without_ttl"] = evaluate(Pipeline([("prep", tree_preprocessor(no_ttl)),
                                              ("clf", HistGradientBoostingClassifier(random_state=SEED))]), no_ttl)
out["A_tree_depth4_ttl_only"] = evaluate(DecisionTreeClassifier(max_depth=4, random_state=SEED), TTL)
out["A_training_file_sttl"] = {
    str(v): {"rows": int((train["sttl"] == v).sum()), "attack_share": round(float(train.loc[train["sttl"] == v, "label"].mean()), 4)}
    for v in train["sttl"].value_counts().index[:6]}

# ---------------------------------------------------------------- B. label conflicts (same definition of
# "identical" as r11: all feature columns equal)
key = cols
tr = train[key + ["label"]].copy()
tr["_k"] = pd.util.hash_pandas_object(tr[key], index=False).values
te = test[key + ["label"]].copy()
te["_k"] = pd.util.hash_pandas_object(te[key], index=False).values
lab = tr.groupby("_k")["label"].agg(["mean", "size"])
m = te.join(lab, on="_k", how="inner")
benign_dup = m[m["label"] == 0]
attack_dup = m[m["label"] == 1]
out["B_label_conflicts"] = {
    "test_rows_identical_to_a_training_row": int(len(m)),
    "benign_test_rows_identical_to_a_training_row": int(len(benign_dup)),
    "of_which_majority_attack_in_training": int((benign_dup["mean"] > 0.5).sum()),
    "of_which_any_attack_in_training": int((benign_dup["mean"] > 0).sum()),
    "attack_test_rows_identical_to_a_training_row": int(len(attack_dup)),
    "of_which_majority_benign_in_training": int((attack_dup["mean"] < 0.5).sum())}
save_json(out, "r26_ttl_and_label_conflicts.json")
for k in ["A_hgb_all_features", "A_hgb_without_ttl", "A_tree_depth4_ttl_only"]:
    d = out[k]["test"]["default_0.5"]
    print(k, out[k]["test_roc_auc"], d["accuracy"], d["false_positive_rate"], d["attack_recall"])
print(out["B_label_conflicts"])
