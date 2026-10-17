"""
R11 - Duplicate flows in UNSW-NB15 and their effect on the CV-vs-test gap (supervisor review 3.1:
"Deduplicate or group correlated flows to reduce train-test contamination").

Rows are compared on all 42 model features (id, label and attack_cat excluded).
Reports:
  * share of rows in the official training file that duplicate another training row;
  * share of official test rows identical to some training row (train-test overlap);
  * HistGradientBoosting 5-fold CV accuracy on the training file with ordinary stratified folds versus
    folds grouped by duplicate signature (identical rows never split across folds);
  * official-test metrics on all rows and on test rows that do NOT appear in the training file.
Output: results/revision/r11_unsw_duplicates.json
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from common import SEED, detection_metrics, feature_cols, load_unsw, save_json, tree_preprocessor

train, test = load_unsw()
cols = feature_cols(train)
sig_tr = pd.util.hash_pandas_object(train[cols], index=False).values
sig_te = pd.util.hash_pandas_object(test[cols], index=False).values
groups = pd.factorize(sig_tr)[0]
in_train = np.isin(sig_te, sig_tr)

# Label conflicts: identical features with different labels inside the training file.
conf = pd.DataFrame({"g": groups, "y": train["label"].values}).groupby("g")["y"].nunique()


def model():
    return Pipeline([("prep", tree_preprocessor(cols)), ("clf", HistGradientBoostingClassifier(random_state=SEED))])


X, y = train[cols], train["label"].values
cv_plain = cross_val_score(model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy")
cv_group = cross_val_score(model(), X, y, groups=groups,
                           cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED), scoring="accuracy")

m = model().fit(X, y)
p = m.predict_proba(test[cols])[:, 1]
yt = test["label"].values
out = {
    "features_compared": len(cols),
    "train_rows": int(len(train)), "test_rows": int(len(test)),
    "train_share_duplicate_of_another_train_row": round(float(pd.Series(sig_tr).duplicated().mean()), 4),
    "train_unique_signatures": int(len(np.unique(sig_tr))),
    "train_signatures_with_conflicting_labels": int((conf > 1).sum()),
    "test_share_identical_to_a_train_row": round(float(in_train.mean()), 4),
    "test_share_duplicate_within_test": round(float(pd.Series(sig_te).duplicated().mean()), 4),
    "cv5_accuracy_stratified": {"mean": round(float(cv_plain.mean()), 4), "sd": round(float(cv_plain.std()), 4)},
    "cv5_accuracy_grouped_by_duplicate": {"mean": round(float(cv_group.mean()), 4), "sd": round(float(cv_group.std()), 4)},
    "test_all_rows_at_0.5": detection_metrics(yt, p, 0.5),
    "test_rows_not_in_train_at_0.5": detection_metrics(yt[~in_train], p[~in_train], 0.5),
    "test_rows_in_train_at_0.5": detection_metrics(yt[in_train], p[in_train], 0.5),
}
out["cv_minus_test_gap_pp"] = {
    "stratified": round(100 * (out["cv5_accuracy_stratified"]["mean"] - out["test_all_rows_at_0.5"]["accuracy"]), 2),
    "grouped": round(100 * (out["cv5_accuracy_grouped_by_duplicate"]["mean"] - out["test_all_rows_at_0.5"]["accuracy"]), 2)}
save_json(out, "r11_unsw_duplicates.json")
print({k: v for k, v in out.items() if not k.startswith("test_")})
