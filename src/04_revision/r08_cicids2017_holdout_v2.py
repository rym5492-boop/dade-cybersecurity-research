"""
R08 - CICIDS2017 unseen attack-family evaluation with validation-chosen thresholds. RUN LOCALLY.

Audit of src/01_predictive_layer/02b_cicids2017_attack_family_holdout.py (results/cicids2017_family_holdout_results.json):
  * unseen-family test = 11,098 held-out attacks (Bots, Brute Force) + 11,098 fresh benign rows, so
    50% prevalence; the model flagged only 34 of 22,196 rows as attack at threshold 0.5
    (recall 0.0024, accuracy 0.5009 = "predict everything benign"), while ROC-AUC was 0.953.
    The ranking signal exists; the fixed 0.5 threshold does not transfer.
  * The paper's sentence "may be improved by adjusting the classification threshold" can only be
    supported if the threshold is chosen WITHOUT the unseen-family test data.
This version keeps the original data handling and model, and adds:
  * a validation split carved from the SEEN training pool (unseen families never touch it);
  * thresholds chosen on that validation split: 0.5, max-F1, and the thresholds giving 1% and 5%
    false-positive rate on validation benign traffic;
  * for each threshold: recall on unseen families, FPR, precision, balanced accuracy, prevalence,
    majority baseline and absolute confusion counts;
  * a check that ROC-AUC and accuracy are computed on the same rows (asserted).
Input : cicids2017_cleaned.csv (same file as the original script). Output: results/revision/r08_cicids2017_holdout_v2.json
"""
import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from common import SEED, detection_metrics, find_file, save_json, threshold_at_fpr, threshold_max_f1

SAMPLE_SIZE = 400_000
HOLDOUT_FRACTION = 0.40

# Memory-lean loading (works on machines with limited RAM): features are read once as float32 and
# all subsets are handled as row-index arrays instead of DataFrame copies. Row order and random
# states are the same as a DataFrame-based version, so the selected rows are identical.
path = find_file("cicids2017_cleaned.csv")
attack_type = pd.read_csv(path, usecols=["Attack Type"])["Attack Type"].to_numpy()
feats = [c for c in pd.read_csv(path, nrows=0).columns if c != "Attack Type"]
X = pd.read_csv(path, usecols=feats, dtype=np.float32)
gc.collect()
y_all = (attack_type != "Normal Traffic").astype(int)
cats = sorted(np.unique(attack_type[y_all == 1]).tolist())
rng = np.random.RandomState(SEED)
held = sorted(rng.choice(cats, size=max(1, round(len(cats) * HOLDOUT_FRACTION)), replace=False).tolist())
print("held-out families:", held)

idx_benign = np.where(attack_type == "Normal Traffic")[0]
idx_seen = np.where(np.isin(attack_type, [c for c in cats if c not in held]))[0]
idx_unseen = np.where(np.isin(attack_type, held))[0]

pool = np.concatenate([idx_benign, idx_seen])
if len(pool) > SAMPLE_SIZE:
    pool, _ = train_test_split(pool, train_size=SAMPLE_SIZE, stratify=y_all[pool], random_state=SEED)
tr, rest = train_test_split(pool, test_size=0.30, stratify=y_all[pool], random_state=SEED)
val, test_seen = train_test_split(rest, test_size=0.50, stratify=y_all[rest], random_state=SEED)

fresh_benign = np.setdiff1d(idx_benign, pool)
fresh_benign = pd.Series(fresh_benign).sample(n=min(len(idx_unseen), len(fresh_benign)), random_state=SEED).to_numpy()
test_unseen = np.concatenate([idx_unseen, fresh_benign])
print(f"train {len(tr)}  val {len(val)}  seen-test {len(test_seen)}  unseen-test {len(test_unseen)} "
      f"(held-out attacks {len(idx_unseen)}, fresh benign {len(fresh_benign)})")

model = HistGradientBoostingClassifier(random_state=SEED, max_iter=100)
model.fit(X.iloc[tr], y_all[tr])   # trees: no scaling needed
p_val = model.predict_proba(X.iloc[val])[:, 1]
thresholds = {"default_0.5": 0.5,
              "val_max_f1": threshold_max_f1(y_all[val], p_val),
              "val_fpr_1pct": threshold_at_fpr(y_all[val], p_val, 0.01),
              "val_fpr_5pct": threshold_at_fpr(y_all[val], p_val, 0.05)}

out = {"held_out_families": held, "thresholds_chosen_on_validation": {k: round(v, 6) for k, v in thresholds.items()},
       "seen_family_test": {}, "unseen_family_test": {}}
for name, rows in [("seen_family_test", test_seen), ("unseen_family_test", test_unseen)]:
    y = y_all[rows]
    p = model.predict_proba(X.iloc[rows])[:, 1]
    assert len(p) == len(y)
    for k, t in thresholds.items():
        out[name][k] = detection_metrics(y, p, t)
    print(name, {k: (v["attack_recall"], v["false_positive_rate"]) for k, v in out[name].items()})
save_json(out, "r08_cicids2017_holdout_v2.json")
