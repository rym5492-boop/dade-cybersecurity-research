"""
Closes supervisor Section 3.7 (second clause) for CICIDS2017: the existing
02_cicids2017_experiment.py uses a stratified RANDOM 80/20 split, which the
supervisor review notes can overestimate generalization. This script adds an
ATTACK-FAMILY HOLDOUT split as a complement: entire attack categories are
excluded from training and used ONLY for testing, directly measuring
generalization to attack types the model has never seen at all (the
strongest form of leakage-resistant evaluation available without a
timestamp column, which the cleaned/preprocessed CICIDS2017 distribution
does not retain).

Protocol:
  1. Identify all attack categories present in "Attack Type" (excluding
     "Normal Traffic").
  2. Fix random_state=42 and randomly hold out ~40% of attack CATEGORIES
     (not rows) entirely: every row belonging to a held-out category is
     removed from the training pool and reserved for the held-out test set.
  3. Train the best-performing model from the random-split experiment
     (HistGradientBoosting) on: Benign + attacks from the remaining
     ("seen") categories only, using the same stratified subsample size
     (400,000 rows total pool, same as 02_cicids2017_experiment.py) and an
     internal 80/20 train/test split of the "seen" pool for a like-for-like
     comparison point.
  4. Evaluate on TWO test sets:
       (a) "seen-family" test set  = the standard internal 20% held-out
           split of the seen-category pool (comparable to the original
           random-split result).
       (b) "unseen-family" test set = ALL rows from the held-out attack
           categories (the model never saw these categories during
           training) + a fresh, unused sample of Benign traffic.
  5. Report Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC/FPR for both, so the
     gap between (a) and (b) is directly comparable to the CV-to-test gap
     already reported for UNSW-NB15's official split in the paper.

Run this after 02_cicids2017_experiment.py, from the same working directory
(where cicids2017_cleaned.csv lives), with the same environment
(requirements.txt). Output: results/cicids2017_family_holdout_results.json
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, confusion_matrix)
import json
import time
import gc
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SAMPLE_SIZE = 400000          # same total pool size as 02_cicids2017_experiment.py
HOLDOUT_FRACTION = 0.40       # fraction of ATTACK CATEGORIES (not rows) held out entirely

print("Loading CICIDS2017 (cleaned)...")
df = pd.read_csv("cicids2017_cleaned.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

df["binary_label"] = (df["Attack Type"] != "Normal Traffic").astype(int)

attack_categories = sorted(df.loc[df["binary_label"] == 1, "Attack Type"].unique().tolist())
print(f"\nFound {len(attack_categories)} attack categories: {attack_categories}")

rng = np.random.RandomState(RANDOM_STATE)
n_holdout = max(1, round(len(attack_categories) * HOLDOUT_FRACTION))
held_out_categories = sorted(rng.choice(attack_categories, size=n_holdout, replace=False).tolist())
seen_categories = [c for c in attack_categories if c not in held_out_categories]
print(f"\nHeld-out (unseen-during-training) categories ({len(held_out_categories)}): {held_out_categories}")
print(f"Seen (used-in-training) categories ({len(seen_categories)}): {seen_categories}")

# ---------- Partition rows ----------
df_benign = df[df["Attack Type"] == "Normal Traffic"]
df_seen_attacks = df[df["Attack Type"].isin(seen_categories)]
df_unseen_attacks = df[df["Attack Type"].isin(held_out_categories)]
del df
gc.collect()

print(f"\nBenign rows: {len(df_benign)}")
print(f"Seen-category attack rows: {len(df_seen_attacks)}")
print(f"Unseen-category attack rows (fully held out): {len(df_unseen_attacks)}")

# ---------- Build the "seen" pool (benign + seen-category attacks), same total size as original experiment ----------
seen_pool = pd.concat([df_benign, df_seen_attacks], axis=0)
if len(seen_pool) > SAMPLE_SIZE:
    seen_pool, _ = train_test_split(
        seen_pool, train_size=SAMPLE_SIZE, stratify=seen_pool["binary_label"], random_state=RANDOM_STATE
    )
print(f"\nSeen pool (for training + seen-family test): {len(seen_pool)} rows")

feature_cols = [c for c in seen_pool.columns if c not in ("Attack Type", "binary_label")]

X_seen = seen_pool[feature_cols].astype(np.float32)
y_seen = seen_pool["binary_label"]

X_train, X_test_seen, y_train, y_test_seen = train_test_split(
    X_seen, y_seen, test_size=0.2, random_state=RANDOM_STATE, stratify=y_seen
)
del seen_pool, X_seen, y_seen
gc.collect()

# ---------- Build the "unseen-family" test set: all held-out attacks + a fresh benign sample not used in training ----------
benign_used_idx = set(df_benign.index) - set(X_train.index) - set(X_test_seen.index)
# df_benign rows not already consumed by the seen pool subsample; fall back to a fresh random draw if pool shrank benign rows
benign_remaining = df_benign.loc[df_benign.index.isin(benign_used_idx)]
if len(benign_remaining) < len(df_unseen_attacks):
    # not enough fresh benign left; sample with replacement=False from full benign pool, excluding train/test rows used above
    benign_remaining = df_benign.drop(index=[i for i in X_train.index if i in df_benign.index] +
                                             [i for i in X_test_seen.index if i in df_benign.index],
                                       errors="ignore")
benign_for_unseen_test = benign_remaining.sample(
    n=min(len(df_unseen_attacks), len(benign_remaining)), random_state=RANDOM_STATE
) if len(benign_remaining) > 0 else benign_remaining

unseen_family_test = pd.concat([df_unseen_attacks, benign_for_unseen_test], axis=0)
X_test_unseen = unseen_family_test[feature_cols].astype(np.float32)
y_test_unseen = unseen_family_test["binary_label"]
del df_benign, df_seen_attacks, df_unseen_attacks, benign_remaining, benign_for_unseen_test, unseen_family_test
gc.collect()

print(f"\nTrain (seen categories only): {len(X_train)}")
print(f"Seen-family test set:         {len(X_test_seen)}")
print(f"Unseen-family test set:       {len(X_test_unseen)}  "
      f"(attacks from held-out categories: {y_test_unseen.sum()}, benign: {(y_test_unseen == 0).sum()})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_seen_scaled = scaler.transform(X_test_seen).astype(np.float32)
X_test_unseen_scaled = scaler.transform(X_test_unseen).astype(np.float32)
del X_train, X_test_seen, X_test_unseen
gc.collect()

model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=100)

print(f"\n{'='*60}\nTraining HistGradientBoosting on SEEN categories only\n{'='*60}")
t0 = time.time()
model.fit(X_train_scaled, y_train)
train_time = time.time() - t0
print(f"Training time: {train_time:.1f}s")


def evaluate(name, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_proba) if len(set(y)) > 1 else float("nan")
    pr_auc = average_precision_score(y, y_proba) if len(set(y)) > 1 else float("nan")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"FPR:       {fpr:.4f}")
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc == roc_auc else None,
        "pr_auc": round(float(pr_auc), 4) if pr_auc == pr_auc else None,
        "false_positive_rate": round(float(fpr), 4) if fpr == fpr else None,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y)),
    }


results = {
    "held_out_categories": held_out_categories,
    "seen_categories": seen_categories,
    "model": "HistGradientBoosting",
    "train_time_sec": round(train_time, 1),
    "seen_family_test": evaluate("Seen-family test (in-distribution)", X_test_seen_scaled, y_test_seen),
    "unseen_family_test": evaluate("Unseen-family test (held-out attack categories)", X_test_unseen_scaled, y_test_unseen),
}

acc_gap = results["seen_family_test"]["accuracy"] - results["unseen_family_test"]["accuracy"]
results["accuracy_gap_seen_minus_unseen"] = round(acc_gap, 4)
print(f"\n{'='*60}\nSEEN-vs-UNSEEN-FAMILY ACCURACY GAP: {acc_gap:.4f} "
      f"({acc_gap*100:.1f} percentage points)\n{'='*60}")

with open("cicids2017_family_holdout_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved cicids2017_family_holdout_results.json")
print("Done.")
