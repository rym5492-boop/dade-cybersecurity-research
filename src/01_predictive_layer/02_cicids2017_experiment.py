"""
Priority 2 continuation: Second modern dataset (CICIDS2017) following the same
methodology as UNSW-NB15 (2 strong models + 1 simple baseline, expanded metrics).

NOTE: Due to compute-environment memory constraints, a large stratified subsample
(400,000 of 2,520,751 records, preserving the original class balance) is used
instead of the full dataset. This is disclosed transparently as a limitation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, confusion_matrix)
import json
import time
import gc
import warnings
warnings.filterwarnings("ignore")

print("Loading CICIDS2017 (cleaned)...")
df = pd.read_csv("cicids2017_cleaned.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

df["binary_label"] = (df["Attack Type"] != "Normal Traffic").astype(int)
print("Full distribution:", dict(df["binary_label"].value_counts()))

# ---------- Stratified subsample for memory feasibility ----------
SAMPLE_SIZE = 400000
df_sample, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df["binary_label"], random_state=42)
del df
gc.collect()
print(f"\nUsing stratified subsample: {len(df_sample)} rows")
print("Subsample distribution:", dict(df_sample["binary_label"].value_counts()))

X = df_sample.drop(columns=["Attack Type", "binary_label"]).astype(np.float32)
y = df_sample["binary_label"]
del df_sample
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
del X, y
gc.collect()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)
del X_train, X_test
gc.collect()

print(f"\nTrain: {X_train_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}")

models = {
    "Logistic Regression (baseline)": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2, max_depth=15),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_iter=100),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)

    cv_sample_idx = np.random.RandomState(42).choice(len(X_train_scaled), size=min(30000, len(X_train_scaled)), replace=False)
    cv_scores = cross_val_score(model, X_train_scaled[cv_sample_idx], y_train.iloc[cv_sample_idx],
                                  cv=cv, scoring="accuracy", n_jobs=2)

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "false_positive_rate": round(fpr, 4),
        "cv_mean_accuracy_subsample": round(cv_scores.mean(), 4),
        "cv_std_accuracy_subsample": round(cv_scores.std(), 4),
        "train_time_sec": round(train_time, 1),
        "confusion_matrix": cm.tolist(),
    }

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"CV accuracy (30k subsample): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    gc.collect()

with open("cicids2017_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
summary_df = pd.DataFrame(results).T
print(summary_df[["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "false_positive_rate"]].to_string())
summary_df.to_csv("cicids2017_summary.csv")
print("\nDone.")
