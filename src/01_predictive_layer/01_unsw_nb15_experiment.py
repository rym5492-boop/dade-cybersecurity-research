"""
Priority 2 (supervisor review, Section 3.6/3.7 and Phase B): Modern multi-dataset
predictive layer validation using UNSW-NB15 (official train/test split, avoiding
our own random re-splitting on this dataset), with 2-3 strong models plus a
simple baseline (per Phase B recommendation), and expanded evaluation metrics
beyond Accuracy/Precision/Recall/F1 (ROC-AUC, PR-AUC, FPR).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, confusion_matrix,
                              roc_curve)
import json
import warnings
warnings.filterwarnings("ignore")

# ---------- 1. Load OFFICIAL train/test split (not our own random split) ----------
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

print(f"Official training set: {train_df.shape[0]} records")
print(f"Official test set: {test_df.shape[0]} records")
print(f"\nAttack category distribution (train):")
print(train_df["attack_cat"].value_counts())

# ---------- 2. Preprocessing ----------
categorical_cols = ["proto", "service", "state"]
drop_cols = ["id", "attack_cat", "label"]

# Fit encoders on combined categories to avoid unseen-category errors at test time
combined = pd.concat([train_df[categorical_cols], test_df[categorical_cols]], axis=0)
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(combined[col].astype(str))
    encoders[col] = le
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["label"]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df["label"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature count: {X_train.shape[1]}")
print(f"Train label distribution: {dict(y_train.value_counts())}")
print(f"Test label distribution: {dict(y_test.value_counts())}")

# ---------- 3. Models: 2 strong + 1 simple baseline (per supervisor Phase B) ----------
models = {
    "Logistic Regression (baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    model.fit(X_train_scaled, y_train)
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

    # 5-fold CV on training set for confidence-interval-style variability
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "false_positive_rate": round(fpr, 4),
        "cv_mean_accuracy": round(cv_scores.mean(), 4),
        "cv_std_accuracy": round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Confusion matrix:\n{cm}")

with open("unsw_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}\nSUMMARY TABLE\n{'='*60}")
summary_df = pd.DataFrame(results).T
print(summary_df[["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "false_positive_rate"]].to_string())
summary_df.to_csv("unsw_summary_results.csv")
print("\nDone.")
