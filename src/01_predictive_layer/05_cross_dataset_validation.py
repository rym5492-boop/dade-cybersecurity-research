"""
Experiment E2 (Cross-dataset / unseen-distribution generalization test).
Since the three datasets use entirely different feature schemas, a harmonized
set of 4 conceptually-common features is constructed for each:
  1. duration        (connection/flow duration)
  2. total_bytes      (total bytes transferred)
  3. rate             (traffic rate: bytes or packets per second)
  4. avg_packet_size  (average packet size)

A Random Forest is trained on EACH dataset's harmonized features and tested on
the OTHER TWO datasets (out-of-distribution), to directly test generalization
across genuinely different network environments and attack distributions --
addressing supervisor Section 3.7 / Experiment E2.

LIMITATION (disclosed honestly): this harmonization is coarse (only 4 shared
concepts out of dozens of dataset-specific features), so results here should be
read as a first, conservative test of cross-domain generalizability, not a
substitute for full feature-engineering alignment.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import json
import gc
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
SAMPLE_SIZE = 100000  # per dataset, for memory feasibility and balanced comparison

# ---------- Load and harmonize UNSW-NB15 ----------
print("Loading and harmonizing UNSW-NB15...")
unsw_train = pd.read_csv("/home/claude/unsw_data/UNSW_NB15_training-set.csv")
unsw_test = pd.read_csv("/home/claude/unsw_data/UNSW_NB15_testing-set.csv")
unsw = pd.concat([unsw_train, unsw_test], ignore_index=True)
del unsw_train, unsw_test
gc.collect()

unsw_h = pd.DataFrame({
    "duration": unsw["dur"],
    "total_bytes": unsw["sbytes"] + unsw["dbytes"],
    "rate": unsw["rate"],
    "avg_packet_size": (unsw["smean"] + unsw["dmean"]) / 2,
    "binary_label": unsw["label"].astype(int),
})
del unsw
gc.collect()
if len(unsw_h) > SAMPLE_SIZE:
    unsw_h, _ = train_test_split(unsw_h, train_size=SAMPLE_SIZE, stratify=unsw_h["binary_label"], random_state=42)
print(f"UNSW-NB15 harmonized: {len(unsw_h)} rows, label balance: {dict(unsw_h['binary_label'].value_counts())}")

# ---------- Load and harmonize CICIDS2017 ----------
print("\nLoading and harmonizing CICIDS2017...")
cicids = pd.read_csv("/home/claude/cicids_data/cicids2017_cleaned.csv")
cicids_h = pd.DataFrame({
    "duration": cicids["Flow Duration"],
    "total_bytes": cicids["Total Length of Fwd Packets"],
    "rate": cicids["Flow Bytes/s"].replace([np.inf, -np.inf], np.nan).fillna(0),
    "avg_packet_size": cicids["Average Packet Size"],
    "binary_label": (cicids["Attack Type"] != "Normal Traffic").astype(int),
})
del cicids
gc.collect()
if len(cicids_h) > SAMPLE_SIZE:
    cicids_h, _ = train_test_split(cicids_h, train_size=SAMPLE_SIZE, stratify=cicids_h["binary_label"], random_state=42)
print(f"CICIDS2017 harmonized: {len(cicids_h)} rows, label balance: {dict(cicids_h['binary_label'].value_counts())}")

# ---------- Load and harmonize CICIoT2023 ----------
print("\nLoading and harmonizing CICIoT2023...")
ciciot = pd.read_csv("/home/claude/ciciot_data/balanced_7classes_500000_each.csv")
ciciot_h = pd.DataFrame({
    "duration": ciciot["flow_duration"],
    "total_bytes": ciciot["Tot sum"],
    "rate": ciciot["Rate"],
    "avg_packet_size": ciciot["AVG"],
    "binary_label": (ciciot["MappedLabel"] != "Benign").astype(int),
})
del ciciot
gc.collect()
if len(ciciot_h) > SAMPLE_SIZE:
    ciciot_h, _ = train_test_split(ciciot_h, train_size=SAMPLE_SIZE, stratify=ciciot_h["binary_label"], random_state=42)
print(f"CICIoT2023 harmonized: {len(ciciot_h)} rows, label balance: {dict(ciciot_h['binary_label'].value_counts())}")

datasets = {
    "UNSW-NB15": unsw_h,
    "CICIDS2017": cicids_h,
    "CICIoT2023": ciciot_h,
}

# ---------- Cross-dataset training/testing matrix ----------
feature_cols = ["duration", "total_bytes", "rate", "avg_packet_size"]
results_matrix = {}

for train_name, train_df in datasets.items():
    print(f"\n{'='*70}\nTraining Random Forest on: {train_name}\n{'='*70}")
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["binary_label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2, max_depth=12)
    clf.fit(X_train_scaled, y_train)

    row = {}
    for test_name, test_df in datasets.items():
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["binary_label"].values
        X_test_scaled = scaler.transform(X_test)  # use TRAIN dataset's scaler (realistic deployment scenario)

        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = float("nan")

        row[test_name] = {"accuracy": round(acc, 4), "f1_score": round(f1, 4), "roc_auc": round(auc, 4)}
        tag = "(within-dataset)" if train_name == test_name else "(cross-dataset)"
        print(f"  Tested on {test_name:12s} {tag:18s} -> Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    results_matrix[train_name] = row

with open("cross_dataset_results.json", "w") as f:
    json.dump(results_matrix, f, indent=2)

print("\n\nSaved cross_dataset_results.json")
