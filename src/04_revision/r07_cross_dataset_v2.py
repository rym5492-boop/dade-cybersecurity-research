"""
R07 - Cross-dataset transfer, corrected (supervisor review 3.1, fourth bullet). RUN LOCALLY.

Problems found in src/01_predictive_layer/05_cross_dataset_validation.py:
  1. The diagonal ("within-dataset" 92.2 / 99.4 / 90.4%) was computed on the SAME rows the model was
     trained on, i.e. training accuracy, not held-out accuracy.
  2. Off-diagonal accuracies equal the class share of the target set (e.g. CICIDS2017 -> CICIoT2023:
     accuracy 0.174 with F1 = 0.000), meaning the model predicted a single class. Accuracy alone
     hides this; the target prevalence and majority baseline must be shown.
  3. Units and semantics differed: UNSW `dur` is seconds but CICIDS2017 `Flow Duration` is
     microseconds; UNSW total bytes summed both directions but CICIDS2017 used forward bytes only;
     UNSW `rate` is packets/s while CICIDS2017 `Flow Bytes/s` is bytes/s.
This version:
  * converts durations to seconds, sums both directions where available, and recomputes
    bytes-per-second from total bytes and duration for every dataset;
  * applies log1p before scaling (heavy-tailed traffic features);
  * evaluates the diagonal on a held-out 30% split;
  * reports prevalence, majority baseline, balanced accuracy, macro-F1, ROC-AUC and the share of rows
    predicted as attack;
  * tests two cheap adaptations instead of assuming retraining is the only remedy:
      (a) threshold re-selected on 1% labelled target data,
      (b) retraining on source + 1% / 5% labelled target data.
Set the three paths below. Datasets whose file is missing are skipped.
CICIoT2023 `flow_duration` units must be checked against the dataset documentation; set
CICIOT_DURATION_TO_SECONDS accordingly before reporting.
Output: results/revision/r07_cross_dataset_v2.json
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import SEED, find_file, save_json

CICIDS_PATH = "cicids2017_cleaned.csv"
CICIOT_PATH = "balanced_7classes_500000_each.csv"
CICIOT_DURATION_TO_SECONDS = 1.0   # VERIFY against CICIoT2023 documentation before reporting
N_PER_DATASET = 100_000
FEATS = ["duration_s", "total_bytes", "bytes_per_s", "avg_packet_size"]


def harmonise(duration_s, total_bytes, avg_pkt, label):
    d = pd.DataFrame({"duration_s": duration_s.astype(float), "total_bytes": total_bytes.astype(float),
                      "avg_packet_size": avg_pkt.astype(float), "label": label.astype(int)})
    d["bytes_per_s"] = d["total_bytes"] / d["duration_s"].clip(lower=1e-6)
    return d.replace([np.inf, -np.inf], np.nan).dropna()


def load_all():
    out = {}
    try:
        u = pd.concat([pd.read_csv(find_file("UNSW_NB15_training-set.csv")),
                       pd.read_csv(find_file("UNSW_NB15_testing-set.csv"))], ignore_index=True)
        out["UNSW-NB15"] = harmonise(u["dur"], u["sbytes"] + u["dbytes"], (u["smean"] + u["dmean"]) / 2, u["label"])
    except FileNotFoundError as e:
        print("skip UNSW:", e)
    try:
        cpath = find_file(CICIDS_PATH)
        have = set(pd.read_csv(cpath, nrows=0).columns)
        want = ["Flow Duration", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                "Average Packet Size", "Attack Type"]
        c = pd.read_csv(cpath, usecols=[w for w in want if w in have])   # only needed columns (low RAM)
        bwd = c["Total Length of Bwd Packets"] if "Total Length of Bwd Packets" in c else 0
        if "Total Length of Bwd Packets" not in c:
            print("WARNING: CICIDS2017 file has no backward-bytes column; total_bytes = forward only")
        out["CICIDS2017"] = harmonise(c["Flow Duration"] / 1e6, c["Total Length of Fwd Packets"] + bwd,
                                      c["Average Packet Size"], (c["Attack Type"] != "Normal Traffic"))
    except FileNotFoundError as e:
        print("skip CICIDS2017:", e)
    try:
        t = pd.read_csv(find_file(CICIOT_PATH), usecols=["flow_duration", "Tot sum", "AVG", "MappedLabel"])
        out["CICIoT2023"] = harmonise(t["flow_duration"] * CICIOT_DURATION_TO_SECONDS, t["Tot sum"], t["AVG"],
                                      (t["MappedLabel"] != "Benign"))
    except FileNotFoundError as e:
        print("skip CICIoT2023:", e)
    for k, d in out.items():
        if len(d) > N_PER_DATASET:
            out[k], _ = train_test_split(d, train_size=N_PER_DATASET, stratify=d["label"], random_state=SEED)
    return out


def prep(X):
    return np.log1p(np.clip(X, 0, None))


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    prev = float(np.mean(y))
    return {"n": int(len(y)), "prevalence_attack": round(prev, 4),
            "majority_baseline_accuracy": round(max(prev, 1 - prev), 4),
            "accuracy": round(float((yhat == y).mean()), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, yhat), 4),
            "macro_f1": round(f1_score(y, yhat, average="macro"), 4),
            "roc_auc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
            "share_predicted_attack": round(float(yhat.mean()), 4), "threshold": round(float(thr), 3)}


def best_threshold(y, p):
    grid = np.linspace(0.02, 0.98, 49)
    return float(grid[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]))])


data = load_all()
splits = {k: train_test_split(d, test_size=0.30, stratify=d["label"], random_state=SEED) for k, d in data.items()}
out = {"note": "diagonal = held-out 30%; off-diagonal = full target test split", "matrix": {}}
for src, (src_tr, _) in splits.items():
    sc = StandardScaler().fit(prep(src_tr[FEATS].values))
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
    clf.fit(sc.transform(prep(src_tr[FEATS].values)), src_tr["label"])
    row = {}
    for tgt, (tgt_tr, tgt_te) in splits.items():
        Xte = sc.transform(prep(tgt_te[FEATS].values))
        p = clf.predict_proba(Xte)[:, 1]
        r = {"no_adaptation": metrics(tgt_te["label"].values, p)}
        if tgt != src:
            for frac in (0.01, 0.05):
                lab, _ = train_test_split(tgt_tr, train_size=frac, stratify=tgt_tr["label"], random_state=SEED)
                if frac == 0.01:
                    p_lab = clf.predict_proba(sc.transform(prep(lab[FEATS].values)))[:, 1]
                    r["threshold_from_1pct_target"] = metrics(tgt_te["label"].values, p, best_threshold(lab["label"].values, p_lab))
                mix = pd.concat([src_tr, lab])
                sc2 = StandardScaler().fit(prep(mix[FEATS].values))
                c2 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
                c2.fit(sc2.transform(prep(mix[FEATS].values)), mix["label"])
                r[f"retrain_with_{int(frac*100)}pct_target"] = metrics(
                    tgt_te["label"].values, c2.predict_proba(sc2.transform(prep(tgt_te[FEATS].values)))[:, 1])
        row[tgt] = r
        print(f"{src:11s} -> {tgt:11s} " + " | ".join(
            f"{k}: bal {v['balanced_accuracy']:.3f} AUC {v['roc_auc']}" for k, v in r.items()))
    out["matrix"][src] = row
save_json(out, "r07_cross_dataset_v2.json")
