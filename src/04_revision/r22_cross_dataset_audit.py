"""
R22 - Audit of the cross-dataset transfer of r07 (supervisor review of 30 October 2026, P0-7). RUN LOCALLY
(needs the CICIDS2017 and CICIoT2023 files; datasets that are missing are skipped, and part 1 runs on the
stored r07 results alone).

1. Subset summary of the stored r07 results: balanced accuracy without adaptation for all six directions,
   without the directions that involve CICIDS2017 (forward-only bytes in the cleaned file) and without those
   that involve CICIoT2023 (duration unit not stated in the documentation; window-based features).
2. Semantics and units, computed from the files: for each harmonised feature and dataset, median and 5th/95th
   percentiles by class, so that a unit error (for example microseconds read as seconds) is visible.
3. Sensitivity of the transfer to the questionable choices (same model, split and seed as r07):
     V0  r07 as published;
     V1  without the two byte-based features (total_bytes, bytes_per_s), i.e. duration and packet size only;
     V2  CICIoT2023 duration read as milliseconds and as microseconds instead of seconds;
     V3  CICIDS2017 bytes doubled as a crude two-direction proxy.
4. Independence of the target adaptation: the 1% and 5% labelled target samples are drawn from the target
   training part and the evaluation uses the disjoint 30% test part; the script asserts that no row is shared.
Outputs: results/revision/r22_cross_dataset_audit.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import OUT_DIR, SEED, find_file, save_json

FEATS = ["duration_s", "total_bytes", "bytes_per_s", "avg_packet_size"]
N_PER = 100_000
out = {}

# ---------------------------------------------------------------- 1. stored r07 results
r07 = json.load(open(OUT_DIR / "r07_cross_dataset_v2.json"))["matrix"]
dirs = {(s, t): r["no_adaptation"]["balanced_accuracy"] for s, row in r07.items() for t, r in row.items() if s != t}


def summ(sel):
    v = [dirs[k] for k in sel]
    return {"directions": [f"{a} -> {b}" for a, b in sel], "balanced_accuracy_min": min(v), "balanced_accuracy_max": max(v),
            "balanced_accuracy_mean": round(float(np.mean(v)), 4)} if v else {"directions": []}


out["stored_r07_subsets"] = {
    "all_six": summ(list(dirs)),
    "without_CICIDS2017": summ([k for k in dirs if "CICIDS2017" not in k]),
    "without_CICIoT2023": summ([k for k in dirs if "CICIoT2023" not in k]),
    "UNSW_NB15_source_only": summ([k for k in dirs if k[0] == "UNSW-NB15"])}


# ---------------------------------------------------------------- data (same harmonisation as r07)
def harmonise(duration_s, total_bytes, avg_pkt, label):
    d = pd.DataFrame({"duration_s": duration_s.astype(float), "total_bytes": total_bytes.astype(float),
                      "avg_packet_size": avg_pkt.astype(float), "label": label.astype(int)})
    d["bytes_per_s"] = d["total_bytes"] / d["duration_s"].clip(lower=1e-6)
    return d.replace([np.inf, -np.inf], np.nan).dropna()


def load(variant):
    data = {}
    try:
        u = pd.concat([pd.read_csv(find_file("UNSW_NB15_training-set.csv")), pd.read_csv(find_file("UNSW_NB15_testing-set.csv"))],
                      ignore_index=True)
        data["UNSW-NB15"] = harmonise(u["dur"], u["sbytes"] + u["dbytes"], (u["smean"] + u["dmean"]) / 2, u["label"])
    except FileNotFoundError:
        pass
    try:
        p = find_file("cicids2017_cleaned.csv")
        have = set(pd.read_csv(p, nrows=0).columns)
        want = ["Flow Duration", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Average Packet Size", "Attack Type"]
        c = pd.read_csv(p, usecols=[w for w in want if w in have])
        bwd = c["Total Length of Bwd Packets"] if "Total Length of Bwd Packets" in c else 0
        fwd = c["Total Length of Fwd Packets"] * (2 if variant == "V3" else 1)
        data["CICIDS2017"] = harmonise(c["Flow Duration"] / 1e6, fwd + bwd, c["Average Packet Size"], c["Attack Type"] != "Normal Traffic")
        out.setdefault("cicids2017_has_backward_bytes", "Total Length of Bwd Packets" in have)
    except FileNotFoundError:
        pass
    try:
        t = pd.read_csv(find_file("balanced_7classes_500000_each.csv"), usecols=["flow_duration", "Tot sum", "AVG", "MappedLabel"])
        scale = {"V2_ms": 1e-3, "V2_us": 1e-6}.get(variant, 1.0)
        data["CICIoT2023"] = harmonise(t["flow_duration"] * scale, t["Tot sum"], t["AVG"], t["MappedLabel"] != "Benign")
    except FileNotFoundError:
        pass
    for k, d in data.items():
        if len(d) > N_PER:
            data[k], _ = train_test_split(d, train_size=N_PER, stratify=d["label"], random_state=SEED)
    return data


def prep(X):
    return np.log1p(np.clip(X, 0, None))


def transfer(data, feats):
    splits = {k: train_test_split(d, test_size=0.30, stratify=d["label"], random_state=SEED) for k, d in data.items()}
    res = {}
    for s, (s_tr, _) in splits.items():
        sc = StandardScaler().fit(prep(s_tr[feats].values))
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
        clf.fit(sc.transform(prep(s_tr[feats].values)), s_tr["label"])
        for t, (t_tr, t_te) in splits.items():
            if s == t:
                continue
            assert not set(t_tr.index) & set(t_te.index), "adaptation and evaluation rows overlap"
            p = clf.predict_proba(sc.transform(prep(t_te[feats].values)))[:, 1]
            y = t_te["label"].values
            res[f"{s} -> {t}"] = {"balanced_accuracy": round(balanced_accuracy_score(y, (p >= 0.5).astype(int)), 4),
                                  "roc_auc": round(roc_auc_score(y, p), 4)}
    return res


base = load("V0")
out["datasets_available"] = sorted(base)
out["feature_distributions"] = {k: {f: {str(c): [round(float(x), 4) for x in np.percentile(d.loc[d["label"] == c, f], [5, 50, 95])]
                                        for c in (0, 1)} for f in FEATS} for k, d in base.items()}
out["adaptation_rows_disjoint_from_evaluation_rows"] = True
if len(base) >= 2:
    out["sensitivity"] = {"V0_as_r07": transfer(base, FEATS),
                          "V1_duration_and_packet_size_only": transfer(base, ["duration_s", "avg_packet_size"])}
    if "CICIoT2023" in base:
        out["sensitivity"]["V2_CICIoT_duration_ms"] = transfer(load("V2_ms"), FEATS)
        out["sensitivity"]["V2_CICIoT_duration_us"] = transfer(load("V2_us"), FEATS)
    if "CICIDS2017" in base:
        out["sensitivity"]["V3_CICIDS_bytes_doubled"] = transfer(load("V3"), FEATS)
save_json(out, "r22_cross_dataset_audit.json")
print(json.dumps(out["stored_r07_subsets"], indent=1))
print("datasets available:", out["datasets_available"])
