"""
R12 - Dataset manifest (supervisor review 3.1 and Week-2 gate: every headline number traceable to code).

Counts rows, attack share and label/family counts for every dataset file used in the revision, so that
Table 4 of the paper is produced by a script rather than typed by hand. RUN LOCALLY: the CICIDS2017 and
CICIoT2023 files are too large for the repository; files that are not found are reported as missing.
Output: results/revision/r12_dataset_manifest.json
"""
import pandas as pd

from common import find_file, save_json

NSL_COLS = list(range(43))  # 41 features + attack name + difficulty


def summary(labels: pd.Series, is_attack: pd.Series) -> dict:
    return {"rows": int(len(labels)), "attacks": int(is_attack.sum()),
            "attack_share": round(float(is_attack.mean()), 4),
            "label_counts": {str(k): int(v) for k, v in labels.value_counts().items()}}


out = {}
jobs = {
    "NSL-KDD KDDTrain+": ("KDDTrain+.txt", None),
    "NSL-KDD KDDTest+": ("KDDTest+.txt", None),
    "UNSW-NB15 training": ("UNSW_NB15_training-set.csv", "attack_cat"),
    "UNSW-NB15 testing": ("UNSW_NB15_testing-set.csv", "attack_cat"),
    "CICIDS2017 cleaned": ("cicids2017_cleaned.csv", "Attack Type"),
    "CICIoT2023 balanced subsample": ("balanced_7classes_500000_each.csv", "MappedLabel"),
}
for name, (fname, col) in jobs.items():
    try:
        path = find_file(fname)
    except FileNotFoundError:
        out[name] = {"file": fname, "status": "missing on this machine"}
        continue
    if col is None:  # NSL-KDD: no header, attack name in column 41
        lab = pd.read_csv(path, header=None, usecols=[41])[41].astype(str).str.strip()
        att = lab != "normal"
    else:
        lab = pd.read_csv(path, usecols=[col])[col].astype(str).str.strip()
        att = {"attack_cat": lab != "Normal", "Attack Type": lab != "Normal Traffic",
               "MappedLabel": lab != "Benign"}[col]
    out[name] = {"file": fname, **summary(lab, att)}
    print(name, out[name]["rows"], out[name]["attack_share"])

save_json(out, "r12_dataset_manifest.json")
