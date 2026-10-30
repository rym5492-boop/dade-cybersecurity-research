"""
R23 - Provenance manifest (supervisor review of 30 October 2026, P0-6): SHA-256 of every dataset file found,
the row identifiers of the FIT / VAL / CAL split of the UNSW-NB15 training file used by r01-r04, r09 and r17,
the test rows shown in the 30 scenarios, the seed, and the software versions of the run.
Outputs: results/revision/r23_provenance_manifest.json, results/revision/splits/unsw_fit_val_cal_ids.csv
"""
import hashlib
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

from common import OUT_DIR, SEED, find_file, load_unsw, save_json, split_fit_val_cal

FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv", "KDDTrain+.txt", "KDDTest+.txt",
         "cicids2017_cleaned.csv", "balanced_7classes_500000_each.csv"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


hashes = {}
for name in FILES:
    try:
        p = find_file(name)
        hashes[name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    except FileNotFoundError:
        hashes[name] = "not present on this machine"

train, test = load_unsw()
fit, val, cal = split_fit_val_cal(train)
(OUT_DIR / "splits").mkdir(exist_ok=True)
ids = pd.concat([pd.DataFrame({"id": fit["id"].values, "split": "FIT"}), pd.DataFrame({"id": val["id"].values, "split": "VAL"}),
                 pd.DataFrame({"id": cal["id"].values, "split": "CAL"})]).sort_values("id")
ids.to_csv(OUT_DIR / "splits" / "unsw_fit_val_cal_ids.csv", index=False)
scen = pd.read_csv(OUT_DIR / "r09_scenarios.csv", encoding="utf-8-sig")
out = {"seed": SEED,
       "python": sys.version.split()[0], "platform": platform.platform(),
       "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
       "datasets": hashes,
       "unsw_split": {"file": "results/revision/splits/unsw_fit_val_cal_ids.csv", "column": "id of UNSW_NB15_training-set.csv",
                      "rows": {"FIT": int(len(fit)), "VAL": int(len(val)), "CAL": int(len(cal))},
                      "attack_share": {"FIT": round(float(fit["label"].mean()), 4), "VAL": round(float(val["label"].mean()), 4),
                                       "CAL": round(float(cal["label"].mean()), 4)},
                      "overlap_between_parts": int(len(set(fit["id"]) & set(val["id"])) + len(set(fit["id"]) & set(cal["id"]))
                                                   + len(set(val["id"]) & set(cal["id"])))},
       "scenario_test_rows": {"file": "results/revision/r09_scenarios.csv", "column": "test_id (id in UNSW_NB15_testing-set.csv)",
                              "test_ids": scen.sort_values("scenario_id")["test_id"].astype(int).tolist()}}
save_json(out, "r23_provenance_manifest.json")
print({k: (v if isinstance(v, str) else v["sha256"][:16]) for k, v in hashes.items()}, out["unsw_split"]["rows"])
