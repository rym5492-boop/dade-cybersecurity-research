"""
R05 - NSL-KDD on the official KDDTrain+ / KDDTest+ split (supervisor review, Section 4 "Title and abstract").

The paper's 99.94% came from a random 80/20 split INSIDE KDDTrain+ (Figure 3's 25,195 test rows =
20% of 125,973). This script reports both protocols with the same leakage-free pipeline so the
paper can show the gap honestly. XGBoost is not used here (HistGradientBoosting is the gradient-
boosted model used for all modern datasets in the paper).
Inputs : KDDTrain+.txt and KDDTest+.txt (official NSL-KDD files) in the repository root or data/.
Output : results/revision/r05_nslkdd.json
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import SEED, detection_metrics, find_file, save_json

NAMES = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
         "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
         "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
         "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
         "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
         "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
         "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
         "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"]
CAT = ["protocol_type", "service", "flag"]

tr = pd.read_csv(find_file("KDDTrain+.txt"), header=None, names=NAMES)
te = pd.read_csv(find_file("KDDTest+.txt"), header=None, names=NAMES)
assert len(tr) == 125973 and len(te) == 22544
for d in (tr, te):
    d["label"] = (d["attack"] != "normal").astype(int)
FEAT = [c for c in NAMES if c not in ("attack", "difficulty")]
NUM = [c for c in FEAT if c not in CAT]
unseen_types = sorted(set(te["attack"]) - set(tr["attack"]))


def model(name):
    if name in ("LogisticRegression",):
        prep = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT), ("n", StandardScaler(), NUM)])
        clf = LogisticRegression(max_iter=3000, random_state=SEED)
    else:
        prep = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
                                  ("n", "passthrough", NUM)])
        clf = {"DecisionTree": DecisionTreeClassifier(random_state=SEED),
               "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
               "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED)}[name]
    return Pipeline([("prep", prep), ("clf", clf)])


out = {"official_split": {"train": len(tr), "test": len(te),
                          "test_attack_types_absent_from_train": unseen_types,
                          "test_rows_with_unseen_attack_types": int(te["attack"].isin(unseen_types).sum())},
       "random_80_20_inside_KDDTrain+ (paper protocol)": {}, "official_KDDTest+": {}}
r_tr, r_te = train_test_split(tr, test_size=0.20, stratify=tr["label"], random_state=SEED)
for name in ["LogisticRegression", "DecisionTree", "RandomForest", "HistGradientBoosting"]:
    m = model(name).fit(r_tr[FEAT], r_tr["label"])
    out["random_80_20_inside_KDDTrain+ (paper protocol)"][name] = detection_metrics(
        r_te["label"], m.predict_proba(r_te[FEAT])[:, 1])
    m = model(name).fit(tr[FEAT], tr["label"])
    p = m.predict_proba(te[FEAT])[:, 1]
    d = detection_metrics(te["label"], p)
    unseen = te["attack"].isin(unseen_types).values
    d["recall_on_attack_types_unseen_in_training"] = round(float((p[unseen] >= .5).mean()), 4)
    d["recall_on_attack_types_seen_in_training"] = round(float((p[(~unseen) & (te["label"].values == 1)] >= .5).mean()), 4)
    out["official_KDDTest+"][name] = d
    a = out["random_80_20_inside_KDDTrain+ (paper protocol)"][name]["accuracy"]
    print(f"{name:22s} random-split acc {a:.4f} | official acc {d['accuracy']:.4f} bal {d['balanced_accuracy']:.4f} "
          f"recall {d['attack_recall']:.4f} unseen-type recall {d['recall_on_attack_types_unseen_in_training']:.4f}")

save_json(out, "r05_nslkdd.json")
