"""
Preliminary predictive-layer experiment on NSL-KDD (paper Section 5.1, Table 1, Figures 2-4).

NOTE: this is a re-implementation of the original preliminary run from the protocol stated in
the paper. With scikit-learn 1.8 it reproduces Table 1 within 0.3 percentage points
(RF 0.9990 vs 0.9992; DT 0.9977 vs 0.9979; LR 0.9520 vs 0.9548) and the same top-2
features (src_bytes, dst_bytes). Small differences come from library versions and
unreported default settings of the original run; no conclusion of the paper depends on them.

Data   : KDDTrain+.txt (125,973 records: 67,343 normal / 58,630 attack), official NSL-KDD release.
Label  : binary (normal = 0, any attack = 1).
Prep   : LabelEncoder on protocol_type, service, flag; StandardScaler (fit on training data only).
Split  : stratified 80/20 train/test, random_state = 42.
Models : Random Forest, Decision Tree, Logistic Regression, XGBoost (random_state = 42).
CV     : 5-fold StratifiedKFold (shuffle, random_state = 42) on the training set.
Output : results/nsl_kdd_reimplementation_results.json, results/nsl_kdd_rf_confusion_matrix.json,
         results/nsl_kdd_rf_feature_importance.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

df = pd.read_csv("KDDTrain+.txt", names=COLS)
y = (df["label"] != "normal").astype(int)
print(f"Records: {len(df)}  normal: {(y == 0).sum()}  attack: {(y == 1).sum()}")
X = df.drop(columns=["label", "difficulty"])
for c in ["protocol_type", "service", "flag"]:
    X[c] = LabelEncoder().fit_transform(X[c])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
except ImportError:
    print("xgboost not installed -- XGBoost skipped (pip install xgboost)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {}
for name, m in models.items():
    m.fit(X_tr_s, y_tr); p = m.predict(X_te_s)
    cvs = cross_val_score(m, X_tr_s, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
    res[name] = {k: round(float(v), 4) for k, v in {
        "accuracy": accuracy_score(y_te, p),
        "precision": precision_score(y_te, p, average="weighted"),
        "recall": recall_score(y_te, p, average="weighted"),
        "f1_score": f1_score(y_te, p, average="weighted"),
        "cv_accuracy": cvs.mean()}.items()}
    print(name, res[name])
    if name == "Random Forest":
        cm = confusion_matrix(y_te, p, labels=[1, 0])  # rows/cols: Attack, Normal (as in Figure 3)
        imp = sorted(zip(X.columns, m.feature_importances_), key=lambda t: -t[1])[:10]
        json.dump({"labels": ["Attack", "Normal"], "matrix": cm.tolist()},
                  open("nsl_kdd_rf_confusion_matrix.json", "w"), indent=2)
        json.dump([{"feature": f, "gini_importance": round(float(v), 4)} for f, v in imp],
                  open("nsl_kdd_rf_feature_importance.json", "w"), indent=2)
        print("RF confusion matrix [Attack, Normal]:", cm.tolist())
        print("Top-10 features:", [f for f, _ in imp])
json.dump(res, open("nsl_kdd_reimplementation_results.json", "w"), indent=2)
print("Saved nsl_kdd_reimplementation_results.json")
