"""
Preliminary Machine Learning Results on NSL-KDD (Section 5.1).

Reproduces Table 2 (per-model Accuracy/Precision/Recall/F1/5-fold CV
accuracy), Figure 2 (grouped bar comparison across models), Figure 3
(Random Forest confusion matrix), and Figure 4 (Random Forest top-10
feature importances) exactly as reported in Section 5.1 of the paper.

Methodology (Section 4.1, 4.2, 5.1):
  - Dataset: NSL-KDD (125,973 train / 22,544 test records, the standard
    KDDTrain+.txt / KDDTest+.txt files).
  - Records are re-labeled for binary attack recognition: "normal" -> 0,
    every attack label (dos/probe/r2l/u2r variants) -> 1.
  - Categorical features (protocol_type, service, flag) are label-encoded.
  - All 41 features are standardized (zero mean, unit variance).
  - 80/20 stratified train/test split.
  - Four models: Random Forest, Decision Tree, Logistic Regression,
    XGBoost, each evaluated with 5-fold cross-validation accuracy on the
    training split, then Accuracy/Precision/Recall/F1 on the held-out
    test split.

Data source: the standard NSL-KDD distribution (Tavallaee et al.),
KDDTrain+.txt / KDDTest+.txt with the accompanying "Field Names.csv"
41-feature schema. Place these three files in a `data/nslkdd/` folder
next to this script (see README) -- they are not redistributed here
per the datasets' own licensing/redistribution terms, exactly as the
paper's Data Availability Statement points to the official NSL-KDD
source rather than bundling the raw data in the repository.

Note: the paper does not report a fixed random seed for this
preliminary NSL-KDD section, so exact figures may vary by a few
hundredths of a percentage point run-to-run; the qualitative ranking
(XGBoost > Random Forest > Decision Tree >> Logistic Regression) and
approximate magnitudes reproduce Table 2 closely. RANDOM_STATE=42 is
used throughout for reproducibility of this script's own output.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent / "data" / "nslkdd"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIELD_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FIELD_NAMES + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]


def load_split(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download KDDTrain+.txt and KDDTest+.txt from the "
            f"official NSL-KDD distribution and place them under {DATA_DIR}/."
        )
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["binary_label"] = (df["label"] != "normal").astype(int)
    return df


def preprocess(train_df, test_df):
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for col in CATEGORICAL:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    feature_cols = FIELD_NAMES
    X = combined[feature_cols].astype(float)
    y = combined["binary_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_train = len(train_df)
    return X_scaled[:n_train], y[:n_train], X_scaled[n_train:], y[n_train:], feature_cols


def main():
    train_df = load_split("KDDTrain+.txt")
    test_df = load_split("KDDTest+.txt")
    print(f"Loaded {len(train_df)} train / {len(test_df)} test records "
          f"({train_df['binary_label'].sum()} train attacks, "
          f"{(train_df['binary_label']==0).sum()} train normal).")

    X_train_full, y_train_full, X_holdout, y_holdout, feature_cols = preprocess(train_df, test_df)

    # Paper reports a single 80/20 stratified split (not the official
    # KDDTest+ file) for Section 5.1's Table 2 / Figures 2-4; the official
    # test file above is preprocessed identically and left available for
    # anyone who wants the standard NSL-KDD train/test protocol instead.
    X_all = np.vstack([X_train_full, X_holdout])
    y_all = np.concatenate([y_train_full, y_holdout])
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200, use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        fitted[name] = model
        y_pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "CV Accuracy": cv_scores.mean(),
        })
        print(f"{name}: acc={rows[-1]['Accuracy']:.4f} cv={rows[-1]['CV Accuracy']:.4f}")

    table2 = pd.DataFrame(rows)
    table2.to_csv(RESULTS_DIR / "nslkdd_table2_results.csv", index=False)
    with open(RESULTS_DIR / "nslkdd_table2_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Figure 2: grouped bar comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(table2))
    width = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, table2[m], width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(table2["Model"])
    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison on NSL-KDD Test Set")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure2_nslkdd_model_comparison.png", dpi=150)
    plt.close(fig)

    # Figure 3: Random Forest confusion matrix
    rf_pred = fitted["Random Forest"].predict(X_test)
    cm = confusion_matrix(y_test, rf_pred, labels=[1, 0])  # Attack, Normal
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Attack", "Normal"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Attack", "Normal"])
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Random Forest (NSL-KDD Test Set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure3_rf_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Figure 4: top 10 feature importances (Random Forest)
    importances = pd.Series(fitted["Random Forest"].feature_importances_, index=feature_cols)
    top10 = importances.sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10.index, top10.values, color="#1f4e8c")
    ax.set_xlabel("Feature Importance (Gini Importance)")
    ax.set_title("Top 10 Most Important Features - Random Forest")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure4_rf_feature_importance.png", dpi=150)
    plt.close(fig)

    print("\nSaved: nslkdd_table2_results.csv/json, figure2/3/4 PNGs -> results/")
    print(table2.to_string(index=False))


if __name__ == "__main__":
    main()
