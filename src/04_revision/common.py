"""
Shared utilities for the revision experiments (supervisor review, October 2026).

Design rules applied in every revision script:
  * Only the official UNSW-NB15 training file is used for fitting. It is split, stratified by
    attack_cat, into FIT (70%), VAL (15%) and CAL (15%). The official testing file is touched
    only for final evaluation.
  * All preprocessing (encoding, scaling) lives inside an sklearn Pipeline and is fitted on FIT
    only. No encoder is fitted on test data.
  * Decision thresholds and calibrators are chosen on VAL / CAL, never on the test file.
  * Every metric table reports prevalence and a majority-class baseline next to accuracy.

Run every script from the repository root, e.g.
    python src/04_revision/r01_unsw_detection.py
UNSW-NB15 CSVs are looked up in the repository root or in data/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "revision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "attack_cat", "label"]

# Same severity convention as the original paper (Section 5.2 / 01_ablation_study.py).
SEVERITY_MAP = {"Normal": 0.00, "Fuzzers": 0.35, "Reconnaissance": 0.40, "Analysis": 0.55,
                "DoS": 0.70, "Backdoor": 0.85, "Exploits": 0.80, "Shellcode": 0.90,
                "Worms": 0.95, "Generic": 0.60}


def find_file(name: str) -> Path:
    for c in (REPO_ROOT / name, REPO_ROOT / "data" / name, Path(name)):
        if c.exists():
            return c
    raise FileNotFoundError(f"{name} not found in repository root or data/.")


def load_unsw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(find_file("UNSW_NB15_training-set.csv"))
    test = pd.read_csv(find_file("UNSW_NB15_testing-set.csv"))
    assert len(train) == 175341 and len(test) == 82332, "Unexpected UNSW-NB15 file sizes"
    for df in (train, test):
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    return train, test


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def split_fit_val_cal(train: pd.DataFrame, seed: int = SEED):
    """70/15/15 split of the official training file, stratified by attack category."""
    fit, rest = train_test_split(train, test_size=0.30, stratify=train["attack_cat"], random_state=seed)
    val, cal = train_test_split(rest, test_size=0.50, stratify=rest["attack_cat"], random_state=seed)
    return fit, val, cal


def tree_preprocessor(cols: list[str]) -> ColumnTransformer:
    num = [c for c in cols if c not in CAT_COLS]
    return ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
         ("num", "passthrough", num)])


def linear_preprocessor(cols: list[str]) -> ColumnTransformer:
    num = [c for c in cols if c not in CAT_COLS]
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_COLS),
         ("num", StandardScaler(), num)])


def detection_metrics(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prev = float(y.mean())
    return {
        "n": int(len(y)), "prevalence_attack": round(prev, 4),
        "majority_baseline_accuracy": round(max(prev, 1 - prev), 4),
        "threshold": round(float(thr), 4),
        "accuracy": round(accuracy_score(y, yhat), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, yhat), 4),
        "macro_f1": round(f1_score(y, yhat, average="macro"), 4),
        "attack_precision": round(precision_score(y, yhat, zero_division=0), 4),
        "attack_recall": round(recall_score(y, yhat, zero_division=0), 4),
        "attack_f1": round(f1_score(y, yhat, zero_division=0), 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "mcc": round(matthews_corrcoef(y, yhat), 4),
        "roc_auc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
        "pr_auc": round(average_precision_score(y, p), 4) if len(set(y)) > 1 else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_max_f1(y, p) -> float:
    """Threshold maximising attack F1 on a validation set (never the test set)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def threshold_at_fpr(y, p, target_fpr: float) -> float:
    """Smallest threshold whose validation false-positive rate is <= target_fpr."""
    y = np.asarray(y).astype(int)
    benign = np.sort(np.asarray(p)[y == 0])
    k = int(np.floor((1 - target_fpr) * len(benign)))
    k = min(max(k, 0), len(benign) - 1)
    return float(np.nextafter(benign[k], 1.0))


def ece(y, p, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def bootstrap_ci(stat, *arrays, n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(stat(*[np.asarray(a)[i] for a in arrays]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return [round(float(lo), 4), round(float(hi), 4)]


def prior_shift(p, pi_train: float, pi_new: float):
    """Bayes prior-shift correction of a calibrated posterior (Saerens et al., 2002)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    a = (pi_new / pi_train) * p
    b = ((1 - pi_new) / (1 - pi_train)) * (1 - p)
    return a / (a + b)


def save_json(obj, name: str) -> Path:
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Saved {path.relative_to(REPO_ROOT)}")
    return path
