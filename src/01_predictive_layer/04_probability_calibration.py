"""
Experiment E3 (Probability Calibration) - Supervisor Report Section 3.8.
Tests whether the Random Forest's predicted probabilities (MLS) are well-calibrated,
since DADE consumes this probability directly as a numeric input to Equation (1).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import json
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

categorical_cols = ["proto", "service", "state"]
drop_cols = ["id", "attack_cat", "label"]

combined = pd.concat([train_df[categorical_cols], test_df[categorical_cols]], axis=0)
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(combined[col].astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["label"]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df["label"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Uncalibrated Random Forest (as used throughout the paper) ----------
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
proba_uncalibrated = rf.predict_proba(X_test_scaled)[:, 1]

brier_uncalibrated = brier_score_loss(y_test, proba_uncalibrated)
frac_pos_unc, mean_pred_unc = calibration_curve(y_test, proba_uncalibrated, n_bins=10, strategy="quantile")

# ---------- Calibrated version (Platt scaling / sigmoid), for comparison ----------
rf_cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                                  method="sigmoid", cv=3)
rf_cal.fit(X_train_scaled, y_train)
proba_calibrated = rf_cal.predict_proba(X_test_scaled)[:, 1]

brier_calibrated = brier_score_loss(y_test, proba_calibrated)
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, proba_calibrated, n_bins=10, strategy="quantile")

# Expected Calibration Error (ECE) - manual computation with equal-width bins
def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1] if i < n_bins-1 else y_prob <= bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return ece

y_test_arr = y_test.values
ece_uncalibrated = compute_ece(y_test_arr, proba_uncalibrated)
ece_calibrated = compute_ece(y_test_arr, proba_calibrated)

print("="*70)
print("CALIBRATION RESULTS (Random Forest, UNSW-NB15 official test set)")
print("="*70)
print(f"\nUncalibrated (used throughout the paper so far):")
print(f"  Brier Score: {brier_uncalibrated:.4f}  (0=perfect, 0.25=uninformative)")
print(f"  ECE:         {ece_uncalibrated:.4f}")
print(f"\nCalibrated (Platt/sigmoid scaling):")
print(f"  Brier Score: {brier_calibrated:.4f}")
print(f"  ECE:         {ece_calibrated:.4f}")

results = {
    "uncalibrated": {
        "brier_score": round(float(brier_uncalibrated), 4),
        "ece": round(float(ece_uncalibrated), 4),
        "calibration_curve": {"mean_predicted": mean_pred_unc.tolist(), "fraction_positive": frac_pos_unc.tolist()},
    },
    "calibrated_platt": {
        "brier_score": round(float(brier_calibrated), 4),
        "ece": round(float(ece_calibrated), 4),
        "calibration_curve": {"mean_predicted": mean_pred_cal.tolist(), "fraction_positive": frac_pos_cal.tolist()},
    },
}
with open("/home/claude/collected_results/calibration_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved calibration_results.json")
