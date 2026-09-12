import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

with open("/home/claude/collected_results/calibration_results.json") as f:
    r = json.load(f)

fig, ax = plt.subplots(figsize=(6.5, 6), dpi=600)

# Perfect calibration reference line
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

unc = r["uncalibrated"]["calibration_curve"]
cal = r["calibrated_platt"]["calibration_curve"]

ax.plot(unc["mean_predicted"], unc["fraction_positive"], marker="o", color="#2c5f8a",
         linewidth=2, markersize=7, label=f"Uncalibrated RF (Brier={r['uncalibrated']['brier_score']})")
ax.plot(cal["mean_predicted"], cal["fraction_positive"], marker="s", color="#e67e22",
         linewidth=2, markersize=7, label=f"Platt-calibrated RF (Brier={r['calibrated_platt']['brier_score']})")

ax.set_xlabel("Mean predicted probability (MLS)", fontsize=11)
ax.set_ylabel("Observed fraction of actual attacks", fontsize=11)
ax.set_title("Calibration Curve: Random Forest on UNSW-NB15\n(official train/test split)", fontsize=11)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig("/home/claude/collected_results/calibration_curve.png", dpi=600, bbox_inches="tight")
print("Saved")
