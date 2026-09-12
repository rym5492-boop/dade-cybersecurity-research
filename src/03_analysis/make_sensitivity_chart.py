import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pert_df = pd.read_csv("sensitivity_weights.csv")
thresh_df = pd.read_csv("sensitivity_thresholds.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=600)

# --- Left: weight sensitivity by criterion (20% perturbation) ---
pert20 = pert_df[pert_df["perturbation"].str.contains("20%")]
grouped = pert20.groupby("criterion")["pct_decisions_changed"].mean().sort_values()
axes[0].barh(grouped.index, grouped.values, color="#2c5f8a")
axes[0].set_xlabel("% of decisions changed", fontsize=10)
axes[0].set_title("Weight Sensitivity (\u00b120% per criterion)", fontsize=10.5)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# --- Right: threshold sensitivity ---
axes[1].bar(thresh_df["threshold_perturbation"], thresh_df["pct_decisions_changed"], color="#e67e22")
axes[1].set_xlabel("Threshold perturbation", fontsize=10)
axes[1].set_ylabel("% of decisions changed", fontsize=10)
axes[1].set_title("Threshold Sensitivity", fontsize=10.5)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("sensitivity_chart.png", dpi=600, bbox_inches="tight")
print("Saved sensitivity_chart.png")
