"""
Ablation study chart (Section 5.5, Figure 10).

Reads results/ablation_results.json (produced by 01_ablation_study.py) so the
chart always reflects whichever conditions were actually run -- (A)/(B)/(C),
or (A)/(B)/(C)/(D) once the expert-weighted condition has been computed with
the real AHP group weights (Section 4.3).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/ablation_results.json") as f:
    results = json.load(f)

LABELS = {
    "A_ML_only": "(A)\nML-only",
    "B_BusinessContext_only": "(B)\nBusiness-context\nonly",
    "C_EqualWeight_DADE": "(C)\nEqual-weight\nDADE",
    "D_ExpertWeighted_DADE": "(D)\nExpert-weighted\nDADE",
}
ORDER = ["A_ML_only", "B_BusinessContext_only", "C_EqualWeight_DADE", "D_ExpertWeighted_DADE"]
present = [k for k in ORDER if k in results]

conditions = [LABELS[k] for k in present]
accuracy = [results[k]["accuracy_vs_ground_truth"] for k in present]
kappa = [results[k]["cohen_kappa_vs_ground_truth"] for k in present]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.3), dpi=600)
bars1 = ax.bar(x - width/2, accuracy, width, label="Accuracy vs. ground truth", color="#2c5f8a")
bars2 = ax.bar(x + width/2, kappa, width, label="Cohen's Kappa", color="#e67e22")

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax.text(len(conditions) - 0.45, 0.51, "chance level", fontsize=7.5, color="gray")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=9.5)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(-0.1, 1.18)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_title("Ablation Study: Attack-Detection Agreement by Decision Condition\n(n=200 UNSW-NB15 test instances)", fontsize=10.5)
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("ablation_chart.png", dpi=600, bbox_inches="tight")
print(f"Saved ablation_chart.png ({len(conditions)} conditions: {', '.join(present)})")
