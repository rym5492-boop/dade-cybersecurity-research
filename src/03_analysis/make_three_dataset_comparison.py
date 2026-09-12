import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

with open("unsw_results.json") as f:
    unsw = json.load(f)
with open("cicids2017_results.json") as f:
    cicids = json.load(f)
with open("ciciot2023_results.json") as f:
    ciciot = json.load(f)

# Use the best model (HistGradientBoosting) from each dataset for comparison
datasets = ["UNSW-NB15\n(official split)", "CICIDS2017\n(random split)", "CICIoT2023\n(random split)"]
best_model = "HistGradientBoosting"

accuracy = [unsw[best_model]["accuracy"], cicids[best_model]["accuracy"], ciciot[best_model]["accuracy"]]
f1 = [unsw[best_model]["f1_score"], cicids[best_model]["f1_score"], ciciot[best_model]["f1_score"]]
roc_auc = [unsw[best_model]["roc_auc"], cicids[best_model]["roc_auc"], ciciot[best_model]["roc_auc"]]
fpr = [unsw[best_model]["false_positive_rate"], cicids[best_model]["false_positive_rate"], ciciot[best_model]["false_positive_rate"]]

x = np.arange(len(datasets))
width = 0.2

fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=600)
ax.bar(x - 1.5*width, accuracy, width, label="Accuracy", color="#2c5f8a")
ax.bar(x - 0.5*width, f1, width, label="F1-Score", color="#4f8fbf")
ax.bar(x + 0.5*width, roc_auc, width, label="ROC-AUC", color="#7fb3d9")
ax.bar(x + 1.5*width, fpr, width, label="False Positive Rate", color="#e67e22")

ax.set_ylabel("Score", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9.5)
ax.set_title(f"Best Model ({best_model}) Performance Across Three Modern Datasets", fontsize=10.5)
ax.legend(loc="upper left", fontsize=8.5, ncol=2, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("three_dataset_comparison.png", dpi=600, bbox_inches="tight")
print("Saved three_dataset_comparison.png")
