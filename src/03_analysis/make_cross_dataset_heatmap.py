import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

with open("cross_dataset_results.json") as f:
    r = json.load(f)

datasets = ["UNSW-NB15", "CICIDS2017", "CICIoT2023"]
matrix = np.zeros((3, 3))
for i, train in enumerate(datasets):
    for j, test in enumerate(datasets):
        matrix[i, j] = r[train][test]["accuracy"]

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=600)
im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        color = "white" if matrix[i, j] < 0.5 or matrix[i, j] > 0.85 else "black"
        marker = " *" if i == j else ""
        ax.text(j, i, f"{matrix[i,j]:.1%}{marker}", ha="center", va="center",
                 color=color, fontsize=12, fontweight="bold")

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(datasets, fontsize=10)
ax.set_yticklabels(datasets, fontsize=10)
ax.set_xlabel("Tested On", fontsize=11)
ax.set_ylabel("Trained On", fontsize=11)
ax.set_title("Cross-Dataset Generalization (Random Forest,\nharmonized 4-feature set)\n* = within-dataset (not cross-dataset)", fontsize=10.5)
plt.colorbar(im, ax=ax, label="Accuracy", fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("cross_dataset_heatmap.png", dpi=600, bbox_inches="tight")
print("Saved")
