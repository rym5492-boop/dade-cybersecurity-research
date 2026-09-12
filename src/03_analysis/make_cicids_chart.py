import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["Logistic\nRegression", "Random\nForest", "HistGradient\nBoosting"]
accuracy = [0.9518, 0.9981, 0.9990]
precision = [0.8704, 0.9969, 0.9959]
recall = [0.8394, 0.9917, 0.9979]
f1 = [0.8546, 0.9943, 0.9969]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=600)
ax.bar(x - 1.5*width, accuracy, width, label="Accuracy", color="#2c5f8a")
ax.bar(x - 0.5*width, precision, width, label="Precision", color="#4f8fbf")
ax.bar(x + 0.5*width, recall, width, label="Recall", color="#7fb3d9")
ax.bar(x + 1.5*width, f1, width, label="F1-Score", color="#b3d4e8")

ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0.8, 1.005)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_title("Model Performance on CICIDS2017 (400,000-instance stratified subsample)", fontsize=10.5)
ax.legend(loc="lower right", fontsize=9, ncol=2, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("cicids2017_chart.png", dpi=600, bbox_inches="tight")
print("Saved")
