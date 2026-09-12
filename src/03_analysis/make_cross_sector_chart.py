import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sectors = ["Healthcare\nprofile", "Finance\nprofile", "Equal-weight\n(baseline)"]
high = [48, 44, 50]
medium = [150, 151, 141]
low = [2, 5, 9]

x = np.arange(len(sectors))
width = 0.5

fig, ax = plt.subplots(figsize=(7, 4.6), dpi=600)
ax.bar(x, high, width, label="High", color="#c0392b")
ax.bar(x, medium, width, bottom=high, label="Medium", color="#e67e22")
ax.bar(x, low, width, bottom=np.array(high)+np.array(medium), label="Low", color="#27ae60")

ax.set_ylabel("Number of incidents (out of 200)", fontsize=11)
ax.set_title("Priority-Level Distribution Under Different Sector Weight Profiles\n(identical set of 200 technical incidents)", fontsize=10.5)
ax.legend(loc="upper right", fontsize=9, frameon=False, title="Priority Level")
ax.set_xticks(x)
ax.set_xticklabels(sectors, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/home/claude/collected_results/cross_sector_chart.png", dpi=600, bbox_inches="tight")
print("Saved")
