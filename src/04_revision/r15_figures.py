"""
R15 - Figures 4 and 5 of the revised paper, drawn only from existing result files (no new analysis).

Figure 4: group AHP weights (AIJ, all six experts) with expert-bootstrap 95% intervals, and the
          weights without E3 for comparison (r06_ahp_robustness.json).
Figure 5: DADE minus each baseline, quadratic-weighted kappa and mean misprioritization cost, with
          95% cluster-bootstrap intervals over scenarios (r10_stage2.json, primary leave-one-expert-out).

Output: results/revision/fig4_ahp_weights.png, results/revision/fig5_decision_differences.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT_DIR

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.6, "xtick.color": "#444444", "ytick.color": "#222222"})
INK, ACCENT, MUTED, GRID = "#222222", "#1f4e99", "#9a9a9a", "#e6e6e6"
CM = 1 / 2.54

# ---------- Figure 4
r6 = json.load(open(OUT_DIR / "r06_ahp_robustness.json"))
var = r6["aggregation_variants"]
w_all = var["AIJ_all_six (paper)"]["weights"]
w_no3 = next(v["weights"] for k, v in var.items() if k.startswith("AIJ_excluding_most_inconsistent"))
ci = r6["expert_bootstrap_AIJ"]["weight_95ci"]
names = {"TS": "Threat severity (TS)", "BI": "Business impact (BI)", "AC": "Asset criticality (AC)",
         "MLS": "Detection score (MLS)", "RTO": "Recovery time objective (RTO)",
         "RPO": "Recovery point objective (RPO)", "HI": "Incident history (HI)"}
order = sorted(w_all, key=lambda c: w_all[c])            # smallest at the bottom
fig, ax = plt.subplots(figsize=(16 * CM, 7.2 * CM))
for y, c in enumerate(order):
    lo, hi = ci[c]
    ax.plot([lo, hi], [y, y], color=ACCENT, lw=2, solid_capstyle="round", alpha=0.35)
    ax.plot(w_all[c], y, "o", color=ACCENT, ms=7, mec="white", mew=1.2, zorder=3)
    ax.plot(w_no3[c], y, "o", mfc="white", mec=INK, ms=6, mew=1.0, zorder=4)
    ax.text(hi + 0.008, y, f"{w_all[c]:.3f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(order)), [names[c] for c in order])
ax.set_xlim(0, 0.42)
ax.set_xlabel("Group weight (AIJ)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h1 = plt.Line2D([], [], marker="o", ls="", color=ACCENT, mec="white", ms=7, label="All six experts (used)")
h2 = plt.Line2D([], [], color=ACCENT, lw=2, alpha=0.35, label="Expert-bootstrap 95% interval")
h3 = plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=6, label="Without E3")
ax.legend(handles=[h1, h2, h3], loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_ahp_weights.png", dpi=300)
plt.close(fig)

# ---------- Figure 5
r10 = json.load(open(OUT_DIR / "r10_stage2.json"))
diff = r10["primary_paired_bootstrap_DADE_minus_baseline"]
labels = {"B1_ML_only": "B1 detector only", "B2_equal_weight": "B2 equal weights", "B3_BIA_only": "B3 BIA only",
          "B4_severity_x_impact": "B4 severity × impact", "B5_ML_x_impact": "B5 detector × impact"}
keys = list(labels)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.6 * CM), sharey=True)
panels = [("quadratic_kappa", "DADE − method, weighted κ\n(> 0: DADE agrees better)"),
          ("mean_cost", "DADE − method, mean cost\n(< 0: DADE costs less)")]
for ax, (m, title) in zip(axes, panels):
    for y, k in enumerate(keys):
        d = diff[k][m]["diff"]; lo, hi = diff[k][m]["ci95"]
        sig = lo > 0 or hi < 0
        col = ACCENT if sig else MUTED
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(d, y, "o", color=col, ms=7, mec="white", mew=1.2, zorder=3)
        ax.text(hi, y, f"  {d:+.2f}".replace("-", "\u2212"), ha="left", va="center", fontsize=8.5, color=INK, bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    lo_all = min(diff[k][m]["ci95"][0] for k in keys); hi_all = max(diff[k][m]["ci95"][1] for k in keys)
    ax.set_xlim(lo_all - 0.08 * (hi_all - lo_all), hi_all + 0.30 * (hi_all - lo_all))
    ax.set_xlabel(title, fontsize=9.5)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(keys)), [labels[k] for k in keys])
axes[0].set_ylim(-0.6, len(keys) - 0.4)
h1 = plt.Line2D([], [], color=ACCENT, lw=2, marker="o", mec="white", label="95% interval excludes 0")
h2 = plt.Line2D([], [], color=MUTED, lw=2, marker="o", mec="white", label="95% interval includes 0")
fig.legend(handles=[h1, h2], loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.6, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT_DIR / "fig5_decision_differences.png", dpi=300)
plt.close(fig)
print("saved fig4_ahp_weights.png and fig5_decision_differences.png")
