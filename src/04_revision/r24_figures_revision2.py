"""
R24 - Figures of the second revision, drawn from the result files only (computes nothing new):
  fig6_workflow.png/.pdf          decision workflow from alert to human action, with the data used at each step
  fig7_threshold_tradeoff.png/.pdf High recall, escalation rate and mean cost against the High cut-off (r18)
  fig8_ablation.png/.pdf          change in mean cost and kappa when each input is removed (r19)
  fig9_error_propagation.png/.pdf share of decisions changed by detector and BIA uncertainty (r20)
600 dpi PNG and vector PDF.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from common import OUT_DIR

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType fonts embedded in the PDFs; no text below 7 pt
INK, MUTED, ACCENT, ACCENT2, GRID = "#1f2933", "#7b8794", "#0b6e4f", "#c05621", "#e4e7eb"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6: workflow
steps = [("1  Alert on\na network flow", "IDS or flow sensor"),
         ("2  Attack probability\n(MLS) and predicted\ncategory", "HGB on FIT, isotonic\nmap on CAL; category\nclassifier, no true label"),
         ("3  Service linkage", "flow \u2192 asset \u2192 service\n(asset inventory\nor CMDB)"),
         ("4  BIA record\nof the service", "AC, BI, RTO, RPO: BIA\nregister; HI: incident log\n(hypothetical here)"),
         ("5  Weighted score\n(DPS)", "AHP group weights;\nseverity of the\npredicted category"),
         ("6  Priority level", "cut-offs fixed in\nadvance, or tuned on\nlabeled cases"),
         ("7  Human decision\nand action", "analyst approves or\noverrides; reason\nlogged; playbook")]
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.set_xlim(0, 4); ax.set_ylim(0, 2.4); ax.axis("off")
pos = [(i, 1.25) for i in range(4)] + [(i + 0.5, 0.1) for i in range(3)]
evaluated = {1, 3, 4, 5}
for i, ((title, note), (x, y)) in enumerate(zip(steps, pos)):
    ax.add_patch(FancyBboxPatch((x + 0.04, y), 0.92, 0.9, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc="#e6f4ee" if i in evaluated else "#f5f7fa", ec=INK, lw=0.6))
    ax.text(x + 0.5, y + 0.67, title, ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold", linespacing=1.15)
    ax.text(x + 0.5, y + 0.25, note, ha="center", va="center", fontsize=7, color="#3e4c59", linespacing=1.15)
arrow = dict(arrowstyle="->", color=INK, lw=0.8)
for i in range(3):
    ax.annotate("", xy=(i + 1.04, 1.7), xytext=(i + 0.96, 1.7), arrowprops=arrow)
for i in range(2):
    ax.annotate("", xy=(i + 1.54, 0.55), xytext=(i + 1.46, 0.55), arrowprops=arrow)
ax.plot([3.5, 3.5, 0.55], [1.25, 1.125, 1.125], color=INK, lw=0.8, solid_capstyle="butt")
ax.annotate("", xy=(0.55, 1.0), xytext=(0.55, 1.13), arrowprops=arrow)
ax.text(2.0, 2.31, "Green: evaluated offline in this study on 30 scenarios.  Grey: specified for deployment, not field-tested.",
        ha="center", fontsize=7.5, color=INK)
save(fig, "fig6_workflow")

# ---------------------------------------------------------------- Figure 7: threshold trade-off
T = pd.read_csv(OUT_DIR / "r18_tradeoff_curves.csv")
names = {"DADE_AHP": "DADE (AHP weights)", "B1_ML_only": "B1 detector only", "B2_equal_weight": "B2 equal weights",
         "B3_BIA_only": "B3 BIA only"}
styles = {"DADE_AHP": (ACCENT, "-"), "B1_ML_only": ("#9f7aea", "--"), "B2_equal_weight": (INK, ":"), "B3_BIA_only": (MUTED, "-.")}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
for m, d in T.groupby("method"):
    c, ls = styles[m]
    for ax, col in zip(axs, ["high_recall", "escalation_rate", "mean_cost"]):
        ax.plot(d["high_cutoff"], d[col], ls, color=c, lw=1.1, label=names[m])
b5 = json.load(open(OUT_DIR / "r19_ablation_and_paired.json"))["A_metrics_with_ci"]["B5_ML_x_impact"]
for ax, col in zip(axs, ["high_recall", "escalation_rate", "mean_cost"]):
    ax.axhline(b5[col]["value"], color=ACCENT2, lw=0.9, ls=(0, (1, 1)), label="B5 detector \u00d7 impact (fixed)" if col == "high_recall" else None)
for ax, t in zip(axs, ["High-priority recall", "Share of alerts escalated to High", "Mean misprioritization cost"]):
    ax.set_title(t, fontsize=8); ax.axvline(0.66, color=GRID, lw=2, zorder=0); ax.set_xlabel("High cut-off")
    ax.grid(axis="y", color=GRID, lw=0.5)
axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "fig7_threshold_tradeoff")

# ---------------------------------------------------------------- Figure 8: ablation
r19 = json.load(open(OUT_DIR / "r19_ablation_and_paired.json"))["B_ablations"]
order = ["equal_weights", "without_MLS", "without_TS", "without_detector_inputs_MLS_TS", "without_AC", "without_BI",
         "without_HI", "without_RTO", "without_RPO", "without_all_BIA"]
lab = {"equal_weights": "AHP replaced by equal weights", "without_MLS": "without MLS", "without_TS": "without TS",
       "without_detector_inputs_MLS_TS": "without MLS and TS", "without_AC": "without AC", "without_BI": "without BI",
       "without_HI": "without HI", "without_RTO": "without RTO", "without_RPO": "without RPO", "without_all_BIA": "without all five BIA criteria"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, key, title in zip(axs, ["mean_cost", "quadratic_kappa"], ["Mean cost: full DADE minus variant\n(< 0: removing the input raises cost)", "Quadratic kappa: full DADE minus variant\n(> 0: removing the input lowers kappa)"]):
    for i, k in enumerate(order):
        d = r19[k]["paired_full_minus_variant"][key]
        ax.errorbar(d["diff"], i, xerr=[[d["diff"] - d["ci95"][0]], [d["ci95"][1] - d["diff"]]], fmt="o", color=ACCENT, ms=3, lw=0.8, capsize=1.5)
    ax.axvline(0, color=MUTED, lw=0.6); ax.set_title(title, fontsize=8); ax.grid(axis="x", color=GRID, lw=0.5)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([lab[k] for k in order], fontsize=7.5); axs[0].invert_yaxis()
fig.tight_layout()
save(fig, "fig8_ablation")

# ---------------------------------------------------------------- Figure 9: error propagation
r20 = json.load(open(OUT_DIR / "r20_error_propagation.json"))
meths = ["DADE_AHP", "B1_ML_only", "B2_equal_weight", "B3_BIA_only", "B4_severity_x_impact", "B5_ML_x_impact"]
short = ["DADE", "B1", "B2", "B3", "B4", "B5"]
fig, ax = plt.subplots(figsize=(4.8, 2.4))
w = 0.26
for j, (exp, lbl, c) in enumerate([("D_detector", "detector inputs", ACCENT), ("B_bia", "BIA record", ACCENT2), ("DB_both", "both", INK)]):
    vals = [r20[exp][m]["share_of_decisions_changed"]["mean"] for m in meths]
    lo = [v - r20[exp][m]["share_of_decisions_changed"]["p2.5"] for v, m in zip(vals, meths)]
    hi = [r20[exp][m]["share_of_decisions_changed"]["p97.5"] - v for v, m in zip(vals, meths)]
    ax.bar([i + (j - 1) * w for i in range(len(meths))], vals, w, yerr=[lo, hi], color=c, label=lbl, error_kw=dict(lw=0.6, capsize=1.5))
ax.set_xticks(range(len(meths))); ax.set_xticklabels(short)
ax.set_ylabel("Share of the 30 decisions changed"); ax.legend(frameon=False, fontsize=7.5); ax.grid(axis="y", color=GRID, lw=0.5)
fig.tight_layout()
save(fig, "fig9_error_propagation")
print("figures written")
