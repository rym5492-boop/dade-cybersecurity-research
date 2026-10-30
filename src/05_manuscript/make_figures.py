"""Figures of the revised IJCIP manuscript (30 October 2026).

Reads only stored result files in results/revision (never writes there) and computes nothing new. Style
follows src/04_revision/r24_figures_revision2.py (DejaVu Sans, no text below 7 pt, 7.2 in wide, 600 dpi PNG
and vector PDF with embedded TrueType fonts). Run from the repository root:
    python src/05_manuscript/make_figures.py [output folder]     (default: results/manuscript/figures)

  Fig. 1  decision workflow            copy of r24 fig6_workflow (a diagram; it plots no data)
  Fig. 2  DADE minus each baseline     r10_stage2.json, primary_paired_bootstrap_DADE_minus_baseline
  Fig. 3  threshold trade-off          copy of r24 fig7_threshold_tradeoff (r18_tradeoff_curves.csv)
  Fig. 4  ablation                     copy of r24 fig8_ablation (r19_ablation_and_paired.json)
  Fig. 6  error propagation            copy of r24 fig9_error_propagation (r20_error_propagation.json)
  Fig. 5  AHP group weights            r06_ahp_robustness.json (same content as r15 Figure 4)
  Fig. 7  cross-dataset transfer       r07_cross_dataset_v2.json
  Fig. 8  reliability diagram          r02_calibration.json, HistGradientBoosting reliability tables
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "revision"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED, ACCENT, ACCENT2, GRID = "#1f2933", "#7b8794", "#0b6e4f", "#c05621", "#e4e7eb"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def load(name):
    with open(RES / name) as f:
        return json.load(f)


# ------------------------------------------------------------------ Fig. 2 DADE minus each baseline (r10)
r10 = load("r10_stage2.json")
diff = r10["primary_paired_bootstrap_DADE_minus_baseline"]
labels = {"B1_ML_only": "B1 detector only", "B2_equal_weight": "B2 equal weights", "B3_BIA_only": "B3 BIA only",
          "B4_severity_x_impact": "B4 severity \u00d7 impact", "B5_ML_x_impact": "B5 detector \u00d7 impact"}
keys = list(labels)[::-1]
panels = [("quadratic_kappa", "Weighted kappa\n(> 0: DADE agrees better)"),
          ("mean_cost", "Mean cost\n(< 0: DADE costs less)"),
          ("high_recall", "High-priority recall\n(> 0: DADE finds more High)")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (m, title) in zip(axes, panels):
    for y, k in enumerate(keys):
        d = diff[k][m]["diff"]; lo, hi = diff[k][m]["ci95"]
        col = ACCENT if (lo > 0 or hi < 0) else MUTED
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(d, y, "o", color=col, ms=4.5, mec="white", mew=0.8, zorder=3)
        ax.text(max(hi, 0) + 0.01 * (hi - lo + 0.5), y + 0.24, f"{d:+.2f}".replace("-", "−"), ha="left", va="center", fontsize=7, color=INK)
    ax.axvline(0, color="#555555", lw=0.6, ls="--")
    lo_all = min(diff[k][m]["ci95"][0] for k in keys); hi_all = max(diff[k][m]["ci95"][1] for k in keys)
    ax.set_xlim(lo_all - 0.08 * (hi_all - lo_all), hi_all + 0.28 * (hi_all - lo_all))
    ax.set_title(title, fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
axes[0].set_yticks(range(len(keys)), [labels[k] for k in keys])
axes[0].set_ylim(-0.6, len(keys) - 0.4)
h1 = plt.Line2D([], [], color=ACCENT, lw=1.6, marker="o", mec="white", label="95% interval excludes 0")
h2 = plt.Line2D([], [], color=MUTED, lw=1.6, marker="o", mec="white", label="95% interval includes 0")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, -0.08))
fig.suptitle("DADE minus each method (95% cluster-bootstrap intervals)", fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig02_decision_differences")

# ------------------------------------------------------------------ Figs. 1, 3, 4, 6: already drawn by r24 (copied)
for src, dst in [("fig6_workflow", "fig01_workflow"), ("fig7_threshold_tradeoff", "fig03_threshold_tradeoff"), ("fig8_ablation", "fig04_ablation"),
                 ("fig9_error_propagation", "fig06_error_propagation")]:
    for ext in ("png", "pdf"):
        shutil.copyfile(RES / f"{src}.{ext}", OUT / f"{dst}.{ext}")

# ------------------------------------------------------------------ Fig. 6 AHP group weights (r06)
r6 = load("r06_ahp_robustness.json")
var = r6["aggregation_variants"]
w_all = var["AIJ_all_six (paper)"]["weights"]
w_no3 = next(v["weights"] for k, v in var.items() if k.startswith("AIJ_excluding_most_inconsistent"))
ci = r6["expert_bootstrap_AIJ"]["weight_95ci"]
names = {"TS": "Threat severity (TS)", "BI": "Business impact (BI)", "AC": "Asset criticality (AC)",
         "MLS": "Detection score (MLS)", "RTO": "Recovery time objective (RTO)",
         "RPO": "Recovery point objective (RPO)", "HI": "Incident history (HI)"}
order = sorted(w_all, key=lambda c: w_all[c])
fig, ax = plt.subplots(figsize=(5.6, 2.8))
for y, c in enumerate(order):
    lo, hi = ci[c]
    ax.plot([lo, hi], [y, y], color=ACCENT, lw=2.2, solid_capstyle="round", alpha=0.35)
    ax.plot(w_all[c], y, "o", color=ACCENT, ms=5, mec="white", mew=0.8, zorder=3)
    ax.plot(w_no3[c], y, "o", mfc="white", mec=INK, ms=4.2, mew=0.8, zorder=4)
    ax.text(hi + 0.006, y, f"{w_all[c]:.3f}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(order)), [names[c] for c in order])
ax.set_xlim(0, 0.42); ax.set_xlabel("Group weight (AIJ)")
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
h1 = plt.Line2D([], [], marker="o", ls="", color=ACCENT, mec="white", ms=5, label="All six experts (used)")
h2 = plt.Line2D([], [], color=ACCENT, lw=2.2, alpha=0.35, label="Expert-bootstrap 95% interval")
h3 = plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=4.2, label="Without E3")
ax.legend(handles=[h1, h2, h3], loc="lower right", frameon=False, fontsize=7.5)
fig.tight_layout()
save(fig, "fig05_ahp_weights")

# ------------------------------------------------------------------ Fig. 7 cross-dataset transfer (r07)
r7 = load("r07_cross_dataset_v2.json")["matrix"]
dsets = ["UNSW-NB15", "CICIDS2017", "CICIoT2023"]
dirs = [(s, t) for s in dsets for t in dsets if s != t]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
shades = {"threshold_from_1pct_target": ("D", "#ffffff", ACCENT2, "Threshold from 1% target labels"),
          "retrain_with_1pct_target": ("o", "#7fb8a4", ACCENT, "Retrain with 1% target labels"),
          "retrain_with_5pct_target": ("o", ACCENT, ACCENT, "Retrain with 5% target labels")}
for y, (s, t) in enumerate(dirs[::-1]):
    r = r7[s][t]
    base = r["no_adaptation"]["balanced_accuracy"]
    own = r7[t][t]["no_adaptation"]["balanced_accuracy"]
    vals = [base] + [r[k]["balanced_accuracy"] for k in shades]
    ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=1.2, zorder=1)
    ax.plot(base, y, "o", mfc="white", mec=INK, ms=5, mew=0.9, zorder=3)
    for k, (mk, fc, ec, _) in shades.items():
        ax.plot(r[k]["balanced_accuracy"], y, mk, mfc=fc, mec=ec, ms=4.6 if mk == "D" else 5.2, mew=0.9, zorder=4)
    ax.plot([own, own], [y - 0.28, y + 0.28], color=INK, lw=1.4, zorder=2)
    ax.text(base - 0.012, y + 0.3, f"{base:.2f}", ha="right", va="center", fontsize=7, color=MUTED)
    ax.text(r["retrain_with_5pct_target"]["balanced_accuracy"], y + 0.33,
            f"{r['retrain_with_5pct_target']['balanced_accuracy']:.2f}", ha="center", va="center", fontsize=7, color=ACCENT)
ax.axvline(0.5, color=MUTED, lw=0.6, ls="--", zorder=0)
ax.set_yticks(range(len(dirs)), [f"{s} \u2192 {t}" for s, t in dirs[::-1]])
ax.set_xlim(0.25, 1.0); ax.set_ylim(-0.6, len(dirs) - 0.3)
ax.set_xlabel("Balanced accuracy on the held-out 30% of the target dataset")
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
hs = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="No adaptation")]
hs += [plt.Line2D([], [], marker=mk, ls="", mfc=fc, mec=ec, label=lab) for mk, fc, ec, lab in shades.values()]
hs += [plt.Line2D([], [], color=INK, lw=1.4, label="Target's own within-dataset score")]
ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.42, 1.24), ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout()
save(fig, "fig07_transfer")

# ------------------------------------------------------------------ Fig. 8 reliability diagram (r02)
r2 = load("r02_calibration.json")["models"]["HistGradientBoosting"]["reliability_table"]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.7, label="Perfect calibration")
for key, col, mfc, lab in [("uncalibrated", MUTED, "#d9dde2", "HGB, uncalibrated"),
                           ("isotonic", ACCENT, "#7fb8a4", "HGB, isotonic map fitted on CAL")]:
    rows = [b for b in r2[key] if b["count"] > 0]
    xs = [b["mean_predicted"] for b in rows]; ys = [b["observed_attack_rate"] for b in rows]
    sz = [max(6, 140 * (b["count"] / 40406) ** 0.5) for b in rows]
    ax.plot(xs, ys, "-", color=col, lw=0.9, zorder=2)
    ax.scatter(xs, ys, s=sz, color=mfc, edgecolor=col, lw=0.6, zorder=3, label=lab)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Mean predicted probability of attack (15 equal-width bins)")
ax.set_ylabel("Observed share of attacks")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=7.5, markerscale=0.6)
fig.tight_layout()
save(fig, "fig08_reliability")
print("figures written to", OUT)
