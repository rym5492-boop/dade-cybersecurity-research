"""Supplementary material (Tables S1 to S10, with parts a to e) generated from the stored result files only.

Every number is read from repo_filtered/results/revision/*.json, from results/ahp_weights_results.json (September
AHP answers, earlier analysis) or from the coverage-search files in results/manuscript; values are rounded half up. Output: paper2/supplementary/Alenezi_IJCIP_Supplementary.docx
"""
import csv
import json
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]          # repository root (script in src/05_manuscript)
RES = ROOT / "results" / "revision"
MAN = ROOT / "results" / "manuscript"
OUTFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else MAN / "Alenezi_IJCIP_Supplementary.docx"
J = lambda n: json.load(open(RES / n))


def r(x, nd=2):
    if x is None:
        return ""
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pct(x, nd=1):
    return r(100 * x, nd) + "%"


def rng(lo, hi, nd=2):
    return f"{r(lo, nd)} to {r(hi, nd)}"


doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, m, Cm(1.8))

doc.add_heading("Supplementary material", level=0)
FINAL_COMMIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
doc.add_paragraph("Intrusion detection and business impact analysis for continuity prioritization: "
                  "a leakage-audited evaluation of an analytic hierarchy process rule against simpler baselines. "
                  "K.K. Alenezi, M. Ayari. Table S1 comes from the related-work coverage search (results/manuscript) and Table S2 "
                  "from the coding check of Table 1 described in Section 2.5 of the paper; every other value is read from the "
                  f"result files of the public repository (commit {FINAL_COMMIT}) by "
                  "src/05_manuscript/build_supplementary.py; "
                  "the file named under each table gives the source.")


def table(title, header, rows, note=None, widths=None):
    doc.add_paragraph().add_run(title).bold = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; run = c.paragraphs[0].add_run(str(h)); run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(8)
    if widths:
        t.autofit = False
        for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w * 567)))
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    if note:
        p = doc.add_paragraph(note); p.runs[0].font.size = Pt(8.5)
    doc.add_paragraph()


M = ["DADE_AHP", "B1_ML_only", "B2_equal_weight", "B3_BIA_only", "B4_severity_x_impact", "B5_ML_x_impact"]
NAME = {"DADE_AHP": "DADE (AHP)", "B1_ML_only": "B1 detector only", "B2_equal_weight": "B2 equal weights",
        "B3_BIA_only": "B3 BIA only", "B4_severity_x_impact": "B4 severity x impact", "B5_ML_x_impact": "B5 detector x impact"}

# ---------------------------------------------------------------- S1 coverage search
cov = json.load(open(MAN / "crossref_coverage_search_2026-10-30.json"))
scr = list(csv.DictReader(open(MAN / "coverage_screening_2026-10-30.csv")))
rows = []
for i, q in enumerate(cov["queries"], 1):
    c = Counter(x["category"] for x in scr if x["query"] == f"Q{i}")
    rows.append([f"Q{i}", q["query"], q["requested_utc"], f"{q['total_results']:,}", c.get("TRIAGE", 0), c.get("CONTEXT", 0), c.get("OTHER", 0) + c.get("CITED", 0)])
COMBINED = ["10.32620/reks.2025.4.15", "10.37547/tajet/volume08issue06-04"]   # titles combining both topics
uniq = {}
for x in scr:
    uniq.setdefault(x["doi"], x["category"])
cu = Counter(uniq.values())
table("Table S1. Coverage search in the Crossref REST API (30 October 2026). Filter: " + cov["filter"] +
      "; sorted by relevance; top 20 records screened by title.",
      ["Query", "Query string (query=)", "Requested (UTC)", "Total hits", "Triage", "Context", "Other"], rows,
      note=(f"Unique records screened: {len(uniq)}; TRIAGE (alert or incident prioritization, triage or scoring): "
            f"{cu.get('TRIAGE', 0)}; CONTEXT (asset, business-impact or continuity context in cyber decision support): "
            f"{cu.get('CONTEXT', 0)}; OTHER: {cu.get('OTHER', 0)}. None of the screened records is coded in Table 1. "
            "Crossref matches any query term, so total hits do not measure recall. Every record and decision is in "
            "coverage_screening_2026-10-30.csv; the raw responses are in crossref_coverage_search_2026-10-30.json. "
            "Screening was done by one screener (the AI assistant, under the first author's direction) with the rules "
            "in src/05_manuscript/coverage_screening.py. Only Crossref was searched. Two titles combine incident "
            "prioritization with business impact or criticality levels; they are cited and described from their abstracts in "
            "Section 2.5 of the paper (full texts could not be opened) and are not coded in Table 1: " + "; ".join(f"{' '.join(x['title'].split())} ({x['year']}), https://doi.org/{x['doi']}" for x in
                                  (next(y for y in scr if y["doi"] == d) for d in COMBINED)) + "."),
      widths=[1.2, 8.5, 4.0, 2.4, 1.6, 1.6, 1.6])

# ---------------------------------------------------------------- S2 coding evidence
ev = [
    ("Yao and García de Soto", "Full text", "Conceptual ML framework, no model trained; SHAP proposed, not implemented; design confirmed by two industry experts (qualitative)."),
    ("Kure et al.", "Accepted manuscript", "Seven classifiers cross-validated on VCDB; fuzzy asset criticality; framework applied in one real CI organization through workshops; controls recommended; experts approve criticality (no agreement measure)."),
    ("Hernández-Bejarano et al.", "Author copy", "Vision paper: ML-supported cyber-resilience framework for continuity; recovery discussed conceptually; no implementation or validation."),
    ("Verma et al. (2025)", "Abstract only", "Conceptual review of resilience concepts and strategies; service continuity discussed; no empirical test."),
    ("Islam et al.", "Full text", "Regression and neural model predict EPSS on CVEjoin; SHAP and LIME produced; service tiers with recovery-time requirements (qualitative); controls prioritized, not evaluated."),
    ("Wang et al.", "Abstract only", "Pythagorean fuzzy AHP-TOPSIS; expert pairwise judgments as inputs; Kendall's W 0.952 reported among methods; expert code P from the abstract."),
    ("Olaore and Alkhwaldi", "Full text", "MCDA over eight criteria to rank frameworks; ML is a conceptual input, no model executed; no expert panel."),
    ("Papastergiou et al.", "Full text", "CodeBERT exploitability prediction on CVEjoin; SHAP; controls and insurance decision; single 5G pilot."),
    ("Mohamed et al.", "arXiv version", "Weighted-sum ranking of ATT&CK mitigations; lab test with three machines; transparent weights."),
    ("Ganin et al.", "Full text", "Additive multi-criteria utility; consequence (CIA) criteria; hypothetical case; Monte Carlo sensitivity; cost and time criteria of countermeasures."),
    ("CRISP (Verma et al. 2026)", "Abstract only", "Threat prediction with urgency levels for business prioritization; Microsoft Malware Prediction data."),
    ("CISA NCISS", "Secondary sources", "Weighted categories (functional and information impact, recoverability and others) to a 0 to 100 score and priority level; analysts score incidents; no published evaluation. Official pages could not be opened on 30 October 2026."),
    ("NIST SP 800-30 Rev. 1", "Full text", "Likelihood and impact combined into a risk level; guidance only."),
    ("Shah et al.", "Abstract only", "Constraint optimization of SOC throughput, quality and cost; simulation."),
    ("Tong et al.", "Full text", "Adversarial reinforcement learning for budgeted alert prioritization; Suricata alerts on CICIDS2017 and fraud data; loss and investigation cost."),
    ("Gonzalez-Granadillo et al.", "Abstract only", "Response selection by financial, operational and threat assessments; one real CI case; responses proposed to operators."),
    ("Zhong et al.", "Abstract and news item", "Triage rules mined from professional analysts' traces; compared with ground truth, not with independent experts."),
    ("Kim et al.", "Author copy", "Active learning chooses which alerts an error-prone analyst investigates under a budget; simulation on an LLDOS-based attack graph."),
    ("van Ede et al. (DeepCASE)", "Full text", "Attention-based contextual analysis; 10.5 million events from 20 organizations; retrospective; operator labels as reference."),
    ("Freitas et al.", "Full text (arXiv version)", "Copilot Guided Response deployed in production; triage and containment recommendations; analysts confirm each one; no business inputs."),
]
table("Table S2. Evidence for the coding of Table 1 (verification report of 30 October 2026; Crossref checked for "
      "correction or retraction notices, none found).", ["Study", "Text used", "Evidence for the codes"],
      [list(e) for e in ev], widths=[4.5, 3.0, 18.5])

# ---------------------------------------------------------------- S3 thresholds
t18 = J("r18_threshold_study.json")
rows = []
for m in ["DADE_AHP", "B1_ML_only", "B2_equal_weight", "B3_BIA_only", "B5_ML_x_impact"]:
    for pol in ["FIXED", "COST", "RECALL90", "NESTED"]:
        a = t18["S1_leave_one_form_out"][m][pol]; b = t18["S2_repeated_grouped_5fold_100x"][m][pol]
        e = t18["external_practitioners_with_locked_thresholds"][m].get(pol, {})
        th = a.get("thresholds_chosen", {})
        rows.append([NAME[m], pol, f"{r(th.get('high_median'))} ({rng(*th['high_range'])})" if th else "",
                     r(a["mean_cost"]), r(a["high_recall"]), r(a["escalation_rate"]), r(a["quadratic_kappa"]),
                     f"{r(b['mean_cost']['mean'])} ({rng(b['mean_cost']['p2.5'], b['mean_cost']['p97.5'])})",
                     r(b["high_recall"]["mean"]), r(b["escalation_rate"]["mean"]), r(b["quadratic_kappa"]["mean"]),
                     (f"{r(e['mean_cost'])} / {r(e['high_recall'])} / {r(e['escalation_rate'])} [{', '.join(r(x) for x in e['thresholds'])}]" if e else "")])
b4 = t18["S1_leave_one_form_out"]["B4_severity_x_impact"]["FIXED"]; e4 = t18["external_practitioners_with_locked_thresholds"]["B4_severity_x_impact"]["FIXED"]
rows.append([NAME["B4_severity_x_impact"], "none", "", r(b4["mean_cost"]), r(b4["high_recall"]), r(b4["escalation_rate"]),
             r(b4["quadratic_kappa"]), r(b4["mean_cost"]), r(b4["high_recall"]), r(b4["escalation_rate"]), r(b4["quadratic_kappa"]),
             f"{r(e4['mean_cost'])} / {r(e4['high_recall'])} / {r(e4['escalation_rate'])}"])
table("Table S3. Threshold policies scored on held-out scenarios (LOFO: leave-one-form-out; CV5×100: grouped 5-fold x 100) and "
      "on the 120 practitioner labels with cut-offs locked from the 120 expert labels (r18_threshold_study.json, "
      "r18_locked_thresholds.json).",
      ["Method", "Policy", "LOFO High cut-off, median (range)", "LOFO cost", "LOFO recall", "LOFO escal.", "LOFO kappa",
       "CV5×100 cost (2.5 to 97.5 pct)", "CV5×100 recall", "CV5×100 escal.", "CV5×100 kappa", "External cost / recall / escal. [cut-offs]"], rows,
      note="The external check reuses the 30 scenarios; the practitioners' labels were known from 20 October 2026, so it is "
           "independent in raters only. NESTED share of chosen policies is in the result file.")

# ---------------------------------------------------------------- S4 ablations and aggregation
t19 = J("r19_ablation_and_paired.json")
rows = []
for k, v in t19["B_ablations"].items():
    mt = v["metrics"]; pf = v.get("paired_full_minus_variant", {})
    rows.append([k.replace("_", " "), r(mt["quadratic_kappa"]), r(mt["mean_cost"]), r(mt["high_recall"]), r(mt["escalation_rate"]),
                 pct(v["decisions_changed_vs_full"]) if "decisions_changed_vs_full" in v else "",
                 f"{r(pf['quadratic_kappa']['diff'],3)} ({rng(*pf['quadratic_kappa']['ci95'],3)})" if pf else "",
                 f"{r(pf['mean_cost']['diff'],3)} ({rng(*pf['mean_cost']['ci95'],3)})" if pf else ""])
table("Table S4a. Ablations of DADE (120 expert labels, leave-one-expert-out weights, fixed cut-offs; "
      "r19_ablation_and_paired.json, B_ablations). Paired values are full DADE minus the variant.",
      ["Variant", "Kappa", "Mean cost", "High recall", "Escalation", "Decisions changed", "Full minus variant, kappa (95% CI)",
       "Full minus variant, cost (95% CI)"], rows)
rows = []
for k, v in t19["C_weight_aggregation"].items():
    w = v["weights"]; vl = v["vs_120_labels"]
    rows.append([k.replace("_", " "), " ".join(f"{c} {r(w[c],3)}" for c in ["TS", "BI", "AC", "MLS", "RTO", "RPO", "HI"]),
                 v["scenarios_changed_vs_used"], f"{v['level_counts']['High']}/{v['level_counts']['Medium']}/{v['level_counts']['Low']}",
                 r(vl["quadratic_kappa"]), r(vl["mean_cost"]), r(vl["high_recall"])])
table("Table S4b. Weight aggregation variants with fixed group weights (not leave-one-expert-out) scored against the 120 "
      "expert labels (r19, C_weight_aggregation).",
      ["Variant", "Weights", "Scenarios changed vs used", "High/Medium/Low", "Kappa", "Mean cost", "High recall"], rows)
t06 = J("r06_ahp_robustness.json"); var = t06["aggregation_variants"]
var["AIJ_consistent_only"] = next(v for k, v in var.items() if k.startswith("AIJ_consistent_only"))
sept = json.load(open(RES.parent / "ahp_weights_results.json"))["group"]
C7 = ["TS", "BI", "AC", "MLS", "RTO", "RPO", "HI"]
rows = [["AIJ, all six (used; equal to AIP)"] + [r(var["AIJ_all_six (paper)"]["weights"][c], 3) for c in C7] + [r(var["AIJ_all_six (paper)"]["group_CR"], 3)],
        ["AIJ without E3 (consistent experts only)"] + [r(var["AIJ_consistent_only"]["weights"][c], 3) for c in C7] + [r(var["AIJ_consistent_only"]["group_CR"], 3)],
        ["AIJ, September answers (earlier analysis)"] + [r(sept["weights"][c], 3) for c in C7] + [r(sept["CR"], 3)],
        ["Expert bootstrap 95% interval (all six)"] + [rng(*t06["expert_bootstrap_AIJ"]["weight_95ci"][c], 2) for c in C7] + [""]]
table("Table S4c. Group AHP weights under aggregation variants, with expert-bootstrap 95% intervals (5,000 resamples; "
      "r06_ahp_robustness.json; September row from results/ahp_weights_results.json of the earlier analysis).",
      ["Variant"] + C7 + ["Group CR"], rows)

# ---------------------------------------------------------------- S4d paired differences, cost matrices, two-way bootstrap
t10 = J("r10_stage2.json")["primary_paired_bootstrap_DADE_minus_baseline"]
ci = lambda d, nd=2: f"{r(d['diff'], nd)} ({rng(*d['ci95'], nd)})"
rows = [[NAME[b], ci(v["quadratic_kappa"]), ci(v["mean_cost"]), ci(v["high_recall"]), ci(v["under_prioritisation"])] for b, v in t10.items()]
table("Table S4d. DADE minus each baseline against the 120 expert labels (leave-one-expert-out weights, fixed cut-offs): "
      "bootstrap mean and 95% interval over 2,000 resamples of scenarios (r10_stage2.json, planned analysis). These are the "
      "differences shown in Fig. 2 of the paper.",
      ["Baseline", "Quadratic kappa", "Mean cost", "High recall", "Under-prioritization"], rows)
t25 = J("r25_decision_sensitivity.json")
VN = {"protocol": "Protocol, Eq. (6): [[0,1,2],[2,0,1],[6,3,0]]", "symmetric_linear": "Symmetric linear |i - j|",
      "symmetric_quadratic": "Symmetric quadratic (i - j)^2"}
rows = []
for k, v in t25["A_cost_matrix_sensitivity"].items():
    nm = VN[k] if k in VN else f"Under-prioritization x{k.split('_x')[1]} over-escalation, linear in levels"
    mc = v["mean_cost"]
    rows.append([nm] + [r(mc[m]) for m in M] + [ci(v["DADE_minus_B5"]), ci(v["DADE_minus_B2"]), ci(v["DADE_minus_B1"]), NAME[v["cheapest"]]])
table("Table S4d (continued). Mean misprioritization cost under other cost matrices (rows: rater's label; columns: method's "
      "level; Low, Medium, High), same 120 leave-one-expert-out decisions, with DADE minus baseline and 95% intervals from "
      "2,000 resamples of scenarios (r25_decision_sensitivity.json, exploratory, 31 October 2026).",
      ["Cost matrix"] + [NAME[m] for m in M] + ["DADE - B5", "DADE - B2", "DADE - B1", "Cheapest"], rows,
      note="Under x1.0 equals the symmetric linear matrix. No interval for DADE - B5 excludes zero. Differences here are point "
           "differences with their own resamples, so the protocol row can differ in the second decimal from the bootstrap means above.")
tw = t25["B_two_way_bootstrap_DADE_minus"]
rows = [[NAME[b], f"{r(v['quadratic_kappa']['mean'])} ({rng(*v['quadratic_kappa']['ci95'])})",
         f"{r(v['mean_cost']['mean'])} ({rng(*v['mean_cost']['ci95'])})", v["mean_cost"]["n_resamples"]] for b, v in tw.items()]
table("Table S4d (continued). Two-way bootstrap: experts and scenarios both resampled with replacement (2,000 resamples; "
      "r25_decision_sensitivity.json, exploratory). Mean and 95% interval of DADE minus the baseline.",
      ["Baseline", "Quadratic kappa", "Mean cost", "Resamples"], rows)

# ---------------------------------------------------------------- S4e by criticality, all methods
bc = t19["A_by_criticality"]
rows = []
for lvl in ["high", "medium", "low"]:
    for m in M:
        v = bc[lvl]["methods"][m]
        rows.append([lvl, NAME[m], r(v["high_recall"]["value"]), r(v["escalation_rate"]["value"]),
                     f"{r(v['mean_cost']['value'])} ({rng(*v['mean_cost']['ci95'])})", r(v["under_prioritisation"]["value"])])
dist = "; ".join(f"{lvl}: " + ", ".join(f"{bc[lvl]['label_distribution'].get(c, 0)} {c}" for c in ("High", "Medium", "Low"))
                 for lvl in ["high", "medium", "low"])
table("Table S4e. High-priority recall, escalation rate, mean cost (95% interval) and under-prioritization by criticality "
      "of the profile, 40 expert labels each (r19_ablation_and_paired.json, A_by_criticality, exploratory).",
      ["Criticality", "Method", "High recall", "Escalation", "Mean cost (95% CI)", "Under-prioritization"], rows,
      note="Expert labels per criticality level: " + dist + ". High: P1, P2; medium: P3, P4; low: P5, P6.")

# ---------------------------------------------------------------- S5 error propagation
t20 = J("r20_error_propagation.json")
rows = []
for exp, lab in [("D_detector", "Detector inputs"), ("B_bia", "BIA ratings"), ("DB_both", "Both")]:
    for m in M:
        v = t20[exp][m]
        rows.append([lab, NAME[m], f"{pct(v['share_of_decisions_changed']['mean'])} ({pct(v['share_of_decisions_changed']['p2.5'])} to {pct(v['share_of_decisions_changed']['p97.5'])})",
                     v["scenarios_changed_in_over_half_of_draws"],
                     f"{r(v['mean_cost_vs_labels']['mean'])} ({rng(v['mean_cost_vs_labels']['p2.5'], v['mean_cost_vs_labels']['p97.5'])})",
                     f"{r(v['high_recall_vs_labels']['mean'])} ({rng(v['high_recall_vs_labels']['p2.5'], v['high_recall_vs_labels']['p97.5'])})"])
table("Table S5. Monte Carlo propagation of input uncertainty (2,000 draws; r20_error_propagation.json). Costs and recall "
      "under redrawn inputs measure dependence on the shown flow, not performance, because the labels refer to the shown inputs.",
      ["Perturbation", "Method", "Decisions changed, mean (95% range)", "Scenarios changed in >50% of draws",
       "Mean cost (95% range)", "High recall (95% range)"], rows)

# ---------------------------------------------------------------- S6 scenarios
t21 = J("r21_panel_audit.json")
ext = {row["scenario_id"]: row for row in csv.DictReader(open(ROOT / "data" / "external_reference_labels.csv"))} \
    if (ROOT / "data" / "external_reference_labels.csv").exists() else {}
rows = []
for sid, v in t21["experts"]["scenarios"].items():
    pv = t21["practitioners"]["scenarios"][sid]
    rows.append([sid, v["profile"], v["true_category"], v["shown_category"], r(v["MLS"], 3),
                 "/".join(str(x) for x in v["votes_LMH"]), v["consensus"], r(v["p_consensus_changes_under_rater_bootstrap"]),
                 "/".join(str(x) for x in pv["votes_LMH"]), pv["consensus"], r(pv["p_consensus_changes_under_rater_bootstrap"]),
                 v["DADE"], v["B5"]])
table("Table S6. Every scenario: profile, true and shown category, MLS, votes (Low/Medium/High), consensus label, probability "
      "that the consensus changes under rater resampling, and the levels of DADE and B5 with group weights (r21_panel_audit.json).",
      ["Scenario", "Profile", "True category", "Shown category", "MLS", "Expert votes L/M/H", "Expert consensus",
       "P(change)", "Practitioner votes L/M/H", "Practitioner label", "P(change)", "DADE", "B5"], rows,
      note="Consensus rule: median of four labels with ties to the higher level (practitioners: majority, ties to the higher "
           "level, 2-1-1 by plurality).")

# ---------------------------------------------------------------- S7 raters
rows = []
for panel in ["experts", "practitioners"]:
    for k, v in t21[panel]["raters"].items():
        d = v["label_distribution"]
        rows.append([k, v["n"], f"{d.get('High', 0)}/{d.get('Medium', 0)}/{d.get('Low', 0)}", r(v["kappa_with_consensus_of_others"]),
                     r(v["kappa_with_DADE"]), r(v["kappa_with_B5"]), r(v["mean_cost_DADE"])])
table("Table S7a. Each rater: number of labels, High/Medium/Low labels, quadratic kappa with the consensus of the other raters "
      "of the same panel, with DADE (leave-one-expert-out for the experts) and with B5, and DADE's mean cost (r21).",
      ["Rater", "Labels", "High/Medium/Low", "Kappa with others", "Kappa with DADE", "Kappa with B5", "DADE cost"], rows)
rows = []
for panel in ["experts", "practitioners"]:
    for k, v in t21[panel]["pairwise"].items():
        rows.append([k, v["n_shared"], r(v["quadratic_kappa"]), pct(v["exact_agreement"])])
es = t21["effective_sample_size_DADE"]; pdf = t21["panel_difference"]
table("Table S7b. Pairwise rater agreement on shared scenarios (r21).", ["Pair", "Shared scenarios", "Quadratic kappa", "Exact agreement"], rows,
      note=(f"Design effect for DADE's exact agreement {r(es['exact']['design_effect'])} (intra-scenario correlation "
            f"{r(es['exact']['icc1_within_scenario'])}, effective n {r(es['exact']['effective_n_of_120'],1)} of 120); for cost "
            f"{r(es['cost']['design_effect'])} (correlation {r(es['cost']['icc1_within_scenario'])}, effective n "
            f"{r(es['cost']['effective_n_of_120'],1)}). Expert consensus versus practitioner reference: exact agreement "
            f"{pct(pdf['exact_agreement'])}, quadratic kappa {r(pdf['quadratic_kappa_between_consensus_labels'])}; experts higher on "
            f"{pdf['scenarios_experts_higher']} and practitioners higher on {pdf['scenarios_practitioners_higher']} scenarios "
            f"(sign test p = {r(pdf['sign_test_p_two_sided'])})."))

# ---------------------------------------------------------------- S8 detection details
t05 = J("r05_nslkdd.json"); t01 = J("r01_unsw_detection.json"); t08 = J("r08_cicids2017_holdout_v2.json"); t02 = J("r02_calibration.json")
rows = []
for m in ["LogisticRegression", "DecisionTree", "RandomForest", "HistGradientBoosting"]:
    a = t05["random_80_20_inside_KDDTrain+ (paper protocol)"][m]; b = t05["official_KDDTest+"][m]
    rows.append([m, pct(a["accuracy"]), pct(b["accuracy"]), pct(b["balanced_accuracy"]),
                 pct(b["recall_on_attack_types_unseen_in_training"]), pct(b["recall_on_attack_types_seen_in_training"]), pct(b["false_positive_rate"])])
table("Table S8a. NSL-KDD: random 80/20 split of KDDTrain+ versus the official KDDTest+ (threshold 0.5; r05_nslkdd.json).",
      ["Model", "Random split accuracy", "Official accuracy", "Official balanced acc.", "Recall, unseen types", "Recall, seen types", "FPR"], rows)
rows = []
for m in ["LogisticRegression", "RandomForest", "HistGradientBoosting"]:
    for th in ["default_0.5", "val_max_f1", "val_fpr_1pct"]:
        v = t01["models"][m]["test"][th]; c = v.get("confusion", {})
        rows.append([m, th, r(v["threshold"], 4), pct(v["accuracy"]), pct(v["balanced_accuracy"]), pct(v["attack_recall"]),
                     pct(v["false_positive_rate"]), pct(v["attack_precision"]), r(v["mcc"], 3), r(v["roc_auc"], 3),
                     f"{c.get('tn', ''):,} / {c.get('fp', ''):,} / {c.get('fn', ''):,} / {c.get('tp', ''):,}" if c else ""])
table("Table S8b. UNSW-NB15 official test file, all models and thresholds (thresholds chosen on VAL, on each model's uncalibrated score; r01_unsw_detection.json).",
      ["Model", "Threshold rule", "Threshold", "Accuracy", "Balanced acc.", "Recall", "FPR", "Precision", "MCC", "ROC-AUC", "TN / FP / FN / TP"], rows)
rows = []
for th in ["default_0.5", "val_max_f1", "val_fpr_1pct", "val_fpr_5pct"]:
    v = t08["unseen_family_test"][th]; c = v["confusion"]
    rows.append([th, str(t08["thresholds_chosen_on_validation"][th]), pct(v["attack_recall"]), pct(v["false_positive_rate"], 2), pct(v["attack_precision"]),
                 pct(v["balanced_accuracy"]), f"{c['tn']:,} / {c['fp']:,} / {c['fn']:,} / {c['tp']:,}"])
table("Table S8c. CICIDS2017 unseen-family test (11,098 withheld Bots and Brute Force flows + 11,098 benign flows; thresholds "
      "chosen on seen-family validation data; r08_cicids2017_holdout_v2.json, run by the first author on his computer).",
      ["Threshold rule", "Threshold", "Recall", "FPR", "Precision", "Balanced acc.", "TN / FP / FN / TP"], rows)
rows = []
for m in ["HistGradientBoosting", "RandomForest"]:
    for meth in ["uncalibrated", "sigmoid", "isotonic"]:
        v = t02["models"][m][meth]
        rows.append([m, meth, r(v["brier"], 3), r(v["log_loss"], 3), f"{r(v['ece_15bins'],3)} ({rng(*v['ece_95ci'],3)})"])
table("Table S8d. Calibration on the UNSW-NB15 test file, maps fitted on CAL (r02_calibration.json).",
      ["Model", "Method", "Brier", "Log loss", "ECE, 15 bins (95% CI)"], rows)
t26 = J("r26_ttl_and_label_conflicts.json")
rows = []
for k, nm in [("A_hgb_all_features", "HGB, all features"), ("A_hgb_without_ttl", "HGB without sttl, dttl, ct_state_ttl"),
              ("A_tree_depth4_ttl_only", "Depth-4 tree on the three TTL features only")]:
    for th in ["default_0.5", "val_fpr_1pct"]:
        v = t26[k]["test"][th]
        rows.append([nm, th, r(v["threshold"], 4), r(t26[k]["test_roc_auc"], 3), pct(v["accuracy"]), pct(v["attack_recall"]), pct(v["false_positive_rate"])])
lc = t26["B_label_conflicts"]
table("Table S8e. UNSW-NB15 testbed TTL features and label conflicts between the training and test files "
      "(r26_ttl_and_label_conflicts.json, exploratory, 31 October 2026; FIT for training, thresholds on VAL).",
      ["Model", "Threshold rule", "Threshold", "Test ROC-AUC", "Accuracy", "Recall", "FPR"], rows,
      note=(f"Label conflicts: {lc['test_rows_identical_to_a_training_row']:,} test rows have all feature values equal to a "
            f"training row; of the {lc['benign_test_rows_identical_to_a_training_row']:,} benign ones, "
            f"{lc['of_which_majority_attack_in_training']} are labeled attack in most of their training copies and "
            f"{lc['of_which_any_attack_in_training']} in at least one; of the {lc['attack_test_rows_identical_to_a_training_row']:,} "
            f"attack ones, {lc['of_which_majority_benign_in_training']} are mostly benign in training. Training-file rows by sttl "
            "value (attack share): " + "; ".join(f"{k}: {v['rows']:,} ({pct(v['attack_share'])})" for k, v in t26["A_training_file_sttl"].items()) + "."))

# ---------------------------------------------------------------- S9 base rate
t17 = J("r17_base_rate_thresholds.json")
rows = []
pol_names = {"P1_uncorrected_0.5": "BR1 uncorrected 0.5", "P2_oracle_prior_0.5": "BR2 oracle prior", "P3_em_prior_0.5": "BR3 EM prior",
             "P4a_prior_x5_0.5": "BR4a prior x5", "P4b_prior_div5_0.5": "BR4b prior /5", "P5_validation_fpr1pct": "BR5 validation threshold"}
for prev, v in t17["prevalence"].items():
    for p, nm in pol_names.items():
        if p not in v["policies"]:
            continue
        q = v["policies"][p]
        pi, pp = t17["calibration_split_prevalence"], float(prev)
        prior = {"P1_uncorrected_0.5": None, "P2_oracle_prior_0.5": pp, "P3_em_prior_0.5": v["em_estimated_prevalence"]["mean"],
                 "P4a_prior_x5_0.5": min(5 * pp, 0.99), "P4b_prior_div5_0.5": pp / 5, "P5_validation_fpr1pct": None}[p]
        thr = ("0.500" if p.startswith("P1") else r(t17["validation_threshold_fpr1pct"], 3) if p.startswith("P5")
               else r(pi * (1 - prior) / (pi * (1 - prior) + prior * (1 - pi)), 3) + ("*" if p.startswith("P3") else ""))
        rows.append([f"{prev} ({v['n_attacks_per_draw']:,} attacks, {v['n_flows_per_draw']:,} flows)", nm, thr,
                     f"{pct(q['precision']['mean'])} ({pct(q['precision']['min'])} to {pct(q['precision']['max'])})",
                     f"{pct(q['recall']['mean'])} ({pct(q['recall']['min'])} to {pct(q['recall']['max'])})",
                     pct(q["fpr"]["mean"], 2), r(q["alerts_per_1000_flows"]["mean"], 1),
                     f"{r(q['tp']['mean'],1)} / {r(q['fp']['mean'],1)} / {r(q['fn']['mean'],1)} / {r(q['tn']['mean'],1)}"])
eq = "; ".join(f"{k}: {v['equivalent_uncorrected_threshold_of_P2']}" for k, v in t17["prevalence"].items())
em = "; ".join(f"{k}: {v['em_estimated_prevalence']['mean']} ({v['em_estimated_prevalence']['min']} to {v['em_estimated_prevalence']['max']})" for k, v in t17["prevalence"].items())
rk = "; ".join(f"{k}: ROC-AUC {v['roc_auc_raw_mean']} / {v['roc_auc_corrected_mean']}, PR-AUC {v['pr_auc_raw_mean']} / {v['pr_auc_corrected_mean']}"
               for k, v in t17["ranking_unchanged_by_correction"].items())
table("Table S9. Base-rate policies, mean and range over ten draws per prevalence (r17_base_rate_thresholds.json). Detector: "
      + t17["detector"] + f"; training prior {t17['calibration_split_prevalence']}; {t17['n_benign_in_every_draw']:,} benign flows in every draw; "
      f"validation threshold {t17['validation_threshold_fpr1pct']}.",
      ["Prevalence", "Policy", "Threshold", "Precision (range)", "Recall (range)", "FPR", "Alerts per 1,000 flows", "TP / FP / FN / TN (means)"], rows,
      note=f"Thresholds are on the uncorrected isotonic score: BR1 uses 0.5, BR5 the validation threshold, and BR2 to BR4 the value of Eq. (4) of the paper with each policy's prior; "
           f"* with the mean EM estimate (it varies between draws). BR4a multiplies the prevalence by 5, capped at 0.99, so at 50% it uses 0.99. "
           f"Policy keys in the result file keep the earlier names P1 to P5. Equivalent uncorrected threshold of BR2 as stored: {eq}. EM-estimated prevalence: {em}. Ranking before / after correction: {rk}.")

bc_ = t25["C_base_rate_at_decision_layer"]
rows = [["uncorrected (training prior " + str(bc_["pi_train"]) + ")"] + ["0 / " + str(bc_["uncorrected_high_decisions"][m]) for m in M]]
for k, v in bc_.items():
    if k.startswith("prior_"):
        rows.append([k.replace("prior_", "deployment prior ")] + [f"{v['decisions_changed_of_30'][m]} / {v['high_decisions'][m]}" for m in M])
table("Table S9b. Base rate at the decision layer: the 30 scenario MLS values corrected by Eq. (3) of the paper from the "
      "training prior to each deployment prior, every rule re-applied with the group weights and fixed cut-offs "
      "(r25_decision_sensitivity.json, exploratory). Cells: decisions changed of 30 / High decisions.",
      ["Scenario MLS"] + [NAME[m] for m in M], rows,
      note="Costs are not reported, because the labels refer to the uncorrected MLS that the raters saw. B3 and B4 do not use the MLS.")

# ---------------------------------------------------------------- S10 leakage audit
t04 = J("r04_ablation_leakage_audit.json")
cond = [("A_ML_only", "Detector only", "none"), ("D_AHP_weight__TS_oracle", "AHP weights", "true category (earlier draft)"),
        ("D_AHP_weight__TS_predicted", "AHP weights", "predicted category"), ("C_equal_weight__TS_predicted", "Equal weights", "predicted category"),
        ("D_AHP_weight__TS_none", "AHP weights", "TS removed"), ("C_equal_weight__TS_oracle", "Equal weights", "true category"),
        ("C_equal_weight__TS_none", "Equal weights", "TS removed")]
rows = [[nm, ts] + [f"{pct(t04[part][k]['accuracy'])} (kappa {r(t04[part][k]['kappa'], 2)})" for part in ("paper_sample_n200", "full_test_n82332")]
        for k, nm, ts in cond]
table("Table S10. Leakage audit: agreement with the attack or benign label (accuracy, Cohen's kappa) with the earlier draft's "
      "business context (r04_ablation_leakage_audit.json).",
      ["Condition", "TS source", "200-flow sample of the earlier draft", "Full UNSW-NB15 test file (82,332)"], rows)

out = OUTFILE
doc.save(out)
print("saved", out)
