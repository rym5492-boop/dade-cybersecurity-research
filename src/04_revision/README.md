# Revision experiments (response to the supervisor review, October 2026)

These scripts replace the analyses the review flagged. The original scripts in
`src/01_predictive_layer` and `src/02_dade_engine` are kept unchanged so every
change can be audited. All outputs go to `results/revision/`.

Run from the repository root, with the dataset files in the root or in `data/`.

| Script | Review item | Runs where | What it fixes |
| --- | --- | --- | --- |
| `r01_unsw_detection.py` | 3.1 | anywhere (UNSW-NB15) | Pipeline fitted on training data only; thresholds chosen on validation; prevalence, majority baseline, balanced accuracy, macro-F1, MCC, FPR, confusion counts, per-category recall; leakage-free CV-vs-test gap |
| `r02_calibration.py` | 3.1 | anywhere (UNSW-NB15) | Calibrator fitted on a disjoint split; sigmoid vs isotonic; ECE with bootstrap CI; reliability table; prevalence shift with prior correction |
| `r03_attack_category_severity.py` | Critical 1 | anywhere (UNSW-NB15) | Threat Severity predicted from a category classifier trained on training data only; severity error reported |
| `r04_ablation_leakage_audit.py` | Critical 1 | anywhere (UNSW-NB15) | Reproduces the published ablation exactly, then swaps only Threat Severity (oracle / predicted / none) |
| `r05_nslkdd_official_split.py` | Section 4, abstract | anywhere (NSL-KDD) | Official KDDTrain+/KDDTest+ split next to the paper's random split |
| `r06_ahp_robustness.py` | Critical 4, 3.3 | anywhere (expert data in `data/`) | CR by two estimators, straight-lining flag, five aggregation variants, expert-bootstrap CIs and rank probabilities |
| `r07_cross_dataset_v2.py` | 3.1 | **your machine** (needs CICIDS2017 and CICIoT2023) | Held-out diagonal, unit and semantic harmonisation, prevalence and majority baseline, two adaptation tests |
| `r08_cicids2017_holdout_v2.py` | 3.1 | **your machine** (needs CICIDS2017) | Thresholds chosen on validation for unseen-family evaluation |
| `r09_build_scenarios.py` | 3.2, Critical 4/5 | anywhere (UNSW-NB15, needs r03) | 30 balanced hypothetical scenarios anchored on real test flows; forms A/B/C |
| `r10_decision_evaluation.py` | 3.2, Critical 4/5 | anywhere (needs r09; stage 2 needs `data/scenario_labels.csv`) | Pre-registered decision evaluation: DADE vs ML-only, equal-weight, BIA-only, severity x impact and ML x impact baselines; results by sector and traffic type, failure cases; monotonicity and weight/threshold stability; Krippendorff's alpha, leave-one-expert-out comparison, asymmetric cost, paired cluster bootstrap |
| `r11_unsw_duplicates.py` | 3.1 | anywhere (UNSW-NB15) | Duplicate and train-test overlap audit; CV with duplicate-grouped folds; test metrics on rows unseen in training |
| `r12_dataset_manifest.py` | 3.1, Week-2 gate | anywhere for NSL-KDD/UNSW-NB15; **your machine** for the CIC files | Rows, attack share and label counts of every dataset file (Table 4) |
| `r13_external_panel.py` | 3.2 (independent validation) | anywhere (needs `data/external_panel_labels.csv`) | Secondary analysis declared in `PREREG_ADDENDUM_external_panel.md`: Krippendorff's alpha among four external raters, majority reference label per scenario, sensitivity checks |
| `r14_final_checks.py` | Critical 4, 3.2 | anywhere (needs r09, r10 stage 1, r13 and `data/scenario_labels.csv`) | Checks declared before the final run, kept out of r10 so r10 stays unchanged: CR of every expert before and after the October revisions, leave-one-expert-out without E3's AHP answers, external panel item 3 (DADE and baselines vs the external reference), exploratory panel-vs-external agreement |
| `r15_figures.py` | Figures 4–5 | anywhere (needs r06 and r10 stage 2 outputs) | Draws Figure 4 (group AHP weights with bootstrap intervals, with and without E3) and Figure 5 (difference between DADE and each baseline in quadratic κ and mean cost, with 95% intervals) from the result files; computes nothing new |

Order: r01, r02, r03, then r04 (needs r03), r05, r06; r07 and r08 on the machine that holds the
CIC datasets; r09, r10, r13, then r14; r15 last (figures only).

Data integrity: the UNSW-NB15 files used to produce `results/revision` have the official sizes
(175,341 / 82,332 rows) and category counts, and re-running the original
`src/02_dade_engine/01_ablation_study.py` on them reproduces the published ablation numbers exactly.

## Where each number in the revised paper comes from

| Paper item | Script | Output file |
| --- | --- | --- |
| Table 4 (data manifest: rows, attack share) | r12 (CIC rows: run on the machine holding the CIC files; run by the author on 20 Oct 2026) | r12_dataset_manifest.json |
| §4.7 external panel (alpha, reference labels) | r13 | r13_external_panel.json, data/external_reference_labels.csv |
| Table 4 partitions, §4.1 duplicates | r01, r08, r11 | r01_unsw_detection.json, r08_cicids2017_holdout_v2.json, r11_unsw_duplicates.json |
| Table 5 (NSL-KDD random vs official split) | r05 | r05_nslkdd.json |
| Table 6, §5.2 (UNSW-NB15 thresholds, CV–test gap) | r01, r11 | r01_unsw_detection.json, r11_unsw_duplicates.json |
| §5.3 (predicted category, severity error) | r03 | r03_attack_category.json |
| Table 7, §5.4 (CICIDS2017 unseen families) | r08 | r08_cicids2017_holdout_v2.json |
| Figure 2, §5.5 (cross-dataset transfer) | r07 | r07_cross_dataset_v2.json |
| Table 8, Figure 3, §5.6 (calibration, base-rate shift) | r02 | r02_calibration.json |
| Table 9, §6.1 (leakage audit of the earlier ablation) | r04 | r04_ablation_leakage_audit.json |
| Tables 10–11, §6.2 (AHP consistency, aggregation, bootstrap) | r06 | r06_ahp_robustness.json |
| §6.3 (monotonicity, weight/threshold/severity checks, rule comparison) | r10 (stage 1) | r10_stage1.json, r10_decisions_stage1.csv |
| Table 12, §4.7 (scenarios) | r09 | r09_scenarios.csv, r09_scenario_cards_ar.md |
| Table 13, §6.4 (agreement with expert labels, thresholds, failure cases, consensus) | r10 (stage 2) | r10_stage2.json |
| Table 10 (CR September vs final), §4.6 and §6.4 (without-E3 sensitivity), §6.4 external panel vs methods | r14 | r14_final_checks.json |
| Figure 4 (§6.2), Figure 5 (§6.4) | r15 | fig4_ahp_weights.png, fig5_decision_differences.png (drawn from r06_ahp_robustness.json and r10_stage2.json) |

All AHP revision answers arrived (E1, E2, E6 targeted pairs; E3, E5 full repeat). They are in
`data/ahp_expert_judgments_anonymized.json` (`pairwise_responses`; the September answers are kept in
`pairwise_responses_original_sept2026`, with a `revision_log`). r06, r10, r13 and r14 were then run on
20–21 October 2026; r10 is unchanged since commit a5788aa.

## When the expert scenario answers arrive (step-by-step; done 21 Oct 2026)

1. In Tally, open each scenario form (A, B, C) → Submissions → Download CSV.
2. Give the three CSV files to the assistant (or do it by hand): they are converted into
   `data/scenario_labels.csv` with columns `expert_id, form, scenario_id, label`, following
   `data/scenario_labels_TEMPLATE.csv`. Labels may be written as High/Medium/Low or مرتفع/متوسط/منخفض.
   Each expert's anonymous ID (E1–E6) must match the ID used in `data/ahp_expert_judgments_anonymized.json`.
3. Run `python r10_decision_evaluation.py` from `src/04_revision`. Stage 2 runs automatically and writes
   `results/revision/r10_stage2.json`.
4. If the revised AHP answers (E1, E2, E6 targeted; E3, E5 full repeat) have arrived, update `data/ahp_expert_judgments_anonymized.json`
   first, then re-run `r06_ahp_robustness.py` and `r10_decision_evaluation.py`.
5. Copy the numbers into Table 13, the abstract and Section 6.4 of the manuscript, replacing every [PENDING].
   Do not edit `r10_decision_evaluation.py` after the labels arrive; any extra analysis goes in a new file
   and is reported as exploratory.

### Decision log — E3 AHP re-elicitation (2026-10-18, revised 2026-10-20)
- E3's original answers: all 21 = 5 ("equal"), so no weights can be derived.
- E3's re-elicitation (Tally pexRNB, submitted 2026-10-18 10:04 +0300): 20 of 21 answers = 6, one answer = 7. CR = 0.134 (> 0.10). The answers always favour the right-hand criterion, which ranks TS last and MLS first. This is a same-side response pattern.
- 18 Oct: E3 was first marked for exclusion. **20 Oct: that decision is withdrawn.** The rule fixed in Section 4.6 before any answer was received is that revised answers replace the originals. Excluding E3 alone would also treat E3 (CR 0.134) differently from E2, whose original answers (CR 0.106) are kept.
- Primary analysis: E3's repeat replaces the original, as the rule requires. Sensitivity: the variant without E3 (in r06 as "AIJ_consistent_only" / "AIJ_excluding_most_inconsistent [E3]", and in r14 for the leave-one-expert-out decisions).
- Effect, checked on 20 Oct with E1 and E6 revised and E2 and E5 original: DADE gives the same level for all 30 scenarios with or without E3; under leave-one-expert-out, 1 of 120 decisions differs.
- Final check (21 Oct, all answers in: E2 accepted its three suggestions, E5's full repeat CR 0.075): without E3, 2 of 30 group decisions (A10, C05) and 3 of 120 leave-one-expert-out decisions differ (r14).
- E1 and E6 accepted every suggested value (CR 0.354 → 0.058 and 0.233 → 0.096). Because the suggestions were shown, their new consistency partly reflects the feedback; the paper says so.
- E3's scenario labels are a separate task and are kept.
