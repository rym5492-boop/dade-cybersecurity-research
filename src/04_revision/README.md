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

Order: r01, r02, r03, then r04 (needs r03), r05, r06; r07 and r08 on the machine that holds the
CIC datasets.

Data integrity: the UNSW-NB15 files used to produce `results/revision` have the official sizes
(175,341 / 82,332 rows) and category counts, and re-running the original
`src/02_dade_engine/01_ablation_study.py` on them reproduces the published ablation numbers exactly.
