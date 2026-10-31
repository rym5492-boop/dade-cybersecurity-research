# Content comparison of the author's re-run (31 October 2026)

The re-run report marked the CSV, Markdown and NPZ outputs "file differs (check by eye)", because reproduce.py then compared those files byte by byte. This file replaces that statement with a check by content. It was produced by the AI assistant in the cloud workspace with

    python -I src/04_revision/compare_outputs.py results/reproduction/20261031_153220/reference results/reproduction/20261031_153220/outputs

Results: every JSON, CSV, Markdown and NPZ output is the same as the stored one except the Logistic Regression results of r01 (metrics by up to 0.006; single test-flow scores by up to 0.19) and r05 (fourth decimal). DADE does not use that model. r22 differs from the stored file of that time because the author's run was the first complete one (CIC parts included); results/revision/r22_cross_dataset_audit.json is now his file.

Dataset hashes: the author's UNSW-NB15 copies (SHA-256 0ebd9eb06d53ae22..., 7ec02e7e44d72bd2...) are the recorded files with LF instead of CRLF line endings:

    python src/04_revision/hash_line_endings.py data/UNSW_NB15_training-set.csv data/UNSW_NB15_testing-set.csv

```
data/UNSW_NB15_training-set.csv
  as stored bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa
  LF        0ebd9eb06d53ae223fbe7948737985e851e00e9c20140508d653de4b1655fdd6
  CRLF      bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa
data/UNSW_NB15_testing-set.csv
  as stored 734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559
  LF        7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c
  CRLF      734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559
```

Reference: `reference/` (the stored results copied by reproduce.py before the run)
New: `outputs/` (the results/revision folder of the author's computer after the run, sent by him on 31 October 2026)

| File | Result |
| --- | --- |
| fig4_ahp_weights.png | figure file, not compared (drawn from the compared result files) |
| fig5_decision_differences.png | figure file, not compared (drawn from the compared result files) |
| fig6_workflow.pdf | figure file, not compared (drawn from the compared result files) |
| fig6_workflow.png | figure file, not compared (drawn from the compared result files) |
| fig7_threshold_tradeoff.pdf | figure file, not compared (drawn from the compared result files) |
| fig7_threshold_tradeoff.png | figure file, not compared (drawn from the compared result files) |
| fig8_ablation.pdf | figure file, not compared (drawn from the compared result files) |
| fig8_ablation.png | figure file, not compared (drawn from the compared result files) |
| fig9_error_propagation.pdf | figure file, not compared (drawn from the compared result files) |
| fig9_error_propagation.png | figure file, not compared (drawn from the compared result files) |
| r01_test_scores.npz | arrays differ: LogisticRegression (largest absolute difference 0.186786); attack_cat (text array, identical stored bytes) |
| r01_unsw_detection.json | 61 values differ (largest absolute difference 210; e.g. /models/LogisticRegression/cv_minus_test_accuracy_gap_pp) |
| r02_calibration.json | identical |
| r03_attack_category.json | identical |
| r03_test_predicted_severity.npz | same content (4 arrays; numeric arrays within 1e-9; pred_cat (text array, identical stored bytes), true_cat (text array, identical stored bytes)) |
| r04_ablation_leakage_audit.json | identical |
| r05_nslkdd.json | 15 values differ (largest absolute difference 1; e.g. /official_KDDTest+/LogisticRegression/accuracy) |
| r06_ahp_robustness.json | identical |
| r07_cross_dataset_v2.json | identical |
| r08_cicids2017_holdout_v2.json | identical |
| r09_scenario_cards_ar.md | same text (line endings ignored) |
| r09_scenarios.csv | same content (30 rows x 24 columns; line endings ignored) |
| r10_decisions_stage1.csv | same content (30 rows x 19 columns; line endings ignored) |
| r10_stage1.json | identical |
| r10_stage2.json | identical |
| r11_unsw_duplicates.json | identical |
| r12_dataset_manifest.json | identical |
| r13_external_panel.json | identical |
| r14_final_checks.json | identical |
| r16_level_bounds.json | identical |
| r17_base_rate_thresholds.json | identical |
| r17_test_isotonic_scores.npz | same content (3 arrays; numeric arrays within 1e-9; attack_cat (text array, identical stored bytes)) |
| r18_locked_thresholds.json | identical |
| r18_threshold_study.json | identical |
| r18_tradeoff_curves.csv | same content (124 rows x 7 columns; line endings ignored) |
| r19_ablation_and_paired.json | identical |
| r20_error_propagation.json | identical |
| r21_panel_audit.json | identical |
| r22_cross_dataset_audit.json | 124 values differ (largest absolute difference 0; e.g. /adaptation_independence_check/by_dataset/CICIDS2017/adaptation_rows_1pct) |
| r23_provenance_manifest.json | identical |
