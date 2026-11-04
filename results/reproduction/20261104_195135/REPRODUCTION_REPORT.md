# Reproduction report 20261104_195135

- Run by: root on vm (Linux-6.18.44-fc-v233-x86_64-with-glibc2.39)
- Started: 2026-11-04T19:51:35
- Python 3.13.18
- numpy 2.5.3
- pandas 3.0.5
- sklearn 1.9.1
- scipy 1.18.1
- matplotlib 3.11.2

## Dataset files

| File | Expected SHA-256 | Found | Match |
| --- | --- | --- | --- |
| UNSW_NB15_training-set.csv | bec7dd5ec88dc2a0... | bec7dd5ec88dc2a0... | yes |
| UNSW_NB15_testing-set.csv | 734fe6642edf758f... | 734fe6642edf758f... | yes |
| KDDTrain+.txt | 1b86d2f957b33082... | 1b86d2f957b33082... | yes |
| KDDTest+.txt | fa46b0935342616a... | fa46b0935342616a... | yes |
| cicids2017_cleaned.csv | not present on this machine | missing | |
| balanced_7classes_500000_each.csv | not present on this machine | missing | |

## Scripts

| Script | Exit | Seconds | Output comparison |
| --- | --- | --- | --- |
| r01_unsw_detection | 0 | 164 | r01_test_scores.npz: same content (5 arrays; numeric arrays within 1e-9; attack_cat (text array, identical stored bytes)); r01_unsw_detection.json: identical |
| r02_calibration | 0 | 45 | r02_calibration.json: identical |
| r03_attack_category_severity | 0 | 20 | r03_attack_category.json: identical; r03_test_predicted_severity.npz: same content (4 arrays; numeric arrays within 1e-9; pred_cat (text array, identical stored bytes), true_cat (text array, identical stored bytes)) |
| r04_ablation_leakage_audit | 0 | 27 | r04_ablation_leakage_audit.json: identical |
| r05_nslkdd_official_split | 0 | 31 | r05_nslkdd.json: 18 values differ, e.g. [('/random_80_20_inside_KDDTrain+ (paper protocol)/LogisticRegression/mcc', 0.9436, 0.9435), ('/official_KDDTest+/LogisticRegression/attack_f1', 0.7431, 0.7426), ('/official_KDDTest+/LogisticRegression/attack_recall', 0.6242, 0.6236)] |
| r06_ahp_robustness | 0 | 1 | r06_ahp_robustness.json: identical |
| r07_cross_dataset_v2 | skipped | | data missing: cicids2017_cleaned.csv, balanced_7classes_500000_each.csv (stored results kept) |
| r08_cicids2017_holdout_v2 | skipped | | data missing: cicids2017_cleaned.csv (stored results kept) |
| r09_build_scenarios | 0 | 5 | r09_scenario_cards_ar.md: byte-identical; r09_scenarios.csv: byte-identical |
| r10_decision_evaluation | 0 | 52 | r10_stage2.json: identical; r10_stage1.json: identical; r10_decisions_stage1.csv: byte-identical |
| r11_unsw_duplicates | 0 | 46 | r11_unsw_duplicates.json: identical |
| r12_dataset_manifest | skipped | | data missing: cicids2017_cleaned.csv, balanced_7classes_500000_each.csv (stored results kept) |
| r13_external_panel | 0 | 1 | r13_external_panel.json: identical |
| r14_final_checks | 0 | 1 | r14_final_checks.json: identical |
| r15_figures | 0 | 2 | fig4_ahp_weights.png: figure file, not compared (drawn from the compared result files); fig5_decision_differences.png: figure file, not compared (drawn from the compared result files) |
| r16_level_bounds | 0 | 0 | r16_level_bounds.json: identical |
| r17_base_rate_thresholds | 0 | 9 | r17_base_rate_thresholds.json: identical; r17_test_isotonic_scores.npz: byte-identical |
| r18_threshold_study | 0 | 513 | r18_locked_thresholds.json: identical; r18_tradeoff_curves.csv: byte-identical; r18_threshold_study.json: identical |
| r19_ablation_and_paired | 0 | 248 | r19_ablation_and_paired.json: identical |
| r20_error_propagation | 0 | 386 | r20_error_propagation.json: identical |
| r21_panel_audit | 0 | 3 | r21_panel_audit.json: identical |
| r22_cross_dataset_audit | skipped | | data missing: cicids2017_cleaned.csv, balanced_7classes_500000_each.csv (stored results kept) |
| r25_decision_sensitivity | 0 | 18 | r25_decision_sensitivity.json: identical |
| r26_ttl_and_label_conflicts | 0 | 11 | r26_ttl_and_label_conflicts.json: identical |

Finished 2026-11-04T20:17:58.
