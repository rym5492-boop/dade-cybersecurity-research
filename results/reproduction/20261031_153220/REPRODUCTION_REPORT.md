# Reproduction report 20261031_153220

- Run by: HP on khalid (Windows-11-10.0.26200-SP0)
- Started: 2026-10-31T15:32:20
- Python 3.13.17
- numpy 2.5.3
- pandas 3.0.5
- sklearn 1.9.1
- scipy 1.18.1
- matplotlib 3.11.2

## Dataset files

| File | Expected SHA-256 | Found | Match |
| --- | --- | --- | --- |
| UNSW_NB15_training-set.csv | bec7dd5ec88dc2a0... | 0ebd9eb06d53ae22... | NO |
| UNSW_NB15_testing-set.csv | 734fe6642edf758f... | 7ec02e7e44d72bd2... | NO |
| KDDTrain+.txt | 1b86d2f957b33082... | 1b86d2f957b33082... | yes |
| KDDTest+.txt | fa46b0935342616a... | fa46b0935342616a... | yes |
| cicids2017_cleaned.csv | (not recorded)... | 0ce5a39edffcde3f... | recorded now |
| balanced_7classes_500000_each.csv | (not recorded)... | e09635e13f7562d0... | recorded now |

## Scripts

| Script | Exit | Seconds | Output comparison |
| --- | --- | --- | --- |
| r01_unsw_detection | 0 | 261 | r01_test_scores.npz: file differs (binary or text; check by eye); r01_unsw_detection.json: 61 values differ, e.g. [('/models/LogisticRegression/test/val_fpr_1pct/roc_auc', 0.9559, 0.9561), ('/models/LogisticRegression/test/default_0.5/confusion/fn', 1257, 1251), ('/models/LogisticRegression/test/val_fpr_1pct/confusion/fn', 11475, 11484)] |
| r02_calibration | 0 | 32 | r02_calibration.json: identical |
| r03_attack_category_severity | 0 | 21 | r03_attack_category.json: identical; r03_test_predicted_severity.npz: file differs (binary or text; check by eye) |
| r04_ablation_leakage_audit | 0 | 13 | r04_ablation_leakage_audit.json: identical |
| r05_nslkdd_official_split | 0 | 35 | r05_nslkdd.json: 15 values differ, e.g. [('/official_KDDTest+/LogisticRegression/false_positive_rate', 0.0739, 0.074), ('/official_KDDTest+/LogisticRegression/balanced_accuracy', 0.7752, 0.7751), ('/official_KDDTest+/LogisticRegression/confusion/fp', 718, 719)] |
| r06_ahp_robustness | 0 | 2 | r06_ahp_robustness.json: identical |
| r07_cross_dataset_v2 | 0 | 41 | r07_cross_dataset_v2.json: identical |
| r08_cicids2017_holdout_v2 | 0 | 25 | r08_cicids2017_holdout_v2.json: identical |
| r09_build_scenarios | 0 | 7 | r09_scenarios.csv: file differs (binary or text; check by eye); r09_scenario_cards_ar.md: file differs (binary or text; check by eye) |
| r10_decision_evaluation | 0 | 46 | r10_decisions_stage1.csv: file differs (binary or text; check by eye); r10_stage1.json: identical; r10_stage2.json: identical |
| r11_unsw_duplicates | 0 | 80 | r11_unsw_duplicates.json: identical |
| r12_dataset_manifest | 0 | 16 | r12_dataset_manifest.json: identical |
| r13_external_panel | 0 | 5 | r13_external_panel.json: identical |
| r14_final_checks | 0 | 7 | r14_final_checks.json: identical |
| r15_figures | 0 | 9 | fig4_ahp_weights.png: file differs (binary or text; check by eye); fig5_decision_differences.png: file differs (binary or text; check by eye) |
| r16_level_bounds | 0 | 1 | r16_level_bounds.json: identical |
| r17_base_rate_thresholds | 0 | 12 | r17_base_rate_thresholds.json: identical; r17_test_isotonic_scores.npz: file differs (binary or text; check by eye) |
| r18_threshold_study | 0 | 796 | r18_locked_thresholds.json: identical; r18_threshold_study.json: identical; r18_tradeoff_curves.csv: file differs (binary or text; check by eye) |
| r19_ablation_and_paired | 0 | 475 | r19_ablation_and_paired.json: identical |
| r20_error_propagation | 0 | 679 | r20_error_propagation.json: identical |
| r21_panel_audit | 0 | 6 | r21_panel_audit.json: identical |
| r22_cross_dataset_audit | 0 | 112 | r22_cross_dataset_audit.json: 124 values differ, e.g. [('/feature_distributions/CICIoT2023/avg_packet_size/1[0]', 'MISSING', 50.0), ('/feature_distributions/CICIoT2023/duration_s/0[0]', 'MISSING', 0.6828), ('/adaptation_independence_check/by_dataset/CICIDS2017/rows_shared_with_evaluation_1pct', 'MISSING', 0)] |

Finished 2026-10-31T16:17:21.
