# Reproduction report 20261104_204819

- Run by: HP on khalid (Windows-11-10.0.26200-SP0)
- Started: 2026-11-04T20:48:20
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
| r01_unsw_detection | 0 | 118 | r01_test_scores.npz: arrays differ: LogisticRegression (largest absolute difference 0.186786); attack_cat (text array, identical stored bytes); r01_unsw_detection.json: 61 values differ, e.g. [('/models/LogisticRegression/test/val_max_f1/balanced_accuracy', 0.7908, 0.7927), ('/models/LogisticRegression/test/default_0.5/confusion/fn', 1257, 1251), ('/models/LogisticRegression/test/val_fpr_1pct/roc_auc', 0.9559, 0.9561)] |
| r02_calibration | 0 | 33 | r02_calibration.json: identical |
| r03_attack_category_severity | 0 | 18 | r03_attack_category.json: identical; r03_test_predicted_severity.npz: same content (4 arrays; numeric arrays within 1e-9; pred_cat (text array, identical stored bytes), true_cat (text array, identical stored bytes)) |
| r04_ablation_leakage_audit | 0 | 15 | r04_ablation_leakage_audit.json: identical |
| r05_nslkdd_official_split | 0 | 37 | r05_nslkdd.json: 15 values differ, e.g. [('/random_80_20_inside_KDDTrain+ (paper protocol)/LogisticRegression/attack_precision', 0.9763, 0.9764), ('/random_80_20_inside_KDDTrain+ (paper protocol)/LogisticRegression/confusion/fp', 274, 273), ('/official_KDDTest+/LogisticRegression/roc_auc', 0.7924, 0.7927)] |
| r06_ahp_robustness | 0 | 1 | r06_ahp_robustness.json: identical |
| r07_cross_dataset_v2 | 0 | 40 | r07_cross_dataset_v2.json: identical |
| r08_cicids2017_holdout_v2 | 0 | 21 | r08_cicids2017_holdout_v2.json: identical |
| r09_build_scenarios | 0 | 5 | r09_scenarios.csv: same content (30 rows x 24 columns; line endings ignored); r09_scenario_cards_ar.md: same text (line endings ignored) |
| r10_decision_evaluation | 0 | 46 | r10_decisions_stage1.csv: same content (30 rows x 19 columns; line endings ignored); r10_stage1.json: identical; r10_stage2.json: identical |
| r11_unsw_duplicates | 0 | 83 | r11_unsw_duplicates.json: identical |
| r12_dataset_manifest | 0 | 16 | r12_dataset_manifest.json: identical |
| r13_external_panel | 0 | 3 | r13_external_panel.json: identical |
| r14_final_checks | 0 | 2 | r14_final_checks.json: identical |
| r15_figures | 0 | 3 | fig4_ahp_weights.png: figure file, not compared (drawn from the compared result files); fig5_decision_differences.png: figure file, not compared (drawn from the compared result files) |
| r16_level_bounds | 0 | 0 | r16_level_bounds.json: identical |
| r17_base_rate_thresholds | 0 | 12 | r17_base_rate_thresholds.json: identical; r17_test_isotonic_scores.npz: same content (3 arrays; numeric arrays within 1e-9; attack_cat (text array, identical stored bytes)) |
| r18_threshold_study | 0 | 748 | r18_locked_thresholds.json: identical; r18_threshold_study.json: identical; r18_tradeoff_curves.csv: same content (124 rows x 7 columns; line endings ignored) |
| r19_ablation_and_paired | 0 | 209 | r19_ablation_and_paired.json: identical |
| r20_error_propagation | 0 | 400 | r20_error_propagation.json: identical |
| r21_panel_audit | 0 | 3 | r21_panel_audit.json: identical |
| r22_cross_dataset_audit | 0 | 74 | r22_cross_dataset_audit.json: identical |
| r25_decision_sensitivity | 0 | 17 | r25_decision_sensitivity.json: identical |
| r26_ttl_and_label_conflicts | 0 | 9 | r26_ttl_and_label_conflicts.json: identical |

Finished 2026-11-04T21:20:17.
