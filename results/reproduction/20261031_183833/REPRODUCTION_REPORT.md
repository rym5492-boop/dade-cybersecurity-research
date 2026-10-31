# Reproduction report 20261031_183833

- Run by: HP on khalid (Windows-11-10.0.26200-SP0)
- Started: 2026-10-31T18:38:33
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
| r25_decision_sensitivity | 0 | 40 | r25_decision_sensitivity.json: identical |
| r26_ttl_and_label_conflicts | 0 | 14 | r26_ttl_and_label_conflicts.json: identical |

Finished 2026-10-31T18:39:33.
