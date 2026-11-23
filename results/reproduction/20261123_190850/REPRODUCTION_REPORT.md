# Reproduction report 20261123_190850

- Run by: HP on khalid (Windows-11-10.0.26200-SP0)
- Started: 2026-11-23T19:08:50
- Python 3.14.7
- numpy 2.5.2
- pandas 3.0.5
- sklearn 1.9.0
- scipy 1.18.0
- matplotlib 3.11.2

## Dataset files

| File | Expected SHA-256 | Found | Match |
| --- | --- | --- | --- |
| UNSW_NB15_training-set.csv | bec7dd5ec88dc2a0... | 0ebd9eb06d53ae22... | NO |
| UNSW_NB15_testing-set.csv | 734fe6642edf758f... | 7ec02e7e44d72bd2... | NO |
| KDDTrain+.txt | 1b86d2f957b33082... | missing | |
| KDDTest+.txt | fa46b0935342616a... | missing | |
| cicids2017_cleaned.csv | not present on this machine | missing | |
| balanced_7classes_500000_each.csv | not present on this machine | missing | |

## Scripts

| Script | Exit | Seconds | Output comparison |
| --- | --- | --- | --- |
| r03_attack_category_severity | 0 | 27 | r03_attack_category.json: identical; r03_test_predicted_severity.npz: same content (4 arrays; numeric arrays within 1e-9; pred_cat (text array, identical stored bytes), true_cat (text array, identical stored bytes)) |

Finished 2026-11-23T19:09:19.
