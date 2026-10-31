# Reproducibility record (30 and 31 October 2026)

Answers item P0-6 of the supervisor's review of 30 October 2026: data and code provenance, an independent
re-run from a fresh environment, and a record of who executed what.

## 1. Data files

The dataset files are not stored here (size and licence). The UNSW-NB15 and NSL-KDD files used by the
re-run below were obtained from public mirrors and checked against the row and label counts of
`results/revision/r12_dataset_manifest.json`; all four match exactly. One mirror swaps the names of the two
UNSW-NB15 files; the training file has 175,341 rows and the testing file 82,332, as in the official release.

| File | Rows | SHA-256 |
| --- | --- | --- |
| UNSW_NB15_training-set.csv | 175,341 | bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa |
| UNSW_NB15_testing-set.csv | 82,332 | 734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559 |
| KDDTrain+.txt | 125,973 | 1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95 |
| KDDTest+.txt | 22,544 | fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84 |
| cicids2017_cleaned.csv | 2,520,751 | 0ce5a39edffcde3f… (first 16 characters only, as printed in the author's re-run report of 31 October 2026) |
| balanced_7classes_500000_each.csv | 2,878,962 | e09635e13f7562d0… (idem) |

The author's own UNSW-NB15 files (re-run of 31 October 2026, section 2b) have other SHA-256 values, beginning with
0ebd9eb06d53ae22 (training) and 7ec02e7e44d72bd2 (testing). They are the recorded files with LF instead of CRLF line
endings: `python src/04_revision/hash_line_endings.py data/UNSW_NB15_training-set.csv data/UNSW_NB15_testing-set.csv`
prints exactly these values for the LF version (log in `results/reproduction/20261031_153220/CONTENT_COMPARISON.md`).

Split manifest: `results/revision/splits/unsw_fit_val_cal_ids.csv` lists the `id` of every row of the
UNSW-NB15 training file and whether it belongs to FIT (122,738 rows), VAL (26,301) or CAL (26,302); the parts
do not overlap. `results/revision/r23_provenance_manifest.json` also lists the test rows shown in the 30
scenarios. Every script that draws random numbers uses the seed 42.

## 2. Clean-room re-run (cloud, 30 October 2026)

A fresh clone of the repository, with only the four files above added, was run from scratch by the AI assistant
(Python 3.13.18, NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, SciPy 1.18.1, Matplotlib 3.11.2;
`requirements-lock.txt`). Every output was compared with the stored one.

Paper table and figure numbers in this section are those of the paper version of 24 October 2026.

| Script | Result of the comparison |
| --- | --- |
| r01 UNSW-NB15 detection (paper Table 6) | identical except the three run times; stored test scores identical |
| r02 calibration and base rate (paper Table 8) | identical (578 values) |
| r03 attack category and severity | identical; stored predictions identical |
| r04 leakage audit (paper Table 9) | identical |
| r05 NSL-KDD | identical for Random Forest, Decision Tree and HGB; Logistic Regression differs in the fourth decimal (official test accuracy 0.7543 stored, 0.7539 re-run; 8 of 12,833 attacks), a solver difference that leaves every reported rounded value unchanged |
| r06 AHP robustness (paper Table 11, Figure 4) | identical |
| r09 scenarios, r10 decision evaluation (paper Table 13, Figure 5), r11, r13, r14, r16 | identical |
| r15 figures | images redrawn from identical data; PNG bytes differ only through the plotting library |
| r07, r08, r12 (CIC files; paper Table 7, Figure 2) | not re-run in the cloud: the CIC files could not be obtained there; re-run by the author (section 2b) |
| r22 transfer audit | run on UNSW-NB15 only in the cloud; complete run by the author (section 2b) |

## 2b. Independent re-run by the author (31 October 2026)

The first author ran `python src/04_revision/reproduce.py` on his own computer (Windows 11, Python 3.13.17 with the
locked versions above) on a fresh download of this repository, with all six dataset files. The unedited report and
the console log of every script are in `results/reproduction/20261031_153220/`.

| Scripts | Result of the comparison |
| --- | --- |
| r02–r04, r06–r08, r10–r14, r16–r21 | every JSON result identical, including the CIC scripts r07, r08, r12 and the second-revision scripts r17–r21 |
| r05 | identical except Logistic Regression, which differs in the fourth decimal, as in the cloud re-run |
| r01 | Random Forest and HGB identical; Logistic Regression differs by up to 0.006 (for example FPR at the F1 threshold 0.3906 stored, 0.3849 re-run; that threshold 0.50 and 0.52), which is new in this run and changes some rounded Logistic Regression values by up to 0.6 percentage points (at the F1 threshold, which moved from 0.50 to 0.52) and by 0.1 point at 0.5 and at the 1% FPR threshold; DADE does not use this model |
| r09, r10, r18 CSV and Markdown outputs | same content, checked by `src/04_revision/compare_outputs.py` on the output files the author sent (`results/reproduction/20261031_153220/outputs/`; log `CONTENT_COMPARISON.md` in the same folder); the report's "file differs (check by eye)" came from a byte comparison |
| score files (`.npz`) | same arrays within 1e-9 (same check), except the Logistic Regression scores of r01, which differ by up to 0.19 for single test flows |
| r15 figures | images redrawn; bytes differ |
| r22 | the CIC parts (adaptation independence on all three datasets, sensitivity variants V1–V3) were run for the first time; `results/revision/r22_cross_dataset_audit.json` now holds this complete run |

## 3. Who executed what

| Work | Executed by | Where |
| --- | --- | --- |
| Scripts r01–r06, r09–r11, r13–r16 (first revision, 16–28 October 2026) | AI assistant, on instructions and review of the author | cloud workspace |
| Scripts r07, r08, r12 (CIC files) | the author | author's computer (library versions not recorded) |
| Expert and practitioner data collection, form content, consent | the author | form service |
| Building the online forms and closing them on 25 October 2026 (with read access to the form account, which holds the named responses) | AI assistant, at the author's request | form service, through a connector |
| Second revision r17–r24 and the clean-room re-run above (30 October 2026) | AI assistant | cloud workspace |
| Manuscript figures and related-work coverage search (`src/05_manuscript`, 30 October 2026) | AI assistant | cloud workspace |
| Independent re-run of every analysis script r01–r22, including r07, r08, r12 and r22 (31 October 2026) | the author, with `reproduce.py` | author's computer; report in `results/reproduction/20261031_153220/` |

The manuscript does not claim reproducibility on the strength of its own statements: the re-run reports
in `results/reproduction/` are the evidence, and the author's re-run is listed separately from the cloud runs.
The re-runs the author reported on 17, 21 and 24 October 2026 left no report or log in this repository and are
not used as evidence. Scripts r17 to r24 were first run on 30 October 2026, so the clean-room comparison above
does not cover them.

## 4. Re-running everything yourself

```
pip install -r requirements-lock.txt
python src/04_revision/reproduce.py
python src/04_revision/reproduce.py --only=r25_decision_sensitivity,r26_ttl_and_label_conflicts   # the two scripts added on 31 October 2026
```
The script records the user, machine and versions, checks the SHA-256 of each dataset file, copies the stored
results, runs every analysis script (r01 to r22, r25, r26) whose data are present, compares each output with the stored version
(JSON values within 1e-9; CSV, NPZ and text files by content with `compare_outputs.py`) and writes
`results/reproduction/<date>/REPRODUCTION_REPORT.md`. Scripts whose data are missing are skipped and their
stored results are kept.

## 5. Repository history

Until 25 and 28 October 2026 the data files recorded the field and experience band of each expert and
practitioner code and the sector of each practitioner. These were then reduced to panel totals, but the old versions stayed in the
history, where a reader could link a code to a person in a small panel. The original repository is therefore
kept unchanged and private as evidence (its GitHub-signed commits date the pre-registration), and this public
repository carries the same history with only those fields removed. `docs/COMMIT_MAP.md` maps every
original commit to its public counterpart. A Git bundle of the original repository is kept by the author
(SHA-256 8a97adaab451b514eea98ff3da50f192d7819761da6b7eff9363e5df6102502e) and can be given to examiners.
