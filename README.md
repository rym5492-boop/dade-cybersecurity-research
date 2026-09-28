# DADE: Dynamic Adaptive Decision Engine — Reproducibility Package

This repository contains the complete, reproducible code pipeline for the paper
**"Dynamic Adaptive Decision Engine for Integrating Machine Learning-Based Cyber
Risk Prediction with Business Continuity"** (Khalid Khadhear Alenezi, MSc
Cybersecurity, Northern Border University).

This package addresses the reproducibility requirements identified in the
supervisor technical review (Section 3.14): explicit preprocessing, feature
definitions, random seeds, hyperparameters, split protocols, normalization
steps, AHP calculations, and threshold selection are documented below and in
the code itself.

## Repository Structure

```
dade_repo/
├── README.md                          <- this file
├── requirements.txt                   <- Python package versions
├── data/
│   └── ahp_expert_judgments_anonymized.json  <- six experts' AHP comparisons + scenario votes (no names)
├── results/                            <- all JSON/CSV output files from every experiment
└── src/
    ├── 01_predictive_layer/           <- ML models (NSL-KDD + three modern datasets)
    │   ├── 00_nsl_kdd_preliminary.py            <- Table 1 / Figures 2-4 (re-implementation, see note)
    │   ├── 01_unsw_nb15_experiment.py
    │   ├── 02_cicids2017_experiment.py
    │   ├── 02b_cicids2017_attack_family_holdout.py
    │   ├── 03_ciciot2023_experiment.py
    │   ├── 03b_ciciot2023_attack_family_holdout.py
    │   ├── 04_probability_calibration.py
    │   └── 05_cross_dataset_validation.py
    ├── 02_dade_engine/                 <- DADE weighting, scoring and validation
    │   ├── 00_ahp_expert_weights.py              <- AHP weights, CR, AIJ, leave-one-out (Section 4.3)
    │   ├── 00b_dade_priority_engine.py           <- Equation (1), thresholds, score-to-action mapping
    │   ├── 01_ablation_study.py
    │   ├── 02_sensitivity_analysis.py
    │   └── 03_cross_sector_analysis.py
    └── 03_analysis/                    <- figure-generation scripts
        └── make_*.py (7 scripts, one per figure)
```

## Data

Due to file size (largest file ~1 GB), raw datasets are **not included** in
this repository. They must be downloaded separately from their official
sources and placed in `data/`:

| Dataset | Official source | Files needed |
|---|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | `KDDTrain+.txt` |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| CICIDS2017 (cleaned) | Kaggle: `ericanacletoribeiro/cicids2017-cleaned-and-preprocessed` | `cicids2017_cleaned.csv` |
| CICIoT2023 (balanced sample) | Kaggle: `kibremogesf/kibremoges` | `balanced_7classes_500000_each.csv` |

All datasets are publicly available under their respective open licenses
(see each source for citation requirements).

## Environment

```
Python 3.11+
pandas==2.x
numpy==1.26+
scikit-learn==1.8+
matplotlib==3.x
```

Install with: `pip install -r requirements.txt`

## Reproducibility Details (per supervisor Section 3.14)

### Preprocessing (all datasets)
- Categorical features (protocol, service, state/flag) encoded with `LabelEncoder`,
  fit jointly on train+test to avoid unseen-category errors at inference time.
- Numeric features standardized with `StandardScaler` (fit on training data only,
  applied to test data — never fit on test data).
- Missing/infinite values: none found in any of the four datasets after using
  their official "cleaned" distributions (verified via `isnull().sum()` and
  explicit infinity checks; see inline comments in each script).

### Random seeds
All scripts fix `random_state=42` (scikit-learn) and `np.random.seed(42)` /
`np.random.RandomState(42)` wherever stochastic sampling occurs, to ensure
exact reproducibility of splits, model initialization, and illustrative
sampling.

### Train/test split protocol
- **UNSW-NB15**: uses the dataset's **official** pre-defined train/test split
  (not a random resplit), since this is the split most comparable to prior
  literature and reveals genuine distribution shift (see Section 5.1 of the
  paper).
- **NSL-KDD, CICIDS2017, CICIoT2023**: 80/20 stratified random split
  (`train_test_split`, `stratify=y`, `random_state=42`), disclosed as a
  limitation relative to grouped/temporal splitting (Section 5.3).
- **CICIDS2017 and CICIoT2023**: due to compute-environment memory limits, a
  stratified subsample of 400,000 records (from ~2.5–2.9 million) is used;
  this is disclosed explicitly in-code and in the paper.

### Hyperparameters
- Random Forest: `n_estimators=100–200`, `max_depth=12–20` (dataset-dependent
  for memory feasibility; exact value stated at the top of each script),
  `random_state=42`.
- HistGradientBoostingClassifier: `max_iter=100`, `random_state=42` (default
  scikit-learn regularization otherwise).
- Logistic Regression (baseline): `max_iter=500–1000`, `random_state=42`.
- 5-fold `StratifiedKFold` (`shuffle=True`, `random_state=42`) used for all
  cross-validation reported in the paper.

### DADE Priority Score calculation (Equation 1)
```
PS = w1*TS + w2*AC + w3*BI + w4*HI + w5*RTO + w6*RPO + w7*MLS
```
- `MLS`: real predicted probability of the attack class from the trained
  Random Forest (`predict_proba`).
- `TS` (Threat Severity): derived **independently** of MLS, from an
  attack-category severity taxonomy (DoS/Probe/R2L/U2R convention for
  NSL-KDD; dataset-native `attack_cat`/`MappedLabel` for UNSW-NB15/CICIoT2023).
  See `02_dade_engine/01_ablation_study.py` for the exact `SEVERITY_MAP`.
- `AC, BI, HI, RTO, RPO`: illustrative representative values (uniform random,
  fixed seed) pending real organizational data or expert elicitation — this
  is disclosed as a limitation throughout the paper (Sections 5.3, 5.4).
- Baseline weights: equal (`1/7` each) unless a specific AHP-informed
  sector profile is being tested (see `03_cross_sector_analysis.py` for the
  Healthcare/Finance illustrative weight vectors and their derivation
  rationale).

### Threshold selection (Priority Level mapping)
- High: PS >= 0.66
- Medium: 0.33 <= PS < 0.66
- Low: PS < 0.33
These thresholds are provisional (evenly spaced terciles), pending the
expert-based validation described in the paper's Section 4.4. Sensitivity
of these exact threshold values is tested in
`02_dade_engine/02_sensitivity_analysis.py`.

### Evaluation metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC (average precision),
False Positive Rate, Brier score, Expected Calibration Error (ECE, 10
equal-width bins), and Cohen's Kappa (for ablation agreement) are all
computed via standard `scikit-learn.metrics` functions; see each script for
exact function calls.

## How to Reproduce a Result

Example — reproducing the UNSW-NB15 predictive-layer results (Table 1-equivalent
for UNSW-NB15, Section 5.1 discussion):

```bash
cd src/01_predictive_layer
python 01_unsw_nb15_experiment.py
```
Output: printed metrics table + `unsw_results.json` (already provided in
`results/` for reference without re-running).

## Citation

If you use this code, please cite the associated paper (full citation to be
finalized upon publication).

## Contact

Khalid Khadhear Alenezi — st202506139@stu.nbu.edu.sa


### AHP expert weighting (Section 4.3)
`src/02_dade_engine/00_ahp_expert_weights.py` reads `data/ahp_expert_judgments_anonymized.json`
(six experts, 21 pairwise comparisons each, collected with a two-part online instrument).
Slider responses (1-9, 5 = equal) are mapped to Saaty intensities {1, 3, 5, 7, 9}; individual
weights use the row geometric-mean method; consistency uses CR = CI / RI with RI(7) = 1.32 and an
acceptance threshold of CR <= 0.10 fixed before analysis. All six matrices are aggregated with
AIJ (element-wise geometric mean). Output (`results/ahp_weights_results.json`) reproduces the
paper exactly: group CR = 0.0495; TS 0.262, BI 0.211, AC 0.189, MLS 0.131, RTO 0.083, RPO 0.077,
HI 0.048. The script also reports leave-one-expert-out stability and inter-expert agreement on
the validation scenarios. Participant names and affiliations are not included.

### Illustrative calculations (Tables 2 and 4)
Tables 2 and 4 are illustrative applications of Equation (1) to example inputs.
`src/02_dade_engine/00b_dade_priority_engine.py` implements the scoring engine used for them
(normalized criteria, equal or AHP weights, thresholds High >= 0.66 and Low < 0.33, and the
score-to-action mapping), so any input vector can be scored reproducibly.

### NSL-KDD preliminary experiment (Table 1, Figures 2-4)
`src/01_predictive_layer/00_nsl_kdd_preliminary.py` re-implements the preliminary NSL-KDD run
from the protocol stated in the paper (binary labels, label-encoded categorical features,
standardization, stratified 80/20 split, random_state = 42, 5-fold CV). With scikit-learn 1.8
it reproduces Table 1 within 0.3 percentage points (Random Forest 0.9990 vs. 0.9992, Decision
Tree 0.9977 vs. 0.9979, Logistic Regression 0.9520 vs. 0.9548) and the same two most important
features (src_bytes, dst_bytes). XGBoost requires `pip install xgboost`. The small differences
reflect library versions and unreported defaults of the original run and do not affect any
conclusion.
