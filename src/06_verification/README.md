# Independent verification (23 November 2026)

`recompute_decision_layer.py` recomputes the main decision-layer and base-rate results of the manuscript from the
raw inputs (coded labels, AHP answers, scenario inputs, isotonic test scores, split manifest and the UNSW-NB15
files). It was written from the definitions in the manuscript (Sections 3.2, 4.3, 4.6, 4.7), not from the analysis
scripts in `src/04_revision`, so that an error shared by a script and its own output would show up as a difference.

```
python src/06_verification/recompute_decision_layer.py
```

Output: `results/verification/recompute_decision_layer.json` (about 20 seconds; the UNSW-NB15 files must be in
`data/`). It covers:

- Table 6 (all six methods, 120 expert labels, leave-one-expert-out AHP weights) and the 30 stored scenario levels;
- Table 7 (confusion against the expert consensus) and the consensus counts;
- Section 5.7 (DADE and the baselines against the practitioners' reference labels, panel agreement) and the
  fixed-cut-off external columns of Table 9;
- Table 10: false and true negatives at each threshold, the threshold of Eq. (4) for the true and the EM prior, the
  expected true positives, and the equivalence of re-applying 0.5 to corrected scores and Eq. (4);
- the split manifest (every training row in exactly one of FIT, VAL and CAL; attack share per split), the use of
  the predicted, not the true, category for TS in all 30 scenarios, and scenario flows identical to a training row.

Result: every recomputed value agrees with the stored results. The recomputation also showed three values that the
manuscript of 1 November 2026 had rounded twice (from four stored decimals): DADE's quadratic kappa (0.74246, printed
0.743, now 0.742), the isotonic ECE (0.09448, printed 0.095, now 0.094) and the share of DADE decisions changed by
BIA perturbations (0.097483 in a full-precision re-run of r20 whose output file was identical to the stored one;
printed 9.8%, now 9.7%). It also found that one scenario flow (C08, benign) is identical to a CAL row labelled
Fuzzers, and none to a FIT row.

Executed by the AI assistant in a cloud workspace at the author's request; the author has not re-run this script.
