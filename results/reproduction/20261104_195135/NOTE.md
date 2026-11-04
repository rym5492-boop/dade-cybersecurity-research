# Clean-room re-run in the cloud, 4 November 2026

Run by the AI assistant at the author's request, to replace the cloud re-run of 30 October 2026, whose outputs were
not kept.

- Fresh clone of the public repository from GitHub at commit 0562efc, with the four NSL-KDD and UNSW-NB15 files
  added to `data/` (SHA-256 as recorded; see the report). The CIC files are not available in the cloud, so r07, r08,
  r12 and r22 were skipped and their stored results were kept.
- Python 3.13.18 with the versions of `requirements-lock.txt` (NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1,
  SciPy 1.18.1, Matplotlib 3.11.2).
- Command: `python src/04_revision/reproduce.py` from the repository root. `REPRODUCTION_REPORT.md` is the unedited
  report, `log_*.txt` the console output of each script, `console.log` the console output of reproduce.py,
  `reference/` the stored results it compared against, and `outputs/` the files the scripts wrote in this run.

Result: every output was identical to the stored one (JSON values equal, CSV and Markdown byte-identical, score
arrays within 1e-9), except

- the two figures of r15, which are redrawn and not compared;
- Logistic Regression in r05 (NSL-KDD), whose metrics differ in the fourth decimal, as in the cloud re-run of
  30 October 2026: on the official test file it classified 8 of the 12,833 attacks differently (TP 8,003 against 8,011
  stored; accuracy 0.7539 against 0.7543). No percentage printed in thesis Table 5.3 changes; thesis Table C.2 gives
  the stored counts. DADE does not use this model.

Two scripts that `reproduce.py` does not run were also run in the same clone: `label_timing.py` and
`ahp_min_changes.py`; both outputs were byte-identical to the stored ones.
