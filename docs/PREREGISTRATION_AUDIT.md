# Pre-registration audit (30 October 2026)

This audit answers item P0-3 of the supervisor's review of 30 October 2026: verify the protocol of
17 October 2026 against the code and the analyses, and separate planned from exploratory results.

## Evidence and how to check it

| Event | Commit (original repository) | Time (UTC+3) | How to check |
| --- | --- | --- | --- |
| Decision evaluation (r10) and scenario template committed before any expert label | a5788aa | 17 Oct 2026 14:16 | Commit signed by GitHub (web upload); `git show a5788aa` |
| First expert scenario label submitted | (Tally scenario forms) | 18 Oct 2026 | Submission times exported from the form service; all scenario submissions fall between 18 and 20 October |
| External-panel addendum committed | 42c1b19 | 18 Oct 2026 00:28 | Signed by GitHub; the first external answer (X2) had been submitted at 00:26, two minutes earlier |
| Timing note added to the addendum | ecee1dd | 20 Oct 2026 19:22 | Signed by GitHub |
| Expert labels committed | 100dca8 | 21 Oct 2026 16:59 | Signed by GitHub |
| r10 stage 2 run and results committed | 88107c1 | 21 Oct 2026 17:12 | `git diff a5788aa 88107c1 -- src/04_revision/r10_decision_evaluation.py` is empty: r10 ran unchanged |

The commit identifiers above are those of the original repository, which is kept unchanged as private
evidence (its history contains participant background data; see `docs/REPRODUCIBILITY.md`, section
"Repository history"). The public repository holds the same files with that background removed from the
history; `docs/COMMIT_MAP.md` gives the public identifier of every original commit.

## Planned versus exploratory

| Result reported | Status | Basis |
| --- | --- | --- |
| Leave-one-expert-out comparison of DADE with B1–B5: exact agreement, macro-F1, quadratic kappa, High recall, under-prioritisation, over-escalation, mean cost | **Planned (primary)** | r10 at a5788aa, stage 2 |
| Paired cluster bootstrap of DADE minus each baseline (2,000 resamples of scenarios) | **Planned** | r10 at a5788aa |
| Consensus-label comparison, Krippendorff's alpha with bootstrap interval, per-form alpha | **Planned (secondary)** | r10 at a5788aa |
| Results by profile criticality, by profile and by true traffic type; failure cases | **Planned** | r10 at a5788aa |
| Cut-offs tuned on one form and tested on the other two | **Planned** | r10 at a5788aa |
| Stage-1 checks: monotonicity, weight and cut-off perturbation, severity table, counterfactual BIA | **Planned** | r10 at a5788aa |
| Designation of "DADE versus the detection score alone (B1)" as the single test of hypothesis H1 | **Not in the protocol** | The protocol fixed the comparison with all five baselines but did not single out one; H1's choice of B1 was made later and is reported as such |
| External panel: alpha, majority reference label, linear kappa and cost against the reference | **Secondary; registered two minutes after the first external answer** | 42c1b19; the answer was not opened before the commit, but this cannot be proven from timestamps alone, so the analysis is labelled secondary and not prospectively registered |
| r14 checks (CR before and after the feedback round, leave-one-expert-out without E3, quadratic kappa against the external reference) | **Exploratory** | committed 21 Oct 2026 17:09, after the labels |
| r16 level bounds per profile | **Exploratory (analytic)** | committed 26 Oct 2026 |
| r17–r22 (second revision: base-rate policies, threshold study, ablations, error propagation, panel audit, transfer audit) | **Exploratory (post hoc)** | written on 30 Oct 2026 in response to the review, after all labels were known |

No threshold, weight or cost-matrix entry used in the primary analysis was changed after the labels were
collected. The thresholds tuned in r18 are evaluated only on scenarios or raters not used to tune them.

## Immutable records to keep

- The original repository, unchanged and private, with its GitHub-signed commits.
- A Git bundle of the original repository (SHA-256 recorded in `docs/REPRODUCIBILITY.md`).
- The form definitions and the submission exports of every form, kept by the author offline (they contain
  names and are not published).
