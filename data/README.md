# data/

Place the downloaded dataset files here or in the repository root (see the main README for sources and file
names). The dataset files are not stored in the repository because of their size.

The anonymized expert data used by the revision scripts are stored here:

| File | Content | Used by |
| --- | --- | --- |
| `ahp_expert_judgments_anonymized.json` | 21 pairwise AHP comparisons per expert (E1–E6): final answers (`pairwise_responses`), the September answers (`pairwise_responses_original_sept2026`) and a revision log; field and experience only as panel totals | r06, r10, r14 |
| `scenario_labels.csv` | 120 blinded priority labels (6 experts × 20 scenarios; columns `expert_id, form, scenario_id, label`) | r10, r14 |
| `scenario_labels_TEMPLATE.csv` | Empty template for the labels file | — |
| `external_panel_labels.csv` | 120 labels from the four external practitioners (X1–X4), with consent answer, submission time and form submission identifier; no names, and no per-rater field, experience or sector | r13 |
| `external_panel_profile.json` | Field, experience and sector of the four external practitioners as panel totals only | r13 |
| `external_reference_labels.csv` | External reference label per scenario, written by r13 | r14 |

No names, e-mail addresses or other personal data are stored in these files.
