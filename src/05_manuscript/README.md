# Manuscript scripts (30 October 2026)

Scripts that produce the figures and the related-work coverage records of the IJCIP manuscript. They read
stored result files and compute no new result.

| Script | Output |
| --- | --- |
| `make_figures.py` | Figs. 1 to 8 of the manuscript (600 dpi PNG and vector PDF). Figs. 1, 3, 4 and 6 are copies of the figures drawn by `src/04_revision/r24_figures_revision2.py`; Figs. 2, 5, 7 and 8 are drawn here from r10, r06, r07 and r02. Fig. 1 is a diagram and plots no data. |
| `crossref_coverage_search.py` | `results/manuscript/crossref_coverage_search_2026-10-30.json`: eight queries to the Crossref REST API with their time, hit count and top 20 records (Supplementary Table S1). |
| `build_supplementary.py` | Supplementary Tables S1 to S10 (Word file) from the stored result files and the coverage records; usage `python src/05_manuscript/build_supplementary.py [output.docx]`. |
| `coverage_screening.py` | `results/manuscript/coverage_screening_2026-10-30.csv`: the 146 unique records with the title screening codes. |

Run from the repository root, for example `python src/05_manuscript/make_figures.py`. These scripts were
written and run by the AI assistant (Claude, Anthropic) under the first author's direction, like r17 to r24
(docs/REPRODUCIBILITY.md).
