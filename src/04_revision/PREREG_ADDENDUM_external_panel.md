# Pre-registration addendum — independent external panel (declared 2026-10-18, before any external answer)

**Status:** secondary analysis. It does not replace or change any primary analysis fixed in the r10 pre-registration (commit a5788aa).

**Why:** the six panel experts both elicited the AHP weights and labelled the scenarios. Practitioners from outside the faculty, who took no part in the AHP elicitation, give an independent check of the engine's decisions.

**Design:**
- 3 or 4 external raters (X1–X4) from outside the faculty.
- Each rater labels all 30 scenarios (A01–C10) on the same three-level scale: High / Medium / Low.
- Form: Tally BE46AQ. The rater ID is held in the hidden URL field `x`.
- Profile questions: field, years of experience, sector.
- Consent is collected in the form, and answers are analysed without names.

**Planned analysis (fixed now):**
1. Inter-rater agreement among the external raters: Krippendorff's α (ordinal).
2. External reference label per scenario = majority vote. Ties: if all votes differ (3 raters), the label is Medium; if two labels tie (4 raters, 2–2), the higher priority wins, because missing a high-priority alert costs more under the asymmetric cost matrix.
3. DADE decision vs the external reference: weighted κ (linear) and cost under the same asymmetric cost matrix as the primary analysis.
4. Results are reported as secondary in Section 4.7. The primary conclusions rest on the pre-registered panel analysis.

**Fixed rules:**
- No external answer will be used to tune weights, thresholds or the cost matrix.
- An incomplete or non-consented response is excluded and reported.
