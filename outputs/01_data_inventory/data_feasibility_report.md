# Data Feasibility Report

## Direct findings
- Inventory covers 7 CSV datasets and 9 SAV datasets present locally.
- Cross-country trend analysis is supported by the multi-year CSV files.
- School-level risk profiling is supported by the supplied 2022 risk file plus the raw 2022 school and student SAV files.
- Within-school SES-gap analysis is supported for 2022 from the raw student questionnaire using `ESCS`, plausible values, and weights.
- Clustering / segmentation is supportable for schools in 2022 once student, school, and teacher features are merged.
- Intervention mapping is supportable because the intervention library is present, but it remains evidence-informed rather than causal.

## Constraints
- The derived multi-year student and school CSVs are too sparse for a strong multi-year driver analysis on their own.
- The richer correlates analysis requires the 2022 SAV files; CSV-only mode would leave Steps 6-8 materially weaker.
- Timing, CRT, and financial-literacy SAV files are best treated as enrichment layers rather than primary outcome sources.

## Sufficiency answer
- Yes, the current datasets are enough for the intended 12-step project in hybrid mode.
- CSV-only is enough for Steps 1-3 and part of Step 4 and Step 9, but not enough for a defensible full driver / within-school / segmentation pipeline.