# Risk Score Diagnostics

## Completion
- Status: fully completed.

## Direct findings
- School risk file covers 79 countries and 21358 schools.
- High-risk top decile threshold: 81.0.
- Low-risk bottom decile threshold: 24.0.

## Evidence-based interpretation
- `school_mean_math` correlates with risk at 0.377; high-risk minus low-risk mean difference is 81.22.
- `low_perf_share_400` correlates with risk at -0.368; high-risk minus low-risk mean difference is -0.34.
- `school_mean_escs` correlates with risk at 0.234; high-risk minus low-risk mean difference is 0.48.
- `teacher_response_count` correlates with risk at 0.231; high-risk minus low-risk mean difference is 6.65.
- `within_school_gap` correlates with risk at 0.212; high-risk minus low-risk mean difference is 29.27.

## Caution
- The supplied risk score is treated as an observed input, not a validated causal construct.
- Diagnostic patterns suggest whether the score aligns more with low performance, inequality, or context strain, but they do not reveal the original score formula.