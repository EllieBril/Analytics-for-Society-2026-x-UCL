# Country Trajectory Analysis

## Completion
- Status: fully completed.

## Key findings
- Latest ranking covers 107 countries/economies with at least one observed wave.
- 36 countries are classed as worsening and 18 as improving, but only where at least 3 waves are available.
- Supplied 2009/2022 trajectory file is internally consistent with the country-year gap file: `True`.
- Median latest gap among ranked countries is 80.2 points.

## Interpretation discipline
- Latest gap levels are directly observed from available country-year summaries.
- Annual change uses an OLS slope over all available waves when at least 3 observations exist, with a first-last fallback only for thinner series.
- Worsening or improving labels are inferred from observed wave-to-wave comparisons, not from causal explanations.
- Countries with 1-2 time points are kept in the level rankings but flagged as weak evidence for trend claims.