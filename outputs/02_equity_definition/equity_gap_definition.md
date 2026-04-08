# Equity Gap Definition

## Primary definition
- Country-level primary metric: weighted mathematics score gap between the top and bottom ESCS quartiles within each country-year.
- School-level primary metric: weighted mathematics score gap between the top and bottom ESCS quartiles within each school.
- Secondary school-level estimate: weighted within-school SES slope in mathematics.

## Why this definition
- It is directly anchored in the available student SES measure (`ESCS`) and mathematics plausible values.
- It is transparent enough for a competition narrative and easier to explain than a fully model-based index.

## Supplied vs rebuilt metric
- Mean absolute difference between supplied `GAP` and rebuilt quartile gap: 6.33 points.
- Maximum absolute difference: 57.66 points.
- Interpretation: the supplied gap behaves like a SES-based mathematics inequality measure, but the rebuilt definition is retained for transparency.

## Secondary operationalisations considered
- Regression-based SES slope: useful for within-school inequality and less sensitive to quartile cut points.
- Between-school vs within-school decomposition: useful conceptually, but not taken as the headline metric in v1 because the competition deliverable needs a simpler explanation.

## ESCS comparability note
- The local `escs_trend.csv` provides a cross-cycle ESCS re-scaling for cycles [5, 6, 7], which supports cautious comparability for 2012-2018.
- For the main multi-year country gap story, the supplied and rebuilt CSV-based series are used; for the rich driver work, the focus narrows to 2022.