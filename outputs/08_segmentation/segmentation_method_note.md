# Segmentation Method Note

## Completion
- Status: fully completed.

## Published method
- Official published segmentation: `rule_based`.
- Unit of segmentation: schools.
- Core inputs: performance, within-school SES gap, overall risk, SES composition, and climate/resource strain proxies.

## Why rule-based was preferred
- The competition use case needs interpretable profiles that can be mapped to transparent intervention logic.
- The school profiles already support a stable rule-based typology without relying on opaque optimisation.
- Earlier unsupervised clustering attempts were not retained as the published method because they were less stable and less interpretable in this execution environment.

## Published segment counts
- `Low-achievement broad support need`: 4676 schools.
- `High-achievement unequal schools`: 4676 schools.
- `Resilient equitable performers`: 3055 schools.
- `Strained high-inequality schools`: 2152 schools.
- `Mixed profile`: 573 schools.
- `Digitally constrained schools`: 330 schools.

## Interpretation
- These segments are descriptive school archetypes, not causal types.
- They support prioritisation and tailored recommendation logic, not deterministic prescriptions for a named school.

## Limitation
- This step deliberately favours interpretability over technical clustering novelty. That is appropriate for the current observational data and competition narrative.