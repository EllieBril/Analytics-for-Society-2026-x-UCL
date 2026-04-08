# SDG 4 Equity Diagnostic Prototype

This repository is the canonical working project for the SDG 4 competition concept:
a data-driven equity diagnostic that combines country trajectory analysis, school-profile
segmentation, and transparent intervention prioritisation using PISA-style data.

## What this repository is for

- Diagnose where socio-economic achievement gaps are largest or worsening.
- Identify school-level risk and within-school inequality patterns.
- Translate those profiles into evidence-informed intervention priorities.
- Support a competition-ready dashboard prototype without making causal claims the data cannot support.

## Canonical project logic

1. `Country diagnostic`
   Use all available years to identify high-gap and worsening systems.
2. `Observed school profile`
   Use the rich 2022 merged dataset to inspect anonymised school profiles.
3. `Rule-based school segmentation`
   Publish interpretable school archetypes rather than opaque clustering output.
4. `Transparent recommendation logic`
   Map each profile to intervention categories with explicit reasoning.
5. `Evidence and limits`
   Keep model-based correlates and V7 findings in a secondary evidence layer.

## Repository structure

- `app.py`
  Streamlit entrypoint.
- `src/`
  Canonical Python source for the analysis pipeline and dashboard logic.
- `outputs/`
  Canonical generated artifacts from the 12-step workflow.
- `v7/`
  Secondary evidence layer from the 2022 cluster-level model. Not the canonical recommender.
- `docs/`
  Architecture, technical review, privacy, and implementation guidance.

Local working copies may also keep a non-public `archive/` directory for legacy materials, but that directory is intentionally excluded from the public competition branch.

## Source of truth

- The official backend is the local 12-step pipeline in `src/`.
- The official presentation layer is the Streamlit dashboard in `app.py` and `src/dashboard_app.py`.
- The official published artifacts are the files in `outputs/`.
- The V7 model is supporting evidence only.

## Run the analysis pipeline

```bash
python3 src/run_pipeline.py --data-root . --focus-year 2022 --min-school-n 20 --seed 42
```

## Run the dashboard

```bash
streamlit run app.py
```

## Method boundary

- Safe claim: the project supports evidence-informed prioritisation.
- Unsafe claim: the project proves why a named school has a given gap or guarantees that an intervention will causally reduce it.

Read `docs/ARCHITECTURE_PLAN.md`, `docs/TECHNICAL_REVIEW.md`, and `docs/PRIVACY_AND_SECURITY.md` before extending the product.
