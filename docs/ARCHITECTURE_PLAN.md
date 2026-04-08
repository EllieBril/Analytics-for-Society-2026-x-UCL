# Architecture Plan

## Objective

Build a competition-ready SDG 4 decision-support prototype that is analytically rigorous, operationally clear, and careful about non-causal interpretation.

## Design decision

The repository follows a three-layer design:

1. `Canonical backend`
   The 12-step local pipeline in `src/` is the only official analysis engine.
2. `Presentation layer`
   The Streamlit dashboard reads canonical outputs and does not invent its own competing risk logic.
3. `Secondary evidence layer`
   The `v7/` model is retained as supporting evidence only.

## Why this design is the best fit

- It preserves the strongest analytical work: raw SAV ingestion, student-school linking, within-school inequality, segmentation, and intervention mapping.
- It keeps the competition representable product-oriented and understandable.
- It avoids the weakest pattern from the reference app: a heuristic school report that looks precise but is not strongly grounded in school-level evidence.

## Canonical data flow

1. Raw and derived input data are discovered locally.
2. `src/run_pipeline.py` generates canonical outputs in `outputs/`.
3. `app.py` and `src/dashboard_app.py` read those outputs.
4. `v7/` is used only inside the evidence layer and supporting documentation.

## Product views

### 1. Country Diagnostic
- Purpose: identify where equity gaps are highest or worsening.
- Unit: country/system.
- Data basis: all available years.

### 2. School Segments
- Purpose: show interpretable archetypes of school-level need.
- Unit: anonymised school profile.
- Data basis: rich 2022 merged dataset.

### 3. Observed Recommendation
- Purpose: map observed school profiles to intervention priorities.
- Unit: observed school profile.
- Interpretation: profile-based prioritisation, not causal proof.

### 4. Scenario Planner
- Purpose: support hypothetical planning in demos.
- Unit: user-defined hypothetical profile.
- Interpretation: planning aid only, not prediction.

### 5. Evidence & Limits
- Purpose: show correlates, V7 support, and explicit claim boundaries.

## Claim boundaries

- Safe: identify observed patterns, correlates, typologies, and evidence-informed priorities.
- Unsafe: claim the definitive cause of a school’s gap or predict a guaranteed intervention effect.

## File ownership

- `src/`
  Official codebase.
- `outputs/`
  Official generated artifacts.
- `docs/`
  Official project guidance and review notes.
- `v7/`
  Evidence supplement.

Local working copies may keep a non-public archive for bulky or legacy materials, but that is not part of the public branch structure.

## Immediate implementation standard

- Keep all explanatory files in English.
- Keep school-level presentation anonymised.
- Keep one source of truth for risk, segments, and recommendations.
- Keep any stronger model-based evidence behind an explicit caveat.
