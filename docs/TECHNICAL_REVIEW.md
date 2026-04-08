# Technical Review

## Findings

### 1. Critical: the reference app’s school risk logic is not strongly school-specific

The reviewed reference implementation uses a heuristic risk formula driven mainly by country gap, trajectory, school type, and student-teacher ratio. That is too weak to anchor a competition claim about school-specific diagnosis.

Why this matters:
- It creates an illusion of precision at the school level.
- It underuses the richer school-profile information already available in the local pipeline.

Required resolution:
- Keep the local 12-step pipeline as the canonical backend.
- Read school risk, performance, and within-school inequality from `outputs/`, not from a separate heuristic calculator.

### 2. Critical: projected school-level “gap reduction” language overclaims

The reference app presents intervention outputs in a way that can be read as predicted causal improvement for a named school. The current data do not support that.

Why this matters:
- The project is observational.
- The competition narrative is much safer when framed as triage and prioritisation.

Required resolution:
- Remove forecast-like language from the main recommendation flow.
- Keep a separate scenario planner for hypothetical planning.

### 3. High: multiple truth sources create narrative drift

There were three parallel logic sources:
- the local 12-step pipeline,
- the reference app’s own simplified data and heuristics,
- the V7 cluster-level model.

Why this matters:
- The same concept can end up with different numbers in different places.
- This weakens technical credibility.

Required resolution:
- Use `outputs/` as the only official artifact source for the dashboard.
- Treat V7 as evidence support only.

### 4. High: V7 is valuable, but it is not the production recommendation engine

V7 improves on V6 by using grouped cross-validation and removing a feature-selection leakage path. That is a genuine methodological improvement.

Why it still cannot be the main engine:
- It is 2022-only.
- It operates at clustered-school level rather than the official school-profile layer.
- It produces coded feature signals that still require careful interpretation.

Required resolution:
- Keep V7 in the evidence layer.
- Do not use it as the primary school recommender.

### 5. Medium: repo structure needed cleanup and archival discipline

The workspace contained active code, bulky raw archives, reference materials, and legacy logs at the same surface level.

Why this matters:
- It obscures the canonical project path.
- It increases the chance of using the wrong file in a submission or demo.

Required resolution:
- Archive non-essential materials without deleting them.
- Add explicit repository guidance.

### 6. Medium: privacy and publication guardrails needed to be documented

The project uses anonymised study data, but the repository still needed explicit documentation on what can and cannot be published or demoed.

Required resolution:
- Add a privacy and security note.
- Avoid committing raw SAV data, local cache, and local secrets.

## Recommended target state

### Canonical backend
- `src/run_pipeline.py`
- `src/build_analytic_datasets.py`
- `src/country_analysis.py`
- `src/segmentation.py`
- `src/interventions.py`

### Canonical app surface
- `app.py`
- `src/dashboard_app.py`

### Secondary evidence
- `v7/`

### Repository hygiene
- local-only archive storage when needed, but not as part of the public branch
- `docs/`
- `.gitignore`

## Changes implemented in this pass

- Kept non-essential logs, probe outputs, raw archive bundles, and runtime cache out of the public branch.
- Added repository-level documentation and privacy guidance.
- Updated the dashboard to separate `Observed Recommendation` from `Scenario Planner`.
- Removed forecast-like language from the main recommendation path.
- Added V7 as a secondary evidence layer rather than a primary decision engine.

## Remaining improvements worth doing later

- Replace hard-coded intervention category rules with a configuration file.
- Add variable-label translation for coded V7 questionnaire features.
- Split dashboard UI concerns into smaller modules if the app grows further.
