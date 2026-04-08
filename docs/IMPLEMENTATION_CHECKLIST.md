# Implementation Checklist

## Completed now

- [x] Keep the local 12-step pipeline as the canonical backend.
- [x] Keep the Streamlit dashboard as the canonical presentation layer.
- [x] Keep legacy logs, probe outputs, raw archive bundles, and runtime cache out of the public branch.
- [x] Add repository-level architecture, review, and privacy documentation.
- [x] Separate observed recommendation logic from hypothetical scenario planning in the dashboard.
- [x] Keep V7 in the evidence layer rather than the primary recommendation path.
- [x] Add a `.gitignore` to reduce accidental publication of raw local-only data and secrets.

## Use this workflow going forward

1. Run the backend pipeline first.
2. Treat `outputs/` as the only official artifact source.
3. Update the dashboard only after the backend outputs are stable.
4. Keep any new model or notebook in a clearly marked secondary-evidence area until validated.
5. Keep clutter in a local-only archive instead of deleting it.

## Next recommended engineering tasks

- [ ] Move segment thresholds and intervention category rules into a config file.
- [ ] Add a small codebook mapping for the most important coded V7 variables.
- [ ] Add a deployment-specific read-only demo dataset if the app will be shared externally.
- [ ] Add automated checks for the presence of required output files before app startup.
- [ ] Add a short presenter script for live competition demos.

## Submission guardrails

- [ ] Do not show raw student-level records.
- [ ] Do not claim causal intervention effects.
- [ ] Do not present scenario-planner output as observed school diagnosis.
- [ ] Do not commit raw SAV files or local secrets to a public repository.
