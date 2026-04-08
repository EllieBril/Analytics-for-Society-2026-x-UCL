# Privacy And Security

## Data handling principles

- Use anonymised study identifiers only.
- Do not expose named schools, named students, or personal records in the dashboard.
- Keep raw SAV data local unless there is an explicit legal and licensing basis to publish them.

## Repository publication rules

- Do not commit local secrets or environment files.
- Do not commit raw `.SAV` files to a public repository.
- Do not commit bulky merged parquet files that materially reconstruct the raw datasets.
- Do not include absolute local filesystem paths in documentation or user-facing outputs.

## App-level guardrails

- The dashboard should present anonymised school profile IDs only.
- The dashboard should state clearly that recommendations are evidence-informed prioritisation, not causal proof.
- Scenario-planner outputs must stay clearly separated from observed school profiles.
- Usage statistics are disabled in the local Streamlit configuration.

## Demo and deployment guidance

- Prefer a read-only deployment for competition demos.
- Prefer derived, presentation-safe artifacts from `outputs/` over raw-data access in a public demo.
- If external deployment is needed, use a minimal dataset subset that supports the narrative without exposing unnecessary detail.

## Safe communication rules

- Say what is directly observed.
- Say what is model-based.
- Mark interpretation separately from evidence.
- Avoid deterministic wording such as “this intervention will reduce the gap by X points for this school.”

## Current repository posture

- Legacy and bulky materials should stay local-only rather than being committed to the public branch.
- A `.gitignore` has been added to reduce accidental publication of raw or local-only artifacts.
- The main dashboard now separates observed diagnosis from hypothetical scenario planning.
