# Dashboard Usage

## Run locally

```bash
streamlit run app.py
```

## What the dashboard includes

- `Overview`: competition-ready summary of systems, segmented schools, and the core prototype framing.
- `Country Diagnostic`: latest-gap and trajectory views for identifying where the equity problem is largest or worsening.
- `School Segments`: the published school typology and the performance-versus-inequality map.
- `Observed Recommendation`: profile-based intervention priorities for a selected observed school profile.
- `Scenario Planner`: hypothetical planning mode kept separate from observed diagnosis.
- `Evidence & Limits`: correlates, V7 support, scope limits, and narrative text that prevent overclaiming.

## Intended framing

- This is a decision-support prototype, not a causal policy engine.
- School identifiers are anonymised PISA school IDs, so the school-level pane should be presented as a profile inspector.
- Scenario output is a planning aid, not a predicted intervention effect.
