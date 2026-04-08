# Dashboard Explainer

## What this representable is

This Streamlit app is a `decision-support dashboard`.
It is not just a visualisation.
Its logic is:

`problem diagnosis -> observed profile -> targeted prioritisation -> optional scenario planning -> evidence and limits`

## Main tabs

### 1. Overview
- Purpose: fast summary of the product and why it is a dashboard rather than a single chart.
- Left chart:
  current high-gap systems.
- Right chart:
  distribution of schools across the published segments.

### 2. Country Diagnostic
- Purpose: answer `where is the equity problem largest or worsening?`
- Use it for the competition problem-diagnosis story.
- Key outputs:
  latest gap, long-run change, trend evidence, and country typology.

### 3. School Segments
- Purpose: answer `what kind of school profile are we dealing with?`
- Each point is an anonymised school profile.
- The key idea is that schools with similar performance can still have very different inequality and support needs.

### 4. Observed Recommendation
- Purpose: answer `given an observed school profile, what intervention categories fit best?`
- This is the safe recommendation mode.
- It uses:
  observed risk,
  observed segment,
  observed performance and inequality profile.
- It does not present projected impact for one school.

### 5. Scenario Planner
- Purpose: answer `if a hypothetical school looked like this, what segment and intervention mix would the rules suggest?`
- This is for demo and planning conversations only.
- It is separate from observed diagnosis on purpose.

### 6. Evidence & Limits
- Purpose: show correlates, secondary V7 evidence, and claim boundaries.
- This tab is what keeps the project rigorous.

## Key terms

- `Latest gap`: most recent observed SES-maths gap for a country/system.
- `Within-school gap`: performance spread inside a school between higher-SES and lower-SES students.
- `Risk score`: supplied school-level indicator used as an observed input.
- `Official segment`: published school typology label from the interpretable segmentation step.
- `Observed recommendation`: recommendation based on a real school profile already in the analysis dataset.
- `Scenario planner`: hypothetical planning mode, not a forecast engine.

## How to present it

1. Start with `Country Diagnostic`.
   Show where the problem is most severe or worsening.
2. Move to `School Segments`.
   Show why one-size-fits-all policy is weak.
3. Use `Observed Recommendation`.
   Show how the same evidence base turns profiles into tailored priorities.
4. Use `Scenario Planner` only if someone asks “what if”.
5. End on `Evidence & Limits`.
   Show that the team understands what the data can and cannot support.

## What not to claim

- Do not say the dashboard identifies the exact cause of a school’s gap.
- Do not say it predicts the exact effect of an intervention for one school.
- Do say it provides `evidence-informed prioritisation` based on observed patterns and transparent rules.
