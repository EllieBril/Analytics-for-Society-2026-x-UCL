# Prototype Logic Note

- Input: school profile metrics already produced by the pipeline.
- Output: risk classification, segment, recommended intervention categories, and a short rationale.
- Design choice: simple inspectable rules, no opaque black-box recommender.
- Risk tiers use percentile-based score cut points from the current school-profile distribution.
- The dashboard uses `dashboard_school_view.csv` as a lightweight public-facing school-profile artifact.
- Interpretation: recommendations are evidence-informed prioritisation suggestions only.