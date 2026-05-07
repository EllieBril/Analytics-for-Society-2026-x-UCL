# Session Change Log

## 1. Equity Risk Score — Formula Rebuilt (Core Change)

**Files:** `app/model.py`, `scripts/regenerate_risk_scores.py`

The old formula gave ±3 pts from school inputs, making scores nearly identical for all schools in the same country. Replaced with an OLS-evidence-based formula.

### Old formula (0–100)
| Component | Range | Basis |
|---|---|---|
| Country gap score | 0–60 | Normalised vs global mean |
| Trajectory score | 0–20 | Closing/Stable/Widening |
| School score | 0–20 | School type + student-teacher ratio |

### New formula (0–100)
| Component | Range | Basis |
|---|---|---|
| Country gap score | 0–50 | Normalised vs global mean (rescaled) |
| Trajectory score | 0–20 | Closing=0, Stable=10, Widening=20 |
| School profile score | 0–30 | OLS regression β weights (PISA 2022, n=15,238) |

### School profile score formula
```
ability_01   = 1.0 if ability_grouping == 'Yes' else 0.0   # ABGMATH,    |β|=1.225
climate_01   = (behaviour_disruption - 1) / 3.0             # NEGSCLIM,   |β|=0.885
bullying_01  = (bullying_severity - 1) / 3.0                # SC061Q05TA, |β|=1.468

contribution = (1.225 * ability_01) + (0.885 * climate_01) + (1.468 * bullying_01)
school_score = clip(contribution / 3.578 * 30, 0, 30)
```

All three inputs use **|β| magnitudes** as weights with all inputs treated as positive risk factors — domain logic, not raw regression sign. The OLS regression target was within-school SES gap (lower = better equity); the risk score is also lower = better, but they are different constructs.

**Critical bug fixed during session:** Original implementation used raw β=−1.468 for bullying, meaning high bullying *lowered* the risk score. Corrected to |β|=1.468 so all bad inputs raise risk.

### New `get_equity_risk_score()` signature
```python
# Old
get_equity_risk_score(country_code, school_type, stratio, df_traj, df_gap)

# New
get_equity_risk_score(country_code, df_traj, df_gap,
                      bullying_severity=2, ability_grouping='No', negsclim=2)
```

---

## 2. Two New School Diagnostic Inputs

**File:** `app/app.py`

Added to the school diagnostic section (second row):

| Input | Widget | PISA variable |
|---|---|---|
| Student bullying / intimidation | `select_slider` 1–4 | SC061Q05TA |
| Ability grouping in maths | `radio` Yes/No | SC042/SC187 (ABGMATH) |

Both feed directly into the new school profile score formula.

---

## 3. school_risk_scores.csv Regenerated

**File:** `scripts/regenerate_risk_scores.py` (new script)

All 21,629 PISA 2022 schools recomputed with the new formula using raw questionnaire items from `school_2022.parquet`:

- **ABGMATH proxy:** `(SC042Q01TA < 3).astype(float)` — ability grouping flag
- **NEGSCLIM proxy:** mean of SC061Q01–SC061Q04TA — negative climate composite
- **SC061Q05TA:** student bullying item (clipped to 1–4, 95=invalid → NaN)
- Missing values imputed with country mean, then global mean fallback

Score distribution after regeneration: mean=44.6, std=17.1, range=6.3–97.4

Cache busting added to `load_data()` using file mtime as `_version` key so Streamlit auto-reloads when the CSV changes.

---

## 4. Segment Classifier — Replaced with Rule-Based Logic

**File:** `app/model.py`

### Problem with the ML model (`segment_classifier.pkl`)
- **LightGBM classifier** trained to reproduce rule-based labels from `segmentation.py`
- `STAFFSHORT_b`, `EDUSHORT_b`, `NEGSCLIM_b` features were binarised against PISA z-score quartile cuts, but the app passed raw slider integers (1–4) — a silent mismatch
- Classifier was dominated by `school_mean_math` and `school_mean_escs`, effectively ignoring the risk score
- "Digitally constrained schools" was never reachable (no digital feature in `feature_cols`)
- Visible symptom: risk score 72/100 (High risk) assigned "Resilient equitable performers"

### New rule-based function
```python
MATH_CUT  = 472   # OECD 2022 average maths score
RISK_CUT  = 55    # elevated risk threshold
STRAIN_CUT = 3    # slider: "To some extent" or "A lot"

high_math    = school_mean_math >= MATH_CUT
high_risk    = risk_score >= RISK_CUT
high_strain  = staffshort >= STRAIN_CUT or edushort >= STRAIN_CUT
high_climate = negsclim >= STRAIN_CUT

if high_math and not high_risk:          → Resilient equitable performers
elif high_math and high_risk:            → High-achievement unequal schools
elif not high_math and edushort == 4:    → Digitally constrained schools
elif not high_math and (high_risk
     or high_strain or high_climate):    → Strained high-inequality schools
else:                                    → Low-achievement broad support need
```

The pkl is still loaded solely for `country_math_refs` (per-country average maths scores used as hint text in the maths score input widget).

---

## 5. UI Layout Changes

**File:** `app/app.py`

### Metric cards (equity report section)
- Order changed: card 2 = National Trajectory, card 3 = Gap already being closed (previously swapped)
- All 4 cards unified to same structure (`_card_center` style) — removed stacked bar from card 1 to fix unequal heights

### Position charts
- **Left:** Histogram x-axis now always includes user's school score (was clipping it off); percentile annotation added ("Your school: 72.0 (100th pct.)")
- **Right:** Replaced meaningless box plot by school type with a **risk score breakdown bar chart** (Country gap / Trend / School profile components with grey background + coloured fill bars)

### Recommended interventions section
- 3 summary cards (Interventions available, Avg evidence strength, If top intervention applied) moved from right sidebar to a flat 3-column row below the section title — matching the equity report card style
- Intervention list changed from single column to **2-column grid** (alternating by rank: 1,3,5… left; 2,4,6… right)
- **Intervention focus badge** added to every card showing category (📚 Learning Support, 🤝 Climate Support, 👨‍👩‍👧 Family Engagement, 💼 Resource Intensive) with colour-coded styling matching the segment card

---

## 6. Global Context Tab — Caption Fixes

**File:** `app/app.py`

- "Japan" → "Macao and Hong Kong" in the performance vs equity gap chart insight box
- "79 countries" → "80 countries" (2 occurrences: OLS regression description and forest plot caption)

---

## 7. Methodology Footnote — Rewritten

**File:** `app/app.py`

Fully rewritten to match current implementation:
- Data sources: PISA 2022 (21,629 schools, 80 countries), PISA 2009–2022 trends, EEF toolkit
- Risk score formula with correct component ranges (0–50 / 0–20 / 0–30) and OLS specs
- Gap reduction formula: target group scaling × disadvantaged %, 0.7^i diminishing returns decay, 60% cap
- Limitations: self-reported inputs vs PISA-measured comparison schools; observational regression; projection estimates
