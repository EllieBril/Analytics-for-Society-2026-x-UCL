# EquiTrack — Change Log

Comparison against the original `app (1).py` baseline. Changes are grouped by area.

---

## 1. Bug fixes (`model.py`)

| # | Location | Problem | Fix |
|---|---|---|---|
| 1 | `get_interventions()` | `inv_to_cat` iterated over characters of the category string (e.g. `'l','e','a','r',...`) so every intervention mapped to `'other'`, breaking segment-priority ranking | Replaced with `{k: v for k, v in INTERVENTION_CATEGORIES.items()}` |
| 2 | `get_equity_risk_score()` | Accepted a `df_scores` parameter that was never used in the function body | Removed from signature and all call sites |
| 3 | `_load_segment_bundle()` | The pkl file was reloaded from disk on every button click via an inline `__import__('joblib').load(...)` hack | Added a module-level `_segment_bundle = None` cache so the file is read once per process |

---

## 2. Bug fixes (`app.py`)

| # | Location | Problem | Fix |
|---|---|---|---|
| 4 | Tab 2, `diag1` block | `__import__('joblib').load(...)` re-read the pkl on every render | Replaced with `from model import _load_segment_bundle` (uses the module cache) |
| 5 | Tab 2, `if generate:` block | `get_interventions()` called without `segment_priorities` so segment-based ranking was never applied | Added `segment_priorities=seg_result['info'].get('priorities')` |
| 6 | Tab 2, `if generate:` block | `get_equity_risk_score()` called with the removed `df_scores` argument | Removed from call site |
| 7 | Tab 2, `if generate:` block | `import joblib` inside the block — not used anywhere, dead code | Removed |
| 8 | Tab 3, cross-reference loop | The `✓ N interventions confirmed` dark card was inside the `for row_start` loop, rendering once per row group | Fixed indentation so card renders once after the loop |
| 9 | Tab 4 | `import plotly.graph_objects as go` duplicated an import already at the top of the file | Removed the duplicate |

---

## 3. Narrative restructure — new content

### Tab 1 — Global Context
- **Added**: Full OLS regression findings section with a forest-plot chart (4 significant variables, 95% CIs) and interpretation cards for each finding (school meaning + practical lever).
- **Added**: "From findings to action" bridge section — a mapping table showing how each OLS finding translates into an intervention category (`climate_support`, `learning_support`, `resource_intensive`), paired with a green insight-box explaining that an independent XGBoost model confirmed the same findings via SHAP. This connects Tab 1 evidence to Tab 2 recommendations and Tab 3 validation before the user reaches those tabs.

### Tab 2 — My School Report
- **Added**: `SEGMENT_RATIONALE` dict (moved to module level) — a segment-specific "why this priority" sentence rendered inside the segment card, explaining *why* the recommended category is the right first lever for that school profile.
- **Changed**: Interventions section title from `"Recommended interventions — filtered for {budget_label} budget"` to `"Recommended interventions — {budget_label} budget"` + a `st.caption` below it that makes the segment→ranking logic explicit to the user.
- **Added**: `mitigation_credit` calculation — existing practices are now scored using the same diminishing-returns formula (capped at 50% of country gap) and shown as a separate metric card ("Current mitigation").
- **Changed**: Metric cards restructured — replaced "Country avg gap" card with "Current mitigation" card; added a stacked-bar breakdown of the risk score components (gap / trend / school) inside the risk score card.
- **Added**: Gap projection waterfall chart — three-step horizontal waterfall showing: baseline country gap → existing practices reduction → recommended interventions reduction → projected gap.
- **Added**: Within-country school comparison — a histogram of equity risk scores for all schools in the selected country (with the user's school marked) and a box plot by school type.
- **Changed**: `projected_gap` now accounts for both existing practices AND new interventions (previously only new interventions).

### Tab 3 — Intervention Evidence Base
- **Changed**: Section titles restructured as "Part 1 — What the evidence says: EEF Teaching & Learning Toolkit" and "Part 2 — Independent confirmation: PISA 2022 machine learning analysis" to frame the tab as a two-source validation story rather than two unrelated exhibits.
- **Changed**: Part 2 insight-box rewritten to lead with "This is a completely independent check" and make the "two methods, one answer" framing explicit.
- **Changed**: SHAP chart moved from a 2-column layout (chart left, cross-reference cards right) to full width, so the plain-English y-axis labels have room to display.
- **Changed**: SHAP chart filtered to labeled features only (`label != feature`) — raw PISA variable codes are excluded. Also added `bargap=0.45`, increased `height` to 900, and repositioned the legend to the bottom-right corner of the plot area (vertical) to prevent overlap with the x-axis.
- **Added**: Four new PISA variable translations added to `sc_labels` in `load_shap()`: `SC172Q02JA`, `SC172Q03JA`, `SC053Q01TA`, `SCHLTYPE`.
- **Changed**: Cross-reference cards moved below the chart into a full-width 3-column grid.
- **Changed**: Model methodology expander — removed all version references (`v7`, `v6`, "corrected evaluation", the "Evaluation fix (v6→v7)" paragraph). Kept the full explanation: What it does, Data, Performance, Limitations.
- **Changed**: Tab 3 footer note — "XGBoost v7" → "XGBoost".

### Tab 4 — Methodology (formerly "Model Validation")
- **Changed**: Tab label from `"✅  Model Validation"` to `"🔬  Methodology"`.
- **Changed**: Section title from "Robustness benchmark: Random Forest vs LightGBM" to "How we built the risk score — and why you can trust it".
- **Added**: Intro insight-box explaining the risk score formula (60% gap size / 20% trajectory / 20% school profile) and interpreting the modest F1 scores before the benchmark chart.

---

## 4. Visual / CSS fixes

| # | Change |
|---|---|
| Force light theme | Added a comprehensive CSS block targeting `stApp`, `stMain`, `stTabs`, all widget labels, input/select backgrounds, expanders, dataframes, and Plotly chart containers to enforce light-mode colours regardless of the user's system theme. |
| Number input buttons | Added CSS for `[data-testid="stNumberInput"] button` — the `−` and `+` stepper buttons were rendering in dark mode. Now light grey with dark icons. |
| Streamlit toolbar | Added CSS for `stHeader`, `stToolbar` buttons, and the three-dot dropdown menu (`[data-testid="main-menu-list"]`, `[data-baseweb="popover"]`) to render in light mode. |
| EEF chart label colour | Added `tickfont=dict(color='#1F2937')` to the EEF bar chart y-axis so intervention names display in dark text. |
| Tab 1 chart layout | Charts pre-computed before column rendering to ensure equal heights and prevent layout jumps. |

---

## 5. Code quality

| # | Change |
|---|---|
| `SEGMENT_RATIONALE` | Moved from inside `if generate:` (recreated on every click) to module level alongside `color_map`. |
| `import joblib` | Removed dead import from inside `if generate:` block. |
| Duplicate `import plotly.graph_objects as go` | Removed from Tab 4 (already imported at top of file). |
