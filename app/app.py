import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import get_equity_risk_score, get_interventions, calculate_realistic_reduction, predict_segment, SEGMENT_INFO, INTERVENTION_CATEGORIES
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EquiTrack",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #0F2A1D;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-size: 1.05rem;
        color: #4A6355;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #F7FAF8;
        border: 1px solid #D4E6DA;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4A6355;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #0F2A1D;
        line-height: 1;
    }

    .metric-sub {
        font-size: 0.8rem;
        color: #7A9585;
        margin-top: 0.3rem;
    }

    .risk-high   { color: #B91C1C; }
    .risk-medium { color: #B45309; }
    .risk-low    { color: #15803D; }

    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #0F2A1D;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #D4E6DA;
    }

    .intervention-card {
        background: white;
        border: 1px solid #E2EDE7;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        border-left: 4px solid #1D9E75;
    }

    .intervention-card.over-budget {
        border-left-color: #9CA3AF;
        opacity: 0.5;
    }

    .intervention-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #0F2A1D;
    }

    .intervention-meta {
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
    }

    .badge-green  { background: #D1FAE5; color: #065F46; }
    .badge-amber  { background: #FEF3C7; color: #92400E; }
    .badge-red    { background: #FEE2E2; color: #991B1B; }
    .badge-gray   { background: #F3F4F6; color: #374151; }
    .badge-blue   { background: #DBEAFE; color: #1E40AF; }

    .traj-closing  { color: #15803D; font-weight: 600; }
    .traj-stable   { color: #B45309; font-weight: 600; }
    .traj-widening { color: #B91C1C; font-weight: 600; }

    .stButton > button {
        background: #1D9E75;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        width: 100%;
    }

    .stButton > button:hover {
        background: #0F6E56;
    }

    .divider {
        border: none;
        border-top: 1px solid #E2EDE7;
        margin: 1.5rem 0;
    }

    .footer-note {
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2EDE7;
    }

    .insight-box {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #166534;
    }

    /* ── FORCE LIGHT THEME ───────────────────────────────────────── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"],
    section[data-testid="stSidebar"],
    .main .block-container {
        background-color: #FAFBFC !important;
        color: #1F2937 !important;
    }

    /* All generic text nodes */
    p, span, div, li, td, th {
        color: #1F2937;
    }

    /* Markdown containers */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #1F2937 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        color: #374151 !important;
        font-weight: 500;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #0F6E56 !important;
        border-bottom-color: #1D9E75 !important;
        font-weight: 700;
    }
    [data-testid="stTabsContent"],
    [data-testid="stTabPanel"] {
        background-color: #FAFBFC !important;
    }

    /* All widget labels */
    label,
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stCheckbox label,
    .stRadio label,
    .stMultiSelect label,
    .stTextInput label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        color: #1F2937 !important;
        font-weight: 500 !important;
    }

    /* Input and select widget backgrounds */
    [data-baseweb="input"],
    [data-baseweb="select"],
    [data-baseweb="textarea"],
    [data-baseweb="input"] input,
    [data-baseweb="select"] div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border-color: #D1D5DB !important;
    }

    /* Number input stepper buttons (– and +) */
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button svg {
        background-color: #F3F4F6 !important;
        color: #1F2937 !important;
        fill: #1F2937 !important;
        border-color: #D1D5DB !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background-color: #E5E7EB !important;
    }

    /* Selectbox dropdown list */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    /* Slider track and labels */
    [data-testid="stSlider"] div,
    [data-testid="stSlider"] p {
        color: #1F2937 !important;
    }

    /* Checkbox text */
    .stCheckbox span,
    [data-testid="stCheckbox"] span {
        color: #1F2937 !important;
    }

    /* Caption / small text */
    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #4B5563 !important;
    }

    /* Expander */
    [data-testid="stExpander"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    /* Column containers */
    [data-testid="stHorizontalBlock"],
    [data-testid="stColumn"],
    [data-testid="column"] {
        background-color: transparent !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }

    /* Plotly chart container — ensure white bg shows through */
    [data-testid="stPlotlyChart"] > div {
        background: white !important;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }

    /* Warning / info boxes */
    [data-testid="stAlert"] {
        background: #FEF9EC !important;
        color: #1F2937 !important;
        border-color: #F59E0B !important;
    }

    /* Streamlit top toolbar */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    header[data-testid="stHeader"] {
        background-color: #FAFBFC !important;
    }
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] button svg {
        color: #374151 !important;
        fill: #374151 !important;
    }

    /* Toolbar dropdown menu (Rerun / Settings / Print …) */
    [data-testid="main-menu-list"],
    [data-testid="main-menu-list"] ul,
    [data-testid="main-menu-list"] li,
    [data-testid="main-menu-list"] a,
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="listitem"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    [data-testid="main-menu-list"] li:hover,
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] li:hover {
        background-color: #F3F4F6 !important;
    }
    /* Keyboard shortcut badges */
    [data-testid="main-menu-list"] span,
    [data-baseweb="popover"] span {
        color: #6B7280 !important;
    }
    /* Popover container background */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div {
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
    }

    /* Metric cards inherit correct colour */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
    }
    .metric-label { color: #374151; }
    .metric-value { color: #0F2A1D; }
    .metric-sub   { color: #6B7280; }

    /* Equal-height metric card row */
    [data-testid="stHorizontalBlock"]:has(.metric-card) {
        align-items: stretch;
    }
    [data-testid="stHorizontalBlock"]:has(.metric-card) [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(.metric-card) [data-testid="stColumn"] > div,
    [data-testid="stHorizontalBlock"]:has(.metric-card) [data-testid="stColumn"] > div > div {
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    [data-testid="stHorizontalBlock"]:has(.metric-card) .metric-card {
        flex: 1;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), 'data')
    df_gap    = pd.read_csv(os.path.join(base, 'equity_gap_by_country_year.csv'))
    df_traj   = pd.read_csv(os.path.join(base, 'country_trajectories.csv'))
    df_scores = pd.read_csv(os.path.join(base, 'school_risk_scores.csv'))
    df_int    = pd.read_csv(os.path.join(base, 'intervention_library.csv'))
    return df_gap, df_traj, df_scores, df_int

df_gap, df_traj, df_scores, df_interventions = load_data()

color_map = {
    'Closing': '#1D9E75',
    'Stable':  '#EF9F27',
    'Widening':'#E3120B',
    'No data': '#9CA3AF'
}

SEGMENT_RATIONALE = {
    'Resilient equitable performers':
        'Your strong average performance means targeted learning support and family engagement can close the remaining gap without wholesale reform.',
    'High-achievement unequal schools':
        'High average scores mask wide inequality — climate and grouping practices are the likely cause; addressing them unlocks gains for disadvantaged students without pulling others back.',
    'Low-achievement broad support need':
        'Broad underperformance calls for evidence-based learning support first, then family engagement to reinforce gains at home.',
    'Strained high-inequality schools':
        'A hostile climate disproportionately harms disadvantaged students who have no alternative safe space — climate repair is the highest-leverage first move before any academic intervention.',
    'Digitally constrained schools':
        "The OLS analysis found computer-to-student ratio independently narrows the equity gap — your school's digital constraint is directly holding back your most disadvantaged students.",
}

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">EquiTrack</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Know your equity gap. Find your lever. '
    'Close the gap between your most and least advantaged students.</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍  Global Context",
    "🏫  My School Report",
    "📚  Intervention Evidence Base",
    "🔬  Methodology"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">The Global Equity Gap — Why This Matters</div>',
                unsafe_allow_html=True)

    # ── Pre-compute all three charts before rendering columns ─────────────────
    global_gap = df_gap.groupby('YEAR').agg(
        MEAN_GAP=('GAP', 'mean'),
        MEDIAN_GAP=('GAP', 'median'),
        STD_GAP=('GAP', 'std')
    ).reset_index()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=global_gap['YEAR'].tolist() + global_gap['YEAR'].tolist()[::-1],
        y=(global_gap['MEAN_GAP'] + global_gap['STD_GAP']).tolist() +
          (global_gap['MEAN_GAP'] - global_gap['STD_GAP']).tolist()[::-1],
        fill='toself', fillcolor='rgba(29,158,117,0.12)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig_trend.add_trace(go.Scatter(
        x=global_gap['YEAR'], y=global_gap['MEAN_GAP'],
        mode='lines+markers+text',
        line=dict(color='#1D9E75', width=3), marker=dict(size=9),
        text=[f'{v:.1f}' for v in global_gap['MEAN_GAP']],
        textposition='top center',
        textfont=dict(size=11, color='#0F6E56', family='DM Sans'),
        hovertemplate='%{x}: %{y:.1f} pts<extra></extra>', showlegend=False
    ))
    fig_trend.update_layout(
        title=dict(text='Global Equity Gap 2009–2022', font=dict(size=13, color='#1F2937'),
                   x=0.5, xanchor='center'),
        xaxis=dict(
            title=dict(text='PISA Cycle', font=dict(color='#374151')),
            tickfont=dict(color='#374151'),
            tickmode='array',
            tickvals=global_gap['YEAR'].tolist(),
            ticktext=[str(y) for y in global_gap['YEAR'].tolist()],
            type='category'
        ),
        yaxis=dict(title=dict(text='Avg Gap (pts)', font=dict(color='#374151')), tickfont=dict(color='#374151')),
        plot_bgcolor='white', paper_bgcolor='white',
        height=320, margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family='DM Sans', size=11, color='#374151')
    )

    gap_2022_ctx = df_gap[df_gap['YEAR'].astype(str) == '2022'].copy()
    gap_2022_ctx = gap_2022_ctx.merge(df_traj[['CNT', 'TRAJECTORY']], on='CNT', how='left')
    gap_2022_ctx['TRAJECTORY'] = gap_2022_ctx['TRAJECTORY'].fillna('No data')

    traj_order = ['Widening', 'Stable', 'Closing']
    traj_counts = (
        gap_2022_ctx[gap_2022_ctx['TRAJECTORY'].isin(traj_order)]
        ['TRAJECTORY'].value_counts().reindex(traj_order).reset_index()
    )
    traj_counts.columns = ['TRAJECTORY', 'COUNT']

    fig_traj_bar = go.Figure(go.Bar(
        x=traj_counts['COUNT'], y=traj_counts['TRAJECTORY'], orientation='h',
        marker_color=[color_map[t] for t in traj_counts['TRAJECTORY']],
        text=traj_counts['COUNT'].astype(str) + ' countries',
        textposition='inside', textfont=dict(color='white', size=12),
        hovertemplate='<b>%{y}</b><br>%{x} countries<extra></extra>'
    ))
    fig_traj_bar.update_layout(
        title=dict(text='Country Trajectories 2009–2022', font=dict(size=13, color='#1F2937'),
                   x=0.5, xanchor='center'),
        xaxis=dict(title=dict(text='Number of Countries', font=dict(color='#374151')), tickfont=dict(color='#374151')),
        yaxis=dict(tickfont=dict(color='#374151')),
        plot_bgcolor='white', paper_bgcolor='white',
        height=320, margin=dict(l=80, r=20, t=50, b=40),
        showlegend=False, font=dict(family='DM Sans', size=12, color='#374151')
    )

    gap_2022_sc = df_gap[df_gap['YEAR'].astype(str) == '2022'].copy()
    gap_2022_sc = gap_2022_sc.merge(df_traj[['CNT', 'TRAJECTORY']], on='CNT', how='left')
    gap_2022_sc['TRAJECTORY'] = gap_2022_sc['TRAJECTORY'].fillna('No data')

    fig_scatter = go.Figure()
    for traj_val in ['Closing', 'Stable', 'Widening', 'No data']:
        df_sub = gap_2022_sc[gap_2022_sc['TRAJECTORY'] == traj_val]
        if len(df_sub) == 0:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=df_sub['GAP'], y=df_sub['AVG_MATH'], mode='markers',
            name=traj_val,
            marker=dict(size=8, color=color_map[traj_val], opacity=0.82,
                        line=dict(width=0.5, color='white')),
            hovertemplate='<b>%{text}</b><br>Gap: %{x:.1f} pts<br>Score: %{y:.0f}<extra></extra>',
            text=df_sub['CNT']
        ))
    fig_scatter.add_vline(x=gap_2022_sc['GAP'].mean(), line_dash='dash', line_color='#9CA3AF', line_width=1)
    fig_scatter.add_hline(y=gap_2022_sc['AVG_MATH'].mean(), line_dash='dash', line_color='#9CA3AF', line_width=1)
    fig_scatter.add_annotation(
        x=gap_2022_sc['GAP'].min() + 3, y=gap_2022_sc['AVG_MATH'].max() - 10,
        text='<b>Ideal</b>', showarrow=False,
        font=dict(size=10, color='#1D9E75'), bgcolor='rgba(29,158,117,0.1)'
    )
    fig_scatter.update_layout(
        title=dict(text='Performance vs Equity Gap (2022)', font=dict(size=13, color='#1F2937'),
                   x=0.5, xanchor='center'),
        xaxis=dict(title=dict(text='Equity Gap (pts) — Lower Is Better', font=dict(color='#374151')),
                   tickfont=dict(color='#374151')),
        yaxis=dict(title=dict(text='Avg Maths Score', font=dict(color='#374151')),
                   tickfont=dict(color='#374151')),
        legend=dict(
            orientation='h', yanchor='bottom', y=0.01,
            xanchor='center', x=0.5,
            font=dict(size=9, color='#374151'),
            bgcolor='rgba(255,255,255,0.85)', borderwidth=0
        ),
        plot_bgcolor='white', paper_bgcolor='white',
        height=320, margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family='DM Sans', size=11, color='#374151')
    )

    # ── Row 1: charts (same height, perfectly aligned) ────────────────────────
    ctx1, ctx2, ctx3 = st.columns(3)
    with ctx1:
        st.plotly_chart(fig_trend, use_container_width=True)
    with ctx2:
        st.plotly_chart(fig_traj_bar, use_container_width=True)
    with ctx3:
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Row 2: insight boxes (separate row, top-aligned) ─────────────────────
    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown(
            '<div class="insight-box">📊 The global equity gap has barely moved in 13 years '
            '— from 83.8 to 81.6 points. Without deliberate intervention, the default is stagnation.</div>',
            unsafe_allow_html=True
        )
    with ins2:
        st.markdown(
            '<div class="insight-box">🔴 Two thirds of countries are widening or stable. '
            'Only one third are actively closing their equity gap.</div>',
            unsafe_allow_html=True
        )
    with ins3:
        st.markdown(
            '<div class="insight-box">✅ Top-left = ideal: high scores, low gap. '
            'Macao and Japan prove high performance and equity can coexist.</div>',
            unsafe_allow_html=True
        )

    # ── OLS REGRESSION FINDINGS ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">What Drives the Equity Gap? OLS Regression Findings</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        <b>Observational OLS regression</b> on 15,238 schools across 79 countries (PISA 2022).
        Target: within-school maths gap (top vs bottom SES quartile). Controls: school SES composition
        + country fixed effects. Standard errors clustered by country. R² = 0.206.
        <b>Positive β = wider gap; negative β = narrower gap.</b>
        Only the 4 school-practice variables significant at p&lt;0.05 are shown.
    </div>
    """, unsafe_allow_html=True)

    # 4 significant features only, sorted by p-value
    ols_sig_features = [
        dict(
            label='Student sense of belonging',
            var='SC061Q05TA · School Climate',
            coef=-1.468, ci_lo=-2.502, ci_hi=-0.435, p=0.005,
            direction='narrows',
            school_meaning=(
                'Students who feel they belong — welcomed, respected, included — show '
                'a smaller gap between affluent and disadvantaged peers. Belonging is not '
                'a "soft" outcome: it predicts attendance, effort, and persistence, all of '
                'which disproportionately affect students from low-income households.'
            ),
            action='Invest in pastoral care, anti-bullying programmes, and inclusive '
                   'classroom norms. Belonging costs little but pays across the whole cohort.'
        ),
        dict(
            label='Ability grouping in maths',
            var='ABGMATH · Grouping & Tracking',
            coef=1.225, ci_lo=0.332, ci_hi=2.118, p=0.007,
            direction='widens',
            school_meaning=(
                'Schools that sort students into separate maths classes by ability tend '
                'to have wider equity gaps. Lower-SES students are disproportionately '
                'placed in lower sets, receive less experienced teachers, and face reduced '
                'expectations — compounding disadvantage rather than addressing it.'
            ),
            action='Review setting policies. Mixed-ability grouping with in-class support '
                   'shows stronger equity outcomes, especially at Key Stage 3.'
        ),
        dict(
            label='Computer-to-student ratio',
            var='RATCMP1 · Digital Access',
            coef=-0.574, ci_lo=-1.034, ci_hi=-0.113, p=0.015,
            direction='narrows',
            school_meaning=(
                'More computers per student is associated with a narrower equity gap. '
                'Digital access in school compensates for the technology deficit at home '
                'that disproportionately affects low-income students — enabling independent '
                'practice, feedback tools, and self-paced learning.'
            ),
            action='Prioritise device and broadband access for disadvantaged students, '
                   'particularly for homework and catch-up. Loan schemes reduce the home gap.'
        ),
        dict(
            label='Negative school climate index',
            var='NEGSCLIM · School Climate',
            coef=0.885, ci_lo=0.051, ci_hi=1.719, p=0.037,
            direction='widens',
            school_meaning=(
                'A climate characterised by intimidation, conflict, and low safety is '
                'associated with a wider equity gap. Disadvantaged students have fewer '
                'alternative safe spaces — they cannot "opt out" of a hostile environment '
                'the way higher-SES students can through tutors or private provision.'
            ),
            action='Measure climate annually with student surveys. Restorative practice '
                   'and structured peer-support programmes improve climate fastest in '
                   'high-deprivation schools.'
        ),
    ]

    # Forest plot — 4 features, bottom-to-top (most significant at top)
    ols_sorted = list(reversed(ols_sig_features))
    ols_labels  = [d['label'] for d in ols_sorted]
    ols_coefs   = [d['coef'] for d in ols_sorted]
    ols_err_lo  = [abs(d['coef'] - d['ci_lo']) for d in ols_sorted]
    ols_err_hi  = [abs(d['ci_hi'] - d['coef']) for d in ols_sorted]
    ols_colors  = ['#1D9E75' if d['direction'] == 'narrows' else '#E3120B' for d in ols_sorted]

    ols_col1, ols_col2 = st.columns([2, 3])

    with ols_col1:
        fig_ols = go.Figure()
        fig_ols.add_vline(x=0, line_color='#6B7280', line_width=1.5, line_dash='dot')
        fig_ols.add_trace(go.Scatter(
            x=ols_coefs,
            y=ols_labels,
            mode='markers',
            marker=dict(
                color=ols_colors,
                size=14,
                symbol='diamond',
                line=dict(width=1.5, color='white')
            ),
            error_x=dict(
                type='data',
                symmetric=False,
                array=ols_err_hi,
                arrayminus=ols_err_lo,
                color='#6B7280',
                thickness=2.5,
                width=8
            ),
            customdata=[[d['p'], d['var']] for d in ols_sorted],
            hovertemplate=(
                '<b>%{y}</b><br>'
                'β = %{x:.3f}<br>'
                'p = %{customdata[0]:.3f}<br>'
                '%{customdata[1]}<extra></extra>'
            ),
            showlegend=False
        ))
        fig_ols.update_layout(
            title=dict(
                text='OLS Coefficients (95% CI)',
                font=dict(size=13, color='#1F2937'),
                x=0.5, xanchor='center'
            ),
            xaxis=dict(
                title=dict(text='β — Change in Within-School Gap (PISA pts)', font=dict(color='#374151', size=10)),
                zeroline=False,
                gridcolor='#F3F4F6',
                tickfont=dict(color='#374151', size=11),
            ),
            yaxis=dict(
                tickfont=dict(color='#1F2937', size=11, family='DM Sans'),
                automargin=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=580,
            margin=dict(l=10, r=30, t=50, b=60),
            font=dict(family='DM Sans', size=11, color='#374151')
        )
        st.plotly_chart(fig_ols, use_container_width=True)
        st.caption(
            'All 4 significant at p<0.05 (country-clustered SEs) | '
            'Green = Narrows Gap · Red = Widens Gap | '
            'Controls: School SES + Country FE | n=15,238 schools, 79 countries'
        )

    with ols_col2:
        for d in ols_sig_features:
            border_color = '#1D9E75' if d['direction'] == 'narrows' else '#E3120B'
            dir_label   = '↓ Narrows Gap' if d['direction'] == 'narrows' else '↑ Widens Gap'
            dir_color   = '#065F46' if d['direction'] == 'narrows' else '#991B1B'
            dir_bg      = '#F0FDF4' if d['direction'] == 'narrows' else '#FEF2F2'
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E5E7EB;
                        border-left:4px solid {border_color};
                        border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.65rem;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div style="font-size:0.88rem; font-weight:700; color:#1F2937;">
                            {d['label']}
                        </div>
                        <div style="font-size:0.72rem; color:#6B7280; margin-top:0.1rem;">
                            {d['var']} &nbsp;·&nbsp; β={d['coef']:+.3f} &nbsp;·&nbsp; p={d['p']:.3f}
                        </div>
                    </div>
                    <span style="background:{dir_bg}; color:{dir_color}; font-size:0.72rem;
                                 font-weight:700; padding:2px 8px; border-radius:12px;
                                 white-space:nowrap; margin-left:0.5rem;">
                        {dir_label}
                    </span>
                </div>
                <div style="font-size:0.8rem; color:#374151; margin-top:0.5rem; line-height:1.5;">
                    {d['school_meaning']}
                </div>
                <div style="font-size:0.78rem; color:{dir_color}; font-weight:600;
                            margin-top:0.4rem; padding-top:0.35rem;
                            border-top:1px solid #F3F4F6;">
                    Practical Lever: <span style="font-weight:400; color:#374151;">{d['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── OLS → INTERVENTION BRIDGE ────────────────────────────────────────────
    st.markdown('<div class="section-title">From findings to action — how the evidence shapes our recommendations</div>',
                unsafe_allow_html=True)

    bridge_left, bridge_right = st.columns([3, 2])

    with bridge_left:
        bridge_rows = [
            ('Student sense of belonging',       'narrows', 'climate_support',    'Social & emotional learning · Mentoring · Behaviour interventions'),
            ('Ability grouping in maths',         'widens',  'learning_support',   'Individualised instruction · Small group tuition · Peer tutoring'),
            ('Computer-to-student ratio',         'narrows', 'resource_intensive', 'Device & connectivity access · Teaching assistant interventions'),
            ('Negative school climate index',     'widens',  'climate_support',    'Behaviour interventions · Collaborative learning · Mentoring'),
        ]
        for label, direction, category, interventions in bridge_rows:
            arrow    = '↓ Narrows gap' if direction == 'narrows' else '↑ Widens gap'
            a_color  = '#065F46'       if direction == 'narrows' else '#991B1B'
            a_bg     = '#F0FDF4'       if direction == 'narrows' else '#FEF2F2'
            b_color  = '#1D4ED8'
            b_bg     = '#EFF6FF'
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px;
                        padding:0.75rem 1rem; margin-bottom:0.55rem;
                        display:flex; align-items:flex-start; gap:0.75rem;">
                <div style="flex:1;">
                    <div style="font-size:0.85rem; font-weight:700; color:#1F2937;">{label}</div>
                    <div style="font-size:0.75rem; color:#6B7280; margin-top:0.15rem;">
                        → <span style="background:{b_bg}; color:{b_color}; font-weight:600;
                                       padding:1px 7px; border-radius:10px; font-size:0.72rem;">
                            {category.replace('_',' ')}
                          </span>
                        &nbsp;{interventions}
                    </div>
                </div>
                <span style="background:{a_bg}; color:{a_color}; font-size:0.72rem; font-weight:700;
                             padding:2px 8px; border-radius:12px; white-space:nowrap;">
                    {arrow}
                </span>
            </div>
            """, unsafe_allow_html=True)

    with bridge_right:
        st.markdown("""
        <div class="insight-box" style="height:100%; box-sizing:border-box;">
            <b>Two methods. The same answer.</b><br><br>
            These four OLS findings define the <b>priority categories</b> used to rank
            interventions in <em>My School Report</em>. Your school's segment determines
            which category sits at the top of your recommendations list.<br><br>
            An independent XGBoost model trained on <b>613,744 students across 80 countries</b>
            (PISA 2022) confirmed 12 of the same school practices via SHAP analysis —
            with no reference to the EEF toolkit or these OLS results.<br><br>
            See the <b>Intervention Evidence Base</b> tab for the full cross-reference.
        </div>
        """, unsafe_allow_html=True)

    # Country rankings table
    st.markdown('<div class="section-title">Country Equity Gap Rankings (2022)</div>',
                unsafe_allow_html=True)

    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.caption('🔴 Largest Gaps — Most Urgent')
        top10 = gap_2022_ctx.nlargest(10, 'GAP')[['CNT', 'GAP', 'AVG_MATH', 'TRAJECTORY']]
        top10.columns = ['Country', 'Gap (pts)', 'Avg Maths', 'Trajectory']
        top10 = top10.reset_index(drop=True)
        top10.index += 1
        st.dataframe(top10, use_container_width=True)

    with col_rank2:
        st.caption('🟢 Smallest Gaps — Best Practice')
        bot10 = gap_2022_ctx.nsmallest(10, 'GAP')[['CNT', 'GAP', 'AVG_MATH', 'TRAJECTORY']]
        bot10.columns = ['Country', 'Gap (pts)', 'Avg Maths', 'Trajectory']
        bot10 = bot10.reset_index(drop=True)
        bot10.index += 1
        st.dataframe(bot10, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MY SCHOOL REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">School profile</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        countries = sorted(df_gap['CNT'].unique().tolist())
        country = st.selectbox(
            'Country code (ISO 3-letter)',
            options=countries,
            index=countries.index('GBR') if 'GBR' in countries else 0,
            help='Select the country where your school is located'
        )
        school_size = st.number_input(
            'Number of students',
            min_value=50,
            max_value=5000,
            value=620,
            step=10,
            help='Total student enrolment'
        )

    with col2:
        school_type = st.selectbox(
            'School type',
            options=['Public', 'Government-dependent private', 'Independent private'],
            index=0
        )
        stratio = st.number_input(
            'Student-teacher ratio',
            min_value=5.0,
            max_value=50.0,
            value=16.5,
            step=0.5,
            help='Number of students per teacher'
        )

    with col3:
        budget_options = {
            'Under £5,000': 5000,
            '£5,000 – £20,000': 20000,
            '£20,000 – £50,000': 50000,
            'Over £50,000': 100000
        }
        budget_label = st.selectbox(
            'Annual discretionary budget',
            options=list(budget_options.keys()),
            index=1,
            help='Annual budget for initiatives targeting disadvantaged students'
        )
        budget = budget_options[budget_label]

        disadvantaged_pct = st.slider(
            '% of students from low-income households',
            min_value=5,
            max_value=80,
            value=34,
            step=1,
            format='%d%%',
            help='Used to estimate the size of your target group'
        )

    # -- School diagnostic questions -----------------------------------------------
    st.markdown('<div class="section-title">School diagnostic</div>',
                unsafe_allow_html=True)
    st.caption('These questions help us identify your school profile and tailor recommendations')

    diag1, diag2, diag3, diag4 = st.columns(4)

    with diag1:
        from model import _load_segment_bundle
        _crefs = _load_segment_bundle().get('country_math_refs', {})
        _ref = f'Your country avg: {_crefs.get(country, "N/A")}' if country in _crefs else 'OECD avg: 472'
        school_math_score = st.number_input(
            'Average school maths score',
            min_value=200, max_value=700, value=480, step=5,
            help=f'Your school maths average (PISA scale). {_ref}'
        )

    with diag2:
        staff_shortage = st.select_slider(
            'Staff shortage impact',
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: ['', 'Not at all', 'Very little', 'To some extent', 'A lot'][x],
            help='How much is learning hindered by lack of teaching staff?'
        )

    with diag3:
        resource_shortage = st.select_slider(
            'Resource/material shortage',
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: ['', 'Not at all', 'Very little', 'To some extent', 'A lot'][x],
            help='How much is learning hindered by lack of educational materials?'
        )

    with diag4:
        behaviour_disruption = st.select_slider(
            'Behaviour disruption',
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: ['', 'Never', 'Some lessons', 'Most lessons', 'Every lesson'][x],
            help='How often does student behaviour disrupt lessons?'
        )


    st.markdown('<div class="section-title">Current practices</div>',
                unsafe_allow_html=True)
    st.caption('Select interventions already in place — these will be excluded from recommendations')

    all_interventions = df_interventions['intervention'].tolist()
    col_p1, col_p2, col_p3 = st.columns(3)
    existing = []
    chunk = len(all_interventions) // 3

    with col_p1:
        for item in all_interventions[:chunk]:
            if st.checkbox(item, key=f'existing_{item}'):
                existing.append(item)
    with col_p2:
        for item in all_interventions[chunk:chunk*2]:
            if st.checkbox(item, key=f'existing_{item}'):
                existing.append(item)
    with col_p3:
        for item in all_interventions[chunk*2:]:
            if st.checkbox(item, key=f'existing_{item}'):
                existing.append(item)

    st.markdown('<br>', unsafe_allow_html=True)
    generate = st.button('Generate my equity report')

    if generate:

        risk_result = get_equity_risk_score(
            country_code=country,
            school_type=school_type,
            stratio=stratio,
            df_traj=df_traj,
            df_gap=df_gap
        )

        # ── Predict school segment ────────────────────────────────────────────
        seg_result = predict_segment(
            risk_score=risk_result['equity_risk'],
            school_mean_math=school_math_score,
            staffshort=staff_shortage,
            edushort=resource_shortage,
            negsclim=behaviour_disruption,
            disadvantaged_pct=disadvantaged_pct / 100
        )

        int_result = get_interventions(
            country_code=country,
            annual_budget_gbp=budget,
            school_size=school_size,
            disadvantaged_pct=disadvantaged_pct / 100,
            existing_practices=existing,
            df_interventions=df_interventions,
            segment_priorities=seg_result['info'].get('priorities')
        )

        target_students = int_result['target_students']
        if len(int_result['interventions']) > 0:
            realistic_reduction = calculate_realistic_reduction(
                int_result['interventions'],
                risk_result['country_gap'],
                target_pct=disadvantaged_pct / 100
            )
        else:
            realistic_reduction = 0

        country_gap = risk_result['country_gap']

        # Mitigation credit from existing practices
        if existing:
            df_exist = df_interventions[df_interventions['intervention'].isin(existing)].copy()
            df_exist = df_exist.sort_values('gap_reduction_pts', ascending=False)
            mitigation_credit = 0.0
            for i, (_, erow) in enumerate(df_exist.iterrows()):
                mitigation_credit += erow['gap_reduction_pts'] * (disadvantaged_pct / 100) * (0.7 ** i)
            mitigation_credit = round(min(mitigation_credit, country_gap * 0.5), 1)
        else:
            mitigation_credit = 0.0

        # Full projected gap accounts for both existing practices AND new interventions
        projected_gap = max(0, round(country_gap - mitigation_credit - realistic_reduction, 1))

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Your equity report</div>',
                    unsafe_allow_html=True)

        # ── TOP METRICS ───────────────────────────────────────────────────────
        risk_score = risk_result['equity_risk']
        gap_score   = risk_result['gap_score']
        traj_score  = risk_result['traj_score']
        school_score = risk_result['school_score']

        if risk_score >= 65:
            risk_class = 'risk-high'
            risk_label = 'High risk'
        elif risk_score >= 40:
            risk_class = 'risk-medium'
            risk_label = 'Moderate risk'
        else:
            risk_class = 'risk-low'
            risk_label = 'Lower risk'

        # ── Segment card ──────────────────────────────────────────────────────
        seg_name = seg_result['segment']
        seg_info = seg_result['info']
        seg_conf = seg_result['confidence']
        top_priority = seg_info.get('priorities', ['learning support'])[0].replace('_', ' ')
        seg_rationale = SEGMENT_RATIONALE.get(seg_name, '')
        st.markdown(f"""
        <div style="background:{seg_info.get('color','#333')}15; border:2px solid {seg_info.get('color','#333')};
                    border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1.5rem;">
            <div style="font-size:0.75rem; font-weight:600; text-transform:uppercase;
                        letter-spacing:0.08em; color:{seg_info.get('color','#333')}; margin-bottom:0.3rem;">
                {seg_info.get('icon','')} Your school profile
            </div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.6rem;
                        color:#0F2A1D; line-height:1.2; margin-bottom:0.4rem;">
                {seg_name}
            </div>
            <div style="font-size:0.85rem; color:#4A6355;">
                {seg_info.get('desc','')}
            </div>
            <div style="font-size:0.82rem; color:#374151; margin-top:0.5rem; line-height:1.5;">
                {seg_rationale}
            </div>
            <div style="font-size:0.75rem; color:#9CA3AF; margin-top:0.5rem;">
                Confidence: {seg_conf}% | Based on your diagnostic inputs
            </div>
            <div style="font-size:0.8rem; color:#374151; margin-top:0.6rem; border-top:1px solid #E5E7EB; padding-top:0.5rem;">
                The interventions below prioritise <b>{top_priority}</b> to match your school's profile.
            </div>
        </div>
        """, unsafe_allow_html=True)

        traj = risk_result['trajectory']

        trajectory = traj
        if traj == 'Closing':
            traj_class = 'traj-closing'
            traj_arrow = '↓'
        elif traj == 'Widening':
            traj_class = 'traj-widening'
            traj_arrow = '↑'
        else:
            traj_class = 'traj-stable'
            traj_arrow = '→'

        total_reduction = round(mitigation_credit + realistic_reduction, 1)
        mit_sub = f'{len(existing)} practice(s) already in place' if existing else 'No existing practices entered'

        st.caption(
            f'The equity gap for {country} is {country_gap} pts — '
            f'the maths score difference between the most and least advantaged 25% of students. '
            f'Cards 2 and 4 show how much of this gap your school can close.'
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card" style="display:flex; flex-direction:column; height:100%;">
                <div class="metric-label">Equity risk score <span style="font-weight:400; text-transform:none; letter-spacing:0; font-size:0.7rem;">(0–100 composite index)</span></div>
                <div class="metric-value {risk_class}">{risk_score}<span style="font-size:1rem">/100</span></div>
                <div class="metric-sub">{risk_label}</div>
                <div style="display:flex; height:6px; border-radius:3px; overflow:hidden; margin-top:0.65rem;">
                    <div style="width:{gap_score}%; background:#2F6690;"></div>
                    <div style="width:{traj_score}%; background:#EF9F27;"></div>
                    <div style="width:{school_score}%; background:#6B5CA5;"></div>
                </div>
                <div style="font-size:0.68rem; color:#6B7280; margin-top:0.35rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
                    <span style="color:#2F6690;">&#9632; Gap: {gap_score}</span>
                    <span style="color:#EF9F27;">&#9632; Trend: {traj_score}</span>
                    <span style="color:#6B5CA5;">&#9632; School: {school_score}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="display:flex; flex-direction:column; height:100%;">
                <div class="metric-label">Gap already being closed <span style="font-weight:400; text-transform:none; letter-spacing:0; font-size:0.7rem;">(by existing practices)</span></div>
                <div class="metric-value risk-low">–{mitigation_credit}<span style="font-size:1rem"> pts</span></div>
                <div class="metric-sub">{mit_sub}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="display:flex; flex-direction:column; height:100%;">
                <div class="metric-label">National trajectory</div>
                <div class="metric-value {traj_class}">{traj_arrow} {traj}</div>
                <div class="metric-sub">Gap trend 2009–2022</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card" style="display:flex; flex-direction:column; height:100%;">
                <div class="metric-label">Additional gap closable <span style="font-weight:400; text-transform:none; letter-spacing:0; font-size:0.7rem;">(recommended interventions)</span></div>
                <div class="metric-value risk-low">–{realistic_reduction}<span style="font-size:1rem"> pts</span></div>
                <div class="metric-sub">Projected gap: {projected_gap} pts (from {country_gap} pts baseline)</div>
            </div>
            """, unsafe_allow_html=True)

        # ── WITHIN-COUNTRY SCHOOL COMPARISON ─────────────────────────────────
        st.markdown(f'<div class="section-title">How your school compares within {country}</div>',
                    unsafe_allow_html=True)

        df_country = df_scores[df_scores['CNT'] == country].copy()

        if len(df_country) < 5:
            st.info(f'Not enough school data for {country} to show a within-country comparison.')
        else:
            n_schools = len(df_country)
            country_median = round(float(df_country['EQUITY_RISK_SCORE'].median()), 1)

            # Dynamic axis range: data extent ± 10% of spread, clamped to 0–100
            scores = df_country['EQUITY_RISK_SCORE']
            spread = scores.max() - scores.min()
            pad = max(spread * 0.12, 3)
            x_min = max(0,   round(scores.min() - pad, 1))
            x_max = min(100, round(scores.max() + pad, 1))

            dist_col, box_col = st.columns(2)

            with dist_col:
                fig_dist = go.Figure()

                # Risk-zone bands clipped to visible range
                fig_dist.add_vrect(x0=x_min, x1=min(40, x_max),
                    fillcolor='rgba(29,158,117,0.10)', line_width=0)
                fig_dist.add_vrect(x0=max(40, x_min), x1=min(65, x_max),
                    fillcolor='rgba(239,159,39,0.10)', line_width=0)
                fig_dist.add_vrect(x0=max(65, x_min), x1=x_max,
                    fillcolor='rgba(227,18,11,0.10)', line_width=0)

                fig_dist.add_trace(go.Histogram(
                    x=scores,
                    xbins=dict(start=x_min, end=x_max, size=max((x_max - x_min) / 25, 0.5)),
                    marker_color='#A8D5C2',
                    marker_line=dict(color='white', width=0.8),
                    hovertemplate='Score %{x:.1f} – %{x:.1f}<br>Schools: %{y}<extra></extra>'
                ))
                fig_dist.add_vline(
                    x=risk_score, line_color='#0F2A1D', line_width=2.5,
                    annotation_text=f'Your school: {risk_score}',
                    annotation_position='top',
                    annotation_font=dict(color='#0F2A1D', size=11, family='DM Sans')
                )
                fig_dist.add_vline(
                    x=country_median, line_color='#6B7280', line_width=1.5, line_dash='dash',
                    annotation_text=f'Median: {country_median}',
                    annotation_position='top right',
                    annotation_font=dict(color='#6B7280', size=10, family='DM Sans')
                )
                fig_dist.update_layout(
                    title=dict(text=f'School equity risk distribution — {country}',
                               font=dict(size=13, color='#1F2937'), x=0.5, xanchor='center'),
                    xaxis=dict(
                        title=dict(text='Equity risk score', font=dict(color='#374151')),
                        tickfont=dict(color='#374151'),
                        range=[x_min, x_max]
                    ),
                    yaxis=dict(
                        title=dict(text='Number of schools', font=dict(color='#374151')),
                        tickfont=dict(color='#374151')
                    ),
                    plot_bgcolor='white', paper_bgcolor='white',
                    height=340, margin=dict(l=40, r=20, t=50, b=40),
                    showlegend=False, font=dict(family='DM Sans', color='#374151')
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption(f'n = {n_schools} schools in {country} · PISA 2022')

            with box_col:
                schtype_map = {1.0: 'Public', 2.0: 'Gov-dependent private', 3.0: 'Independent private'}
                type_colors = {
                    'Public': '#2F6690',
                    'Gov-dependent private': '#D96C06',
                    'Independent private': '#6B5CA5'
                }
                df_box = df_country.dropna(subset=['SCHTYPE']).copy()
                df_box['School Type'] = df_box['SCHTYPE'].map(schtype_map).fillna('Other')

                # Dynamic y-range across all school types
                box_scores = df_box['EQUITY_RISK_SCORE']
                bpad = max((box_scores.max() - box_scores.min()) * 0.15, 3)
                y_min = max(0,   round(box_scores.min() - bpad, 1))
                y_max = min(100, round(box_scores.max() + bpad, 1))

                fig_box = go.Figure()
                for stype in ['Public', 'Gov-dependent private', 'Independent private']:
                    grp = df_box[df_box['School Type'] == stype]
                    if len(grp) == 0:
                        continue
                    fig_box.add_trace(go.Box(
                        y=grp['EQUITY_RISK_SCORE'],
                        name=stype,
                        marker_color=type_colors[stype],
                        boxpoints='outliers',
                        marker=dict(size=3, opacity=0.5),
                        boxmean='sd',
                        hovertemplate=f'<b>{stype}</b><br>Score: %{{y:.1f}}<extra></extra>'
                    ))
                fig_box.add_hline(
                    y=risk_score, line_color='#0F2A1D', line_width=2, line_dash='dot',
                    annotation_text=f'Your school: {risk_score}',
                    annotation_position='right',
                    annotation_font=dict(color='#0F2A1D', size=10, family='DM Sans')
                )
                fig_box.update_layout(
                    title=dict(text=f'Risk score by school type — {country}',
                               font=dict(size=13, color='#1F2937'), x=0.5, xanchor='center'),
                    yaxis=dict(
                        title=dict(text='Equity risk score', font=dict(color='#374151')),
                        tickfont=dict(color='#374151'),
                        range=[y_min, y_max]
                    ),
                    xaxis=dict(tickfont=dict(color='#374151')),
                    plot_bgcolor='white', paper_bgcolor='white',
                    height=340, margin=dict(l=40, r=20, t=50, b=40),
                    showlegend=False, font=dict(family='DM Sans', color='#374151')
                )
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption('Box = IQR · centre line = median · dotted = your school · outliers shown · PISA 2022')

        # ── GAP PROJECTION WATERFALL ──────────────────────────────────────────
        st.markdown('<div class="section-title">Projected gap reduction pathway</div>',
                    unsafe_allow_html=True)

        fig_wf = go.Figure(go.Waterfall(
            orientation='h',
            measure=['absolute', 'relative', 'relative'],
            x=[country_gap, -mitigation_credit, -realistic_reduction],
            y=['Baseline country gap', 'Existing practices', 'Recommended interventions'],
            textposition='outside',
            text=[f'{country_gap} pts', f'–{mitigation_credit} pts', f'–{realistic_reduction} pts'],
            textfont=dict(family='DM Sans', color='#1F2937', size=11),
            connector=dict(line=dict(color='#E5E7EB', width=1, dash='dot')),
            decreasing=dict(marker=dict(color='#1D9E75', line=dict(width=0))),
            increasing=dict(marker=dict(color='#E3120B', line=dict(width=0))),
            totals=dict(marker=dict(color='#2F6690', line=dict(width=0)))
        ))
        fig_wf.update_layout(
            xaxis=dict(
                title=dict(text='Equity gap (maths score points)', font=dict(color='#374151')),
                tickfont=dict(color='#374151'),
                range=[0, country_gap * 1.15]
            ),
            yaxis=dict(tickfont=dict(color='#1F2937', size=11)),
            plot_bgcolor='white', paper_bgcolor='white',
            height=200, margin=dict(l=10, r=80, t=20, b=40),
            font=dict(family='DM Sans', color='#374151'),
            showlegend=False
        )
        st.plotly_chart(fig_wf, use_container_width=True)
        st.caption(
            f'Baseline: {country_gap} pts country avg gap · '
            f'After existing practices: {round(country_gap - mitigation_credit, 1)} pts · '
            f'After recommended interventions: {projected_gap} pts'
        )

        # ── INTERVENTIONS ─────────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-title">Recommended interventions — {budget_label} budget</div>',
            unsafe_allow_html=True
        )
        st.caption(
            f'Ranked by your segment profile ({seg_name}): '
            f'{top_priority} interventions appear first, '
            f'then by cost-effectiveness within each category.'
        )

        df_rec = int_result['interventions']

        if len(df_rec) == 0:
            st.warning(
                'No interventions found within your budget after excluding existing practices. '
                'Try increasing your budget or removing some existing practices.'
            )
        else:
            evidence_stars = {1: '●○○○○', 2: '●●○○○', 3: '●●●○○', 4: '●●●●○', 5: '●●●●●'}
            cost_labels = {1: '£ very low', 2: '££ low', 3: '£££ moderate',
                           4: '££££ high', 5: '£££££ very high'}

            col_int1, col_int2 = st.columns([3, 1])

            with col_int1:
                st.caption(
                    f'Target group: {target_students} disadvantaged students '
                    f'({disadvantaged_pct}% of {school_size})'
                )
                for _, row in df_rec.iterrows():
                    # Realistic contribution = EEF impact scaled to target group
                    # with diminishing returns for each additional intervention
                    realistic_contrib = round(
                        row['gap_reduction_pts'] * (disadvantaged_pct / 100) * (0.7 ** (row['rank'] - 1)), 1
                    )
                    cost_color = (
                        'badge-green' if row['cost_rating'] == 1
                        else 'badge-amber' if row['cost_rating'] <= 3
                        else 'badge-red'
                    )
                    profile_badge = (
                        '<span style="background:#EFF6FF; color:#1D4ED8; font-size:0.68rem; '
                        'font-weight:700; padding:2px 7px; border-radius:10px; margin-left:6px;">'
                        'Profile Match</span>'
                    ) if row['priority_tier'] == 0 else ''
                    st.markdown(f"""
                    <div class="intervention-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start">
                            <div class="intervention-name">
                                #{int(row['rank'])} {row['intervention']}{profile_badge}
                            </div>
                            <div style="text-align:right;">
                                <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#1D9E75; line-height:1.1;">
                                    –{realistic_contrib:.1f} pts
                                </div>
                                <div style="font-size:0.68rem; color:#9CA3AF;">
                                    to your equity gap
                                </div>
                                <div style="font-size:0.68rem; color:#9CA3AF;">
                                    EEF raw: –{row['gap_reduction_pts']:.0f} pts
                                </div>
                            </div>
                        </div>
                        <div class="intervention-meta" style="margin-top:0.5rem">
                            <span class="badge {cost_color}">{cost_labels.get(int(row['cost_rating']), '£')}</span>
                            <span class="badge badge-blue">£{int(row['total_cost']):,} total</span>
                            <span class="badge badge-gray">Evidence: {evidence_stars.get(int(row['evidence']), '○○○○○')}</span>
                            <span class="badge badge-gray">{int(row['impact_months'])} months impact</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Over budget items
                df_over = df_interventions[
                    (~df_interventions['intervention'].isin(existing)) &
                    (df_interventions['impact_months'] > 0) &
                    (df_interventions['evidence'] >= 2) &
                    (df_interventions['cost_per_pupil'] * round(school_size * disadvantaged_pct / 100) > budget)
                ].copy()
                df_over['total_cost'] = df_over['cost_per_pupil'] * round(
                    school_size * disadvantaged_pct / 100
                )
                if len(df_over) > 0:
                    st.markdown(
                        '<div style="font-size:0.85rem; color:#9CA3AF; '
                        'margin-top:1rem; margin-bottom:0.5rem">'
                        '🔒 Over budget — shown for reference</div>',
                        unsafe_allow_html=True
                    )
                    for _, row in df_over.iterrows():
                        st.markdown(f"""
                        <div class="intervention-card over-budget">
                            <div class="intervention-name">{row['intervention']}</div>
                            <div class="intervention-meta">
                                £{int(row['total_cost']):,} total ·
                                {int(row['impact_months'])} months impact
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            with col_int2:
                total_in_budget = len(df_rec)
                avg_evidence = df_rec['evidence'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Interventions available</div>
                    <div class="metric-value">{total_in_budget}</div>
                    <div class="metric-sub">Within your budget</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg evidence strength</div>
                    <div class="metric-value">{avg_evidence:.1f}<span style="font-size:1rem">/5</span></div>
                    <div class="metric-sub">Padlocks (EEF scale)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">If top intervention applied</div>
                    <div class="metric-value risk-low">
                        –{round(df_rec.iloc[0]['gap_reduction_pts'] * (disadvantaged_pct / 100), 1):.1f}<span style="font-size:1rem"> pts</span>
                    </div>
                    <div class="metric-sub">{df_rec.iloc[0]['intervention']} · scaled to your {disadvantaged_pct}% target group</div>
                </div>
                """, unsafe_allow_html=True)

        # ── FOOTER ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="footer-note">
            <b>Data sources:</b> OECD PISA 2009–2022 (734,122 students, 107 countries, stratified sample) |
            Education Endowment Foundation Teaching & Learning Toolkit<br>
            <b>Methodology:</b> Equity gap = maths score difference between top and bottom SES quartile.
            Risk score weighted by gap size (60%), trajectory (20%) and school profile (20%).
            Gap reduction applies EEF impact to disadvantaged students only with diminishing returns.
            Margin of error: ±12.4 PISA points (95% CI).<br>
            <b>Note:</b> Projections are estimates based on published evidence.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INTERVENTION EVIDENCE BASE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    # ── Load SHAP data ────────────────────────────────────────────────────────
    @st.cache_data
    def load_shap():
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'shap_importance_v7.csv'))
        sc_labels = {
            # ── Teaching practices ──────────────────────────────────────────
            'SC012Q02TA': 'Teachers regularly give feedback to students',
            'SC012Q05TA': 'Teachers use assessments to monitor student learning',
            'SC012Q08JA': 'Teachers adapt instruction to individual student needs',
            'SC016Q03TA': 'Teachers receive regular professional development',
            # ── School academic expectations & student self-regulation ──────
            'SC017Q09JA': 'School has clear and high academic expectations',
            'SC017Q10JA': 'School promotes student self-regulation',
            # ── Student grouping practices ──────────────────────────────────
            'SC172Q02JA': 'School groups students by ability between classes (some subjects)',
            'SC172Q03JA': 'School groups students by ability within maths classes',
            'SC172Q04JA': 'School uses ability grouping between classes',
            'SC172Q05JA': 'School uses mixed-ability grouping',
            'SC172Q07JA': 'School offers support groups for struggling students',
            # ── School admissions & selection ───────────────────────────────
            'SC213Q01JA': 'School selects students by academic ability',
            'SC213Q02JA': 'School accepts students with special educational needs',
            # ── School climate & culture ────────────────────────────────────
            'SC188Q03JA': 'School has a competitive academic culture',
            'SC188Q05JA': 'School emphasises student wellbeing',
            # ── Data use & assessment ───────────────────────────────────────
            'SC190Q02JA': 'School uses student data to improve teaching',
            'SC190Q07JA': 'School tracks student progress over time',
            'SC192Q01JA': 'School shares assessment results with parents',
            # ── Parental & family engagement ────────────────────────────────
            'SC180Q01JA': 'School involves parents in educational decisions',
            # ── Additional academic programmes ──────────────────────────────
            'SC202Q09JA': 'School offers after-school tutoring',
            'SC202Q10JA': 'School offers extracurricular academic support',
            # ── Digital resources ───────────────────────────────────────────
            'SC053Q01TA': 'School has computers available for student instruction',
            # ── Leadership & management ─────────────────────────────────────
            'SC004Q05NA': 'Principal monitors teacher instructional practices',
            'SC212Q01JA': 'Principal provides instructional leadership',
            # ── School autonomy & governance ────────────────────────────────
            'SCHAUTO':    'School has high curriculum autonomy',
            'SCHLTYPE':   'School governance type (public vs. private)',
            # ── Artefacts (filtered from chart) ─────────────────────────────
            'MEAN_ESCS':  'Cluster avg socioeconomic status (artefact)',
            'N_STUDENTS': 'Students in cluster (artefact)',
            'N_SCHOOLS':  'Schools in cluster (artefact)',
            'LOG_GDP_PC': 'Country GDP per capita (country-level)',
            'SD_ESCS':    'SES spread in cluster',
        }
        eef_mapping = {
            'SC012Q05TA': 'Feedback',
            'SC012Q02TA': 'Feedback',
            'SC012Q08JA': 'Individualised instruction',
            'SC172Q07JA': 'Small group tuition',
            'SC172Q05JA': 'Collaborative learning',
            'SC190Q02JA': 'Metacognition & self-regulation',
            'SC180Q01JA': 'Parental engagement',
            'SC016Q03TA': 'Teaching assistant interventions',
            'SC017Q09JA': 'Behaviour interventions',
            'SC202Q09JA': 'One to one tuition',
            'SC202Q10JA': 'Extending school time',
            'SC017Q10JA': 'Metacognition & self-regulation',
            'SC192Q01JA': 'Parental engagement',
        }
        df['label'] = df['feature'].map(sc_labels).fillna(df['feature'])
        df['eef_equivalent'] = df['feature'].map(eef_mapping).fillna('')
        artefacts = ['MEAN_ESCS', 'N_STUDENTS', 'N_SCHOOLS',
                     'LOG_GDP_PC', 'SD_ESCS', 'EDU_PCT', 'GINI']
        df['is_artefact'] = df['feature'].isin(artefacts)
        return df

    df_shap = load_shap()

    # ── EEF CHART ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Part 1 — What the evidence says: EEF Teaching &amp; Learning Toolkit</div>',
                unsafe_allow_html=True)
    st.caption(
        '18 interventions from the EEF Toolkit, filtered to those with ≥2 padlocks of evidence and positive impact. '
        'Gap reduction converted from EEF months using 1 month ≈ 3.5 PISA points (OECD benchmark). '
        'Cost-effectiveness = PISA points gained per £100 per pupil.'
    )

    df_chart = df_interventions[
        (df_interventions['evidence'] >= 2) &
        (df_interventions['impact_months'] > 0)
    ].copy().sort_values('cost_effectiveness', ascending=True)

    cost_color_map = {1: '#1D9E75', 2: '#5DCAA5', 3: '#EF9F27', 4: '#D85A30', 5: '#E3120B'}
    df_chart['COLOR'] = df_chart['cost_rating'].map(cost_color_map)

    fig_eff = go.Figure(go.Bar(
        x=df_chart['cost_effectiveness'],
        y=df_chart['intervention'],
        orientation='h',
        marker_color=df_chart['COLOR'],
        customdata=df_chart[['gap_reduction_pts', 'cost_per_pupil',
                              'evidence', 'impact_months']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Gap reduction: %{customdata[0]:.1f} pts<br>'
            'Cost per pupil: £%{customdata[1]}<br>'
            'Evidence: %{customdata[2]}/5 padlocks<br>'
            'Impact: %{customdata[3]} months<br>'
            'Cost-effectiveness: %{x:.1f}<extra></extra>'
        )
    ))
    fig_eff.update_layout(
        xaxis=dict(
            title=dict(text='Cost-effectiveness (PISA points per £100 per pupil)', font=dict(color='#374151')),
            tickfont=dict(color='#374151'),
        ),
        yaxis=dict(
            tickfont=dict(color='#1F2937', size=12),
            automargin=True,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=520,
        margin=dict(l=230, r=40, t=20, b=60),
        font=dict(family='DM Sans', size=12, color='#374151')
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    # Cost legend
    leg1, leg2, leg3, leg4, leg5 = st.columns(5)
    for col, label, color in [
        (leg1, '£ Very low cost',   '#1D9E75'),
        (leg2, '££ Low cost',       '#5DCAA5'),
        (leg3, '£££ Moderate',      '#EF9F27'),
        (leg4, '££££ High cost',    '#D85A30'),
        (leg5, '£££££ Very high',   '#E3120B'),
    ]:
        col.markdown(
            f'<div style="background:{color}; color:white; padding:6px 10px; '
            f'border-radius:6px; font-size:0.78rem; font-weight:600; text-align:center">'
            f'{label}</div>',
            unsafe_allow_html=True
        )

    # ── V7 CROSS-VALIDATION SECTION ───────────────────────────────────────────
    st.markdown('<div class="section-title">Part 2 — Independent confirmation: PISA 2022 machine learning analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        🔬 <b>This is a completely independent check.</b><br>
        We trained an XGBoost model on PISA 2022 microdata (<b>613,744 students, 80 countries</b>)
        using 369 school questionnaire variables — with no reference to the EEF toolkit or the
        OLS regression above. SHAP analysis identified which school practices most strongly reduce
        the socioeconomic performance gap in the data.
        <b>12 of those practices map directly to EEF interventions recommended in Part 1.</b>
        Two methods, one answer.
    </div>
    """, unsafe_allow_html=True)

    # SHAP bar chart — full width so plain-English labels have room
    # Filter to features that have a plain-English label (exclude raw codes)
    df_reduces = df_shap[
        (~df_shap['is_artefact']) &
        (df_shap['direction'] == 'reduces_slope') &
        (df_shap['label'] != df_shap['feature']) &
        (df_shap['mean_abs_shap'] > 0)
    ].sort_values('mean_abs_shap', ascending=True)

    df_increases = df_shap[
        (~df_shap['is_artefact']) &
        (df_shap['direction'] == 'increases_slope') &
        (df_shap['label'] != df_shap['feature']) &
        (df_shap['mean_abs_shap'] > 0)
    ].sort_values('mean_abs_shap', ascending=True)

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=df_reduces['mean_abs_shap'],
        y=df_reduces['label'],
        orientation='h',
        marker_color='#1D9E75',
        name='Reduces gap (better equity)',
        hovertemplate='<b>%{y}</b><br>|SHAP|: %{x:.3f}<extra></extra>'
    ))
    fig_shap.add_trace(go.Bar(
        x=df_increases['mean_abs_shap'],
        y=df_increases['label'],
        orientation='h',
        marker_color='#E3120B',
        name='Increases gap (worse equity)',
        hovertemplate='<b>%{y}</b><br>|SHAP|: %{x:.3f}<extra></extra>'
    ))
    fig_shap.update_layout(
        title=dict(text='School practices ranked by SHAP importance', font=dict(size=13, color='#1F2937'),
                   x=0.5, xanchor='center'),
        xaxis=dict(
            title=dict(text='Mean |SHAP| value', font=dict(color='#374151')),
            tickfont=dict(color='#374151'),
            gridcolor='#F3F4F6',
        ),
        yaxis=dict(
            tickfont=dict(color='#1F2937', size=12, family='DM Sans'),
            automargin=True,
        ),
        bargap=0.45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=900,
        margin=dict(l=20, r=20, t=50, b=60),
        legend=dict(
            orientation='v',
            yanchor='bottom', y=0.01,
            xanchor='right', x=0.99,
            font=dict(size=11, color='#374151'),
            bgcolor='rgba(255,255,255,0.88)',
            bordercolor='#E5E7EB',
            borderwidth=1,
        ),
        font=dict(family='DM Sans', size=12, color='#374151'),
        barmode='overlay'
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Cross-reference cards — full width, 3-column grid
    st.markdown('**Cross-reference: PISA model confirms EEF recommendations**')
    st.caption(
        'School practices identified by XGBoost SHAP as reducing the equity gap, '
        'mapped to corresponding EEF interventions'
    )

    df_cross = df_shap[
        (~df_shap['is_artefact']) &
        (df_shap['direction'] == 'reduces_slope') &
        (df_shap['eef_equivalent'] != '')
    ].sort_values('mean_abs_shap', ascending=False)

    cross_rows = list(df_cross.iterrows())
    n_cols = 3
    for row_start in range(0, len(cross_rows), n_cols):
        cols = st.columns(n_cols)
        for col_idx, (_, row) in enumerate(cross_rows[row_start:row_start + n_cols]):
            with cols[col_idx]:
                st.markdown(f"""
                <div style="background:#F7FAF8; border:1px solid #D4E6DA; border-radius:8px;
                            padding:0.7rem 1rem; margin-bottom:0.5rem;
                            border-left:3px solid #1D9E75;">
                    <div style="font-size:0.82rem; font-weight:600; color:#0F2A1D">
                        {row['label']}
                    </div>
                    <div style="font-size:0.75rem; color:#4A6355; margin-top:0.2rem">
                        SHAP: {row['mean_abs_shap']:.3f} →
                        <span style="color:#1D9E75; font-weight:600">
                            EEF: {row['eef_equivalent']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0F2A1D; color:white; border-radius:8px;
                padding:0.8rem 1rem; margin-top:0.5rem; font-size:0.85rem;">
        <b>✓ {len(df_cross)} interventions confirmed</b> by both PISA model
        and EEF randomised controlled trial evidence
    </div>
    """, unsafe_allow_html=True)

    # ── MODEL NOTES ───────────────────────────────────────────────────────────
    with st.expander("📋 Model methodology and limitations"):
        st.markdown("""
        **What it does:** Predicts the ESCS slope (PISA points gained per unit of
        socioeconomic status) at the cluster level within each country. A lower slope
        means a smaller equity gap. Schools are grouped into k=5 clusters per country
        using k-means on 369 school questionnaire variables.

        **Data:** Full PISA 2022 microdata — 613,744 students, 21,629 schools, 80 countries.

        **Performance:**
        - Hold-out R² = 0.599 (n=77 test clusters)
        - Grouped 10-fold CV R² = 0.487 ± 0.081 (primary — respects country boundaries)
        - 5-fold CV R² = 0.423 ± 0.064 (ungrouped, for reference)

        **Limitations:** The model uses GroupKFold cross-validation which holds out ~9
        countries per fold, giving stable and interpretable R² estimates. The SHAP
        feature importance is used only for cross-validation of the EEF findings,
        not for prediction. While the model cannot extrapolate to wholly unseen countries
        — since between-country variation is driven by structural factors not captured
        by school features — the patterns it identifies within the training set are
        meaningful and independently validated by EEF randomised controlled trials.
        """)

    # ── FULL INTERVENTION TABLE ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Full intervention library</div>',
                unsafe_allow_html=True)

    df_display = df_interventions[[
        'intervention', 'impact_months', 'gap_reduction_pts',
        'cost_per_pupil', 'cost_effectiveness', 'evidence'
    ]].copy()
    df_display.columns = [
        'Intervention', 'EEF Months', 'Gap Reduction (pts)',
        'Cost per Pupil (£)', 'Cost-Effectiveness', 'Evidence (padlocks)'
    ]
    df_display = df_display.sort_values('Cost-Effectiveness', ascending=False)
    df_display = df_display.reset_index(drop=True)
    df_display.index += 1
    st.dataframe(df_display, use_container_width=True)

    st.markdown("""
    <div class="footer-note">
        <b>EEF Source:</b> Education Endowment Foundation Teaching & Learning Toolkit —
        Closing the Disadvantage Gap tab (educationendowmentfoundation.org.uk)<br>
        <b>PISA Model:</b> XGBoost trained on full PISA 2022 microdata
        (613,744 students, 80 countries). SHAP importance used for cross-validation only.<br>
        <b>Cost definitions:</b> £ = up to £80/pupil/year | ££ = £80–200 |
        £££ = £200–700 | ££££ = £700–1,200 | £££££ = over £1,200 (EEF definitions)<br>
        <b>Conversion:</b> 1 EEF month ≈ 3.5 PISA points (OECD benchmark: 1 year ≈ 35–40 pts)
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4 - MODEL VALIDATION
# ==============================================================================
with tab4:

    st.markdown('<div class="section-title">How we built the risk score — and why you can trust it</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <b>The equity risk score is not a black box.</b><br>
        It is a transparent formula: <b>60%</b> from your country's SES performance gap (PISA data),
        <b>20%</b> from whether that gap is closing or widening (trajectory),
        <b>20%</b> from your school's profile (type and student-teacher ratio).<br><br>
        The benchmark below tests whether operational school features have genuine predictive power
        for equity outcomes — they do, modestly (F1 0.38–0.59). This is why the risk score
        combines data-derived signals with an interpretable formula rather than using a pure
        ML prediction: the formula is auditable, the ML confirms the signal is real.
        <b>Both Random Forest and LightGBM agree within 0.05 F1</b>, confirming the signal
        is model-agnostic, not an artefact of one algorithm.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Robustness benchmark: Random Forest vs LightGBM</div>',
                unsafe_allow_html=True)

    # Load benchmark results
    df_bench = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'v2_comparison.csv'))

    targets = ['trajectory', 'gap_band', 'risk_tier']
    target_labels = {'trajectory': 'Trajectory', 'gap_band': 'Gap Band', 'risk_tier': 'Risk Tier'}

    fig_bench = go.Figure()
    for model_name, color in [('RF', '#2C6E49'), ('LGBM', '#D96C06')]:
        model_data = df_bench[df_bench['model'] == model_name]
        fig_bench.add_trace(go.Bar(
            x=[target_labels.get(t, t) for t in model_data['target']],
            y=model_data['f1'],
            name=model_name,
            marker_color=color,
            text=[f'{v:.3f}' for v in model_data['f1']],
            textposition='outside',
        ))

    fig_bench.update_layout(
        title='F1 Score comparison: RF vs LightGBM (leak-free longitudinal)',
        yaxis_title='F1 Score (weighted)',
        yaxis_range=[0, 0.75],
        plot_bgcolor='white',
        paper_bgcolor='white',
        barmode='group',
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family='DM Sans', size=12),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    # Key metrics cards
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best target</div>
            <div class="metric-value">Risk Tier</div>
            <div class="metric-sub">RF F1 = 0.592 | LGBM F1 = 0.579</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model agreement</div>
            <div class="metric-value risk-low">< 0.05</div>
            <div class="metric-sub">Max F1 difference between RF and LGBM</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Segment classifier</div>
            <div class="metric-value">F1 = 0.626</div>
            <div class="metric-sub">LightGBM on 14,889 labeled schools</div>
        </div>
        """, unsafe_allow_html=True)

    # Methodology section
    st.markdown('<div class="section-title">Validation methodology</div>',
                unsafe_allow_html=True)

    meth1, meth2 = st.columns(2)

    with meth1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Leak-free design</div>
            <div style="font-size:0.85rem; color:#374151; margin-top:0.5rem">
                <b>Solution:</b> We removed all derived/leaky features and used only
                genuine school-level operational features: funding ratios, staffing
                levels, and enrollment patterns.<br><br>
                <b>Validation:</b> GroupKFold by country ensures no within-country
                data leakage between train and test splits.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with meth2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">What the modest F1 scores mean</div>
            <div style="font-size:0.85rem; color:#374151; margin-top:0.5rem">
                F1 scores of 0.38-0.59 are <b>expected and valid</b>. School-level
                features alone cannot fully predict country-level equity outcomes
                because structural factors (culture, policy, history) dominate.<br><br>
                This validates EquiTrack's <b>hybrid architecture</b>: the formula-based
                risk score (gap + trajectory + school profile) captures what pure ML
                cannot, while the school-level ML confirms that operational features
                provide genuine, modest predictive signal.<br><br>
                <b>The segment classifier</b> achieves higher F1 (0.626) because it
                predicts school-level profiles rather than country-level targets.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-note">
        <b>Benchmark data:</b> 84,232 students across 5 PISA cycles (2009, 2012, 2015, 2018, 2022) |
        2,000 stratified students per country per cycle |
        Margin of error: +/-12.4 PISA points (95% CI)<br>
        <b>Models:</b> Random Forest (500 trees) and LightGBM (200 estimators) |
        Evaluated with GroupKFold (country isolation) and StratifiedKFold<br>
        <b>Segment classifier:</b> LightGBM trained on 14,889 labeled schools from
        PISA 2022 school profiles | 5-fold stratified CV F1 = 0.626
    </div>
    """, unsafe_allow_html=True)
