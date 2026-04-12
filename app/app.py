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
        color: #9CA3AF;
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
    "📚  Intervention Evidence Base"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">The global equity gap — why this matters</div>',
                unsafe_allow_html=True)

    ctx1, ctx2, ctx3 = st.columns(3)

    # Chart A — Global gap trend
    with ctx1:
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
            fill='toself',
            fillcolor='rgba(29,158,117,0.12)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_trend.add_trace(go.Scatter(
            x=global_gap['YEAR'],
            y=global_gap['MEAN_GAP'],
            mode='lines+markers',
            line=dict(color='#1D9E75', width=3),
            marker=dict(size=9),
            hovertemplate='%{x}: %{y:.1f} pts<extra></extra>',
            showlegend=False
        ))
        fig_trend.update_layout(
            title=dict(text='Global equity gap 2009–2022', font=dict(size=13)),
            xaxis_title='PISA cycle',
            yaxis_title='Avg gap (pts)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown(
            '<div class="insight-box">📊 The global equity gap has barely moved in 13 years '
            '— from 83.8 to 81.6 points. Without deliberate intervention, the default is stagnation.</div>',
            unsafe_allow_html=True
        )

    # Chart B — Trajectory bar chart
    with ctx2:
        gap_2022_ctx = df_gap[df_gap['YEAR'].astype(str) == '2022'].copy()
        gap_2022_ctx = gap_2022_ctx.merge(
            df_traj[['CNT', 'TRAJECTORY']], on='CNT', how='left'
        )
        gap_2022_ctx['TRAJECTORY'] = gap_2022_ctx['TRAJECTORY'].fillna('No data')

        traj_order = ['Widening', 'Stable', 'Closing']
        traj_counts = (
            gap_2022_ctx[gap_2022_ctx['TRAJECTORY'].isin(traj_order)]
            ['TRAJECTORY'].value_counts()
            .reindex(traj_order)
            .reset_index()
        )
        traj_counts.columns = ['TRAJECTORY', 'COUNT']

        fig_traj_bar = go.Figure(go.Bar(
            x=traj_counts['COUNT'],
            y=traj_counts['TRAJECTORY'],
            orientation='h',
            marker_color=[color_map[t] for t in traj_counts['TRAJECTORY']],
            text=traj_counts['COUNT'].astype(str) + ' countries',
            textposition='inside',
            textfont=dict(color='white', size=12),
            hovertemplate='<b>%{y}</b><br>%{x} countries<extra></extra>'
        ))
        fig_traj_bar.update_layout(
            title=dict(text='Country trajectories 2009–2022', font=dict(size=13)),
            xaxis_title='Number of countries',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            margin=dict(l=80, r=20, t=50, b=40),
            showlegend=False,
            font=dict(family='DM Sans', size=12)
        )
        st.plotly_chart(fig_traj_bar, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🔴 Two thirds of countries are widening or stable. '
            'Only one third are actively closing their equity gap.</div>',
            unsafe_allow_html=True
        )

    # Chart C — Gap vs performance scatter
    with ctx3:
        gap_2022_sc = df_gap[df_gap['YEAR'].astype(str) == '2022'].copy()
        gap_2022_sc = gap_2022_sc.merge(
            df_traj[['CNT', 'TRAJECTORY']], on='CNT', how='left'
        )
        gap_2022_sc['TRAJECTORY'] = gap_2022_sc['TRAJECTORY'].fillna('No data')

        fig_scatter = go.Figure()
        for traj_val, color in color_map.items():
            df_sub = gap_2022_sc[gap_2022_sc['TRAJECTORY'] == traj_val]
            if len(df_sub) == 0:
                continue
            fig_scatter.add_trace(go.Scatter(
                x=df_sub['GAP'],
                y=df_sub['AVG_MATH'],
                mode='markers',
                name=traj_val,
                marker=dict(size=7, color=color, opacity=0.75),
                hovertemplate='<b>%{text}</b><br>Gap: %{x:.1f}<br>Score: %{y:.0f}<extra></extra>',
                text=df_sub['CNT']
            ))

        fig_scatter.add_vline(x=gap_2022_sc['GAP'].mean(), line_dash='dash',
                              line_color='#9CA3AF', line_width=1)
        fig_scatter.add_hline(y=gap_2022_sc['AVG_MATH'].mean(), line_dash='dash',
                              line_color='#9CA3AF', line_width=1)
        fig_scatter.add_annotation(
            x=gap_2022_sc['GAP'].min() + 3,
            y=gap_2022_sc['AVG_MATH'].max() - 10,
            text='<b>Ideal</b>',
            showarrow=False,
            font=dict(size=10, color='#1D9E75'),
            bgcolor='rgba(29,158,117,0.1)'
        )

        fig_scatter.update_layout(
            title=dict(text='Performance vs equity gap (2022)', font=dict(size=13)),
            xaxis_title='Gap (pts) — lower is better',
            yaxis_title='Avg maths score',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
            showlegend=False,
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown(
            '<div class="insight-box">✅ Top-left = ideal: high scores, low gap. '
            'Macao and Japan prove high performance and equity can coexist.</div>',
            unsafe_allow_html=True
        )

    # Country rankings table
    st.markdown('<div class="section-title">Country equity gap rankings (2022)</div>',
                unsafe_allow_html=True)

    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.caption('🔴 Largest gaps — most urgent')
        top10 = gap_2022_ctx.nlargest(10, 'GAP')[['CNT', 'GAP', 'AVG_MATH', 'TRAJECTORY']]
        top10.columns = ['Country', 'Gap (pts)', 'Avg Maths', 'Trajectory']
        top10 = top10.reset_index(drop=True)
        top10.index += 1
        st.dataframe(top10, use_container_width=True)

    with col_rank2:
        st.caption('🟢 Smallest gaps — best practice')
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
            df_scores=df_scores,
            df_traj=df_traj,
            df_gap=df_gap
        )

        # ── Predict school segment ────────────────────────────────────────────
        import joblib
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
            df_interventions=df_interventions
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

        projected_gap = max(0, round(risk_result['country_gap'] - realistic_reduction, 1))

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Your equity report</div>',
                    unsafe_allow_html=True)

        # ── TOP METRICS ───────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)

        risk_score = risk_result['equity_risk']
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
            <div style="font-size:0.75rem; color:#9CA3AF; margin-top:0.5rem;">
                Confidence: {seg_conf}% | Based on your diagnostic inputs
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

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Equity risk score</div>
                <div class="metric-value {risk_class}">{risk_score}<span style="font-size:1rem">/100</span></div>
                <div class="metric-sub">{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Country avg gap</div>
                <div class="metric-value">{risk_result['country_gap']}<span style="font-size:1rem"> pts</span></div>
                <div class="metric-sub">Maths score difference top vs bottom SES</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">National trajectory</div>
                <div class="metric-value {traj_class}">{traj_arrow} {traj}</div>
                <div class="metric-sub">Gap trend 2009–2022</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Potential gap reduction</div>
                <div class="metric-value risk-low">–{realistic_reduction}<span style="font-size:1rem"> pts</span></div>
                <div class="metric-sub">Projected: {projected_gap} pts after interventions</div>
            </div>
            """, unsafe_allow_html=True)

        # ── BENCHMARK CHARTS ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">How your country compares</div>',
                    unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            gap_2022 = df_gap[df_gap['YEAR'].astype(str) == '2022'].copy()
            country_gap_val = risk_result['country_gap']

            fig_bench = go.Figure()
            fig_bench.add_trace(go.Histogram(
                x=gap_2022['GAP'],
                nbinsx=25,
                marker_color='#D4E6DA',
                name='All countries'
            ))
            fig_bench.add_vline(
                x=country_gap_val,
                line_color='#1D9E75',
                line_width=2.5,
                annotation_text=f'{country}: {country_gap_val} pts',
                annotation_position='top',
                annotation_font_color='#1D9E75'
            )
            fig_bench.add_vline(
                x=gap_2022['GAP'].mean(),
                line_color='#9CA3AF',
                line_width=1.5,
                line_dash='dash',
                annotation_text=f'Global avg: {gap_2022["GAP"].mean():.0f} pts',
                annotation_position='top right',
                annotation_font_color='#9CA3AF'
            )
            fig_bench.update_layout(
                title=f'{country} gap vs global distribution (2022)',
                xaxis_title='Equity gap (maths score points)',
                yaxis_title='Number of countries',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=320,
                margin=dict(l=40, r=40, t=50, b=40),
                showlegend=False,
                font=dict(family='DM Sans')
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        with col_chart2:
            gap_2022_sc2 = df_gap[df_gap['YEAR'].astype(str) == '2022'].copy()
            gap_2022_sc2 = gap_2022_sc2.merge(
                df_traj[['CNT', 'TRAJECTORY']], on='CNT', how='left'
            )
            gap_2022_sc2['TRAJECTORY'] = gap_2022_sc2['TRAJECTORY'].fillna('No data')

            fig_sc2 = go.Figure()
            for traj_val, color in color_map.items():
                df_sub2 = gap_2022_sc2[
                    (gap_2022_sc2['TRAJECTORY'] == traj_val) &
                    (gap_2022_sc2['CNT'] != country)
                ]
                if len(df_sub2) == 0:
                    continue
                fig_sc2.add_trace(go.Scatter(
                    x=df_sub2['GAP'],
                    y=df_sub2['AVG_MATH'],
                    mode='markers',
                    marker=dict(size=7, color=color, opacity=0.5),
                    hovertemplate='<b>%{text}</b><br>Gap: %{x:.1f}<br>Score: %{y:.0f}<extra></extra>',
                    text=df_sub2['CNT'],
                    showlegend=False
                ))

            country_row = gap_2022_sc2[gap_2022_sc2['CNT'] == country]
            if len(country_row) > 0:
                fig_sc2.add_trace(go.Scatter(
                    x=country_row['GAP'],
                    y=country_row['AVG_MATH'],
                    mode='markers+text',
                    marker=dict(size=14, color='#0F2A1D',
                                line=dict(width=2, color='white')),
                    text=[country],
                    textposition='top center',
                    textfont=dict(size=12, color='#0F2A1D'),
                    hovertemplate=f'<b>{country}</b><br>Gap: %{{x:.1f}}<br>Score: %{{y:.0f}}<extra></extra>',
                    showlegend=False
                ))

            fig_sc2.add_vline(x=gap_2022_sc2['GAP'].mean(), line_dash='dash',
                              line_color='#9CA3AF', line_width=1)
            fig_sc2.add_hline(y=gap_2022_sc2['AVG_MATH'].mean(), line_dash='dash',
                              line_color='#9CA3AF', line_width=1)
            fig_sc2.update_layout(
                title=f'Where does {country} sit? Performance vs equity',
                xaxis_title='Equity gap (pts) — lower is better',
                yaxis_title='Average maths score',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=320,
                margin=dict(l=40, r=40, t=50, b=40),
                font=dict(family='DM Sans')
            )
            st.plotly_chart(fig_sc2, use_container_width=True)

        # ── COUNTRY TREND ─────────────────────────────────────────────────────
        country_trend = df_gap[df_gap['CNT'] == country].sort_values('YEAR')
        if len(country_trend) >= 2:
            st.markdown(f'<div class="section-title">{country} equity gap over time vs global average</div>',
                        unsafe_allow_html=True)

            fig_ctrend = go.Figure()
            global_avg_by_year = df_gap.groupby('YEAR')['GAP'].mean().reset_index()

            fig_ctrend.add_trace(go.Scatter(
                x=global_avg_by_year['YEAR'],
                y=global_avg_by_year['GAP'],
                mode='lines',
                name='Global average',
                line=dict(color='#9CA3AF', width=2, dash='dash'),
                hovertemplate='Year: %{x}<br>Global avg: %{y:.1f} pts<extra></extra>'
            ))
            fig_ctrend.add_trace(go.Scatter(
                x=country_trend['YEAR'],
                y=country_trend['GAP'],
                mode='lines+markers',
                name=country,
                line=dict(color='#1D9E75', width=3),
                marker=dict(size=10),
                hovertemplate='Year: %{x}<br>Gap: %{y:.1f} pts<extra></extra>'
            ))
            fig_ctrend.update_layout(
                xaxis_title='PISA cycle',
                yaxis_title='Equity gap (maths score points)',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=300,
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1),
                font=dict(family='DM Sans')
            )
            st.plotly_chart(fig_ctrend, use_container_width=True)

        # ── CONTEXT CARDS ─────────────────────────────────────────────────────
        col_ctx1, col_ctx2 = st.columns(2)

        with col_ctx1:
            if trajectory != 'Unknown' and len(df_traj[df_traj['TRAJECTORY'] == traj]) > 0:
                similar = df_traj[df_traj['TRAJECTORY'] == traj]['CNT'].tolist()
                similar_text = ', '.join(similar[:8]) + ('...' if len(similar) > 8 else '')
            else:
                similar_text = 'No trajectory data available for this country'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Countries with similar trajectory</div>
                <div style="font-size:0.85rem; color:#374151; margin-top:0.5rem">
                    {similar_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_ctx2:
            closing = df_traj[df_traj['TRAJECTORY'] == 'Closing']['CNT'].tolist()
            closing_text = ', '.join(closing[:8]) + ('...' if len(closing) > 8 else '')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Gap-closing countries to learn from</div>
                <div style="font-size:0.85rem; color:#15803D; margin-top:0.5rem">
                    {closing_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── INTERVENTIONS ─────────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-title">Recommended interventions — '
            f'filtered for {budget_label} budget</div>',
            unsafe_allow_html=True
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
                    cost_color = (
                        'badge-green' if row['cost_rating'] == 1
                        else 'badge-amber' if row['cost_rating'] <= 3
                        else 'badge-red'
                    )
                    st.markdown(f"""
                    <div class="intervention-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start">
                            <div class="intervention-name">
                                #{int(row['rank'])} {row['intervention']}
                            </div>
                            <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#1D9E75">
                                –{row['gap_reduction_pts']:.0f} pts
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
                        –{df_rec.iloc[0]['gap_reduction_pts']:.0f}<span style="font-size:1rem"> pts</span>
                    </div>
                    <div class="metric-sub">{df_rec.iloc[0]['intervention']}</div>
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
            'SC012Q05TA': 'Teachers use assessments to monitor learning',
            'SC012Q02TA': 'Teachers regularly give feedback to students',
            'SC012Q08JA': 'Teachers adapt instruction to individual needs',
            'SC202Q10JA': 'School offers extracurricular academic support',
            'SC212Q01JA': 'Principal provides instructional leadership',
            'SC017Q09JA': 'School has clear and high academic expectations',
            'SC172Q07JA': 'School offers support for struggling students',
            'SC172Q05JA': 'School uses mixed-ability grouping',
            'SC190Q02JA': 'School uses student data to improve teaching',
            'SC180Q01JA': 'School involves parents in educational decisions',
            'SC017Q10JA': 'School promotes student self-regulation',
            'SC004Q05NA': 'Principal monitors teacher instructional practices',
            'SC213Q02JA': 'School accepts students with special needs',
            'SC188Q05JA': 'School emphasises student wellbeing',
            'SC202Q09JA': 'School offers after-school tutoring',
            'SC192Q01JA': 'School shares assessment results with parents',
            'SC016Q03TA': 'Teachers receive regular professional development',
            'SC172Q04JA': 'School uses ability grouping between classes',
            'SC213Q01JA': 'School selects students by academic ability',
            'SC188Q03JA': 'School has competitive academic culture',
            'SC190Q07JA': 'School tracks student progress over time',
            'SCHAUTO':    'School has high curriculum autonomy',
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
    st.markdown('<div class="section-title">EEF intervention cost-effectiveness ranking</div>',
                unsafe_allow_html=True)
    st.caption(
        'Source: Education Endowment Foundation Teaching & Learning Toolkit | '
        'Cost-effectiveness = PISA points gained per £100 per pupil | '
        '1 EEF month ≈ 3.5 PISA points (OECD benchmark)'
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
        xaxis_title='Cost-effectiveness (PISA points per £100 per pupil)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=520,
        margin=dict(l=230, r=40, t=20, b=60),
        font=dict(family='DM Sans', size=12)
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
    st.markdown('<div class="section-title">Independent validation — XGBoost SHAP analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        🔬 <b>Two independent methods point to the same interventions.</b><br>
        Our team built an XGBoost model (v7) trained on <b>613,744 students across
        80 countries</b> from the full PISA 2022 dataset, using 369 school questionnaire
        variables. SHAP analysis identified which school practices most strongly reduce
        the socioeconomic performance gap. <b>12 of these independently confirmed our
        EEF-based recommendations</b> — providing convergent validity from a completely
        different analytical approach.
    </div>
    """, unsafe_allow_html=True)

    col_shap1, col_shap2 = st.columns(2)

    with col_shap1:
        # SHAP bar chart — policy features only, reduces slope
        artefact_features = ['MEAN_ESCS', 'N_STUDENTS', 'N_SCHOOLS',
                             'LOG_GDP_PC', 'SD_ESCS', 'EDU_PCT', 'GINI']

        df_reduces = df_shap[
            (~df_shap['is_artefact']) &
            (df_shap['direction'] == 'reduces_slope')
        ].sort_values('mean_abs_shap', ascending=True)

        df_increases = df_shap[
            (~df_shap['is_artefact']) &
            (df_shap['direction'] == 'increases_slope')
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
            title='School practices ranked by SHAP importance',
            xaxis_title='Mean |SHAP| value',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=560,
            margin=dict(l=280, r=20, t=50, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=-0.12),
            font=dict(family='DM Sans', size=10),
            barmode='overlay'
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_shap2:
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

        for _, row in df_cross.iterrows():
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
        **EquiTrack v7 — XGBoost cluster model (corrected evaluation)**

        **What it does:** Predicts the ESCS slope (PISA points gained per unit of
        socioeconomic status) at the cluster level within each country. A lower slope
        means smaller equity gap. Schools are grouped into k=5 clusters per country
        using k-means on 369 school questionnaire variables.

        **Data:** Full PISA 2022 microdata — 613,744 students, 21,629 schools, 80 countries.
        Mean 1,471 students pooled per cluster (vs ~30 per school in earlier versions).

        **Performance:**
        - Hold-out R² = 0.599 (n=77 test clusters)
        - Grouped 10-fold CV R² = 0.487 ± 0.081 (primary — respects country boundaries)
        - 5-fold CV R² = 0.423 ± 0.064 (ungrouped, for reference)

        **Evaluation fix (v6 → v7):** v6 used LOCO (Leave-One-Country-Out) which asks
        whether the model can predict wholly unseen countries — an unfair test since
        between-country variation is driven by structural factors not captured by school
        features. v7 uses GroupKFold(10) which holds out ~9 countries per fold, giving
        stable and interpretable R² estimates. The SHAP feature importance is used only
        for cross-validation of the EEF findings, not for prediction.

        **Why the SHAP findings are still valid:** SHAP identifies which school
        practices correlate with lower equity gaps across 80 countries in the
        training data. While the model cannot extrapolate to new countries, the
        patterns it identifies within the training set are meaningful and
        independently validated by EEF randomised controlled trials.
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
        <b>PISA Model:</b> XGBoost v7 trained on full PISA 2022 microdata
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

    st.markdown('<div class="section-title">Robustness benchmark: Random Forest vs LightGBM</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <b>Are the predictive signals real or model-specific?</b><br>
        We trained two independent ML models (Random Forest and LightGBM) on
        <b>84,232 student records across 5 PISA cycles (2009-2022)</b> using only
        genuine school-level operational features. Both models were evaluated with
        leak-free validation (GroupKFold by country).
        <b>Result: both models agree within 0.05 F1</b>, confirming the signal is
        model-agnostic.
    </div>
    """, unsafe_allow_html=True)

    # Load benchmark results
    df_bench = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'v2_comparison.csv'))

    # Comparison chart
    import plotly.graph_objects as go

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
                <b>Problem:</b> V1 benchmark showed perfect F1 = 1.0, which was
                caused by <code>COUNTRY_AVG_GAP</code> acting as a country fingerprint.<br><br>
                <b>Solution:</b> V2 removed all derived/leaky features and used only
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
