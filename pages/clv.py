"""
NexusPredict — Financial Projections (CLV Predictions)
Customer Lifetime Value analysis with RFM breakdown.
Model 2: Regression (Random Forest / Gradient Boosting)
"""

import streamlit as st
import pandas as pd
from utils.mock_data import (
    get_clv_predictions, get_clv_metrics, get_clv_tier_summary,
)
from utils.charts import (
    create_scatter_chart, create_bar_chart, create_donut_chart,
    get_theme_colors,
)
import plotly.graph_objects as go
import plotly.express as px


TIER_COLORS = {
    "Platinum": "#8b5cf6",
    "Gold": "#f59e0b",
    "Silver": "#94a3b8",
    "Bronze": "#d97706",
}

TIER_ICONS = {
    "Platinum": "💎",
    "Gold": "🥇",
    "Silver": "🥈",
    "Bronze": "🥉",
}


def render():
    theme = st.session_state.get("theme", "dark")
    colors = get_theme_colors(theme)

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">💰 Financial Projections</div>
        <div class="hero-subtitle">
            Estimate 6-month customer lifetime value using RFM analysis. Target predicted
            VIP spenders with high-tier support options and optimize acquisition spend.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Metrics ──
    metrics = get_clv_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card indigo">
            <div class="metric-label">📐 Mean Absolute Error</div>
            <div class="metric-value">₹{metrics['mae']:,.2f}</div>
            <div class="metric-delta" style="color: var(--text-muted);">Avg prediction deviation</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card cyan">
            <div class="metric-label">📏 Root Mean Sq Error</div>
            <div class="metric-value">₹{metrics['rmse']:,.2f}</div>
            <div class="metric-delta" style="color: var(--text-muted);">Penalizes large errors</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card emerald">
            <div class="metric-label">🎯 R² Score</div>
            <div class="metric-value">{metrics['r2_score']:.3f}</div>
            <div class="metric-delta positive">Variance explained</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card amber">
            <div class="metric-label">📊 MAPE</div>
            <div class="metric-value">{metrics['mape']:.1f}%</div>
            <div class="metric-delta" style="color: var(--text-muted);">Mean % error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Tier Summary & CLV Distribution ──
    clv_df = get_clv_predictions()
    tier_summary = get_clv_tier_summary()

    col_tiers, col_dist = st.columns([1, 1.5])

    with col_tiers:
        st.markdown('<div class="section-header">🏆 CLV Tier Breakdown</div>',
                    unsafe_allow_html=True)

        for _, row in tier_summary.iterrows():
            tier = row["Tier"]
            color = TIER_COLORS.get(tier, "#6366f1")
            icon = TIER_ICONS.get(tier, "•")
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 0.75rem; padding: 1rem 1.3rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <span style="font-family: var(--font-display); font-weight: 700; color: {color}; font-size: 1.05rem; margin-left: 0.4rem;">
                            {tier}
                        </span>
                        <span style="color: var(--text-muted); font-size: 0.82rem; margin-left: 0.6rem;">
                            {row['Count']:,} customers
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-family: var(--font-display); font-weight: 700; color: var(--text-primary); font-size: 1.1rem;">
                            ₹{row['Avg CLV (₹)']:,.0f}
                        </div>
                        <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">avg clv</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_dist:
        st.markdown('<div class="section-header">📊 CLV Distribution</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # CLV Histogram
        fig_hist = px.histogram(
            clv_df, x="Predicted CLV (₹)", nbins=30,
            color="Tier", color_discrete_map=TIER_COLORS,
            height=340,
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Predicted CLV Distribution by Tier", font=dict(size=15, color=colors["text"])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                       font=dict(color=colors["text_secondary"], size=11)),
            bargap=0.05,
        )
        fig_hist.update_xaxes(
            showgrid=False, gridcolor=colors["grid"],
            tickfont=dict(color=colors["text_muted"], size=11),
        )
        fig_hist.update_yaxes(
            showgrid=True, gridcolor=colors["grid"],
            tickfont=dict(color=colors["text_muted"], size=11),
            zeroline=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── RFM Scatter Plot ──
    st.markdown('<div class="section-header">🔬 RFM Analysis — Recency vs Monetary</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_scatter = create_scatter_chart(
        clv_df,
        x="Recency (Days)", y="Avg Monetary (₹)",
        color_col="Tier", size_col="Frequency",
        title="Customer Segments: Recency × Monetary (sized by Frequency)",
        height=440,
        color_map=TIER_COLORS,
        theme=theme,
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Top Customers & Individual Lookup ──
    col_top, col_lookup = st.columns([1, 1])

    with col_top:
        st.markdown('<div class="section-header">🌟 Top 10 Highest-Value Customers</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        top10 = clv_df.head(10)
        st.dataframe(
            top10[["User ID", "Customer Name", "Country", "Frequency",
                   "Avg Monetary (₹)", "Predicted CLV (₹)", "Tier"]],
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "Predicted CLV (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "Avg Monetary (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            },
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_lookup:
        st.markdown('<div class="section-header">🔎 Individual Customer Lookup</div>',
                    unsafe_allow_html=True)

        selected_user = st.selectbox(
            "Select a customer",
            clv_df["User ID"].tolist(),
            key="clv_user_select",
        )

        user = clv_df[clv_df["User ID"] == selected_user].iloc[0]
        tier_color = TIER_COLORS.get(user["Tier"], "#6366f1")
        tier_icon = TIER_ICONS.get(user["Tier"], "•")

        st.markdown(f"""
        <div class="glass-card" style="margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <h3 style="margin: 0 0 0.3rem 0; font-family: var(--font-display); color: var(--text-primary);">
                        {user['Customer Name']}
                    </h3>
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">
                        {user['User ID']} · {user['Country']}
                    </p>
                </div>
                <div style="text-align: right;">
                    <span class="badge" style="background: {tier_color}22; color: {tier_color}; border: 1px solid {tier_color}44; font-size: 0.85rem; padding: 0.3rem 0.9rem;">
                        {tier_icon} {user['Tier']}
                    </span>
                </div>
            </div>
            <div style="margin-top: 1.5rem; text-align: center;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Predicted 6-Month CLV</div>
                <div style="font-family: var(--font-display); font-size: 2.5rem; font-weight: 800; color: {tier_color}; margin: 0.3rem 0;">
                    ₹{user['Predicted CLV (₹)']:,.2f}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.5rem; text-align: center;">
                <div style="background: rgba(99, 102, 241, 0.06); border-radius: var(--radius-md); padding: 1rem;">
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;">Recency</div>
                    <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 700; font-family: var(--font-display);">
                        {user['Recency (Days)']}d
                    </div>
                </div>
                <div style="background: rgba(34, 211, 238, 0.06); border-radius: var(--radius-md); padding: 1rem;">
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;">Frequency</div>
                    <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 700; font-family: var(--font-display);">
                        {user['Frequency']}x
                    </div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.06); border-radius: var(--radius-md); padding: 1rem;">
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;">Monetary</div>
                    <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 700; font-family: var(--font-display);">
                        ₹{user['Avg Monetary (₹)']:,.0f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── CLV by Tier Bar Chart ──
    st.markdown('<div class="section-header">📊 Total Revenue Contribution by Tier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    fig_bar = go.Figure(go.Bar(
        x=tier_summary["Tier"],
        y=tier_summary["Total Revenue (₹)"],
        marker=dict(
            color=[TIER_COLORS[t] for t in tier_summary["Tier"]],
            cornerradius=8,
        ),
        text=tier_summary["Total Revenue (₹)"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="outside",
        textfont=dict(color=colors["text_secondary"], size=11),
        hovertemplate="<b>%{x}</b><br>Total Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        title=dict(text="Revenue Contribution per CLV Tier", font=dict(size=15, color=colors["text"])),
    )
    fig_bar.update_xaxes(
        showgrid=False, tickfont=dict(color=colors["text_muted"], size=12),
    )
    fig_bar.update_yaxes(
        showgrid=True, gridcolor=colors["grid"],
        tickfont=dict(color=colors["text_muted"], size=11), zeroline=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
