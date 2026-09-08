"""
NexusPredict — Retention Risk Radar (Churn Prediction)
Customer churn risk analysis, filtering, and export.
Model 1: Binary Classification (XGBoost / Logistic Regression)
"""

import streamlit as st
import pandas as pd
from utils.mock_data import (
    get_churn_predictions, get_churn_feature_importance, get_churn_summary,
)
from utils.charts import (
    create_gauge_chart, create_horizontal_importance_chart,
    create_donut_chart, create_bar_chart, get_theme_colors,
)


def _risk_badge(label):
    """Generate HTML badge based on risk level."""
    color_map = {"High": "danger", "Medium": "warning", "Low": "success"}
    return f'<span class="badge badge-{color_map.get(label, "info")}">{label}</span>'


def render():
    theme = st.session_state.get("theme", "dark")
    colors = get_theme_colors(theme)

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🛡️ Retention Risk Radar</div>
        <div class="hero-subtitle">
            Identify high-risk customers before they disengage. Filter by risk score, 
            export target lists for win-back campaigns, and understand key churn drivers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary Metrics ──
    summary = get_churn_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card rose">
            <div class="metric-label">⚠️ High Risk Customers</div>
            <div class="metric-value">{summary['high_risk']:,}</div>
            <div class="metric-delta negative">
                {summary['high_risk'] / summary['total_customers'] * 100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card amber">
            <div class="metric-label">⏳ Medium Risk</div>
            <div class="metric-value">{summary['medium_risk']:,}</div>
            <div class="metric-delta" style="color: var(--accent-amber);">
                {summary['medium_risk'] / summary['total_customers'] * 100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card emerald">
            <div class="metric-label">✅ Low Risk</div>
            <div class="metric-value">{summary['low_risk']:,}</div>
            <div class="metric-delta positive">
                {summary['low_risk'] / summary['total_customers'] * 100:.1f}% of total
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card indigo">
            <div class="metric-label">🎯 Model ROC-AUC</div>
            <div class="metric-value">{summary['roc_auc']:.3f}</div>
            <div class="metric-delta positive">
                Recall: {summary['recall']:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Gauge & Feature Importance Row ──
    col_gauge, col_importance = st.columns([1, 1.5])

    with col_gauge:
        st.markdown('<div class="section-header">📊 Overall Risk Distribution</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Risk distribution donut
        risk_dist = pd.DataFrame({
            "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
            "Count": [summary['high_risk'], summary['medium_risk'], summary['low_risk']],
        })
        fig_donut = create_donut_chart(
            risk_dist, names="Risk Level", values="Count",
            title="Customer Risk Segments",
            height=340,
            theme=theme,
        )
        # Override colors for risk-specific palette
        fig_donut.update_traces(
            marker=dict(colors=["#f43f5e", "#f59e0b", "#10b981"],
                        line=dict(color=colors["donut_border"], width=2))
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_importance:
        st.markdown('<div class="section-header">🧠 Churn Feature Importance</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        feat_df = get_churn_feature_importance()
        fig_feat = create_horizontal_importance_chart(
            feat_df, feature_col="Feature", value_col="Importance",
            title="Top Predictive Features",
            height=340,
            theme=theme,
        )
        st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Filters ──
    st.markdown('<div class="section-header">🔍 Customer Risk Explorer</div>',
                unsafe_allow_html=True)

    churn_df = get_churn_predictions()

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])

    with filter_col1:
        risk_threshold = st.slider(
            "Minimum Risk Score (%)", min_value=0, max_value=100,
            value=50, step=5, key="churn_risk_slider",
        )
    with filter_col2:
        countries = ["All"] + sorted(churn_df["Country"].unique().tolist())
        selected_country = st.selectbox("Country Filter", countries, key="churn_country")
    with filter_col3:
        risk_levels = ["All", "High", "Medium", "Low"]
        selected_level = st.selectbox("Risk Level", risk_levels, key="churn_level")

    # Apply filters
    filtered = churn_df[churn_df["Risk Score (%)"] >= risk_threshold]
    if selected_country != "All":
        filtered = filtered[filtered["Country"] == selected_country]
    if selected_level != "All":
        filtered = filtered[filtered["Risk Label"] == selected_level]

    # ── Results Count & Export ──
    res_col1, res_col2 = st.columns([3, 1])
    with res_col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="color: var(--text-primary); font-weight: 600; font-size: 1rem;">
                {len(filtered)} customers found
            </span>
            <span style="color: var(--text-muted); font-size: 0.85rem;">
                Showing risk score ≥ {risk_threshold}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    with res_col2:
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name="churn_risk_customers.csv",
            mime="text/csv",
            key="churn_export",
        )

    # ── Customer Table ──
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    display_df = filtered[[
        "User ID", "Customer Name", "Country", "Age",
        "Days Since Last Purchase", "Login Frequency (monthly)",
        "Support Tickets", "Total Spend (₹)", "Risk Score (%)", "Risk Label"
    ]].head(50)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Risk Score (%)": st.column_config.ProgressColumn(
                "Risk Score (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Total Spend (₹)": st.column_config.NumberColumn(format="₹%.2f"),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Individual Customer Deep Dive ──
    st.markdown('<div class="section-header">🔎 Customer Deep Dive</div>',
                unsafe_allow_html=True)

    if len(filtered) > 0:
        selected_user = st.selectbox(
            "Select a customer to view details",
            filtered["User ID"].tolist(),
            key="churn_user_select",
        )

        user = filtered[filtered["User ID"] == selected_user].iloc[0]
        risk_color = "#f43f5e" if user["Risk Label"] == "High" else "#f59e0b" if user["Risk Label"] == "Medium" else "#10b981"
        risk_bar_class = "high" if user["Risk Label"] == "High" else "medium" if user["Risk Label"] == "Medium" else "low"

        st.markdown(f"""
        <div class="glass-card" style="margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1.5rem;">
                <div>
                    <h3 style="margin: 0 0 0.3rem 0; font-family: var(--font-display); color: var(--text-primary);">
                        {user['Customer Name']}
                    </h3>
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">
                        {user['User ID']} · {user['Country']} · Age {user['Age']}
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2rem; font-weight: 800; color: {risk_color}; font-family: var(--font-display);">
                        {user['Risk Score (%)']:.1f}%
                    </div>
                    <div>{_risk_badge(user['Risk Label'])}</div>
                </div>
            </div>
            <div style="margin-top: 1.2rem;">
                <div class="risk-bar-container">
                    <div class="risk-bar {risk_bar_class}" style="width: {user['Risk Score (%)']}%;"></div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
                <div>
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Days Since Purchase</div>
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">{user['Days Since Last Purchase']}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Login Freq/Month</div>
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">{user['Login Frequency (monthly)']}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Support Tickets</div>
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">{user['Support Tickets']}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Total Spend</div>
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">₹{user['Total Spend (₹)']:,.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No customers match the current filters.")
