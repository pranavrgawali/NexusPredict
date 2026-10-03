"""
NexusPredict — Financial Projections (CLV Predictions)
Customer Lifetime Value analysis with RFM breakdown.
Model 2: Regression (Random Forest / Gradient Boosting)

Integrates with models/clv_model.joblib when available.
Falls back to mock data if the model is not found.

Layout: Simple insights first → Technical details in expander.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.mock_data import (
    get_clv_predictions, get_clv_metrics, get_clv_tier_summary,
    get_raw_customer_data,
)
from utils.charts import (
    create_scatter_chart, create_bar_chart, create_donut_chart,
    create_horizontal_importance_chart, get_theme_colors,
)
from utils.model_loader import (
    is_clv_model_available, predict_clv,
    get_clv_model_metrics, get_clv_feature_importance_from_model,
)
import plotly.graph_objects as go
import plotly.express as px

# ── Determine if real model is available ──
LIVE_MODEL = is_clv_model_available()


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


def _model_status_badge(is_live):
    """Show whether real ML model or demo mock data is being used."""
    if is_live:
        model_metrics = get_clv_model_metrics()
        model_name = model_metrics["model_name"] if model_metrics else "Regressor"
        # Shorten name for badge
        short_name = model_name.replace("Regressor", "").replace("Gradient", "GB").replace("RandomForest", "RF").strip()
        return f"""
        <div style="display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(16, 185, 129, 0.1);
                    border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 0.3rem 0.9rem;
                    font-size: 0.78rem; color: #10b981; font-weight: 600; margin-left: 1rem;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;
                         box-shadow: 0 0 6px #10b981;"></span>
            Live Model · {short_name}
        </div>
        """
    else:
        return """
        <div style="display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(245, 158, 11, 0.1);
                    border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 20px; padding: 0.3rem 0.9rem;
                    font-size: 0.78rem; color: #f59e0b; font-weight: 600; margin-left: 1rem;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; display: inline-block;
                         box-shadow: 0 0 6px #f59e0b;"></span>
            Demo Mode
        </div>
        """


def _get_clv_data():
    """Get CLV predictions — uploaded data > live model > mock data."""
    # 1. Try uploaded data + live model
    if LIVE_MODEL and st.session_state.get("uploaded_data") is not None:
        uploaded_df = st.session_state.uploaded_data.copy()
        result = predict_clv(uploaded_df)
        if result is not None:
            col_map = {
                "recency_days": "Recency (Days)",
                "transaction_count": "Frequency",
                "average_order_value": "Avg Monetary (₹)",
            }
            for raw, display in col_map.items():
                if raw in result.columns and display not in result.columns:
                    result[display] = result[raw]
            if "User ID" not in result.columns:
                result["User ID"] = [f"USR-{i+1}" for i in range(len(result))]
            if "Customer Name" not in result.columns:
                result["Customer Name"] = [f"Customer {i+1}" for i in range(len(result))]
            if "Country" not in result.columns:
                result["Country"] = "N/A"
            # Ensure Frequency column exists for scatter sizing
            if "Frequency" not in result.columns:
                result["Frequency"] = 1
            if "Avg Monetary (₹)" not in result.columns:
                result["Avg Monetary (₹)"] = 0
            if "Recency (Days)" not in result.columns:
                result["Recency (Days)"] = 0
            return result

    # 2. Try live model with mock raw data
    if LIVE_MODEL:
        raw_df = get_raw_customer_data(200)
        result = predict_clv(raw_df)
        if result is not None:
            # Map raw feature cols to display-friendly names
            result["Recency (Days)"] = result["recency_days"]
            result["Frequency"] = result["transaction_count"]
            result["Avg Monetary (₹)"] = result["average_order_value"]
            return result

    # 3. Fallback to mock data
    return get_clv_predictions()


def _get_metrics():
    """Get CLV model metrics — from real model or mock."""
    if LIVE_MODEL:
        metrics = get_clv_model_metrics()
        if metrics:
            return {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2_score": 0.0,   # Not stored in artifact — will show N/A style
                "mape": 0.0,       # Not stored in artifact
                "model_name": metrics["model_name"],
            }
    mock = get_clv_metrics()
    mock["model_name"] = "Mock"
    return mock


def _get_tier_summary(clv_df):
    """Build tier summary from actual predictions or fallback to mock."""
    if LIVE_MODEL and "Tier" in clv_df.columns and "Predicted CLV (₹)" in clv_df.columns:
        tiers = []
        for tier_name in ["Platinum", "Gold", "Silver", "Bronze"]:
            tier_data = clv_df[clv_df["Tier"] == tier_name]
            count = len(tier_data)
            avg_clv = tier_data["Predicted CLV (₹)"].mean() if count > 0 else 0
            total_rev = tier_data["Predicted CLV (₹)"].sum() if count > 0 else 0
            tiers.append({
                "Tier": tier_name,
                "Count": count,
                "Avg CLV (₹)": round(avg_clv, 2),
                "Total Revenue (₹)": round(total_rev, 2),
                "Color": TIER_COLORS.get(tier_name, "#6366f1"),
            })
        return pd.DataFrame(tiers)
    return get_clv_tier_summary()


def _generate_insights(clv_df, tier_summary):
    """Generate plain-English actionable insights from CLV data."""
    insights = []

    total = len(clv_df)

    # VIP customers insight
    if "Tier" in clv_df.columns:
        platinum = clv_df[clv_df["Tier"] == "Platinum"]
        if len(platinum) > 0:
            top_value = platinum["Predicted CLV (₹)"].sum()
            insights.append({
                "icon": "💎",
                "text": f"Your <strong>{len(platinum)} VIP customers</strong> are worth a combined "
                        f"<strong>₹{top_value:,.0f}</strong> over the next 6 months. "
                        f"Give them priority support and exclusive offers to keep them happy!"
            })

    # Top 10 insight
    if len(clv_df) >= 10:
        top10_value = clv_df.head(10)["Predicted CLV (₹)"].sum()
        insights.append({
            "icon": "🌟",
            "text": f"Your <strong>top 10 customers alone</strong> are projected to bring in "
                    f"<strong>₹{top10_value:,.0f}</strong>. These are your most important relationships."
        })

    # Average CLV insight
    avg_clv = clv_df["Predicted CLV (₹)"].mean()
    insights.append({
        "icon": "📊",
        "text": f"On average, each customer is worth <strong>₹{avg_clv:,.0f}</strong> over 6 months. "
                f"Increasing purchase frequency or order size can boost this number."
    })

    # Bronze upgrade opportunity
    if "Tier" in clv_df.columns:
        bronze = clv_df[clv_df["Tier"] == "Bronze"]
        silver = clv_df[clv_df["Tier"] == "Silver"]
        if len(bronze) > 0:
            insights.append({
                "icon": "📈",
                "text": f"<strong>{len(bronze)} Bronze-tier customers</strong> could be upgraded. "
                        f"Targeted promotions and engagement campaigns can move them to Silver tier, "
                        f"significantly increasing their lifetime value."
            })

    # Total projected revenue
    total_projected = clv_df["Predicted CLV (₹)"].sum()
    insights.append({
        "icon": "💰",
        "text": f"Total projected revenue from all {total} customers: "
                f"<strong>₹{total_projected:,.0f}</strong> over the next 6 months."
    })

    return insights


def render():
    theme = st.session_state.get("theme", "dark")
    colors = get_theme_colors(theme)

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">💰 Customer Value</div>
        <div class="hero-subtitle">
            Know how much each customer is worth to your business over the next 6 months,
            so you can invest your time and money where it matters most.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──
    clv_df = _get_clv_data()
    metrics = _get_metrics()
    tier_summary = _get_tier_summary(clv_df)

    # ═══════════════════════════════════════════════════════════
    # SIMPLE VIEW — What a shop owner sees first
    # ═══════════════════════════════════════════════════════════

    # ── Simple Summary Cards ──
    # Calculate key numbers
    total_customers = len(clv_df)
    avg_clv = clv_df["Predicted CLV (₹)"].mean()
    top_clv = clv_df["Predicted CLV (₹)"].max()
    platinum_count = len(clv_df[clv_df["Tier"] == "Platinum"]) if "Tier" in clv_df.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="simple-stat-card purple">
            <span class="simple-stat-icon">💎</span>
            <div class="simple-stat-number">{platinum_count:,}</div>
            <div class="simple-stat-label">VIP Customers</div>
            <div class="simple-stat-sublabel">Your highest-value buyers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="simple-stat-card success">
            <span class="simple-stat-icon">🏆</span>
            <div class="simple-stat-number">₹{top_clv:,.0f}</div>
            <div class="simple-stat-label">Most Valuable Customer</div>
            <div class="simple-stat-sublabel">Highest projected worth</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="simple-stat-card info">
            <span class="simple-stat-icon">📊</span>
            <div class="simple-stat-number">₹{avg_clv:,.0f}</div>
            <div class="simple-stat-label">Average Customer Value</div>
            <div class="simple-stat-sublabel">Per customer, next 6 months</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="simple-stat-card warning">
            <span class="simple-stat-icon">👥</span>
            <div class="simple-stat-number">{total_customers:,}</div>
            <div class="simple-stat-label">Customers Analyzed</div>
            <div class="simple-stat-sublabel">With value predictions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Actionable Insights Box ──
    insights = _generate_insights(clv_df, tier_summary)
    if insights:
        items_html = "".join([
            f'<div class="insight-item"><span class="insight-bullet">{ins["icon"]}</span><span>{ins["text"]}</span></div>'
            for ins in insights
        ])
        st.html(f'<div class="insights-box"><div class="insights-box-title"><span class="insights-icon">💡</span>What You Should Know</div>{items_html}</div>')

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Tier Summary & CLV Distribution ──
    col_tiers, col_dist = st.columns([1, 1.5])

    with col_tiers:
        st.markdown('<div class="section-header">🏆 Customer Tiers</div>',
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
                        <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">avg value</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_dist:
        st.markdown('<div class="section-header">📊 Value Distribution</div>',
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
            title=dict(text="How Customer Values Are Spread", font=dict(size=15, color=colors["text"])),
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

    # ── Top Customers & Individual Lookup ──
    col_top, col_lookup = st.columns([1, 1])

    with col_top:
        st.markdown('<div class="section-header">🌟 Your Top 10 Most Valuable Customers</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        top10 = clv_df.head(10)
        display_cols = ["User ID", "Customer Name", "Country", "Frequency",
                        "Avg Monetary (₹)", "Predicted CLV (₹)", "Tier"]
        display_cols = [c for c in display_cols if c in top10.columns]
        st.dataframe(
            top10[display_cols],
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
        st.markdown('<div class="section-header">🔎 Look Up a Customer</div>',
                    unsafe_allow_html=True)

        selected_user = st.selectbox(
            "Select a customer",
            clv_df["User ID"].tolist(),
            key="clv_user_select",
        )

        user = clv_df[clv_df["User ID"] == selected_user].iloc[0]
        tier_color = TIER_COLORS.get(user["Tier"], "#6366f1")
        tier_icon = TIER_ICONS.get(user["Tier"], "•")

        # Safe column access
        user_recency = user.get("Recency (Days)", user.get("recency_days", "N/A"))
        user_frequency = user.get("Frequency", user.get("transaction_count", "N/A"))
        user_monetary = user.get("Avg Monetary (₹)", user.get("average_order_value", 0))

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
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Projected 6-Month Value</div>
                <div style="font-family: var(--font-display); font-size: 2.5rem; font-weight: 800; color: {tier_color}; margin: 0.3rem 0;">
                    ₹{user['Predicted CLV (₹)']:,.2f}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.5rem; text-align: center;">
                <div style="background: rgba(99, 102, 241, 0.06); border-radius: var(--radius-md); padding: 1rem;">
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;">Last Visit</div>
                    <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 700; font-family: var(--font-display);">
                        {user_recency}d ago
                    </div>
                </div>
                <div style="background: rgba(34, 211, 238, 0.06); border-radius: var(--radius-md); padding: 1rem;">
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;">Purchases</div>
                    <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 700; font-family: var(--font-display);">
                        {user_frequency}x
                    </div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.06); border-radius: var(--radius-md); padding: 1rem;">
                    <div style="color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;">Avg Spend</div>
                    <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 700; font-family: var(--font-display);">
                        ₹{user_monetary:,.0f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # TECHNICAL DETAILS — Collapsed by default
    # ═══════════════════════════════════════════════════════════

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    with st.expander("🔧 Technical Details — For Data Teams & Developers"):
        st.markdown("""
        <div class="tech-details-header">
            <span class="tech-icon">📊</span>
            Model Performance & Analysis
            <span class="tech-badge">ML Reference</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # Model status badge
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span style="color: var(--text-muted); font-size: 0.85rem; font-weight: 500;">Data Source:</span>
            {_model_status_badge(LIVE_MODEL)}
        </div>
        """, unsafe_allow_html=True)

        # ── Model Metrics ──
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
            if metrics.get("r2_score", 0) > 0:
                r2_display = f"{metrics['r2_score']:.3f}"
                r2_delta = "Variance explained"
            else:
                r2_display = "—"
                r2_delta = "Not stored in artifact"
            st.markdown(f"""
            <div class="metric-card emerald">
                <div class="metric-label">🎯 R² Score</div>
                <div class="metric-value">{r2_display}</div>
                <div class="metric-delta positive">{r2_delta}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            if metrics.get("mape", 0) > 0:
                mape_display = f"{metrics['mape']:.1f}%"
                mape_delta = "Mean % error"
            else:
                mape_display = "—"
                mape_delta = "Not stored in artifact"
            st.markdown(f"""
            <div class="metric-card amber">
                <div class="metric-label">📊 MAPE</div>
                <div class="metric-value">{mape_display}</div>
                <div class="metric-delta" style="color: var(--text-muted);">{mape_delta}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # ── Feature Importance (only if live model) ──
        if LIVE_MODEL:
            fi_df = get_clv_feature_importance_from_model()
            if fi_df is not None:
                st.markdown('<div class="section-header">🧠 CLV Feature Importance</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig_fi = create_horizontal_importance_chart(
                    fi_df, feature_col="Feature", value_col="Importance",
                    title="Top Predictive Features (from trained model)",
                    height=340,
                    theme=theme,
                )
                st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
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
