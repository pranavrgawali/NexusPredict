"""
NexusPredict — Executive Dashboard Page
Overview of key business metrics, revenue trends, and operational KPIs.
Includes CSV/Excel upload that persists data across all pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.mock_data import (
    get_dashboard_kpis, get_revenue_trend, get_user_acquisition,
    get_category_sales, get_recent_transactions, get_monthly_stats,
)
from utils.charts import (
    create_area_chart, create_bar_chart, create_donut_chart,
    get_theme_colors,
)
import plotly.express as px


def _metric_card(label, value, delta, icon, color_class):
    """Render a premium metric card with HTML."""
    delta_class = "positive" if delta >= 0 else "negative"
    delta_icon = "↑" if delta >= 0 else "↓"
    delta_sign = "+" if delta >= 0 else ""
    st.markdown(f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta {delta_class}">
            {delta_icon} {delta_sign}{delta}% vs last month
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_upload_section(theme):
    """Render the data upload section with drag-and-drop styling."""
    colors = get_theme_colors(theme)

    has_data = st.session_state.get("uploaded_data") is not None

    if has_data:
        # ── Show uploaded data summary ──
        df = st.session_state.uploaded_data
        fname = st.session_state.uploaded_filename or "data"

        st.markdown(f"""
        <div class="glass-card" style="border: 1px solid rgba(16, 185, 129, 0.25); margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: #10b981;
                                     display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
                        <span style="color: #10b981; font-weight: 700; font-size: 1rem; font-family: var(--font-display);">
                            Dataset Active
                        </span>
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.85rem;">
                        📄 <strong style="color: var(--text-secondary);">{fname}</strong>
                        &nbsp;·&nbsp; {len(df):,} rows &nbsp;·&nbsp; {len(df.columns)} columns
                    </div>
                </div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <div style="background: rgba(99, 102, 241, 0.08); border-radius: 8px; padding: 0.4rem 0.8rem; text-align: center;">
                        <div style="color: var(--text-muted); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em;">Numeric</div>
                        <div style="color: #6366f1; font-weight: 700; font-size: 1rem;">{len(df.select_dtypes(include=[np.number]).columns)}</div>
                    </div>
                    <div style="background: rgba(34, 211, 238, 0.08); border-radius: 8px; padding: 0.4rem 0.8rem; text-align: center;">
                        <div style="color: var(--text-muted); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em;">Text</div>
                        <div style="color: #22d3ee; font-weight: 700; font-size: 1rem;">{len(df.select_dtypes(include=['object', 'category']).columns)}</div>
                    </div>
                    <div style="background: rgba(245, 158, 11, 0.08); border-radius: 8px; padding: 0.4rem 0.8rem; text-align: center;">
                        <div style="color: var(--text-muted); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em;">Missing</div>
                        <div style="color: #f59e0b; font-weight: 700; font-size: 1rem;">{int(df.isnull().sum().sum()):,}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Column info ──
        col_info, col_preview = st.columns([1, 2])

        with col_info:
            st.markdown('<div class="section-header">🏷️ Column Schema</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            schema_data = []
            for col_name in df.columns:
                dtype = str(df[col_name].dtype)
                non_null = int(df[col_name].notna().sum())
                unique = int(df[col_name].nunique())
                schema_data.append({
                    "Column": col_name,
                    "Type": dtype,
                    "Non-Null": f"{non_null}/{len(df)}",
                    "Unique": unique,
                })
            schema_df = pd.DataFrame(schema_data)
            st.dataframe(schema_df, use_container_width=True, hide_index=True, height=340)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_preview:
            st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.dataframe(df.head(50), use_container_width=True, hide_index=True, height=340)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # ── Numeric column distribution charts ──
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown('<div class="section-header">📊 Numeric Distributions</div>', unsafe_allow_html=True)

            # Show up to 4 histograms
            chart_cols = st.columns(min(len(numeric_cols), 4))
            for i, col_name in enumerate(numeric_cols[:4]):
                with chart_cols[i]:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = px.histogram(
                        df, x=col_name, nbins=25, height=260,
                        color_discrete_sequence=["#6366f1"],
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
                        margin=dict(l=10, r=10, t=35, b=10),
                        title=dict(text=col_name, font=dict(size=13, color=colors["text"])),
                        bargap=0.08,
                    )
                    fig.update_xaxes(showgrid=False, tickfont=dict(color=colors["text_muted"], size=10))
                    fig.update_yaxes(showgrid=True, gridcolor=colors["grid"],
                                     tickfont=dict(color=colors["text_muted"], size=10), zeroline=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # ── Replace data option ──
        with st.expander("📤 Upload Different Dataset"):
            _render_uploader()

    else:
        # ── Upload prompt ──
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 2.5rem 2rem; border: 2px dashed rgba(99, 102, 241, 0.25);
                    background: linear-gradient(135deg, rgba(99, 102, 241, 0.04), rgba(139, 92, 246, 0.03));">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">📂</div>
            <div style="font-family: var(--font-display); font-size: 1.3rem; font-weight: 700;
                        color: var(--text-primary); margin-bottom: 0.5rem;">
                Upload Your Dataset
            </div>
            <div style="color: var(--text-muted); font-size: 0.88rem; max-width: 500px; margin: 0 auto 1.5rem auto;
                        line-height: 1.6;">
                Import a CSV or Excel file to analyze your own data across all modules —
                Churn Radar, CLV Projections, and Supply Chain will use your dataset for predictions.
            </div>
            <div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: center; gap: 0.4rem; color: var(--text-secondary); font-size: 0.8rem;">
                    <span style="color: #10b981;">✓</span> .csv
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem; color: var(--text-secondary); font-size: 0.8rem;">
                    <span style="color: #10b981;">✓</span> .xlsx
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem; color: var(--text-secondary); font-size: 0.8rem;">
                    <span style="color: #10b981;">✓</span> .xls
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _render_uploader()


def _render_uploader():
    """Render the file uploader widget."""
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        key="data_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.uploaded_data = df
            st.session_state.uploaded_filename = uploaded_file.name
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Failed to parse file: {e}")


def _generate_dashboard_insights(kpis):
    """Generate plain-English actionable insights from dashboard KPI data."""
    insights = []

    # Revenue growth insight
    rev_delta = kpis['revenue_delta']
    if rev_delta > 0:
        insights.append({
            "icon": "📈",
            "text": f"Your revenue is <b>up {rev_delta}%</b> compared to last month. "
                    f"Keep up the momentum with your current strategies!"
        })
    else:
        insights.append({
            "icon": "📉",
            "text": f"Revenue is <b>down {abs(rev_delta)}%</b> from last month. "
                    f"Consider running a promotion or reaching out to inactive customers."
        })

    # Churn rate insight
    churn = kpis['churn_rate']
    churn_delta = kpis['churn_delta']
    if churn > 15:
        insights.append({
            "icon": "⚠️",
            "text": f"Your churn rate is <b>{churn}%</b> — that means roughly 1 in 7 customers "
                    f"might stop buying. Check the Retention tab for details on who to reach out to."
        })
    elif churn_delta < 0:
        insights.append({
            "icon": "✅",
            "text": f"Great news! Customer churn dropped by <b>{abs(churn_delta)}%</b>. "
                    f"Your retention efforts are paying off."
        })

    # Order growth
    orders_delta = kpis['orders_delta']
    if orders_delta > 10:
        insights.append({
            "icon": "🛒",
            "text": f"Orders are <b>up {orders_delta}%</b> this month — "
                    f"make sure your inventory can handle the increased demand."
        })

    # Conversion rate
    conv = kpis['conversion_rate']
    conv_delta = kpis['conversion_delta']
    if conv < 3:
        insights.append({
            "icon": "🎯",
            "text": f"Your conversion rate is <b>{conv}%</b>. "
                    f"Improving your product pages or adding reviews could help turn more visitors into buyers."
        })
    elif conv_delta > 0:
        insights.append({
            "icon": "🎯",
            "text": f"Conversion rate improved by <b>{conv_delta}%</b> to <b>{conv}%</b>. "
                    f"Whatever you changed is working — keep it up!"
        })

    # AOV insight
    aov = kpis['avg_order_value']
    insights.append({
        "icon": "💡",
        "text": f"Average order value is <b>₹{aov:,.0f}</b>. "
                f"Offering bundle deals or 'frequently bought together' suggestions can increase this."
    })

    return insights


def render():
    theme = st.session_state.get("theme", "dark")

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📊 Your Shop Overview</div>
        <div class="hero-subtitle">
            See how your business is doing at a glance — revenue, orders, customers,
            and what needs your attention today.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data Upload Section ──
    st.markdown('<div class="section-header">📂 Data Source</div>', unsafe_allow_html=True)
    _render_upload_section(theme)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── KPI Cards (simplified labels) ──
    kpis = get_dashboard_kpis()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_class = "positive" if kpis['revenue_delta'] >= 0 else "negative"
        delta_icon = "↑" if kpis['revenue_delta'] >= 0 else "↓"
        st.markdown(f"""
        <div class="simple-stat-card info">
            <span class="simple-stat-icon">💰</span>
            <div class="simple-stat-number">₹{kpis['total_revenue']:,.0f}</div>
            <div class="simple-stat-label">Total Revenue</div>
            <div class="simple-stat-sublabel {delta_class}">{delta_icon} {abs(kpis['revenue_delta'])}% vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        delta_class = "positive" if kpis['users_delta'] >= 0 else "negative"
        delta_icon = "↑" if kpis['users_delta'] >= 0 else "↓"
        st.markdown(f"""
        <div class="simple-stat-card success">
            <span class="simple-stat-icon">👥</span>
            <div class="simple-stat-number">{kpis['active_users']:,}</div>
            <div class="simple-stat-label">Active Customers</div>
            <div class="simple-stat-sublabel {delta_class}">{delta_icon} {abs(kpis['users_delta'])}% vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        delta_class = "positive" if kpis['orders_delta'] >= 0 else "negative"
        delta_icon = "↑" if kpis['orders_delta'] >= 0 else "↓"
        st.markdown(f"""
        <div class="simple-stat-card warning">
            <span class="simple-stat-icon">📦</span>
            <div class="simple-stat-number">{kpis['monthly_orders']:,}</div>
            <div class="simple-stat-label">Orders This Month</div>
            <div class="simple-stat-sublabel {delta_class}">{delta_icon} {abs(kpis['orders_delta'])}% vs last month</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        churn_class = "negative" if kpis['churn_rate'] > 15 else "positive"
        st.markdown(f"""
        <div class="simple-stat-card danger">
            <span class="simple-stat-icon">⚠️</span>
            <div class="simple-stat-number">{kpis['churn_rate']}%</div>
            <div class="simple-stat-label">Customer Churn</div>
            <div class="simple-stat-sublabel">Customers who may leave</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Actionable Insights Box ──
    insights = _generate_dashboard_insights(kpis)
    if insights:
        items_html = "".join([
            f'<div class="insight-item"><span class="insight-bullet">{ins["icon"]}</span><span>{ins["text"]}</span></div>'
            for ins in insights
        ])
        st.html(f'<div class="insights-box"><div class="insights-box-title"><span class="insights-icon">💡</span>What You Should Know</div>{items_html}</div>')

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Revenue Trend ──
    st.markdown('<div class="section-header">📈 Revenue Trend (12 Months)</div>',
                unsafe_allow_html=True)

    revenue_df = get_revenue_trend()

    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_area_chart(
            revenue_df, x="Month", y="Revenue", y2="Target",
            title="Monthly Revenue vs Target",
            height=370,
            theme=theme,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        cat_df = get_category_sales()
        fig2 = create_donut_chart(
            cat_df, names="Category", values="Sales",
            title="Revenue by Category",
            height=370,
            theme=theme,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Quick Stats (simplified) ──
    stats = get_monthly_stats()
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">📊 Avg Monthly Revenue</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                ₹{:,.0f}
            </div>
        </div>
        """.format(stats["Revenue"].mean()), unsafe_allow_html=True)

    with col_s2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">🛒 Avg Order Value</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                ₹{:,.0f}
            </div>
        </div>
        """.format(kpis['avg_order_value']), unsafe_allow_html=True)

    with col_s3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">🎯 Conversion Rate</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                {}%
            </div>
        </div>
        """.format(kpis['conversion_rate']), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # TECHNICAL DETAILS — Collapsed by default
    # ═══════════════════════════════════════════════════════════

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    with st.expander("🔧 Technical Details — For Data Teams & Developers"):
        st.markdown("""
        <div class="tech-details-header">
            <span class="tech-icon">📊</span>
            Detailed Metrics & Acquisition Data
            <span class="tech-badge">Analytics</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # Detailed KPI cards
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            _metric_card("Avg Order Value", f"₹{kpis['avg_order_value']:,.2f}",
                          kpis['aov_delta'], "🛒", "emerald")
        with tc2:
            _metric_card("Monthly Orders", f"{kpis['monthly_orders']:,}",
                          kpis['orders_delta'], "📦", "violet")
        with tc3:
            _metric_card("Churn Rate", f"{kpis['churn_rate']}%",
                          kpis['churn_delta'], "⚠️", "rose")
        with tc4:
            _metric_card("Conversion", f"{kpis['conversion_rate']}%",
                          kpis['conversion_delta'], "🎯", "amber")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # User Acquisition & Transactions
        col_acq, col_txn = st.columns([1, 1.5])

        with col_acq:
            st.markdown('<div class="section-header">🚀 User Acquisition Channels</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            acq_df = get_user_acquisition()
            fig3 = create_bar_chart(
                acq_df, x="Channel", y="Users",
                title="Users by Acquisition Channel",
                orientation="h", text_auto=True,
                height=340,
                theme=theme,
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_txn:
            st.markdown('<div class="section-header">🧾 Recent Transactions</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            txn_df = get_recent_transactions(12)
            st.dataframe(
                txn_df,
                use_container_width=True,
                hide_index=True,
                height=340,
                column_config={
                    "Amount (₹)": st.column_config.NumberColumn(
                        format="₹%.2f",
                    ),
                    "Transaction ID": st.column_config.TextColumn(
                        width="small",
                    ),
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Monthly stats
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Monthly Trend Data</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.dataframe(
            stats,
            use_container_width=True,
            hide_index=True,
            height=240,
            column_config={
                "Revenue": st.column_config.NumberColumn(format="₹%.0f"),
            },
        )
        st.markdown('</div>', unsafe_allow_html=True)

