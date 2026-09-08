"""
NexusPredict — Executive Dashboard Page
Overview of key business metrics, revenue trends, and operational KPIs.
"""

import streamlit as st
from utils.mock_data import (
    get_dashboard_kpis, get_revenue_trend, get_user_acquisition,
    get_category_sales, get_recent_transactions, get_monthly_stats,
)
from utils.charts import create_area_chart, create_bar_chart, create_donut_chart


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


def render():
    theme = st.session_state.get("theme", "dark")

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Executive Dashboard</div>
        <div class="hero-subtitle">
            Real-time overview of your e-commerce performance — revenue, user engagement, 
            and operational metrics at a glance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    kpis = get_dashboard_kpis()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        _metric_card("Total Revenue", f"₹{kpis['total_revenue']:,.0f}",
                      kpis['revenue_delta'], "💰", "indigo")
    with col2:
        _metric_card("Active Users", f"{kpis['active_users']:,}",
                      kpis['users_delta'], "👥", "cyan")
    with col3:
        _metric_card("Avg Order Value", f"₹{kpis['avg_order_value']:,.2f}",
                      kpis['aov_delta'], "🛒", "emerald")
    with col4:
        _metric_card("Monthly Orders", f"{kpis['monthly_orders']:,}",
                      kpis['orders_delta'], "📦", "violet")
    with col5:
        _metric_card("Churn Rate", f"{kpis['churn_rate']}%",
                      kpis['churn_delta'], "⚠️", "rose")
    with col6:
        _metric_card("Conversion", f"{kpis['conversion_rate']}%",
                      kpis['conversion_delta'], "🎯", "amber")

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

    # ── User Acquisition & Transactions ──
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

    # ── Bottom Stats ──
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

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
            <div class="metric-label">👤 Avg Monthly New Users</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                {:,}
            </div>
        </div>
        """.format(int(stats["Users"].mean())), unsafe_allow_html=True)

    with col_s3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">📦 Avg Monthly Orders</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                {:,}
            </div>
        </div>
        """.format(int(stats["Orders"].mean())), unsafe_allow_html=True)
