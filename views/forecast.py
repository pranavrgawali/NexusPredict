"""
NexusPredict — Supply Chain Forecasts (Demand Forecasting)
30-day demand predictions with confidence intervals.
Model 3: Time-Series Forecasting (Prophet)
"""

import streamlit as st
import pandas as pd
from utils.mock_data import (
    get_forecast_data, get_inventory_status, get_forecast_metrics,
    get_product_categories,
)
from utils.charts import (
    create_forecast_chart, create_grouped_bar_chart,
)
import plotly.graph_objects as go


def render():
    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📦 Supply Chain Forecasts</div>
        <div class="hero-subtitle">
            Project item sales velocity for the upcoming 30 days. Avoid under-stocking or
            wasting working capital — make data-driven purchase order decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Category Selector ──
    categories = get_product_categories()
    inventory_df = get_inventory_status()
    metrics_df = get_forecast_metrics()

    col_select, col_spacer = st.columns([1, 2])
    with col_select:
        selected_category = st.selectbox(
            "🏷️ Select Product Category",
            categories,
            key="forecast_category",
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Category Metrics ──
    cat_inv = inventory_df[inventory_df["Category"] == selected_category].iloc[0]
    cat_metric = metrics_df[metrics_df["Category"] == selected_category].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        stockout_color = "rose" if cat_inv["Days Until Stockout"] < 15 else "amber" if cat_inv["Days Until Stockout"] < 25 else "emerald"
        st.markdown(f"""
        <div class="metric-card {stockout_color}">
            <div class="metric-label">⏰ Days Until Stockout</div>
            <div class="metric-value">{cat_inv['Days Until Stockout']:.0f}</div>
            <div class="metric-delta {'negative' if cat_inv['Days Until Stockout'] < 15 else 'positive'}">
                {"⚠️ Critical" if cat_inv['Days Until Stockout'] < 15 else "✅ Healthy"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card cyan">
            <div class="metric-label">📦 Current Stock</div>
            <div class="metric-value">{cat_inv['Current Stock']:,}</div>
            <div class="metric-delta" style="color: var(--text-muted);">units available</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card violet">
            <div class="metric-label">📈 30-Day Demand</div>
            <div class="metric-value">{cat_inv['Projected 30-Day Demand']:,}</div>
            <div class="metric-delta" style="color: var(--text-muted);">projected units</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        mape_color = "emerald" if cat_metric["MAPE (%)"] < 10 else "amber" if cat_metric["MAPE (%)"] < 15 else "rose"
        st.markdown(f"""
        <div class="metric-card {mape_color}">
            <div class="metric-label">🎯 Forecast MAPE</div>
            <div class="metric-value">{cat_metric['MAPE (%)']}%</div>
            <div class="metric-delta" style="color: var(--text-muted);">
                MAE: {cat_metric['MAE']:.1f} · Coverage: {cat_metric['Coverage (%)']}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Forecast Chart ──
    st.markdown(f'<div class="section-header">📊 30-Day Demand Forecast — {selected_category}</div>',
                unsafe_allow_html=True)

    hist_df, forecast_df = get_forecast_data(selected_category)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_forecast = create_forecast_chart(
        hist_df, forecast_df,
        title=f"{selected_category} — Historical vs Forecast",
        height=440,
    )
    st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": False})

    # Legend
    st.markdown("""
    <div class="forecast-legend" style="margin-top: 0.5rem;">
        <div><span class="forecast-legend-dot" style="background: #6366f1;"></span> Historical Actuals</div>
        <div><span class="forecast-legend-dot" style="background: #22d3ee;"></span> Forecast Prediction</div>
        <div><span class="forecast-legend-dot" style="background: rgba(34,211,238,0.2);"></span> 95% Confidence Band</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Stock vs Demand Overview (All Categories) ──
    col_stock, col_alerts = st.columns([1.5, 1])

    with col_stock:
        st.markdown('<div class="section-header">📦 Stock vs Demand — All Categories</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=inventory_df["Category"],
            y=inventory_df["Current Stock"],
            name="Current Stock",
            marker=dict(color="#6366f1", cornerradius=6),
            hovertemplate="<b>%{x}</b><br>Stock: %{y:,} units<extra></extra>",
        ))
        fig_compare.add_trace(go.Bar(
            x=inventory_df["Category"],
            y=inventory_df["Projected 30-Day Demand"],
            name="30-Day Demand",
            marker=dict(color="#22d3ee", cornerradius=6),
            hovertemplate="<b>%{x}</b><br>Demand: %{y:,} units<extra></extra>",
        ))
        fig_compare.add_trace(go.Bar(
            x=inventory_df["Category"],
            y=inventory_df["Safety Stock Level"],
            name="Safety Stock",
            marker=dict(color="rgba(244, 63, 94, 0.5)", cornerradius=6),
            hovertemplate="<b>%{x}</b><br>Safety: %{y:,} units<extra></extra>",
        ))

        fig_compare.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
            margin=dict(l=20, r=20, t=40, b=60),
            height=400,
            barmode="group",
            title=dict(text="Inventory Health Overview", font=dict(size=15, color="#e2e8f0")),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#94a3b8", size=11),
            ),
            hoverlabel=dict(bgcolor="#1e1b4b", font_size=13, font_family="Inter",
                           font_color="#e2e8f0", bordercolor="rgba(99,102,241,0.3)"),
        )
        fig_compare.update_xaxes(
            showgrid=False, tickfont=dict(color="#64748b", size=10),
            tickangle=-35,
        )
        fig_compare.update_yaxes(
            showgrid=True, gridcolor="rgba(148,163,184,0.08)",
            tickfont=dict(color="#64748b", size=11), zeroline=False,
        )
        st.plotly_chart(fig_compare, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_alerts:
        st.markdown('<div class="section-header">🚨 Restock Alerts</div>',
                    unsafe_allow_html=True)

        restock_items = inventory_df[inventory_df["Restock Needed"] == True]
        safe_items = inventory_df[inventory_df["Restock Needed"] == False]

        if len(restock_items) > 0:
            for _, row in restock_items.iterrows():
                deficit = row["Projected 30-Day Demand"] - row["Current Stock"]
                st.markdown(f"""
                <div class="alert-card danger">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; color: var(--accent-rose); font-size: 0.95rem;">
                                ⚠️ {row['Category']}
                            </div>
                            <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 0.25rem;">
                                Stock: {row['Current Stock']:,} · Demand: {row['Projected 30-Day Demand']:,}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 700; color: var(--accent-rose); font-family: var(--font-display); font-size: 1.1rem;">
                                +{deficit:,}
                            </div>
                            <div style="color: var(--text-muted); font-size: 0.7rem;">units needed</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-card success">
                <div style="font-weight: 600; color: var(--accent-emerald);">
                    ✅ All categories well-stocked
                </div>
                <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.25rem;">
                    No restock alerts at this time.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if len(safe_items) > 0:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            for _, row in safe_items.iterrows():
                st.markdown(f"""
                <div class="alert-card success">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; color: var(--accent-emerald); font-size: 0.9rem;">
                                ✅ {row['Category']}
                            </div>
                            <div style="color: var(--text-secondary); font-size: 0.78rem; margin-top: 0.2rem;">
                                {row['Days Until Stockout']:.0f} days of stock remaining
                            </div>
                        </div>
                        <div>
                            <span class="badge badge-success">{row['Current Stock']:,} units</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Forecast Accuracy Table (All Categories) ──
    st.markdown('<div class="section-header">📋 Model Accuracy — All Categories</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "MAPE (%)": st.column_config.ProgressColumn(
                "MAPE (%)",
                min_value=0,
                max_value=25,
                format="%.1f%%",
            ),
            "Coverage (%)": st.column_config.ProgressColumn(
                "Coverage (%)",
                min_value=80,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)
