"""
NexusPredict — Supply Chain & Inventory Forecasting
30-day demand predictions using pre-trained Prophet models.
Model 3: Time-Series Forecasting (Facebook Prophet)

Integrates with timeseries/ folder containing serialized .joblib model artifacts.
Falls back to mock data if Prophet is not installed or models are not found.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

from utils.charts import (
    create_prophet_forecast_chart, create_forecast_chart,
    create_bar_chart, get_theme_colors, COLOR_SEQUENCE,
)
from utils.mock_data import (
    get_forecast_data, get_inventory_status, get_forecast_metrics,
)
import plotly.graph_objects as go

# ── Determine if Prophet is available ──
PROPHET_AVAILABLE = False
try:
    from timeseries.utils.model_loader import (
        load_prophet_model, get_available_categories, load_evaluation_summary,
    )
    from timeseries.utils.forecast_utils import (
        generate_forecast, calculate_stock_gap, format_forecast_for_chart,
    )
    PROPHET_AVAILABLE = True
except ImportError:
    pass


# ── Category display name mapping ──
CATEGORY_DISPLAY = {
    "Electronics": "Electronics",
    "Clothing": "Clothing",
    "Home_Kitchen": "Home & Kitchen",
    "Beauty": "Beauty",
    "Sports": "Sports & Outdoors",
    "Books": "Books",
    "Grocery": "Grocery",
    "Toys": "Toys & Games",
}

CATEGORY_ICONS = {
    "Electronics": "💻",
    "Clothing": "👕",
    "Home_Kitchen": "🏠",
    "Beauty": "💄",
    "Sports": "⚽",
    "Books": "📚",
    "Grocery": "🛒",
    "Toys": "🧸",
}

# ── Mock stock levels per category (placeholder until DB integration) ──
MOCK_STOCK_LEVELS = {
    "Electronics": 320,
    "Clothing": 450,
    "Home_Kitchen": 280,
    "Beauty": 380,
    "Sports": 190,
    "Books": 520,
    "Grocery": 600,
    "Toys": 250,
}


@st.cache_resource(show_spinner=False)
def _load_model_cached(category):
    """Cache-loaded Prophet model to avoid reloading on every interaction."""
    return load_prophet_model(category)


@st.cache_data(show_spinner=False, ttl=3600)
def _generate_forecast_cached(category, periods=30):
    """Cache forecast results for 1 hour."""
    bundle = _load_model_cached(category)
    model = bundle["model"]
    forecast_df = generate_forecast(model, periods=periods)
    return forecast_df, bundle.get("mape", "N/A")


def _get_evaluation_data():
    """Load evaluation summary with caching."""
    if PROPHET_AVAILABLE:
        return load_evaluation_summary()
    return {}


def _metric_card(label, value, subtitle, icon, color_class):
    """Render a premium metric card."""
    st.markdown(f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta" style="color: var(--text-muted);">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render():
    theme = st.session_state.get("theme", "dark")

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📦 Supply Chain Forecast</div>
        <div class="hero-subtitle">
            Project demand for the next 30 days using pre-trained Prophet models.
            Identify stockout risks, optimize purchase orders, and keep your
            supply chain running smoothly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data Source Indicator ──
    using_real_models = False
    available_categories = []

    if PROPHET_AVAILABLE:
        available_categories = get_available_categories()
        if len(available_categories) > 0:
            using_real_models = True

    if using_real_models:
        st.markdown("""
        <div class="info-banner">
            <span class="info-icon">🧠</span>
            <div class="info-text">
                <strong>Prophet ML Models Active</strong> — Forecasts are generated
                from pre-trained time-series models. Models are loaded from serialized
                artifacts and cached for performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-banner">
            <span class="info-icon">📊</span>
            <div class="info-text">
                <strong>Demo Mode</strong> — Showing simulated forecast data.
                Install <code>prophet</code> and place trained model artifacts in
                <code>timeseries/models/prophet/</code> to enable real ML predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Category Selector ──
    if using_real_models:
        display_to_key = {CATEGORY_DISPLAY.get(c, c): c for c in available_categories}
        display_names = list(display_to_key.keys())
    else:
        mock_categories = list(CATEGORY_DISPLAY.values())
        display_to_key = {v: k for k, v in CATEGORY_DISPLAY.items()}
        display_names = mock_categories

    col_select, col_status = st.columns([1, 2])
    with col_select:
        selected_display = st.selectbox(
            "🏷️ Select Product Category",
            display_names,
            key="sc_category",
        )
        selected_key = display_to_key.get(selected_display, selected_display)

    with col_status:
        st.markdown('<div style="height: 1.8rem;"></div>', unsafe_allow_html=True)
        cat_icon = CATEGORY_ICONS.get(selected_key, "📦")
        if using_real_models:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 1.5rem;">{cat_icon}</span>
                <span class="model-status loaded">● Model Loaded</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 1.5rem;">{cat_icon}</span>
                <span class="model-status mock">● Demo Data</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # FORECAST GENERATION
    # ═══════════════════════════════════════════════════════════

    forecast_df = None
    mape_score = None
    current_stock = MOCK_STOCK_LEVELS.get(selected_key, 250)
    gap_metrics = None

    if using_real_models:
        try:
            with st.spinner(f"Loading Prophet model for {selected_display}..."):
                forecast_df, mape_score = _generate_forecast_cached(selected_key)
                gap_metrics = calculate_stock_gap(forecast_df, current_stock)
        except FileNotFoundError:
            st.error(f"Model artifact not found for '{selected_key}'. Check timeseries/models/prophet/")
            using_real_models = False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            using_real_models = False

    # Fall back to mock if needed
    if not using_real_models:
        mock_cat_mapping = {v: k for k, v in {
            "Electronics": "Electronics", "Clothing": "Clothing",
            "Home & Kitchen": "Home & Kitchen", "Books": "Books",
            "Sports & Outdoors": "Sports & Outdoors", "Beauty": "Beauty",
            "Toys & Games": "Toys & Games", "Grocery": "Grocery",
        }.items()}
        mock_cat = selected_display if selected_display in mock_cat_mapping else "Electronics"
        hist_df, mock_fc_df = get_forecast_data(mock_cat)
        # Generate fake gap metrics
        total_demand = mock_fc_df["Units Sold"].sum()
        gap_metrics = {
            "forecast_units": int(total_demand),
            "current_stock": current_stock,
            "recommended_order": max(0, int(total_demand - current_stock)),
            "surplus_or_deficit": int(current_stock - total_demand),
        }

    # ── Key Metrics Row ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        demand_val = gap_metrics["forecast_units"] if gap_metrics else 0
        _metric_card(
            "30-Day Demand", f"{demand_val:,}",
            "projected units", "📈", "indigo",
        )

    with col2:
        _metric_card(
            "Current Stock", f"{current_stock:,}",
            "units available", "📦", "cyan",
        )

    with col3:
        if gap_metrics and gap_metrics["recommended_order"] > 0:
            _metric_card(
                "Restock Needed", f"+{gap_metrics['recommended_order']:,}",
                "⚠️ units to order", "🛒", "rose",
            )
        else:
            surplus = gap_metrics["surplus_or_deficit"] if gap_metrics else 0
            _metric_card(
                "Surplus Stock", f"{surplus:,}",
                "✅ well-stocked", "✅", "emerald",
            )

    with col4:
        if using_real_models and mape_score is not None:
            mape_display = f"{mape_score}%" if isinstance(mape_score, (int, float)) else str(mape_score)
            mape_color = "emerald" if isinstance(mape_score, (int, float)) and mape_score < 10 else "amber"
            _metric_card(
                "Model MAPE", mape_display,
                "prediction error rate", "🎯", mape_color,
            )
        else:
            _metric_card(
                "Forecast MAPE", "~12%",
                "simulated accuracy", "🎯", "amber",
            )

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    # ── Forecast Chart ──
    st.markdown(f'<div class="section-header">📊 30-Day Demand Forecast — {selected_display}</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    if using_real_models and forecast_df is not None:
        fig = create_prophet_forecast_chart(
            forecast_df,
            title=f"{selected_display} — Prophet Model Forecast",
            height=440,
            theme=theme,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Legend
        st.markdown("""
        <div class="forecast-legend" style="margin-top: 0.5rem;">
            <div><span class="forecast-legend-dot" style="background: #6366f1;"></span> Predicted Demand</div>
            <div><span class="forecast-legend-dot" style="background: rgba(99,102,241,0.3);"></span> Confidence Band</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Use mock forecast chart
        mock_cat = selected_display if selected_display in [
            "Electronics", "Clothing", "Home & Kitchen", "Books",
            "Sports & Outdoors", "Beauty", "Toys & Games", "Grocery",
        ] else "Electronics"
        hist_df, mock_fc_df = get_forecast_data(mock_cat)
        fig = create_forecast_chart(
            hist_df, mock_fc_df,
            title=f"{selected_display} — Historical vs Forecast (Demo)",
            height=440,
            theme=theme,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
        <div class="forecast-legend" style="margin-top: 0.5rem;">
            <div><span class="forecast-legend-dot" style="background: #6366f1;"></span> Historical Actuals</div>
            <div><span class="forecast-legend-dot" style="background: #22d3ee;"></span> Forecast Prediction</div>
            <div><span class="forecast-legend-dot" style="background: rgba(34,211,238,0.2);"></span> 95% Confidence Band</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    # ── Stock Gap Analysis Detail ──
    if gap_metrics:
        st.markdown('<div class="section-header">📋 Stock Gap Analysis</div>',
                    unsafe_allow_html=True)

        is_deficit = gap_metrics["recommended_order"] > 0
        stock_ratio = min(100, (current_stock / max(gap_metrics["forecast_units"], 1)) * 100)
        bar_color = "var(--accent-rose)" if stock_ratio < 50 else "var(--accent-amber)" if stock_ratio < 80 else "var(--accent-emerald)"

        st.markdown(f"""
        <div class="glass-card">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; text-align: center;">
                <div>
                    <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;">Forecasted Demand</div>
                    <div style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; color: var(--accent-indigo);">
                        {gap_metrics['forecast_units']:,}
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.78rem;">units in 30 days</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;">Current Inventory</div>
                    <div style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; color: var(--accent-cyan);">
                        {gap_metrics['current_stock']:,}
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.78rem;">units on hand</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;">
                        {"Recommended Order" if is_deficit else "Surplus"}
                    </div>
                    <div style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; color: {"var(--accent-rose)" if is_deficit else "var(--accent-emerald)"};">
                        {"+" if is_deficit else ""}{gap_metrics['recommended_order'] if is_deficit else gap_metrics['surplus_or_deficit']:,}
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.78rem;">
                        {"units to order" if is_deficit else "units surplus"}
                    </div>
                </div>
            </div>
            <div style="margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: var(--text-muted); font-size: 0.75rem;">Stock Coverage</span>
                    <span style="color: var(--text-secondary); font-size: 0.75rem; font-weight: 600;">{stock_ratio:.0f}%</span>
                </div>
                <div class="stock-bar-bg">
                    <div class="stock-bar-fill" style="width: {stock_ratio}%; background: {bar_color};"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    # ── Inventory Overview — All Categories ──
    col_stock, col_alerts = st.columns([1.5, 1])

    with col_stock:
        st.markdown('<div class="section-header">📦 Stock vs Demand — All Categories</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        inventory_df = get_inventory_status()
        colors = get_theme_colors(theme)

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
            font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
            margin=dict(l=20, r=20, t=40, b=60),
            height=400,
            barmode="group",
            title=dict(text="Inventory Health Overview", font=dict(size=15, color=colors["text"])),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color=colors["text_secondary"], size=11),
            ),
            hoverlabel=dict(bgcolor=colors["hover_bg"], font_size=13, font_family="Inter",
                           font_color=colors["text"], bordercolor="rgba(99,102,241,0.3)"),
        )
        fig_compare.update_xaxes(
            showgrid=False, tickfont=dict(color=colors["text_muted"], size=10),
            tickangle=-35,
        )
        fig_compare.update_yaxes(
            showgrid=True, gridcolor=colors["grid"],
            tickfont=dict(color=colors["text_muted"], size=11), zeroline=False,
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
            st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
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

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    # ── Model Accuracy Table ──
    st.markdown('<div class="section-header">📋 Model Performance — All Categories</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    if using_real_models:
        eval_data = _get_evaluation_data()
        if eval_data and "categories" in eval_data:
            rows = []
            for cat_key, metrics in eval_data["categories"].items():
                rows.append({
                    "Category": CATEGORY_DISPLAY.get(cat_key, cat_key),
                    "MAPE (%)": metrics.get("mape", "N/A"),
                    "Training Days": metrics.get("train_days", "N/A"),
                    "Test Days": metrics.get("test_days", "N/A"),
                    "Status": metrics.get("status", "N/A").title(),
                })
            eval_df = pd.DataFrame(rows)

            st.dataframe(
                eval_df,
                use_container_width=True,
                hide_index=True,
                height=340,
                column_config={
                    "MAPE (%)": st.column_config.ProgressColumn(
                        "MAPE (%)",
                        min_value=0,
                        max_value=20,
                        format="%.1f%%",
                    ),
                },
            )
        else:
            st.info("Evaluation summary not found. Place evaluation_summary.json in timeseries/models/prophet/")
    else:
        metrics_df = get_forecast_metrics()
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
