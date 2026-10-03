"""
NexusPredict — Retention Risk Radar (Churn Prediction)
Customer churn risk analysis, filtering, and export.
Model 1: Binary Classification (XGBoost / Logistic Regression)

Integrates with models/churn_model.joblib when available.
Falls back to mock data if the model is not found.

Layout: Simple insights first → Technical details in expander.
"""

import streamlit as st
import pandas as pd
from utils.mock_data import (
    get_churn_predictions, get_churn_feature_importance, get_churn_summary,
    get_raw_customer_data,
)
from utils.charts import (
    create_gauge_chart, create_horizontal_importance_chart,
    create_donut_chart, create_bar_chart, get_theme_colors,
)
from utils.model_loader import (
    is_churn_model_available, predict_churn,
    get_churn_model_metrics, get_churn_feature_importance_from_model,
)

# ── Determine if real model is available ──
LIVE_MODEL = is_churn_model_available()


def _risk_badge(label):
    """Generate HTML badge based on risk level."""
    color_map = {"High": "danger", "Medium": "warning", "Low": "success"}
    return f'<span class="badge badge-{color_map.get(label, "info")}">{label}</span>'


def _model_status_badge(is_live):
    """Show whether real ML model or demo mock data is being used."""
    if is_live:
        return """
        <div style="display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(16, 185, 129, 0.1);
                    border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 0.3rem 0.9rem;
                    font-size: 0.78rem; color: #10b981; font-weight: 600; margin-left: 1rem;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;
                         box-shadow: 0 0 6px #10b981;"></span>
            Live Model · XGBoost
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


def _get_churn_data():
    """Get churn predictions — uploaded data > live model > mock data."""
    # 1. Try uploaded data + live model
    if LIVE_MODEL and st.session_state.get("uploaded_data") is not None:
        uploaded_df = st.session_state.uploaded_data.copy()
        result = predict_churn(uploaded_df)
        if result is not None:
            # Map raw feature cols to display-friendly names if they exist
            col_map = {
                "age": "Age",
                "recency_days": "Days Since Last Purchase",
                "login_frequency_per_month": "Login Frequency (monthly)",
                "support_tickets_raised": "Support Tickets",
                "total_spend": "Total Spend (₹)",
            }
            for raw, display in col_map.items():
                if raw in result.columns and display not in result.columns:
                    result[display] = result[raw]
            # Ensure User ID and Customer Name exist
            if "User ID" not in result.columns:
                result["User ID"] = [f"USR-{i+1}" for i in range(len(result))]
            if "Customer Name" not in result.columns:
                result["Customer Name"] = [f"Customer {i+1}" for i in range(len(result))]
            if "Country" not in result.columns:
                result["Country"] = "N/A"
            return result

    # 2. Try live model with mock raw data
    if LIVE_MODEL:
        raw_df = get_raw_customer_data(200)
        result = predict_churn(raw_df)
        if result is not None:
            # Map raw feature cols to display-friendly names
            result["Age"] = result["age"]
            result["Days Since Last Purchase"] = result["recency_days"]
            result["Login Frequency (monthly)"] = result["login_frequency_per_month"]
            result["Support Tickets"] = result["support_tickets_raised"]
            result["Total Spend (₹)"] = result["total_spend"]
            return result

    # 3. Fallback to mock data
    return get_churn_predictions()


def _get_summary(churn_df):
    """Build summary stats from the actual predictions or fallback to mock."""
    if LIVE_MODEL and "Risk Label" in churn_df.columns:
        high = int((churn_df["Risk Label"] == "High").sum())
        medium = int((churn_df["Risk Label"] == "Medium").sum())
        low = int((churn_df["Risk Label"] == "Low").sum())
        total = len(churn_df)

        metrics = get_churn_model_metrics()
        return {
            "total_customers": total,
            "high_risk": high,
            "medium_risk": medium,
            "low_risk": low,
            "avg_risk_score": round(churn_df["Risk Score (%)"].mean(), 1),
            "roc_auc": metrics["roc_auc"] if metrics else 0.0,
            "recall": metrics["recall"] if metrics else 0.0,
        }
    return get_churn_summary()


def _get_feature_importance():
    """Get feature importance — from real model or mock."""
    if LIVE_MODEL:
        fi = get_churn_feature_importance_from_model()
        if fi is not None:
            return fi
    return get_churn_feature_importance()


def _generate_insights(summary, churn_df):
    """Generate plain-English actionable insights from churn data."""
    insights = []

    # High-risk customers insight
    high = summary["high_risk"]
    total = summary["total_customers"]
    high_pct = high / total * 100 if total > 0 else 0
    if high > 0:
        insights.append({
            "icon": "🚨",
            "text": f"<strong>{high} customers ({high_pct:.0f}%)</strong> are at high risk of leaving. "
                    f"Consider reaching out with a special offer or personal message to bring them back."
        })

    # Medium-risk insight
    medium = summary["medium_risk"]
    if medium > 0:
        insights.append({
            "icon": "⏳",
            "text": f"<strong>{medium} customers</strong> need attention — they're showing early signs of "
                    f"disengagement. A loyalty reward or check-in email could make the difference."
        })

    # Top churn driver insight
    feat_df = _get_feature_importance()
    if len(feat_df) > 0:
        top_feature = feat_df.iloc[0]["Feature"]
        insights.append({
            "icon": "🔍",
            "text": f"The #1 reason customers leave is <strong>{top_feature}</strong>. "
                    f"Focus on improving this area to reduce customer loss."
        })

    # Inactive customers insight
    if "Days Since Last Purchase" in churn_df.columns:
        inactive_30 = int((churn_df["Days Since Last Purchase"] > 30).sum())
        if inactive_30 > 0:
            insights.append({
                "icon": "📅",
                "text": f"<strong>{inactive_30} customers</strong> haven't bought anything in over 30 days. "
                        f"A \"We miss you\" email with a discount code could re-engage them."
            })

    # Good news insight
    low = summary["low_risk"]
    if low > 0:
        low_pct = low / total * 100 if total > 0 else 0
        insights.append({
            "icon": "🎉",
            "text": f"Good news! <strong>{low} customers ({low_pct:.0f}%)</strong> are loyal and active. "
                    f"Keep them happy with consistent quality and occasional surprises."
        })

    return insights


def render():
    theme = st.session_state.get("theme", "dark")
    colors = get_theme_colors(theme)

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🛡️ Customer Retention</div>
        <div class="hero-subtitle">
            See which customers might stop buying from you, so you can bring them back
            before it's too late.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──
    churn_df = _get_churn_data()
    summary = _get_summary(churn_df)

    # ═══════════════════════════════════════════════════════════
    # SIMPLE VIEW — What a shop owner sees first
    # ═══════════════════════════════════════════════════════════

    # ── Simple Summary Cards ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="simple-stat-card danger">
            <span class="simple-stat-icon">⚠️</span>
            <div class="simple-stat-number">{summary['high_risk']:,}</div>
            <div class="simple-stat-label">Customers At Risk</div>
            <div class="simple-stat-sublabel">May stop buying soon</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="simple-stat-card warning">
            <span class="simple-stat-icon">⏳</span>
            <div class="simple-stat-number">{summary['medium_risk']:,}</div>
            <div class="simple-stat-label">Need Attention</div>
            <div class="simple-stat-sublabel">Showing early warning signs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="simple-stat-card success">
            <span class="simple-stat-icon">💚</span>
            <div class="simple-stat-number">{summary['low_risk']:,}</div>
            <div class="simple-stat-label">Loyal Customers</div>
            <div class="simple-stat-sublabel">Active and happy</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="simple-stat-card info">
            <span class="simple-stat-icon">👥</span>
            <div class="simple-stat-number">{summary['total_customers']:,}</div>
            <div class="simple-stat-label">Total Customers</div>
            <div class="simple-stat-sublabel">Analyzed for risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Actionable Insights Box ──
    insights = _generate_insights(summary, churn_df)
    if insights:
        items_html = "".join([
            f'<div class="insight-item"><span class="insight-bullet">{ins["icon"]}</span><span>{ins["text"]}</span></div>'
            for ins in insights
        ])
        st.html(f'<div class="insights-box"><div class="insights-box-title"><span class="insights-icon">💡</span>What You Should Do</div>{items_html}</div>')

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Risk Distribution Donut (visual and intuitive) ──
    col_gauge, col_empty = st.columns([1, 1])

    with col_gauge:
        st.markdown('<div class="section-header">📊 Customer Risk Overview</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

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
        fig_donut.update_traces(
            marker=dict(colors=["#f43f5e", "#f59e0b", "#10b981"],
                        line=dict(color=colors["donut_border"], width=2))
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_empty:
        # Risk summary in plain language
        st.markdown('<div class="section-header">📝 Risk Summary</div>',
                    unsafe_allow_html=True)

        total = summary['total_customers']
        high_pct = summary['high_risk'] / total * 100 if total > 0 else 0
        med_pct = summary['medium_risk'] / total * 100 if total > 0 else 0
        low_pct = summary['low_risk'] / total * 100 if total > 0 else 0

        # Health indicator
        if high_pct > 25:
            health_color = "#f43f5e"
            health_icon = "🔴"
            health_text = "Needs Immediate Attention"
        elif high_pct > 15:
            health_color = "#f59e0b"
            health_icon = "🟡"
            health_text = "Room for Improvement"
        else:
            health_color = "#10b981"
            health_icon = "🟢"
            health_text = "Looking Healthy"

        st.markdown(f"""
        <div class="glass-card" style="padding: 1.5rem;">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">{health_icon}</div>
                <div style="font-family: var(--font-display); font-size: 1.2rem; font-weight: 700; color: {health_color};">
                    {health_text}
                </div>
                <div style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.3rem;">
                    Overall customer retention health
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.8rem; background: rgba(244, 63, 94, 0.06); border-radius: 10px;">
                    <span style="color: var(--text-secondary); font-size: 0.88rem;">⚠️ At Risk</span>
                    <span style="font-family: var(--font-display); font-weight: 700; color: #f43f5e; font-size: 1.05rem;">{high_pct:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.8rem; background: rgba(245, 158, 11, 0.06); border-radius: 10px;">
                    <span style="color: var(--text-secondary); font-size: 0.88rem;">⏳ Need Attention</span>
                    <span style="font-family: var(--font-display); font-weight: 700; color: #f59e0b; font-size: 1.05rem;">{med_pct:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.8rem; background: rgba(16, 185, 129, 0.06); border-radius: 10px;">
                    <span style="color: var(--text-secondary); font-size: 0.88rem;">💚 Loyal</span>
                    <span style="font-family: var(--font-display); font-weight: 700; color: #10b981; font-size: 1.05rem;">{low_pct:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Filters ──
    st.markdown('<div class="section-header">🔍 Find Specific Customers</div>',
                unsafe_allow_html=True)

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
    display_cols = [
        "User ID", "Customer Name", "Country", "Age",
        "Days Since Last Purchase", "Login Frequency (monthly)",
        "Support Tickets", "Total Spend (₹)", "Risk Score (%)", "Risk Label"
    ]
    # Only keep columns that exist in the DataFrame
    display_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[display_cols].head(50)

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

        # Safe column access — handle both live model and mock data column names
        user_age = user.get("Age", user.get("age", "N/A"))
        user_days = user.get("Days Since Last Purchase", user.get("recency_days", "N/A"))
        user_login = user.get("Login Frequency (monthly)", user.get("login_frequency_per_month", "N/A"))
        user_tickets = user.get("Support Tickets", user.get("support_tickets_raised", "N/A"))
        user_spend = user.get("Total Spend (₹)", user.get("total_spend", 0))

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
                        {user['User ID']} · {user['Country']} · Age {user_age}
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
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">{user_days}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Login Freq/Month</div>
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">{user_login}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Support Tickets</div>
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">{user_tickets}</div>
                </div>
                <div>
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Total Spend</div>
                    <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">₹{user_spend:,.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No customers match the current filters.")

    # ═══════════════════════════════════════════════════════════
    # TECHNICAL DETAILS — Collapsed by default
    # ═══════════════════════════════════════════════════════════

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    with st.expander("🔧 Technical Details — For Data Teams & Developers"):
        st.markdown("""
        <div class="tech-details-header">
            <span class="tech-icon">📊</span>
            Model Performance & Feature Analysis
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

        # Model metrics
        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            st.markdown(f"""
            <div class="metric-card indigo">
                <div class="metric-label">🎯 Model ROC-AUC</div>
                <div class="metric-value">{summary['roc_auc']:.3f}</div>
                <div class="metric-delta positive">
                    Area under the ROC curve — higher is better
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tech_col2:
            st.markdown(f"""
            <div class="metric-card cyan">
                <div class="metric-label">🔬 Model Recall</div>
                <div class="metric-value">{summary['recall']:.1%}</div>
                <div class="metric-delta positive">
                    % of actual churners correctly identified
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Feature Importance Chart
        st.markdown('<div class="section-header">🧠 Churn Feature Importance</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        feat_df = _get_feature_importance()
        fig_feat = create_horizontal_importance_chart(
            feat_df, feature_col="Feature", value_col="Importance",
            title="Top Predictive Features",
            height=340,
            theme=theme,
        )
        st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
