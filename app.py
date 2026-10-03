"""
NexusPredict — E-Commerce Smart Analytics Dashboard
Main application entry point with Dark/Light theme toggle.
Run: streamlit run app.py
"""

import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd

# ── Page Configuration ──
st.set_page_config(
    page_title="NexusPredict | Smart Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme State Initialization ──
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ── Uploaded Data State ──
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# ── Load Custom CSS ──
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Inject Theme Script ──
# This script sets the data-theme attribute on the root HTML element
# based on the current theme state, enabling CSS variable overrides
theme_val = st.session_state.theme
st.markdown(f"""
<script>
    (function() {{
        const theme = "{theme_val}";
        document.documentElement.setAttribute("data-theme", theme);

        // Also set on the Streamlit app container
        const appContainer = document.querySelector('[data-testid="stAppViewContainer"]');
        if (appContainer) {{
            appContainer.setAttribute("data-theme", theme);
        }}

        // Set on body for maximum compatibility
        document.body.setAttribute("data-theme", theme);

        // Apply to sidebar
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {{
            sidebar.setAttribute("data-theme", theme);
        }}
    }})();
</script>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    # ── Brand Header Card ──
    st.markdown("""
    <div class="sidebar-brand-card">
        <div class="brand-badge-icon">🔮</div>
        <div class="brand-text-block">
            <div class="brand-title">NexusPredict</div>
            <div class="brand-subtitle">Shop Intelligence Cockpit</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Selector ──
    st.markdown('<div class="sidebar-section-title"><span>APPEARANCE</span></div>', unsafe_allow_html=True)
    theme_choice = st.segmented_control(
        "Theme",
        options=["🌙 Dark", "☀️ Light"],
        default="🌙 Dark" if st.session_state.theme == "dark" else "☀️ Light",
        key="sidebar_theme_toggle",
        label_visibility="collapsed",
        width="stretch",
    )

    if theme_choice:
        selected_theme = "light" if "Light" in theme_choice else "dark"
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()

    # ── Navigation Menu ──
    st.markdown('<div class="sidebar-section-title" style="margin-top: 1.1rem;"><span>NAVIGATION</span></div>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Churn Radar", "CLV Projections", "Supply Chain"],
        icons=["speedometer2", "shield-check", "graph-up-arrow", "box-seam"],
        default_index=0,
        styles={
            "container": {
                "padding": "0",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#818cf8",
                "font-size": "1.05rem",
            },
            "nav-link": {
                "font-size": "0.92rem",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "0.75rem 1rem",
                "border-radius": "12px",
                "color": "#94a3b8",
                "background-color": "transparent",
                "--hover-color": "rgba(99, 102, 241, 0.08)",
                "transition": "all 0.2s ease",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.22) 0%, rgba(139, 92, 246, 0.14) 100%)",
                "color": "#ffffff",
                "font-weight": "600",
                "border": "1px solid rgba(99, 102, 241, 0.35)",
            },
        },
    )

    # ── Store Data Status Card ──
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        fname = st.session_state.uploaded_filename or "custom_data.csv"
        st.markdown(f"""
        <div class="sidebar-status-card loaded">
            <div class="status-header">
                <span class="status-dot green-pulse"></span>
                <span class="status-title">Custom Data Active</span>
            </div>
            <div class="status-detail">
                <span class="status-file">📄 {fname}</span>
                <span class="status-meta">{len(df):,} rows · {len(df.columns)} columns</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕ Reset to Demo Data", key="clear_data_btn", use_container_width=True):
            st.session_state.uploaded_data = None
            st.session_state.uploaded_filename = None
            st.rerun()
    else:
        st.markdown("""
        <div class="sidebar-status-card demo">
            <div class="status-header">
                <span class="status-dot blue-pulse"></span>
                <span class="status-title">Demo Store Active</span>
            </div>
            <div class="status-detail">
                <span class="status-meta">Retail simulation dataset loaded</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar Footer (Clean, elegant, no unwanted credits) ──
    st.markdown("""
    <div class="sidebar-footer-card">
        <div class="footer-status-line">
            <span class="footer-dot-green"></span>
            <span class="footer-status-text">AI Engine Online</span>
        </div>
        <div class="footer-meta-line">
            <span class="footer-brand">NexusPredict Pro</span>
            <span class="footer-badge">v2.4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Page Router ──
if selected == "Dashboard":
    from views.dashboard import render
    render()
elif selected == "Churn Radar":
    from views.churn import render
    render()
elif selected == "CLV Projections":
    from views.clv import render
    render()
elif selected == "Supply Chain":
    from views.supply_chain import render
    render()
