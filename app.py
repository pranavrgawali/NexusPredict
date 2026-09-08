"""
NexusPredict — E-Commerce Smart Analytics Dashboard
Main application entry point with Dark/Light theme toggle.
Run: streamlit run app.py
"""

import streamlit as st
from streamlit_option_menu import option_menu
import os

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
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-text">🔮 NexusPredict</div>
        <div class="logo-sub">Smart Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Toggle ──
    st.markdown("""
    <div class="theme-toggle-container">
        <span class="theme-toggle-label moon">🌙</span>
    """, unsafe_allow_html=True)

    # Use a real Streamlit toggle for reactivity
    is_light = st.toggle(
        "Theme",
        value=(st.session_state.theme == "light"),
        key="theme_toggle",
        label_visibility="collapsed",
    )

    # Update theme state
    new_theme = "light" if is_light else "dark"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("""
        <span class="theme-toggle-label sun">☀️</span>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Churn Radar", "CLV Projections", "Supply Chain"],
        icons=["speedometer2", "shield-exclamation", "currency-dollar", "box-seam"],
        default_index=0,
        styles={
            "container": {
                "padding": "0.5rem 0",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#818cf8",
                "font-size": "1.05rem",
            },
            "nav-link": {
                "font-size": "0.92rem",
                "text-align": "left",
                "margin": "3px 0",
                "padding": "0.7rem 1rem",
                "border-radius": "12px",
                "color": "#94a3b8",
                "background-color": "transparent",
                "--hover-color": "rgba(99, 102, 241, 0.08)",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.12))",
                "color": "#ffffff",
                "font-weight": "600",
            },
        },
    )

    # Sidebar footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <p style="font-size: 0.7rem; color: var(--text-dim); margin: 0;">
            Powered by ML · Built with Streamlit
        </p>
        <p style="font-size: 0.65rem; color: var(--text-muted); margin: 0.3rem 0 0 0;">
            v2.0.0 · NexusPredict
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Page Router ──
if selected == "Dashboard":
    from pages.dashboard import render
    render()
elif selected == "Churn Radar":
    from pages.churn import render
    render()
elif selected == "CLV Projections":
    from pages.clv import render
    render()
elif selected == "Supply Chain":
    from pages.supply_chain import render
    render()
