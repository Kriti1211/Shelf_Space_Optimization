import streamlit as st

# ─── load CSS ───────────────────────────────────────────────────────


def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(layout="wide", page_title="Retail Shelf Optimization")
load_css()

# ─── initialize session state ───────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# ─── menu labels ────────────────────────────────────────────────────
MENU_ITEMS = {
    "Optimize": "🧮 Shelf Optimization",
    "Dashboard": "📊 Dashboard",
    "Forecast": "📈 Supply Forecast",

}

# ─── determine page (using only session state) ──────────────────────
current_page = st.session_state["current_page"]

# ─── HOME / MENU ────────────────────────────────────────────────────
if current_page == "home":
    st.markdown("<h1 class='title'>🛍️ Retail Shelf Space Optimization System</h1><hr>",
                unsafe_allow_html=True)
    st.markdown('<div class="menu-container">', unsafe_allow_html=True)
    for key, label in MENU_ITEMS.items():
        # Using st.button preserves your UI styling
        if st.button(label, key=f"btn_{key}"):
            st.session_state["current_page"] = key
    st.markdown('</div>', unsafe_allow_html=True)

# ─── DASHBOARD ─────────────────────────────────────────────────────
elif current_page == "Dashboard":
    import importlib
    dashboard_mod = importlib.import_module("pages.dashboard")
    dashboard_mod.show_dashboard()

# ─── FORECAST ──────────────────────────────────────────────────────
elif current_page == "Forecast":
    from pages.forecast import show_forecast
    show_forecast()

# ─── OPTIMIZATION ──────────────────────────────────────────────────
elif current_page == "Optimize":
    from pages.optimization import show_optimization
    show_optimization()

# ─── BACK LINK (to go back to home) ──────────────────────────────────
if current_page != "home":
    if st.button("⬅️ Home", key="back"):
        st.session_state["current_page"] = "home"
