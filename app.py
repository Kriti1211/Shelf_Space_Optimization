import streamlit as st

# ─── load CSS ───────────────────────────────────────────────────────
def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Retail Shelf Optimization")
load_css()

# ─── routing helpers ────────────────────────────────────────────────
def get_current_page() -> str:
    return st.query_params.get("page", "home")

def set_query(page: str) -> str:
    """Returns href string that sets ?page=page in the URL."""
    # preserve other params if you have any:
    return f"?page={page}"

# ─── menu labels ────────────────────────────────────────────────────
MENU_ITEMS = {
    "Dashboard": "📊 Dashboard",
    "Forecast" : "📈 Supply Forecast",
    "Optimize" : "🧮 Shelf Optimization",
}

# ─── determine page ─────────────────────────────────────────────────
current_page = get_current_page()

# ─── HOME / MENU ────────────────────────────────────────────────────
if current_page == "home":
    st.markdown("<h1 class='title'>🛍️ Retail Shelf Optimization System</h1><hr>", unsafe_allow_html=True)
    st.markdown('<div class="menu-container">', unsafe_allow_html=True)
    for key, label in MENU_ITEMS.items():
        href = set_query(key)
        st.markdown(
            f'<a class="menu-btn" href="{href}">{label}</a>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ─── DASHBOARD ─────────────────────────────────────────────────────
elif current_page == "Dashboard":
    from pages.dashboard import show_dashboard
    show_dashboard()

# ─── FORECAST ──────────────────────────────────────────────────────
elif current_page == "Forecast":
    from pages.forecast import show_forecast
    show_forecast()

# ─── OPTIMIZATION ──────────────────────────────────────────────────
elif current_page == "Optimize":
    from pages.optimization import show_optimization
    show_optimization()

# ─── BACK LINK ─────────────────────────────────────────────────────
if current_page != "home":
    st.markdown("---")
    st.markdown(
        f'<a class="back-btn" href="{set_query("home")}">⬅️ Home</a>',
        unsafe_allow_html=True
    )
