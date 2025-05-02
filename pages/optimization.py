import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import upload_and_preview_data, filter_by_season
import utils.optimization_methods.linear_programming as lp_mod
import utils.optimization_methods.genetic_algorithm as ga_mod
import utils.optimization_methods.greedy as greedy_mod
from utils.optimization_methods.ppo_quick import quick_optimize

# ─────────────── LOAD CUSTOM STYLES ───────────────
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


def display_planogram_interactive(opt_df: pd.DataFrame):
    if opt_df.empty:
        st.info("No products to display.")
        return

    # Category-level bar
    st.subheader("📦 Shelf Space by Category")
    cat_agg = (
        opt_df.groupby("Category", as_index=False)["Allocated_Space"]
              .sum().sort_values("Allocated_Space", ascending=False)
    )
    fig_cat = px.bar(
        cat_agg, x="Allocated_Space", y="Category",
        orientation="h", text="Allocated_Space",
        labels={"Allocated_Space":"Units Allocated","Category":"Category"},
        title="Shelf Space by Category"
    )
    fig_cat.update_layout(margin=dict(l=150, t=20))
    fig_cat.update_traces(textposition="outside")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Product-level bar
    st.subheader("📦 Shelf Space by Product")
    prod_df = opt_df.sort_values("Allocated_Space", ascending=False)
    fig_prod = px.bar(
        prod_df, x="Allocated_Space", y="Product_Name",
        orientation="h", color="Category",
        text="Allocated_Space", hover_data=["Profit_Per_Unit"],
        labels={"Allocated_Space":"Units","Product_Name":"Product"},
        title="Shelf Space by Product"
    )
    fig_prod.update_layout(margin=dict(l=200, t=20), yaxis={'categoryorder':'total descending'})
    fig_prod.update_traces(textposition="outside")
    st.plotly_chart(fig_prod, use_container_width=True)

    # Treemap view
    st.subheader("📊 Proportional View (Treemap)")
    fig_treemap = px.treemap(
        opt_df, path=["Category","Product_Name"],
        values="Allocated_Space", hover_data=["Profit_Per_Unit"],
        title="Shelf Space Allocation (Treemap)"
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # KPIs and context
    st.subheader("Why this layout helps")
    st.markdown(
        "- Category view for high-level allocation.  \n"
        "- Product view for item detail.  \n"
        "- Treemap for proportional insights."
    )
    total_units  = int(opt_df['Allocated_Space'].sum())
    total_profit = (opt_df['Allocated_Space'] * opt_df['Profit_Per_Unit']).sum()
    st.markdown(f"**Total units used:** {total_units}")
    st.markdown(f"**Estimated profit:** ${total_profit:,.2f}")

# ───────────────── MAIN APP ────────────────────
def show_optimization():
    st.title("🧮 Shelf Space Optimization")
    st.markdown(
        "This app helps you allocate limited shelf units across your products to maximize profit based on recent sales trends. "
        "Upload your retail data, select a season, adjust total capacity and category weights, then choose an optimization method to see your ideal planogram."
    )

    df = upload_and_preview_data()
    if df is None or df.empty:
        return

    df = df.reset_index(drop=True)

    season = st.selectbox(
        "Select Season",
        ["Select Season","Winter","Summer","Monsoon","Spring","Autumn"],
        key="opt_season"
    )
    if season == "Select Season":
        st.info("Please select a season to analyze.")
        return

    seasonal_df = filter_by_season(df, season).reset_index(drop=True)
    if seasonal_df.empty:
        st.warning("No data for this season.")
        return

    total_space = st.slider(
        "Total Shelf Units Available", 100, 5000, 1000, 100,
        key="total_space"
    )

    # Dynamic category weights based on seasonal sales
    cat_sales = seasonal_df.groupby("Category")["Sales_Last_30_Days"].sum()
    min_s, max_s = cat_sales.min(), cat_sales.max()
    default_weights = {
        cat: float(((sales - min_s) / (max(1e-9, max_s - min_s))) * (2.0 - 0.5) + 0.5)
        if max_s != min_s else 1.0
        for cat, sales in cat_sales.items()
    }

    st.subheader("Category Priority Weights")
    weights = {}
    for c in sorted(seasonal_df["Category"].unique()):
        default = round(default_weights.get(c, 1.0), 1)
        weights[c] = st.slider(f"Weight: {c}", 0.5, 2.0, default, 0.1)

    # Explanation of chosen defaults
    st.markdown(
        """
        **Why these default weights?**  
        We normalize each category’s total recent sales into the [0.5, 2.0] range, so categories selling more this season start with higher weights. You can still tweak them, but these defaults reflect seasonal demand.
        """
    )

    weighted_df = seasonal_df.copy()
    weighted_df["Profit_Per_Unit"] = weighted_df.apply(
        lambda r: r.Profit_Per_Unit * weights[r.Category], axis=1
    )

    # Select optimization method
    method = st.radio("Method", ["Linear Programming", "Genetic Algorithm", "PPO (Reinforcement Learning)"], key="method")
    if method == "Linear Programming":
        lp_mod.TOTAL_SHELF_SPACE = total_space
        result = lp_mod.optimize_lp(weighted_df)
    elif method == "Genetic Algorithm":
        ga_mod.TOTAL_SHELF_SPACE = total_space
        result = ga_mod.optimize_ga(weighted_df)
    else:
        if len(weighted_df) > 5000:
            greedy_mod.TOTAL_SHELF_SPACE = total_space
            result = greedy_mod.optimize_greedy(weighted_df)
        else:
            lp_mod.TOTAL_SHELF_SPACE = total_space
            result = quick_optimize(weighted_df, total_space)


    display_planogram_interactive(result)

    # bring the sales column into result so both live in the same DataFrame
    result["Sales_Last_30_Days"] = df["Sales_Last_30_Days"]

    alerts = result[result["Allocated_Space"] < result["Sales_Last_30_Days"]]
    if not alerts.empty:
        st.warning("⚠️ Potential stock-outs:")
        st.table(alerts)

    # Fill‑rate metric
    fill_rate = result.Allocated_Space.sum() / total_space * 100
    st.metric("Overall Fill Rate", f"{fill_rate:.1f}%")

    # Downloadable CSV
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("Download allocation CSV", csv,
                       file_name="planogram.csv", mime="text/csv")

    # JSON summary for debugging or integration
    st.json({"season": season, "total_space": total_space,
             "weights": weights, "method": method})

if __name__ == "__main__":
    show_optimization()