import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import upload_and_preview_data, filter_by_season
import utils.optimization_methods.linear_programming as lp_mod
import utils.optimization_methods.genetic_algorithm as ga_mod
import altair as alt
from utils.optimization_methods.ppo_rl import optimize_ppo_speedup

# Constants
DEFAULT_SHELF_VOLUME = 100_000  # cm³ per shelf

# Cache genetic algorithm output


@st.cache_data(show_spinner=False)
def run_genetic(df: pd.DataFrame, total_volume: int) -> pd.DataFrame:
    return ga_mod.optimize_ga(df, total_volume)


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

    # Cluster by association group if available
    has_assoc = 'Assoc_Group' in opt_df.columns

    # Category-level bar chart
    st.subheader("📦 Shelf Space by Category")
    cat_agg = (
        opt_df.groupby("Category", as_index=False)["Allocated_Units"]
        .sum().sort_values("Allocated_Units", ascending=False)
    )
    fig_cat = px.bar(
        cat_agg, x="Allocated_Units", y="Category",
        orientation="h", text="Allocated_Units",
        labels={"Allocated_Units": "Units Allocated", "Category": "Category"},
        title="Shelf Space by Category"
    )
    fig_cat.update_layout(margin=dict(l=150, t=20))
    fig_cat.update_traces(textposition="outside")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Product-level bar chart
    st.subheader("📦 Shelf Space by Product")
    prod_df = opt_df.sort_values("Allocated_Units", ascending=False)
    color_arg = "Assoc_Group" if has_assoc else "Category"
    fig_prod = px.bar(
        prod_df, x="Allocated_Units", y="Product_Name",
        orientation="h", color=color_arg,
        text="Allocated_Units",
        hover_data=["Profit_Per_Unit", "Category"] if has_assoc else [
            "Profit_Per_Unit"],
        labels={"Allocated_Units": "Units",
                "Product_Name": "Product", color_arg: color_arg},
        title="Shelf Space by Product"
    )
    fig_prod.update_layout(margin=dict(l=200, t=20), yaxis={
                           'categoryorder': 'total descending'})
    fig_prod.update_traces(textposition="outside")
    st.plotly_chart(fig_prod, use_container_width=True)

    # Treemap view
    st.subheader("📊 Proportional View (Treemap)")
    path = ["Category", "Product_Name"]
    if has_assoc:
        path.insert(0, "Assoc_Group")
    fig_treemap = px.treemap(
        opt_df, path=path,
        values="Allocated_Units", hover_data=["Profit_Per_Unit"],
        title="Shelf Space Allocation (Treemap)"
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # KPIs & contextual explanation
    st.subheader("Why this layout helps")
    explanation = [
        "- Category view for high-level allocation.",
        "- Product view (grouped by association if provided) for detail.",
        "- Treemap for proportional insights."
    ]
    st.markdown("  ".join(explanation))
    total_units = int(opt_df['Allocated_Units'].sum())
    total_profit = (opt_df['Allocated_Units'] *
                    opt_df['Profit_Per_Unit']).sum()
    st.markdown(f"Total units used: {total_units}")
    #st.markdown(f"Estimated profit: ₹{total_profit:,.2f}")
    st.markdown(f"""
    <div style='color: white;'>
        <div style='font-size:18px;'>Estimated profit</div>
        <div style='font-size:36px;'>₹{total_profit:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)


def show_optimization():
    st.title("🧮 Shelf Space Optimization")
    st.markdown(
        "This app helps you allocate limited shelf volume across your products to maximize profit based on recent sales trends. "
        "Upload your retail data (including dimensions), select a season, set shelf count, then choose an optimization method."
    )

    df = upload_and_preview_data()
    if df is None or df.empty:
        return

    df = df.reset_index(drop=True)
    season = st.selectbox(
        "Select Season",
        ["Select Season", "Winter", "Summer", "Monsoon", "Spring", "Autumn"],
        key="opt_season"
    )
    if season == "Select Season":
        st.info("Please select a season to analyze.")
        return

    seasonal_df = filter_by_season(df, season).reset_index(drop=True)
    if seasonal_df.empty:
        st.warning("No data for this season.")
        return

    # Inject custom CSS for number input text color
    st.markdown(
        """
    <style>
    /* Target number inputs in Streamlit */
    div[data-testid="stNumberInput"] input {
        color: white !important;
    }
    </style>
    """,
        unsafe_allow_html=True
    )

    # Shelf capacity inputs
    st.subheader("Shelf Capacity Settings")
    shelf_height = st.number_input("Shelf Height (cm)", min_value=1, value=10)
    shelf_width = st.number_input("Shelf Width (cm)", min_value=1, value=50)
    shelf_depth = st.number_input("Shelf Depth (cm)", min_value=1, value=30)
    num_shelves = st.slider("Number of Shelves", 1, 50, 10)
    total_volume = shelf_height * shelf_width * shelf_depth * num_shelves
    st.markdown(f"Total Available Volume: {total_volume:,} cm³")

    # Category Priority Weights
    cat_sales = seasonal_df.groupby("Category")["Sales_Last_30_Days"].sum()
    min_s, max_s = cat_sales.min(), cat_sales.max()
    default_weights = {
        cat: float(((sales - min_s) / (max(1e-9, max_s - min_s)))
                   * (2.0 - 0.5) + 0.5)
        if max_s != min_s else 1.0
        for cat, sales in cat_sales.items()
    }
    st.subheader("Category Priority Weights")
    weights = {}
    for c in sorted(seasonal_df["Category"].unique()):
        default = round(default_weights.get(c, 1.0), 1)
        weights[c] = st.slider(f"Weight: {c}", 0.5, 2.0, default, 0.1)

    weighted_df = seasonal_df.copy()
    weighted_df["Profit_Per_Unit"] = weighted_df.apply(
        lambda r: r.Profit_Per_Unit * weights[r.Category], axis=1
    )
    if "Unit_Volume" not in weighted_df.columns and {"Height_cm", "Width_cm", "Depth_cm"}.issubset(weighted_df.columns):
        weighted_df["Unit_Volume"] = weighted_df["Height_cm"] * \
            weighted_df["Width_cm"] * weighted_df["Depth_cm"]

    method = st.radio("Method", [
                      "Linear Programming", "Genetic Algorithm", "Proximal Policy Optimization"], key="method")
    inputs = (season, total_volume, tuple(sorted(weights.items())), method)

    if st.session_state.get("last_inputs") != inputs:
        st.session_state.last_inputs = inputs
        st.session_state.pop("preview_n", None)
        st.session_state.pop("category_pie_topn", None)
        if method == "LP":
            result = lp_mod.optimize_lp(weighted_df, total_volume)
        elif method == "Genetic":
            result = run_genetic(weighted_df, total_volume)
        else:
            result = optimize_ppo_speedup(weighted_df, total_volume)
        st.session_state.result = result

    result = st.session_state.result
    display_planogram_interactive(result)

    # STOCK-OUT & SHORTFALL ANALYSIS
    alloc_df = result.copy().reset_index(drop=True)
    alloc_df = pd.merge(
        alloc_df,
        seasonal_df[["Product_Name", "Category",
                     "Sales_Last_30_Days", "Profit_Per_Unit"]],
        on=["Product_Name", "Category"],
        how="left",
        suffixes=("", "_season")
    )
    alloc_df["Sales_Last_30_Days"] = alloc_df["Sales_Last_30_Days"].fillna(
        alloc_df["Sales_Last_30_Days_season"])
    alloc_df["Profit_Per_Unit"] = alloc_df["Profit_Per_Unit"].fillna(
        alloc_df["Profit_Per_Unit_season"])
    alloc_df["Category_Weight"] = alloc_df["Category"].map(weights)
    alloc_df["Weighted_Sales"] = alloc_df["Sales_Last_30_Days"] * \
        alloc_df["Category_Weight"]
    total_weighted_demand = alloc_df["Weighted_Sales"].sum()
    alloc_df["Ideal_Allocation"] = np.ceil(
        alloc_df["Weighted_Sales"] / total_weighted_demand * total_volume).astype(int)
    alloc_df["Shortfall_Units"] = np.ceil(
        (alloc_df["Ideal_Allocation"] - alloc_df["Allocated_Units"]).clip(lower=0)).astype(int)
    alloc_df["Shortfall_Revenue"] = (
        alloc_df["Shortfall_Units"] * alloc_df["Profit_Per_Unit"])

    alerts = (
        alloc_df[alloc_df["Shortfall_Units"] > 0]
        .sort_values("Shortfall_Revenue", ascending=False)
        .groupby(["Product_Name", "Category"], as_index=False)
        .agg({
            "Sales_Last_30_Days": "sum",
            "Allocated_Units": "sum",
            "Ideal_Allocation": "sum",
            "Shortfall_Units": "sum",
            "Shortfall_Revenue": "sum",
            "Profit_Per_Unit": "first"
        })
        .sort_values("Shortfall_Revenue", ascending=False)
    )

    st.header("⚠️ Potential Stock-Out Alerts")
    if alerts.empty:
        st.success(
            "✅ All products are sufficiently allocated—no potential stock-outs detected!")
    else:
        st.metric("Products at Risk", len(alerts))
        st.metric("Total Lost Revenue",
                  f"₹{alerts['Shortfall_Revenue'].sum():,.2f}")
        max_n = len(alerts)
        default_n = min(10, max_n)
        start_n = st.session_state.get("preview_n", default_n)
        n = st.slider("How many products to preview?", min_value=1,
                      max_value=max_n, value=start_n, step=1, key="preview_n")
        expander_label = f"Showing top {st.session_state.preview_n} products by lost revenue"
        with st.expander(expander_label):
            st.dataframe(
                alerts.head(st.session_state.preview_n)[[
                    "Product_Name", "Category", "Sales_Last_30_Days",
                    "Allocated_Units", "Ideal_Allocation", "Shortfall_Units", "Shortfall_Revenue"
                ]],
                use_container_width=True
            )
        top5 = alerts.head(5).set_index("Product_Name")["Shortfall_Revenue"]
        fig = px.bar(
            top5.reset_index().rename(
                columns={"Product_Name": "Product",
                         "Shortfall_Revenue": "Lost Revenue"}
            ),
            x="Lost Revenue", y="Product",
            orientation="h",
            title="🔍 Top-5 Lost-Revenue Shortfalls"
        )
        fig.update_layout(margin=dict(l=150, t=30))
        st.plotly_chart(fig, use_container_width=True)
        max_slices = len(alerts)
        default_pie = min(5, max_slices)
        start_pie = st.session_state.get("category_pie_topn", default_pie)
        top_n = st.slider("Compute pie on top how many products by lost revenue?",
                          min_value=1, max_value=max_slices, value=start_pie, step=1, key="category_pie_topn")
        top_alerts = alerts.head(top_n)
        cat_losses = (
            top_alerts.groupby("Category", as_index=False)["Shortfall_Revenue"]
            .sum().sort_values("Shortfall_Revenue", ascending=False)
        )
        fig_cat_pie = px.pie(
            cat_losses, names="Category", values="Shortfall_Revenue",
            title=f"🔍 Category Share of Lost Revenue (Top {top_n} Products)"
        )
        fig_cat_pie.update_traces(
            textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_cat_pie, use_container_width=True)

    fill_rate = result.Allocated_Units.sum() / total_volume * 100
    st.metric("Overall Fill Rate", f"{fill_rate:.1f}%")
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("Download allocation CSV", csv,
                       file_name="planogram.csv", mime="text/csv")
    st.json({"season": season, "total_volume": total_volume,
            "weights": weights, "method": method})


if __name__ == "__main__":
    show_optimization()
