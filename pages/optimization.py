import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import upload_and_preview_data, filter_by_season
import utils.optimization_methods.linear_programming as lp_mod
import utils.optimization_methods.genetic_algorithm as ga_mod
import altair as alt
from utils.optimization_methods.ppo_rl import optimize_ppo_speedup

# at the top of your pages/optimization.py
@st.cache_data(show_spinner=False)
def run_genetic(df: pd.DataFrame, total_space: int) -> pd.DataFrame:
    return ga_mod.optimize_ga(df,total_space)


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
    st.markdown(f"**Estimated profit:** ₹{total_profit:,.2f}")

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
        
        result = lp_mod.optimize_lp(weighted_df,total_space)
    elif method == "Genetic Algorithm":
        result = run_genetic(weighted_df, total_space)
    else:
        with st.spinner("Training PPO with profit/lost‐revenue trade‐off…"):
            result = optimize_ppo_speedup(
                weighted_df, total_space,
                timesteps=5000,
                num_envs=4
            )


    display_planogram_interactive(result)

# ------------- STOCK-OUT & SHORTFALL ANALYSIS --------------

    # 1) Start from your allocation results
    alloc_df = result.copy().reset_index(drop=True)

    # 2) Re-attach the 30-day sales
    df2 = df.reset_index(drop=True)
    alloc_df["Sales_Last_30_Days"] = df2["Sales_Last_30_Days"]

    # 3) Apply category weights → weighted demand
    alloc_df["Category_Weight"] = alloc_df["Category"].map(weights)
    alloc_df["Weighted_Sales"]   = alloc_df["Sales_Last_30_Days"] * alloc_df["Category_Weight"]
    total_weighted_demand       = alloc_df["Weighted_Sales"].sum()

    # 4) Ideal_Allocation (ceiled)
    alloc_df["Ideal_Allocation"] = np.ceil(
        alloc_df["Weighted_Sales"] / total_weighted_demand * total_space
    ).astype(int)

    # 5) Shortfall_Units (ceiled)
    alloc_df["Shortfall_Units"] = np.ceil(
        (alloc_df["Ideal_Allocation"] - alloc_df["Allocated_Space"]).clip(lower=0)
    ).astype(int)

    # 6) Shortfall_Revenue
    alloc_df["Shortfall_Revenue"] = (
        alloc_df["Shortfall_Units"] * alloc_df["Profit_Per_Unit"]
    )

    # 7) Filter + aggregate duplicates
    alerts = (
        alloc_df[alloc_df["Shortfall_Units"] > 0]
        .sort_values("Shortfall_Revenue", ascending=False)
        .groupby(["Product_Name","Category"], as_index=False)
        .agg({
            "Sales_Last_30_Days":   "sum",
            "Allocated_Space":      "sum",
            "Ideal_Allocation":     "sum",
            "Shortfall_Units":      "sum",
            "Shortfall_Revenue":    "sum",
            "Profit_Per_Unit":      "first"
        })
        .sort_values("Shortfall_Revenue", ascending=False)
    )

    st.header("⚠️ Potential Stock-Out Alerts")
    if alerts.empty:
        st.success("✅ All products are sufficiently allocated—no potential stock-outs detected!")
    else:
        st.metric("Products at Risk", len(alerts))
        st.metric("Total Lost Revenue", f"₹{alerts['Shortfall_Revenue'].sum():,.2f}")

        max_n     = len(alerts)
        default_n = min(10, max_n)

        n = st.slider(
            "How many products to preview?",
            min_value=1,
            max_value=max_n,
            value=default_n,
            step=1,
        )

        with st.expander(f"Show top {n} by lost revenue"):
            st.dataframe(
                alerts.head(n)[[
                    "Product_Name","Category","Sales_Last_30_Days",
                    "Allocated_Space","Ideal_Allocation",
                    "Shortfall_Units","Shortfall_Revenue"
                ]],
                use_container_width=True
            )

        # 9) Horizontal bar chart
        top5 = alerts.head(5).set_index("Product_Name")["Shortfall_Revenue"]
        fig = px.bar(
            top5.reset_index().rename(
                columns={"Product_Name":"Product","Shortfall_Revenue":"Lost Revenue"}
            ),
            x="Lost Revenue", y="Product",
            orientation="h",
            title="🔍 Top-5 Lost-Revenue Shortfalls"
        )
        fig.update_layout(margin=dict(l=150,t=30))
        st.plotly_chart(fig, use_container_width=True)

        # 10) Category pie chart
        max_slices = len(alerts)
        default_pie = min(5, max_slices)
        top_n = st.slider(
            "Compute pie on top how many products by lost revenue?",
            1, max_slices, default_pie, step=1,
            key="category_pie_topn"
        )
        top_alerts = alerts.head(top_n)
        cat_losses = (
            top_alerts.groupby("Category", as_index=False)["Shortfall_Revenue"]
            .sum().sort_values("Shortfall_Revenue", ascending=False)
        )
        fig_cat_pie = px.pie(
            cat_losses, names="Category", values="Shortfall_Revenue",
            title=f"🔍 Category Share of Lost Revenue (Top {top_n} Products)"
        )
        fig_cat_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_cat_pie, use_container_width=True)
        
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