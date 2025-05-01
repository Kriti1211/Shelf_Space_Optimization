import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.express as px
from utils.helpers import upload_and_preview_data, filter_by_season

# ─────────────── LOAD CUSTOM STYLES ───────────────
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ─────────────── CONFIGURATION ────────────────
LP_MAX_BUFFER      = 1.2    # allow up to 20% above last-30-day sales
LP_MIN_BUFFER      = 0.2    # at least 20% of last-30-day sales
LP_MIN_STOCK_MIN   = 5

GA_POPULATION_SIZE = 50
GA_GENERATIONS     = 100
GA_MUTATION_RATE   = 0.1

# ─────────────── LP OPTIMIZER ────────────────
def optimize_shelf_space_lp(df: pd.DataFrame, total_space: int) -> pd.DataFrame:
    for col in ('Product_Name','Category','Profit_Per_Unit','Sales_Last_30_Days'):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'")
    prob = pulp.LpProblem("ShelfSpaceLP", pulp.LpMaximize)
    vars_ = [pulp.LpVariable(f"space_{i}", lowBound=0, cat='Integer') for i in df.index]
    profits = df['Profit_Per_Unit'].to_numpy()
    prob += pulp.lpSum(vars_[i] * profits[i] for i in df.index)
    prob += pulp.lpSum(vars_) <= total_space

    sales = df['Sales_Last_30_Days'].to_numpy()
    for i in df.index:
        prob += vars_[i] <= sales[i] * LP_MAX_BUFFER
        prob += vars_[i] >= min(LP_MIN_STOCK_MIN, sales[i] * LP_MIN_BUFFER)

    status = prob.solve()
    if pulp.LpStatus[status] != 'Optimal':
        st.warning(f"LP solver status: {pulp.LpStatus[status]}")
    allocated = [int(v.value() or 0) for v in vars_]

    out = df.copy()
    out['Allocated_Space'] = allocated
    return out[['Product_Name','Category','Allocated_Space','Profit_Per_Unit']]

# ─────────────── GA OPTIMIZER ─────────────────
def optimize_shelf_space_ga(df: pd.DataFrame, total_space: int) -> pd.DataFrame:
    profits = df['Profit_Per_Unit'].to_numpy()
    sales   = df['Sales_Last_30_Days'].to_numpy()
    n       = len(df)

    def create_individual():
        a = np.random.rand(n)
        a = (a / a.sum() * total_space).astype(int)
        return a

    def fitness(a):
        profit  = a.dot(profits)
        penalty = 0
        total   = a.sum()
        if total > total_space:
            penalty += (total - total_space) * 100
        over  = a > sales * LP_MAX_BUFFER
        under = a < np.minimum(LP_MIN_STOCK_MIN, sales * LP_MIN_BUFFER)
        penalty += over.sum()*100 + under.sum()*100
        return profit - penalty

    pop = [create_individual() for _ in range(GA_POPULATION_SIZE)]
    for _ in range(GA_GENERATIONS):
        scores  = np.array([fitness(ind) for ind in pop])
        parents = []
        for _ in range(GA_POPULATION_SIZE):
            contenders = np.random.choice(GA_POPULATION_SIZE, 3, replace=False)
            parents.append(pop[contenders[np.argmax(scores[contenders])]])

        new_pop = []
        for i in range(0, GA_POPULATION_SIZE, 2):
            p1, p2 = parents[i], parents[i+1]
            cp = np.random.randint(n)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            if np.random.rand() < GA_MUTATION_RATE:
                c1[np.random.randint(n)] = np.random.randint(total_space)
            if np.random.rand() < GA_MUTATION_RATE:
                c2[np.random.randint(n)] = np.random.randint(total_space)
            new_pop.extend([c1, c2])
        pop = new_pop

    best = max(pop, key=fitness).astype(int)
    out  = df.copy()
    out['Allocated_Space'] = best
    return out[['Product_Name','Category','Allocated_Space','Profit_Per_Unit']]

# ─────────── PLANOGRAM DISPLAY ────────────────
def display_planogram_interactive(opt_df: pd.DataFrame):
    if opt_df.empty:
        st.info("No products to display.")
        return

    # Category-level planogram
    st.subheader("📦 Shelf Space by Category")
    cat_agg = (
        opt_df.groupby("Category", as_index=False)["Allocated_Space"]
              .sum().sort_values("Allocated_Space", ascending=False)
    )
    fig_cat = px.bar(
        cat_agg, x="Allocated_Space", y="Category",
        orientation="h", text="Allocated_Space",
        labels={"Allocated_Space":"Units Allocated","Category":"Category"}
    )
    fig_cat.update_layout(margin=dict(l=150,t=20))
    fig_cat.update_traces(textposition="outside")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Product-level planogram
    st.subheader("📦 Shelf Space by Product")
    prod_df = opt_df.sort_values("Allocated_Space", ascending=False)
    fig_prod = px.bar(
        prod_df, x="Allocated_Space", y="Product_Name",
        orientation="h", color="Category",
        text="Allocated_Space", hover_data=["Profit_Per_Unit"],
        labels={"Allocated_Space":"Units","Product_Name":"Product"}
    )
    fig_prod.update_layout(margin=dict(l=200,t=20), yaxis={'categoryorder':'total descending'})
    fig_prod.update_traces(textposition="outside")
    st.plotly_chart(fig_prod, use_container_width=True)

    # Treemap
    st.subheader("📊 Proportional View (Treemap)")
    fig_treemap = px.treemap(
        opt_df, path=["Category","Product_Name"],
        values="Allocated_Space", hover_data=["Profit_Per_Unit"]
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # Contextual notes & KPIs
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

    method = st.radio("Method", ["Linear Programming", "Genetic Algorithm"], key="method")
    if method == "Linear Programming":
        result = optimize_shelf_space_lp(weighted_df, total_space)
    else:
        result = optimize_shelf_space_ga(weighted_df, total_space)

    display_planogram_interactive(result)

    # Alerts for potential stock‑outs
    alerts = result[result.Allocated_Space < seasonal_df["Sales_Last_30_Days"]]
    if not alerts.empty:
        st.warning("⚠️ Potential stock‑outs:")
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
