import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.express as px
from utils.helpers import upload_and_preview_data, filter_by_season

# ─────────────── CONFIGURATION ────────────────
TOTAL_SHELF_SPACE  = 1000
LP_MAX_BUFFER      = 1.2    # allow up to 20% above last-30-day sales
LP_MIN_BUFFER      = 0.2    # at least 20% of last-30-day sales
LP_MIN_STOCK_MIN   = 5

GA_POPULATION_SIZE = 50
GA_GENERATIONS     = 100
GA_MUTATION_RATE   = 0.1

# ─────────────── LP OPTIMIZER ────────────────
def optimize_shelf_space_lp(df: pd.DataFrame) -> pd.DataFrame:
    for col in ('Product_Name','Category','Profit_Per_Unit','Sales_Last_30_Days'):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'")
    prob = pulp.LpProblem("ShelfSpaceLP", pulp.LpMaximize)
    vars_ = [pulp.LpVariable(f"space_{i}", lowBound=0, cat='Integer') for i in df.index]
    profits = df['Profit_Per_Unit'].to_numpy()
    prob += pulp.lpSum(vars_[i] * profits[i] for i in df.index)
    prob += pulp.lpSum(vars_) <= TOTAL_SHELF_SPACE
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
def optimize_shelf_space_ga(df: pd.DataFrame) -> pd.DataFrame:
    profits = df['Profit_Per_Unit'].to_numpy()
    sales   = df['Sales_Last_30_Days'].to_numpy()
    n       = len(df)

    def create_individual():
        a = np.random.rand(n)
        a = (a / a.sum() * TOTAL_SHELF_SPACE).astype(int)
        return a

    def fitness(a):
        profit  = a.dot(profits)
        penalty = 0
        total   = a.sum()
        if total > TOTAL_SHELF_SPACE:
            penalty += (total - TOTAL_SHELF_SPACE) * 100
        over  = a > sales * LP_MAX_BUFFER
        under = a < np.minimum(LP_MIN_STOCK_MIN, sales * LP_MIN_BUFFER)
        penalty += over.sum()*100 + under.sum()*100
        return profit - penalty

    pop = [create_individual() for _ in range(GA_POPULATION_SIZE)]
    for _ in range(GA_GENERATIONS):
        scores  = np.array([fitness(ind) for ind in pop])
        parents = []
        for _ in range(GA_POPULATION_SIZE):
            c = np.random.choice(GA_POPULATION_SIZE, 3, replace=False)
            winner = c[np.argmax(scores[c])]
            parents.append(pop[winner])
        new_pop = []
        for i in range(0, GA_POPULATION_SIZE, 2):
            p1, p2 = parents[i], parents[i+1]
            cp = np.random.randint(n)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            if np.random.rand() < GA_MUTATION_RATE:
                c1[np.random.randint(n)] = np.random.randint(TOTAL_SHELF_SPACE)
            if np.random.rand() < GA_MUTATION_RATE:
                c2[np.random.randint(n)] = np.random.randint(TOTAL_SHELF_SPACE)
            new_pop.extend((c1, c2))
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

    st.markdown(
        "**How to read this planogram:**  \n"
        "- **Bar length** = shelf units allocated  \n"
        "- **Y-axis** = product name  \n"
        "- **Color** = category  \n"
        "- **Hover** for profit per unit"
    )

    # Bar chart
    opt_df = opt_df.sort_values('Allocated_Space', ascending=False)
    fig = px.bar(
        opt_df,
        x='Allocated_Space',
        y='Product_Name',
        orientation='h',
        color='Category',
        text='Allocated_Space',
        hover_data=['Profit_Per_Unit'],
        labels={'Allocated_Space':'Units','Product_Name':'Product'},
        title='Shelf Space Allocation'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total descending'}, margin=dict(l=200,t=50))
    st.plotly_chart(fig, use_container_width=True)

    # Treemap
    fig2 = px.treemap(
        opt_df,
        path=['Category','Product_Name'],
        values='Allocated_Space',
        hover_data=['Profit_Per_Unit'],
        title='Shelf Space Allocation (Treemap)',
        color='Category'
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Why this positioning helps")
    st.markdown(
        "- **Top-down sort**: Shows your highest-priority (top sellers) first.  \n"
        "- **Bar length ⇒ importance**: Longer bars = more space = more focus.  \n"
        "- **Color grouping**: Reveals category patterns instantly.  \n"
        "- **Labels & hover**: Precise numbers on bar or in tooltip.  \n"
        "- **Treemap**: Offers a block-area view for proportional comparison."
    )

    total  = int(opt_df['Allocated_Space'].sum())
    profit = (opt_df['Allocated_Space'] * opt_df['Profit_Per_Unit']).sum()
    st.markdown(f"**Total products:** {len(opt_df)}  ")
    st.markdown(f"**Total units used:** {total}  ")
    st.markdown(f"**Estimated profit:** ${profit:,.2f}")

# ───────────────── MAIN APP ────────────────────
def show_optimization():
    st.title("🧮 Shelf Space Optimization")
    df = upload_and_preview_data()
    if df is None or df.empty:
        return

    df = df.reset_index(drop=True)
    season = st.selectbox("Select Season", ['All','Winter','Summer','Monsoon','Spring','Autumn'])
    seasonal_df = filter_by_season(df, season).reset_index(drop=True)
    if seasonal_df.empty:
        st.warning("No data for this season.")
        return

    method = st.radio("Method", ["Linear Programming","Genetic Algorithm"])
    if method == "Linear Programming":
        result = optimize_shelf_space_lp(seasonal_df)
    else:
        result = optimize_shelf_space_ga(seasonal_df)

    display_planogram_interactive(result)

if __name__ == "__main__":
    show_optimization()
