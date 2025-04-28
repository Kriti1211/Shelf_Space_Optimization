import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.express as px
from utils.helpers import upload_and_preview_data, filter_by_season
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ─────────────── CONFIGURATION ────────────────
LP_MAX_BUFFER = 1.2    # allow up to 20% above last-30-day sales
LP_MIN_BUFFER = 0.2    # at least 20% of last-30-day sales
LP_MIN_STOCK_MIN = 5

GA_POPULATION_SIZE = 50
GA_GENERATIONS = 100
GA_MUTATION_RATE = 0.1

# ─────────────── LP OPTIMIZER ────────────────


def optimize_shelf_space_lp(df: pd.DataFrame, total_shelf_space: int) -> pd.DataFrame:
    # Linear Programming Shelf Space Allocation with dynamic total shelf space
    prob = pulp.LpProblem("ShelfSpaceLP", pulp.LpMaximize)
    vars_ = [pulp.LpVariable(
        f"space_{i}", lowBound=0, cat='Integer') for i in df.index]
    profits = df['Profit_Per_Unit'].to_numpy()
    prob += pulp.lpSum(vars_[i] * profits[i] for i in df.index)
    # use the dynamic total shelf space
    prob += pulp.lpSum(vars_) <= total_shelf_space

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
    return out[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]

# ─────────────── GA OPTIMIZER ─────────────────


def optimize_shelf_space_ga(df: pd.DataFrame, total_shelf_space: int) -> pd.DataFrame:
    # Genetic Algorithm Shelf Space Allocation with dynamic total shelf space
    profits = df['Profit_Per_Unit'].to_numpy()
    sales = df['Sales_Last_30_Days'].to_numpy()
    n = len(df)

    def create_individual():
        a = np.random.rand(n)
        # dynamic total shelf space
        a = (a / a.sum() * total_shelf_space).astype(int)
        return a

    def fitness(a):
        profit = a.dot(profits)
        penalty = 0
        total = a.sum()
        if total > total_shelf_space:
            penalty += (total - total_shelf_space) * 100
        over = a > sales * LP_MAX_BUFFER
        under = a < np.minimum(LP_MIN_STOCK_MIN, sales * LP_MIN_BUFFER)
        penalty += over.sum() * 100 + under.sum() * 100
        return profit - penalty

    pop = [create_individual() for _ in range(GA_POPULATION_SIZE)]
    for _ in range(GA_GENERATIONS):
        scores = np.array([fitness(ind) for ind in pop])
        parents = []
        for _ in range(GA_POPULATION_SIZE):
            c = np.random.choice(GA_POPULATION_SIZE, 3, replace=False)
            winner = c[np.argmax(scores[c])]
            parents.append(pop[winner])
        new_pop = []
        for i in range(0, GA_POPULATION_SIZE, 2):
            p1, p2 = parents[i], parents[i + 1]
            cp = np.random.randint(n)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            if np.random.rand() < GA_MUTATION_RATE:
                c1[np.random.randint(n)] = np.random.randint(total_shelf_space)
            if np.random.rand() < GA_MUTATION_RATE:
                c2[np.random.randint(n)] = np.random.randint(total_shelf_space)
            new_pop.extend((c1, c2))
        pop = new_pop

    best = max(pop, key=fitness).astype(int)
    out = df.copy()
    out['Allocated_Space'] = best
    return out[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]


# ─────────── PLANOGRAM DISPLAY ────────────────


def plot_shelf_barchart(opt_df: pd.DataFrame):
    """Display horizontal bar chart for shelf layout."""
    if opt_df.empty:
        st.info("No products to display.")
        return

    opt_df = opt_df.sort_values('Allocated_Space', ascending=False)

    fig_shelf = px.bar(
        opt_df,
        x='Allocated_Space',
        y='Product_Name',
        orientation='h',
        color='Category',
        text='Allocated_Space',
        hover_data=['Profit_Per_Unit', 'Category'],
        labels={'Allocated_Space': 'Units', 'Product_Name': 'Product'},
        title='Shelf Layout View: Product Placement'
    )
    fig_shelf.update_traces(textposition='outside')
    fig_shelf.update_layout(
        yaxis={'categoryorder': 'total descending'},
        margin=dict(l=200, t=50),
        plot_bgcolor='rgb(245,245,245)'
    )
    st.plotly_chart(fig_shelf, use_container_width=True)

    st.markdown(
        "**How to read this visualization:**  \n"
        "- **Bar length** = shelf units allocated  \n"
        "- **Y-axis** = product name (ordered by space allocation)  \n"
        "- **Color** = category (for group identification)  \n"
        "- **Hover** for profit per unit and allocations"
    )


def plot_shelf_treemap(opt_df: pd.DataFrame):
    """Display treemap for shelf space allocation."""
    if opt_df.empty:
        st.info("No products to display.")
        return

    fig_tree = px.treemap(
        opt_df,
        path=['Category', 'Product_Name'],
        values='Allocated_Space',
        hover_data=['Profit_Per_Unit'],
        title='Shelf Space Allocation (Treemap)',
        color='Category'
    )
    st.plotly_chart(fig_tree, use_container_width=True)


def display_optimization_details(opt_df: pd.DataFrame):
    """Show summary stats and explanatory notes for shelf optimization."""
    if opt_df.empty:
        st.info("No optimization results to display.")
        return

    st.subheader("📊 Summary")
    total_units = int(opt_df['Allocated_Space'].sum())
    total_profit = (opt_df['Allocated_Space'] *
                    opt_df['Profit_Per_Unit']).sum()
    total_products = opt_df[opt_df['Allocated_Space'] > 0].shape[0]

    st.markdown(f"**Total products:** {total_products}")
    st.markdown(f"**Total shelf units allocated:** {total_units}  ")
    st.markdown(f"**Projected profit:** ${total_profit:,.2f}")


def display_planogram_shelves(opt_df: pd.DataFrame, shelves: int = 5, units_per_shelf: int = 250, shelf_height_px: int = 80):
    """
    Display a planogram layout: products arranged shelf by shelf
    based on Allocated_Space, with hover info and dynamic colors.
    """
    if opt_df.empty:
        st.info("No products to display on planogram.")
        return

    # Sort by allocated space
    df = opt_df.sort_values(by='Allocated_Space',
                            ascending=False).reset_index(drop=True)

    shelf_blocks = []
    shelf_index = 0
    current_units = 0

    for _, row in df.iterrows():
        product_units = row['Allocated_Space']
        while product_units > 0 and shelf_index < shelves:
            space_left = units_per_shelf - current_units
            placed = min(product_units, space_left)

            shelf_blocks.append({
                'Shelf': shelf_index,
                'Start': current_units,
                'End': current_units + placed,
                'Product': row['Product_Name'],
                'Category': row['Category'],
                'Profit': row['Profit_Per_Unit'],
                'Allocated': placed
            })

            current_units += placed
            product_units -= placed

            if current_units >= units_per_shelf:
                shelf_index += 1
                current_units = 0

    # Plotly figure
    fig = go.Figure()
    category_colors = {cat: px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
                       for i, cat in enumerate(df['Category'].unique())}

    for block in shelf_blocks:
        fig.add_trace(go.Scatter(
            x=[block['Start'], block['End'], block['End'],
                block['Start'], block['Start']],
            y=[-block['Shelf'], -block['Shelf'], -block['Shelf'] -
                1, -block['Shelf'] - 1, -block['Shelf']],
            fill="toself",
            fillcolor=category_colors[block['Category']],
            line=dict(color='black'),
            hoveron='fills',
            name=block['Product'],
            hoverlabel=dict(bgcolor='white', font=dict(
                color='black', size=12)),
            text=(f"<b>{block['Product']}</b><br>"
                  f"Category: {block['Category']}<br>"
                  f"Allocated Units: {block['Allocated']}<br>"
                  f"Profit/Unit: ${block['Profit']:.2f}"),
            hoverinfo='text',
            showlegend=False
        ))

    fig.update_layout(
        title="🧱 Visual Shelf Planogram",
        xaxis=dict(title="Shelf Units", range=[0, units_per_shelf]),
        yaxis=dict(
            title="Shelves (Top to Bottom)",
            autorange='reversed',
            tickvals=[-i - 0.5 for i in range(shelves)],
            ticktext=[f"Shelf {i+1}" for i in range(shelves)]
        ),
        height=shelf_height_px + shelves * 80,
        plot_bgcolor='white',
        margin=dict(t=60, l=50, r=50, b=50),
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_planogram_shelves_3d(opt_df: pd.DataFrame, shelves: int = 5, units_per_shelf: int = 250):
    """
    3D planogram with products placed across shelf levels and widths.
    Each product is shown as a 3D block with hover tooltips.
    This version tries to fit products more tightly on the shelf.
    """
    if opt_df.empty:
        st.info("No products to display on 3D planogram.")
        return

    df = opt_df.sort_values(by='Allocated_Space',
                            ascending=False).reset_index(drop=True)
    category_colors = {
        cat: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
        for i, cat in enumerate(df['Category'].unique())
    }

    shelf_blocks = []
    shelf_index = 0
    current_units = 0
    unit_depth = 10  # depth per unit (z-axis)
    unit_height = 20  # fixed height for simplicity

    fig = go.Figure()

    for _, row in df.iterrows():
        product_units = row['Allocated_Space']
        product_height = row.get("Height", unit_height)

        while product_units > 0 and shelf_index < shelves:
            space_left = units_per_shelf - current_units
            placed = min(product_units, space_left)

            # Coordinates
            x0, x1 = current_units, current_units + placed
            y0, y1 = 0, product_height
            z0, z1 = shelf_index * unit_depth, (shelf_index + 1) * unit_depth

            color = category_colors[row['Category']]

            fig.add_trace(go.Mesh3d(
                x=[x0, x1, x1, x0, x0, x1, x1, x0],
                y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[z0, z0, z0, z0, z1, z1, z1, z1],
                i=[0, 0, 0, 4, 4, 4, 2, 2, 6, 1, 1, 5],
                j=[1, 2, 3, 5, 6, 7, 3, 6, 7, 5, 6, 7],
                k=[2, 3, 0, 6, 7, 4, 0, 7, 4, 6, 7, 4],
                color=color,
                opacity=0.9,
                hovertext=(f"<b>{row['Product_Name']}</b><br>"
                           f"Category: {row['Category']}<br>"
                           f"Units: {placed}<br>"
                           f"Profit/Unit: ${row['Profit_Per_Unit']:.2f}<br>"
                           f"Height: {product_height}px"),
                hoverinfo='text',
                showscale=False
            ))

            current_units += placed
            product_units -= placed

            if current_units >= units_per_shelf:
                # When one shelf is filled, move to the next shelf
                shelf_index += 1
                current_units = 0

    # Update layout and add axes titles for 3D plot
    fig.update_layout(
        title="📦 3D Shelf Planogram",
        scene=dict(
            xaxis_title="Shelf Width (Units)",
            yaxis_title="Product Height (px)",
            zaxis_title="Shelf Depth (Levels)",
            xaxis=dict(backgroundcolor="white"),
            yaxis=dict(backgroundcolor="white"),
            zaxis=dict(backgroundcolor="white")
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
    )

    # Display the 3D plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)


# ───────────────── MAIN APP ────────────────────


def show_optimization():
    st.title("🧮 Shelf Space Optimization")
    df = upload_and_preview_data()
    if df is None or df.empty:
        return

    df = df.reset_index(drop=True)
    season = st.selectbox(
        "Select Season", ['All', 'Winter', 'Summer', 'Monsoon', 'Spring', 'Autumn'])
    seasonal_df = filter_by_season(df, season).reset_index(drop=True)
    if seasonal_df.empty:
        st.warning("No data for this season.")
        return

    method = st.radio("Method", ["Linear Programming", "Genetic Algorithm"])

    # 🧮 Get shelf dimensions from user
    shelves = st.number_input("Number of Shelves", min_value=1, value=5)
    shelf_width = st.number_input(
        "Shelf Width (units)", min_value=1, value=250)

    # 🧮 Calculate total shelf space based on user input
    total_space = shelves * shelf_width

    # 🧠 Optimization
    if method == "Linear Programming":
        result = optimize_shelf_space_lp(seasonal_df, total_space)
    else:
        result = optimize_shelf_space_ga(seasonal_df, total_space)

    # 📊 Other visualizations and details
    plot_shelf_barchart(result)
    plot_shelf_treemap(result)

    display_planogram_shelves(
        result, shelves=shelves, units_per_shelf=shelf_width)
    # 📦 Display 3D Planogram
    st.subheader("🧱 3D Planogram View")
    display_planogram_shelves_3d(
        result, shelves=shelves, units_per_shelf=shelf_width)

    display_optimization_details(result)


if __name__ == "__main__":
    show_optimization()
