import streamlit as st
import pandas as pd
from prophet import Prophet
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from deap import base, creator, tools, algorithms
import random
import math

# Data upload and preprocessing


def upload_and_preview_data():
    uploaded_file = st.file_uploader("Upload retail data CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Shelf_Area'] = df['Width_cm'] * df['Height_cm']
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                df['Expiry_Date'] = pd.to_datetime(
                    df['Expiry_Date'], format=fmt)
                break
            except ValueError:
                continue
        df['Expiry_Date'] = pd.to_datetime(df['Expiry_Date'], errors='coerce')
        df = df.dropna(subset=['Expiry_Date'])
        return df
    return None

# Sales forecasting using Prophet


def forecast_sales(df, product):
    index = df[df['Product_Name'] == product].index[0]
    base_sales = df.loc[index, 'Sales_Last_30_Days'] // 30
    ts = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=30),
        'y': [base_sales + (i % 5) for i in range(30)]
    })
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=15)
    forecast = model.predict(future)

    st.write("### Forecast Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=ts['ds'], y=ts['y'], mode='markers', name='Historical Data'))
    st.plotly_chart(fig)
    return model, forecast

# Filter data based on season


def filter_by_season(df, current_season):
    if current_season == "All":
        return df[df['Season_Used'].str.lower() == "all season"]
    return df[df['Season_Used'].str.contains(current_season, case=False) |
              (df['Season_Used'].str.lower() == "all season")]

# Linear Programming Optimization


def optimize_shelf_space_lp(df):
    forecasted_sales = []
    for i in df.index:
        base = df.loc[i, 'Sales_Last_30_Days'] // 30
        ts = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=30),
            'y': [base + (j % 5) for j in range(30)]
        })
        model = Prophet()
        model.fit(ts)
        future = model.make_future_dataframe(periods=15)
        forecast = model.predict(future)
        avg = forecast[['ds', 'yhat']].tail(7)['yhat'].mean()
        forecasted_sales.append(avg * 7)
    df['Forecasted_Sales'] = forecasted_sales

    df = df.sort_values(by="Expiry_Date").drop_duplicates(
        subset="Product_Name", keep="first")

    lp_model = LpProblem(name="shelf-space-optimization", sense=LpMaximize)
    x = {i: LpVariable(name=f"x_{i}", cat='Binary') for i in df.index}
    lp_model += lpSum(df.loc[i, 'Forecasted_Sales'] *
                      df.loc[i, 'Profit_Per_Unit'] * x[i] for i in df.index)
    lp_model += lpSum(df.loc[i, 'Shelf_Area'] * x[i] for i in df.index) <= 3000
    lp_model.solve()

    selected = []
    for i in df.index:
        if x[i].value() == 1:
            selected.append({
                "Product": df.loc[i, 'Product_Name'],
                "Forecasted Sales": int(df.loc[i, 'Forecasted_Sales']),
                "Profit/Unit": df.loc[i, 'Profit_Per_Unit'],
                "Expiry Date": df.loc[i, 'Expiry_Date']
            })

    selected_df = pd.DataFrame(selected)
    st.write("### ✅ LP-Optimized Product Selection")
    st.dataframe(selected_df)
    return selected_df

# Genetic Algorithm Optimization


def optimize_shelf_space_ga(df, shelf_area_limit=3000, generations=50, population_size=100):
    forecasted_sales = []
    for i in df.index:
        base_sales = df.loc[i, 'Sales_Last_30_Days'] // 30
        ts = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=30),
            'y': [base_sales + (j % 5) for j in range(30)]
        })
        model = Prophet()
        model.fit(ts)
        future = model.make_future_dataframe(periods=15)
        forecast = model.predict(future)
        avg = forecast[['ds', 'yhat']].tail(7)['yhat'].mean()
        forecasted_sales.append(avg * 7)
    df['Forecasted_Sales'] = forecasted_sales

    df = df.sort_values(by="Expiry_Date").drop_duplicates(
        subset="Product_Name", keep="first"
    )
    n_products = len(df)

    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, n=n_products)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(ind):
        total_area = sum(ind[i] * df.iloc[i]['Shelf_Area']
                         for i in range(n_products))
        if total_area > shelf_area_limit:
            return -1e6,
        total_profit = sum(ind[i] * df.iloc[i]['Profit_Per_Unit'] *
                           df.iloc[i]['Forecasted_Sales'] for i in range(n_products))
        return total_profit,

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                        ngen=generations, verbose=False)

    best_ind = tools.selBest(pop, 1)[0]
    selected = []
    for i in range(n_products):
        if best_ind[i] == 1:
            selected.append({
                "Product": df.iloc[i]['Product_Name'],
                "Forecasted Sales": int(df.iloc[i]['Forecasted_Sales']),
                "Profit/Unit": df.iloc[i]['Profit_Per_Unit'],
                "Expiry Date": df.iloc[i]['Expiry_Date']
            })

    selected_df = pd.DataFrame(selected)
    st.write("### 🧬 GA-Optimized Product Selection")
    st.dataframe(selected_df)
    return selected_df

# Interactive Planogram Display


def display_planogram_interactive(selected_df, df):
    st.write("### 🛒 Interactive Shelf Planogram ")
    # Parameters:
    shelf_width = 100                # overall shelf width per shelf (in cm)
    shelf_vertical_spacing = 0      # vertical spacing between shelf rows
    global_shelf_height = 150       # entire shelf height available (in cm)

    fig = go.Figure()

    # Use a consistent color for each product type.
    color_palette = px.colors.qualitative.Plotly
    color_map = {
        prod: color_palette[i % len(color_palette)]
        for i, prod in enumerate(selected_df['Product'])
    }

    # Ensure "Forecasted Sales" exists and compute the Quantity (forecast divided by 20)
    if 'Quantity' not in selected_df.columns:
        selected_df['Quantity'] = selected_df['Forecasted Sales'] // 20

    # Group similar products together (each group is a product type)
    grouped = selected_df.groupby("Product")

    # Initialize shelf row parameters.
    # vertical offset for current shelf row (row's baseline)
    overall_y_offset = 0
    current_x_cursor = 0  # horizontal cursor within current shelf row
    current_row_max_height = 0  # maximum column height in the current row

    import math
    # We'll store details for groups placed in the last shelf row.
    last_row_groups = []

    # For each product group, place all its units in one vertical column.
    for product, group in grouped:
        # Get product dimensions and expiry info.
        prod_info = df[df['Product_Name'] == product].iloc[0]
        width = prod_info['Width_cm']
        unit_height = prod_info['Height_cm']
        expiry_val = group.iloc[0]['Expiry Date']
        expiry = pd.to_datetime(expiry_val) if not isinstance(
            expiry_val, datetime) else expiry_val
        # Total quantity for this group.
        total_qty = group['Quantity'].sum()

        # If placing this column would exceed the shelf width, start a new shelf row.
        if current_x_cursor + width > shelf_width:
            # Draw a brown horizontal shelf line flush with the current row.
            fig.add_shape(
                type="line",
                x0=0,
                y0=overall_y_offset + current_row_max_height,
                x1=shelf_width,
                y1=overall_y_offset + current_row_max_height,
                line=dict(color="brown", width=4)
            )
            overall_y_offset += current_row_max_height + shelf_vertical_spacing
            current_x_cursor = 0
            current_row_max_height = 0
            last_row_groups = []  # reset for the new shelf row

        # Record group details; "Placed" will be updated after drawing.
        group_details = {
            "Product": product,
            "Quantity": total_qty,
            "Unit_Height": unit_height,
            "Width": width,
            "Expiry": expiry,
            "Placed": 0
        }
        last_row_groups.append(group_details)

        # Draw the product units in this column.
        y_cursor = overall_y_offset
        placed_units = 0
        for _ in range(total_qty):
            # Place this unit only if a full unit fits within the global shelf height.
            if y_cursor + unit_height > global_shelf_height:
                break
            bottom = y_cursor
            top = y_cursor + unit_height
            fig.add_shape(
                type="rect",
                x0=current_x_cursor,
                y0=bottom,
                x1=current_x_cursor + width,
                y1=top,
                fillcolor=color_map.get(product, "gray"),
                line=dict(color="black", width=1)
            )
            fig.add_trace(go.Scatter(
                x=[current_x_cursor + width/2],
                y=[bottom + unit_height/2],
                text=[
                    f"<b>{product}</b><br>Expiry: {expiry.strftime('%Y-%m-%d')}<br>Size: {width}×{unit_height} cm"],
                mode="markers",
                marker=dict(size=0.1, color='rgba(0,0,0,0)'),
                hoverinfo="text"
            ))
            y_cursor += unit_height
            placed_units += 1

        # Update group details with placed count.
        group_details["Placed"] = placed_units
        # The effective column height is the number of full units placed times unit_height.
        column_height = placed_units * unit_height

        # Update the current row parameters.
        current_row_max_height = max(current_row_max_height, column_height)
        current_x_cursor += width  # no extra horizontal spacing

    # Draw final brown shelf line for the last shelf row.
    fig.add_shape(
        type="line",
        x0=0,
        y0=overall_y_offset + current_row_max_height,
        x1=shelf_width,
        y1=overall_y_offset + current_row_max_height,
        line=dict(color="brown", width=4)
    )

    # Total computed shelf height.
    total_shelf_height = overall_y_offset + current_row_max_height

    # Prepare details for products not fully placed in the available shelf height.
    not_placed_details = []
    for grp in last_row_groups:
        if grp["Placed"] < grp["Quantity"]:
            not_placed_details.append({
                "Product": grp["Product"],
                "Total Quantity": grp["Quantity"],
                "Placed": grp["Placed"],
                "Not Placed": grp["Quantity"] - grp["Placed"]
            })

    # Use the lesser of the computed and available height in the layout.
    displayed_shelf_height = min(total_shelf_height, global_shelf_height)

    fig.update_layout(
        title="🧠 Optimized Shelf Layout ",
        xaxis=dict(title="Shelf Width (cm)", range=[
                   0, shelf_width], showgrid=False),
        yaxis=dict(title="Shelf Height (cm)", range=[
                   0, displayed_shelf_height], showgrid=False),
        height=800,
        plot_bgcolor='white',
        margin=dict(t=60, l=40, r=40, b=40),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show warning if there are any groups with not fully placed products.
    if not_placed_details:
        st.warning(
            "Warning: Some products cannot be fully placed due to shelf height limitations.")
        # st.dataframe(pd.DataFrame(not_placed_details))
    # Otherwise, also warn if overall height exceeds global shelf height.
    elif total_shelf_height > global_shelf_height:
        st.warning(
            f"Warning: The total shelf height ({total_shelf_height} cm) exceeds the available shelf height ({global_shelf_height} cm)."
        )


# Main Dashboard Logic
st.set_page_config(layout="wide")
st.title("🛍️ Retail Shelf Optimization Dashboard")

df = upload_and_preview_data()
if df is not None:
    season = st.selectbox("🌤️ Select Current Season", [
                          'All', 'Winter', 'Summer', 'Monsoon', 'Spring', 'Autumn'])
    seasonal_df = filter_by_season(df, season)
    product = st.selectbox("🔍 Select a product to forecast",
                           seasonal_df['Product_Name'].unique())
    forecast_sales(seasonal_df, product)

    opt_method = st.radio("Select Optimization Method", [
                          "Linear Programming", "Genetic Algorithm"])
    if opt_method == "Linear Programming":
        selected_df = optimize_shelf_space_lp(seasonal_df)
    else:
        selected_df = optimize_shelf_space_ga(seasonal_df)

    display_planogram_interactive(selected_df, seasonal_df)
