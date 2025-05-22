import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from utils.helpers import upload_and_preview_data, filter_by_season

# Monte Carlo inventory simulation metrics
def simulate_inventory(forecast_df, reorder_point, lead_time, initial_inventory, days, num_simulations=500):
    stats = []
    means = forecast_df['yhat'].values[-days:]
    stds = ((forecast_df['yhat_upper'] - forecast_df['yhat_lower']) / 4).clip(lower=0).values[-days:]
    for sim in range(num_simulations):
        inventory = initial_inventory
        stockouts = 0
        inv_levels = []
        for day in range(days):
            demand = max(0, np.random.normal(means[day], stds[day]))
            if demand > inventory:
                stockouts += 1
                demand = inventory
            inventory -= demand
            inv_levels.append(inventory)
            if inventory <= reorder_point:
                inventory += initial_inventory
        stats.append({
            'sim_id': sim,
            'stockouts': stockouts,
            'avg_inventory': np.mean(inv_levels),
            'trajectory': inv_levels
        })
    return pd.DataFrame(stats)

# Extract trajectories DataFrame
def get_trajectories_df(stats_df):
    records = []
    for _, row in stats_df.iterrows():
        for day_index, inv in enumerate(row['trajectory']):
            records.append({
                'sim_id': row['sim_id'],
                'day': day_index + 1,
                'inventory': inv
            })
    return pd.DataFrame(records)

# Main page function
def show_forecast():
    st.title("📈 Supply Forecast & Demand Planning")
    st.markdown(
        "This dashboard lets you upload sales data, filter by season/product, forecast demand using Prophet (with Indian holidays), compute revenue, inventory metrics, and simulate stock dynamics."
    )

    # STEP 1: Upload & preview data
    st.header("1️⃣ Upload your data")
    st.markdown("- Requires columns: **Product_Name**, **Season**, **Sales_Last_30_Days**, optional **Profit_Per_Unit**.")
    df = upload_and_preview_data()
    if df is None or df.empty:
        st.warning("Please upload a valid CSV to proceed.")
        return
    df = df.reset_index(drop=True)

    # STEP 2: Filter by season
    st.header("2️⃣ Filter by season")
    season = st.selectbox(
        "Choose a season to focus on", 
        ["Select season", "Winter", "Summer", "Monsoon", "Spring", "Autumn"],
        key="forecast_season"
    )
    if season == "Select season":
        st.info("Select a season to continue.")
        return
    seasonal_df = filter_by_season(df, season)
    if seasonal_df.empty:
        st.warning(f"No data for **{season}** season.")
        return

    # STEP 3: Select product & key metrics
    st.header("3️⃣ Select a product & review key metrics")
    product = st.selectbox("Which product to forecast?", seasonal_df['Product_Name'].unique())
    subset = seasonal_df[seasonal_df['Product_Name'] == product]
    if subset.empty:
        st.error("No records found for selected product.")
        return
    total_sales = subset['Sales_Last_30_Days'].sum()
    avg_daily = total_sales / 30.0
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Sales (30d)", f"{total_sales:,.0f} units")
    with col2:
        st.metric("Average Daily Sales", f"{avg_daily:.2f} units")

    # STEP 4: Forecast settings
    st.header("4️⃣ Forecast settings")
    horizon = st.slider("Forecast horizon (days)", 7, 60, 15)
    volatility = st.slider("Synthetic history volatility (%)", 0, 100, 10) / 100.0

    # STEP 5: Generate synthetic 30-day history
    st.header("5️⃣ Generate synthetic 30-day history")
    st.markdown("• A synthetic series is generated using your average daily sales and randomized noise.")
    st.markdown("• This imitates the past 30 days of sales data, providing context for the forecast.")
    st.markdown("• It visually demonstrates recent historical trends to inform the forecasting model.")
    seed = abs(hash(f"{product}-{season}")) % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, avg_daily * volatility, size=30)
    hist = np.clip(avg_daily + noise, 0, None).round(2)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq='D')
    ts = pd.DataFrame({'ds': dates, 'y': hist})
    fig_hist = go.Figure(go.Scatter(x=ts['ds'], y=ts['y'], mode='lines+markers', name='Synthetic History'))
    fig_hist.update_layout(title='Last 30 Days Synthetic History', xaxis_title='Date', yaxis_title='Daily Sales')
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown(f"• Volatility: **{volatility*100:.0f}%**")

    # STEP 6: Fit Prophet model & forecast next 15 days
    st.header("6️⃣ Fit Prophet & forecast next 15 days")
    st.markdown("• The Prophet model is trained on the synthetic history to capture trends and seasonality.")
    st.markdown("• A forecast is generated for the next 15 days based on the historical patterns.")
    st.markdown("• This step predicts future sales and highlights expected trends over the short term.")
    with st.spinner("Training Prophet model…"):
        model = Prophet(daily_seasonality=True)
        model.add_country_holidays(country_name='IN')
        model.fit(ts)
        future = model.make_future_dataframe(periods=15)
        forecast = model.predict(future)

    # STEP 7: Plot forecast vs history
    st.subheader("Forecast vs. History")
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=ts['ds'], y=ts['y'], mode='markers', name='History'))
    fig_fc.add_trace(go.Scatter(x=forecast['ds'].tail(horizon), y=forecast['yhat'].tail(horizon), mode='lines', name='Forecast'))
    fig_fc.update_layout(xaxis_title='Date', yaxis_title='Daily Sales')
    st.plotly_chart(fig_fc, use_container_width=True)

    # STEP 8: Forecast components
    st.header("7️⃣ Inspect trend & seasonality")
    st.markdown("• This component plot breaks down the forecast into trend, seasonal, and holiday effects.")
    st.markdown("• It helps you understand the underlying drivers of the forecasted sales patterns.")
    st.markdown("• Use these insights to adjust your business strategy accordingly.")
    comp_fig = model.plot_components(forecast)
    st.pyplot(comp_fig)

    # STEP 10: Revenue forecast if available
    if 'Profit_Per_Unit' in subset.columns:
        st.header("8️⃣ Revenue Forecast")
        profit_unit = subset['Profit_Per_Unit'].iloc[0]
        forecast['revenue'] = forecast['yhat'] * profit_unit
        cum_rev = forecast['revenue'].tail(horizon).sum()
        st.metric("Total Forecasted Revenue", f"₹{cum_rev:,.0f}")
        fig_rev = go.Figure(go.Bar(x=forecast['ds'].tail(horizon), y=forecast['revenue'].tail(horizon)))
        fig_rev.update_layout(title='Revenue Forecast', xaxis_title='Date', yaxis_title='Revenue (₹)')
        st.plotly_chart(fig_rev, use_container_width=True)

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

    # STEP 11: Demand planning calculators
    st.header("🔢 Demand Planning Calculators")
    lead_time = st.number_input("Lead Time (days)", min_value=1, value=7)
    service_z = st.slider("Service Level (z-score)", 1.0, 2.5, 1.65)
    std_dev = ts['y'].std()
    safety_stock = service_z * std_dev * np.sqrt(lead_time)
    reorder_pt = avg_daily * lead_time + safety_stock
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Safety Stock", f"{safety_stock:,.0f} units")
    with col4:
        st.metric("Reorder Point", f"{reorder_pt:,.0f} units")
    D = st.number_input("Annual Demand (units)", value=int(total_sales*12))
    S = st.number_input("Ordering Cost per Order (₹)", value=500.0)
    H = st.number_input("Holding Cost per Unit/Year (₹)", value=2.0)
    EOQ = np.sqrt(2 * D * S / H)
    st.metric("Economic Order Quantity", f"{EOQ:,.0f} units")

    # STEP 13: Interactive Inventory Simulation
    with st.expander("🔄 Interactive Inventory Simulation"):
        st.markdown("**Configure simulation parameters and explore inventory dynamics.**")
        st.markdown(
            """
This Monte Carlo simulation runs multiple scenarios of daily demand vs. inventory levels:
- For each simulation, it draws demand from forecasted distributions.
- Reorders occur when inventory falls below the reorder point.
- We track how often you stock out (miss demand) and your average on-hand inventory.

The results help you understand service level risk and inventory requirements.
            """
        )
        st.markdown(
            """**Interpretation:**
- **Avg. Stockouts per Sim:** Average number of days where no stock was available across scenarios.
- **Avg. Ending Inventory:** Typical leftover stock at the forecast horizon’s end.
- **Stockouts Distribution:** Variability of stockout counts across simulations.
- **Avg Inventory Distribution:** Variability in average inventory levels.
- **Sample Inventory Trajectories:** Day-by-day inventory trends for example scenarios."""
        )
        init_inv = st.number_input("Initial Inventory (units)", value=int(reorder_pt * 2))
        sims = st.slider("Number of Simulations", 10, 1000, 200, step=10)
        samples = st.slider("Sample Trajectories to Display", 1, 10, 5)

        stats_df = simulate_inventory(forecast, reorder_pt, lead_time, init_inv, horizon, sims)
        st.metric("Avg. Stockouts per Sim", f"{stats_df['stockouts'].mean():.1f}")
        st.metric("Avg. Ending Inventory", f"{stats_df['avg_inventory'].mean():.0f}")

        col5, col6 = st.columns(2)
        with col5:
            fig1 = go.Figure(go.Histogram(x=stats_df['stockouts']))
            fig1.update_layout(title='Stockouts Distribution', xaxis_title='Stockouts', yaxis_title='Count')
            st.plotly_chart(fig1, use_container_width=True)
        with col6:
            fig2 = go.Figure(go.Histogram(x=stats_df['avg_inventory']))
            fig2.update_layout(title='Avg Inventory Distribution', xaxis_title='Avg Inventory', yaxis_title='Count')
            st.plotly_chart(fig2, use_container_width=True)

        traj_df = get_trajectories_df(stats_df)
        sample_ids = stats_df['sim_id'].sample(samples).tolist()
        fig3 = go.Figure()
        for sid in sample_ids:
            sim_data = traj_df[traj_df['sim_id'] == sid]
            fig3.add_trace(go.Scatter(x=sim_data['day'], y=sim_data['inventory'], mode='lines', name=f"Sim {sid}"))
        fig3.update_layout(title='Sample Inventory Trajectories', xaxis_title='Day', yaxis_title='Inventory')
        st.plotly_chart(fig3, use_container_width=True)

# End of multipage module
