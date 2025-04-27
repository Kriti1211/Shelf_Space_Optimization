import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from utils.helpers import upload_and_preview_data, filter_by_season

def show_forecast():
    st.title("📈 Supply Forecast & Insights")
    st.markdown(
        """
        This tool helps you **upload your sales data**, filter by season, 
        select a product, and then **forecast** the next days’ demand so you 
        can plan inventory, promotions, or restocking.
        """
    )

    # ── STEP 1: Upload data ────────────────────────────────
    st.header("1️⃣ Upload your data")
    st.markdown("- Must contain **Product_Name**, **Season**, **Sales_Last_30_Days** columns.")
    df = upload_and_preview_data()
    if df is None or df.empty:
        st.warning("No data uploaded yet.")
        return
    df = df.reset_index(drop=True)

    # ── STEP 2: Season filter ──────────────────────────────
    st.header("2️⃣ Filter by season")
    season = st.selectbox(
        "Choose a season to focus on",
        ["Select season", "All", "Winter", "Summer", "Monsoon", "Spring", "Autumn"],
        key="forecast_season"
    )
    if season == "Select season":
        st.info("Please select a season to continue.")
        return
    seasonal_df = filter_by_season(df, season).reset_index(drop=True)
    if seasonal_df.empty:
        st.warning(f"No records for **{season}** season.")
        return

    # ── STEP 3: Product selector ────────────────────────────
    st.header("3️⃣ Select a product")
    product = st.selectbox("Which product to forecast?", seasonal_df["Product_Name"].unique())
    subset = seasonal_df[seasonal_df["Product_Name"] == product]
    if subset.empty:
        st.error("Selected product missing after filtering—please try again.")
        return

    # ── SHOW KEY METRICS ────────────────────────────────────
    st.subheader("🔢 Key metrics (last 30 days)")
    total_sales = subset["Sales_Last_30_Days"].sum()
    avg_daily = total_sales / 30.0
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Sales", f"{total_sales:,.0f} units")
    with col2:
        st.metric("Avg Daily Sales", f"{avg_daily:.2f} units")

    # ── USER CONTROLS ──────────────────────────────────────
    st.header("4️⃣ Forecast settings")
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=15, step=1)
    volatility = st.slider(
        "Demand volatility (%) – higher = more randomness",
        min_value=0, max_value=100, value=10, step=5
    ) / 100.0

    # ── BUILD SYNTHETIC HISTORY ─────────────────────────────
    st.header("5️⃣ Generate synthetic 30-day history")
    seed = abs(hash(f"{product}-{season}")) % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=avg_daily * volatility, size=30)
    history = np.clip(avg_daily + noise, 0, None).round(2)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq="D")
    ts = pd.DataFrame({"ds": dates, "y": history})
    fig_hist = go.Figure(go.Scatter(
        x=ts["ds"], y=ts["y"], mode="lines+markers", name="Synthetic History"
    ))
    fig_hist.update_layout(
        title="Last 30 Days (Synthetic)",
        xaxis_title="Date",
        yaxis_title="Daily Sales"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown(
        "• We seed the noise so each **(product, season)** combo is repeatable.  \n"
        f"• Volatility = **{volatility*100:.0f}%** controls ups & downs."
    )

    # ── FIT MODEL & FORECAST ───────────────────────────────
    st.header(f"6️⃣ Fit Prophet & forecast next {horizon} days")
    with st.spinner("Training model…"):
        model = Prophet(daily_seasonality=True)
        model.fit(ts)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

    # ── PLOT FORECAST ──────────────────────────────────────
    st.subheader("Forecast vs. History")
    fig_fc = go.Figure([
        go.Scatter(x=ts["ds"], y=ts["y"], mode="markers", name="History"),
        go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast")
    ])
    fig_fc.update_layout(
        xaxis_title="Date",
        yaxis_title="Daily Sales",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown(
        f"• The **blue dots** are your synthetic history, the **line** is the Prophet forecast for the next {horizon} days."
    )

    # ── FORECAST COMPONENTS ────────────────────────────────
    st.header("7️⃣ Inspect trend & seasonality")
    comp_fig = model.plot_components(forecast)
    st.write(comp_fig)
    st.markdown(
        "- **Trend:** overall increase/decrease in demand  \n"
        "- **Weekly/seasonal:** recurring patterns (e.g. weekends, holiday seasons)"
    )

    # ── FORECAST TABLE ─────────────────────────────────────
    st.header("8️⃣ Preview forecast data")
    st.dataframe(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .tail(horizon)
        .reset_index(drop=True)
    )

    # ── WHY THIS MATTERS ────────────────────────────────────
    st.header("🔍 Why this helps")
    st.markdown(
        """
        - **Plan ahead:** see expected demand so you avoid stockouts.  
        - **Adjust safety stock:** tune volatility for conservative vs. aggressive forecasts.  
        - **Visual insight:** components plot shows if seasonality really drives sales.  
        """
    )


if __name__ == "__main__":
    show_forecast()
