import streamlit as st
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from utils.helpers import upload_and_preview_data, filter_by_season

def show_forecast():
    st.title("📈 Supply Forecast")

    # 1) Upload & validate
    df = upload_and_preview_data()
    if df is None or df.empty:
        st.warning("No data uploaded.")
        return
    df = df.reset_index(drop=True)

    # 2) Season filter
    season = st.selectbox(
        "Select Season",
        ["All", "Winter", "Summer", "Monsoon", "Spring", "Autumn"]
    )
    seasonal_df = filter_by_season(df, season).reset_index(drop=True)
    if seasonal_df.empty:
        st.warning(f"No products found for season: {season}")
        return

    # 3) Product selector
    products = seasonal_df["Product_Name"].unique().tolist()
    product = st.selectbox("Select Product to Forecast", products)

    # 4) Safely locate the chosen product
    product_rows = seasonal_df[seasonal_df["Product_Name"] == product]
    if product_rows.empty:
        st.error("Error: selected product missing after filtering. Try again.")
        return
    row = product_rows.iloc[0]

    # 5) Build a 30-day daily time series from last-30-day sales
    daily_avg = row["Sales_Last_30_Days"] / 30.0
    end_date = pd.Timestamp.today().normalize()
    ts = pd.DataFrame({
        "ds": pd.date_range(end=end_date, periods=30, freq="D"),
        "y": [daily_avg + ((i % 5) - 2) * 0.1 for i in range(30)]
    })

    # 6) Fit Prophet and forecast 15 days ahead
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=15)
    forecast = model.predict(future)

    # 7) Render the chart
    st.write("### Forecast vs. Historical")
    fig = go.Figure([
        go.Scatter(
            x=ts["ds"], y=ts["y"],
            mode="markers", name="Historical"
        ),
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines", name="Forecast"
        )
    ])
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Daily Sales",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    show_forecast()
