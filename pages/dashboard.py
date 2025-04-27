import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from utils.helpers import upload_and_preview_data, filter_by_season
from Pages.dashboard import show_dashboard  # Note the capital P

def show_forecast():
    st.title("📈 Supply Forecast")
    st.markdown("""
    This tool predicts daily sales for your chosen product over the next 15 days.

    **Steps**  
    1. Upload your dataset (needs “Product_Name”, “Season”, “Sales_Last_30_Days”).  
    2. Filter by season.  
    3. Pick a product.  
    4. See the last 30 days (synthetic) and a 15-day forecast.  
    """)

    # 1️⃣ Upload & validate
    df = upload_and_preview_data()
    if df is None or df.empty:
        st.info("Please upload a dataset first.")
        return
    df = df.reset_index(drop=True)

    # 2️⃣ Filter by season
    st.subheader("1️⃣ Filter by season")
    season = st.selectbox("Choose a season", ["All", "Winter", "Summer", "Monsoon", "Spring", "Autumn"])
    seasonal_df = filter_by_season(df, season).reset_index(drop=True)
    if seasonal_df.empty:
        st.warning(f"No products found for: **{season}**")
        return

    # 3️⃣ Select product
    st.subheader("2️⃣ Select a product")
    products = seasonal_df["Product_Name"].unique().tolist()
    product = st.selectbox("Product", ["-- pick one --"] + products)
    if product == "-- pick one --":
        return

    # 4️⃣ Generate synthetic daily history
    st.subheader("3️⃣ Last 30 days (synthetic)")
    subset = seasonal_df[seasonal_df["Product_Name"] == product]
    total_sales = subset["Sales_Last_30_Days"].sum()
    base_daily = total_sales / 30.0

    # Use product+season as seed so series changes when either changes
    seed = abs(hash(f"{product}-{season}")) % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=base_daily*0.1, size=30)
    daily_sales = np.clip(base_daily + noise, 0, None).round(2)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq="D")
    ts = pd.DataFrame({"ds": dates, "y": daily_sales})
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=ts["ds"], y=ts["y"], mode="lines+markers", name="Historical"))
    fig_hist.update_layout(
        title="Last 30 Days Sales",
        xaxis_title="Date",
        yaxis_title="Daily Sales"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 5️⃣ Forecast next 15 days
    st.subheader("4️⃣ 15-day forecast")
    with st.spinner("Calculating forecast..."):
        model = Prophet()
        model.fit(ts.rename(columns={"ds":"ds","y":"y"}))
        future = model.make_future_dataframe(periods=15)
        forecast = model.predict(future)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=ts["ds"], y=ts["y"], mode="markers", name="Historical"
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"
    ))
    fig_fc.update_layout(
        title="Forecast vs. Historical",
        xaxis_title="Date",
        yaxis_title="Daily Sales"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Why it matters
    st.subheader("Why this helps")
    st.markdown("""
    - **Seasonal filter** ensures you only model relevant data.  
    - **Synthetic history** varies per product and season (seeded), so you see different patterns.  
    - **Seeded randomness** makes each product/season combination unique and repeatable.  
    - **15-day horizon** gives a quick view to avoid stockouts or overstock.  
    """)

def show_dashboard():
    st.title("📊 Sales Dashboard")
    
    # Upload data
    df = upload_and_preview_data()
    if df is None or df.empty:
        st.info("Please upload a dataset first.")
        return
        
    df = df.reset_index(drop=True)
    
    # Filter by season
    st.subheader("Filter by Season")
    season = st.selectbox("Choose season", ["All", "Winter", "Summer", "Monsoon", "Spring", "Autumn"])
    seasonal_df = filter_by_season(df, season)
    
    if seasonal_df.empty:
        st.warning(f"No data available for {season} season")
        return
        
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_sales = seasonal_df["Sales_Last_30_Days"].sum()
        st.metric("Total Sales", f"{total_sales:,.0f} units")
        
    with col2:
        avg_profit = seasonal_df["Profit_Per_Unit"].mean()
        st.metric("Avg Profit/Unit", f"${avg_profit:.2f}")
        
    with col3:
        total_profit = (seasonal_df["Sales_Last_30_Days"] * seasonal_df["Profit_Per_Unit"]).sum()
        st.metric("Total Profit", f"${total_profit:,.2f}")
    
    # Category breakdown
    st.subheader("Sales by Category")
    cat_sales = seasonal_df.groupby("Category")["Sales_Last_30_Days"].sum().sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=cat_sales.values,
        y=cat_sales.index,
        orientation='h',
        text=cat_sales.values.round(0),
        textposition='outside',
    ))
    
    fig.update_layout(
        title="Sales Volume by Category",
        xaxis_title="Total Units Sold",
        yaxis_title="Category",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top selling products
    st.subheader("Top 10 Products")
    top_products = seasonal_df.nlargest(10, "Sales_Last_30_Days")[
        ["Product_Name", "Category", "Sales_Last_30_Days", "Profit_Per_Unit"]
    ]
    st.dataframe(top_products)

if __name__ == "__main__":
    show_forecast()
    show_dashboard()
