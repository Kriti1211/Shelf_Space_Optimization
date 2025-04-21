import streamlit as st
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from utils.helpers import upload_and_preview_data, filter_by_season

def show_forecast():
    st.title("📈 Supply Forecast")

    df = upload_and_preview_data()
    if df is not None:
        season = st.selectbox("Select Season", ['All', 'Winter', 'Summer', 'Monsoon', 'Spring', 'Autumn'])
        seasonal_df = filter_by_season(df, season)

        product = st.selectbox("Select Product to Forecast", seasonal_df['Product_Name'].unique())
        index = seasonal_df[seasonal_df['Product_Name'] == product].index[0]
        base_sales = seasonal_df.loc[index, 'Sales_Last_30_Days'] // 30

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
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=ts['ds'], y=ts['y'], mode='markers', name='Historical'))
        st.plotly_chart(fig)
