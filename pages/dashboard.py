import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard():
    st.title("🛍️ Retail Sales Insight Dashboard")

    df = pd.read_csv("retail_dataset.csv")  # or use utils if needed

    st.subheader("Total Sales Over Last 30 Days")
    st.metric("Total Sales", f"{df['Sales_Last_30_Days'].sum():,} units")
    st.markdown("<div class='total-sales'>Total Sales</div>", unsafe_allow_html=True)
    st.markdown("<div class='sales-units'>51,922 units</div>", unsafe_allow_html=True)

    st.subheader("Top 10 Products by Sales")
    top_products = df.sort_values(by='Sales_Last_30_Days', ascending=False).head(10)
    st.plotly_chart(px.bar(top_products, x='Product_Name', y='Sales_Last_30_Days',
                           color='Category', text='Sales_Last_30_Days'), use_container_width=True)

    st.subheader("Sales by Category")
    cat_sales = df.groupby('Category')['Sales_Last_30_Days'].sum().reset_index()
    st.plotly_chart(px.pie(cat_sales, names='Category', values='Sales_Last_30_Days'), use_container_width=True)

    st.subheader("📦 Inventory Alert (Low Stock)")
    st.dataframe(df[df['Quantity_In_Stock'] < 5][['Product_Name', 'Quantity_In_Stock', 'Sales_Last_30_Days']])

    st.subheader("📁 Explore by Category")
    st.markdown("<div class='choose-category'>Choose Category</div>", unsafe_allow_html=True)
    selected = st.selectbox("Choose Category", df['Category'].unique())
    st.dataframe(df[df['Category'] == selected][['Product_Name', 'Sales_Last_30_Days', 'Profit_Per_Unit']])
