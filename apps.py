import streamlit as st
import pandas as pd
import plotly.express as px

# Load Data
df = pd.read_csv("retail_dataset.csv")

st.title("🛍️ Retail Sales Insight Dashboard")

# Total Sales Overview
st.subheader("Total Sales Over Last 30 Days")
total_sales = df['Sales_Last_30_Days'].sum()
st.metric("Total Sales", f"{total_sales:,} units")

# Top Products by Sales
st.subheader("Top 10 Products by Sales")
top_products = df.sort_values(
    by='Sales_Last_30_Days', ascending=False).head(10)
fig1 = px.bar(top_products, x='Product_Name', y='Sales_Last_30_Days',
              color='Category', text='Sales_Last_30_Days')
st.plotly_chart(fig1, use_container_width=True)

# Category-wise Sales
st.subheader("Sales by Category")
category_sales = df.groupby(
    'Category')['Sales_Last_30_Days'].sum().reset_index()
fig2 = px.pie(category_sales, names='Category',
              values='Sales_Last_30_Days', title='Category-wise Sales')
st.plotly_chart(fig2, use_container_width=True)

# Inventory Alert
st.subheader("📦 Inventory Status (Low Stock Alert)")
low_stock = df[df['Quantity_In_Stock'] < 5]
st.dataframe(
    low_stock[['Product_Name', 'Quantity_In_Stock', 'Sales_Last_30_Days']])

# Filter by Category
st.subheader("📁 Explore by Category")
categories = df['Category'].unique()
selected = st.selectbox("Select Category", categories)
filtered = df[df['Category'] == selected]
st.dataframe(
    filtered[['Product_Name', 'Sales_Last_30_Days', 'Profit_Per_Unit']])
