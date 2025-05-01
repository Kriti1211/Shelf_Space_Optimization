import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import upload_and_preview_data, filter_by_season

# ── inject our custom styles ─────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def show_dashboard():
    st.title("📊 Interactive Sales Dashboard & Insights")
    st.markdown(
        """
        **Upload** your sales data → **filter** by season & category →  
        **explore** metrics, chart types, top-N products →  
        **drill into** synthetic daily trends → **download** your view.  
        """
    )

    # ── STEP 1: Upload & preview ─────────────────────────────
    st.header("1️⃣ Upload your dataset")
    st.markdown("- Needs **Product_Name**, **Season**, **Category**, **Sales_Last_30_Days**, **Profit_Per_Unit**")
    df = upload_and_preview_data()
    if df is None or df.empty:
        st.warning("No data uploaded yet.")
        return
    df = df.reset_index(drop=True)

    # ── STEP 2: Filters ───────────────────────────────────────
    st.header("2️⃣ Apply filters")
    season = st.selectbox(
        "Choose season to analyze",
        ["Select season", "Winter", "Summer", "Monsoon", "Spring", "Autumn"],
        key="dashboard_season"
    )
    if season == "Select season":
        st.info("Please select a season to analyze.")
        return
    seasonal_df = filter_by_season(df, season)

    cats = st.multiselect(
        "• Category (multi-select)",
        sorted(seasonal_df["Category"].unique().tolist()),
        default=sorted(seasonal_df["Category"].unique().tolist())
    )
    filtered_df = seasonal_df[seasonal_df["Category"].isin(cats)]
    if filtered_df.empty:
        st.warning("No data after filtering — tweak season/category.")
        return

    min_p, max_p = float(filtered_df["Profit_Per_Unit"].min()), float(filtered_df["Profit_Per_Unit"].max())
    profit_range = st.slider(
        f"• Profit per unit (${min_p:.2f}–${max_p:.2f})",
        min_value=min_p, max_value=max_p, value=(min_p, max_p)
    )
    filtered_df = filtered_df[
        filtered_df["Profit_Per_Unit"].between(profit_range[0], profit_range[1])
    ]
    if filtered_df.empty:
        st.warning("No records in that profit range.")
        return

    df = filtered_df

    # ── STEP 3: Key Metrics ───────────────────────────────────
    st.header("3️⃣ Key Metrics (filtered)")
    total_sales = df["Sales_Last_30_Days"].sum()
    avg_profit = df["Profit_Per_Unit"].mean()
    total_profit = (df["Sales_Last_30_Days"] * df["Profit_Per_Unit"]).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales (30d)", f"{total_sales:,.0f} units")
    c2.metric("Avg Profit/Unit", f"${avg_profit:.2f}")
    c3.metric("Total Profit (30d)", f"${total_profit:,.2f}")

    # ── STEP 4: Sales by Category ────────────────────────────
    st.header("4️⃣ Sales by Category")
    chart_type = st.selectbox("Chart Type", ["Horizontal Bar", "Pie Chart"], key="cat_chart_type")
    cat_sales = df.groupby("Category")["Sales_Last_30_Days"].sum().sort_values()

    if chart_type == "Horizontal Bar":
        fig = go.Figure(go.Bar(
            x=cat_sales.values,
            y=cat_sales.index,
            orientation='h',
            text=cat_sales.values.round(0),
            textposition='outside'
        ))
        fig.update_layout(xaxis_title="Units Sold", yaxis_title="Category")
    else:
        fig = px.pie(
            names=cat_sales.index,
            values=cat_sales.values,
            title="Sales Share by Category"
        )
    st.plotly_chart(fig, use_container_width=True)

    # ── STEP 5: Sales vs Profit Scatter ───────────────────────
    st.header("5️⃣ Sales vs. Profit per Unit")
    st.markdown("Check if higher-profit products sell more or less.")
    fig2 = px.scatter(
        df,
        x="Profit_Per_Unit",
        y="Sales_Last_30_Days",
        hover_data=["Product_Name", "Category"],
        trendline="ols"
    )
    fig2.update_layout(xaxis_title="Profit per Unit ($)", yaxis_title="Sales (30d)")
    st.plotly_chart(fig2, use_container_width=True)

    # ── STEP 6: Top-N Products ───────────────────────────────
    st.header("6️⃣ Top-N Products by Sales")
    top_n = st.slider("Choose N", min_value=3, max_value=20, value=10, key="top_n")
    top_df = df.nlargest(top_n, "Sales_Last_30_Days")[[
        "Product_Name", "Category", "Sales_Last_30_Days", "Profit_Per_Unit"
    ]]
    st.dataframe(top_df, use_container_width=True)

    # ── STEP 7: Synthetic Daily Trend ─────────────────────────
    st.header("7️⃣ Show synthetic daily trend for a product")
    if st.checkbox("Enable trend view", key="daily_trend"):
        prod = st.selectbox("Pick product for trend", df["Product_Name"].unique(), key="trend_prod")
        subset = df[df["Product_Name"] == prod]
        base = float(subset["Sales_Last_30_Days"].iloc[0]) / 30.0
        rng = np.random.default_rng(abs(hash(prod)) % 2**32)
        noise = rng.normal(loc=0, scale=base * 0.1, size=30)
        series = np.clip(base + noise, 0, None).round(2)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
        trend_df = pd.DataFrame({"Date": dates, "Sales": series})

        st.markdown(f"**30-day synthetic sales for:** {prod}")
        fig3 = px.line(trend_df, x="Date", y="Sales", markers=True)
        fig3.update_layout(yaxis_title="Units Sold")
        st.plotly_chart(fig3, use_container_width=True)

    # ── STEP 8: Download filtered data ────────────────────────
    st.header("8️⃣ Download your filtered data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download as CSV",
        csv,
        file_name="filtered_sales.csv",
        mime="text/csv"
    )

    st.markdown(
        """
        ---
        👉 Use the **season**, **category**, and **profit** filters to zero-in on products you care about.  
        👉 Switch chart types or drill into individual product trends.  
        👉 Download your exact view for sharing or further analysis!
        """
    )

if __name__ == "__main__":
    show_dashboard()
