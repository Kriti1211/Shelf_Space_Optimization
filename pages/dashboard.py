import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    st.markdown(
        "- Needs **Product_Name**, **Season**, **Category**, **Sales_Last_30_Days**, **Profit_Per_Unit**")
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

    min_p, max_p = float(filtered_df["Profit_Per_Unit"].min()), float(
        filtered_df["Profit_Per_Unit"].max())
    profit_range = st.slider(
        f"• Profit per unit (₹{min_p:.2f}–₹{max_p:.2f})",
        min_value=min_p, max_value=max_p, value=(min_p, max_p)
    )
    filtered_df = filtered_df[
        filtered_df["Profit_Per_Unit"].between(
            profit_range[0], profit_range[1])
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
    c2.metric("Avg Profit/Unit", f"₹{avg_profit:.2f}")
    c3.metric("Total Profit (30d)", f"₹{total_profit:,.2f}")

    # ── STEP 4: Sales by Category ────────────────────────────
    st.header("4️⃣ Sales by Category")
    chart_type = st.selectbox(
        "Chart Type", ["Horizontal Bar", "Pie Chart"], key="cat_chart_type")
    cat_sales = df.groupby("Category")[
        "Sales_Last_30_Days"].sum().sort_values()

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
    fig2.update_layout(xaxis_title="Profit per Unit (₹)",
                       yaxis_title="Sales (30d)")
    st.plotly_chart(fig2, use_container_width=True)

    # ── STEP 6: Top-N Products ───────────────────────────────
    st.header("6️⃣ Top-N Products by Sales")
    top_n = st.slider("Choose N", min_value=3,
                      max_value=20, value=10, key="top_n")
    top_df = df.nlargest(top_n, "Sales_Last_30_Days")[[
        "Product_Name", "Category", "Sales_Last_30_Days", "Profit_Per_Unit"
    ]]
    st.dataframe(top_df, use_container_width=True)

    # ── STEP 7: Synthetic Daily Trend ─────────────────────────
    st.header("7️⃣ Show synthetic daily trend for a product")
    if st.checkbox("Enable trend view", key="daily_trend"):
        prod = st.selectbox("Pick product for trend",
                            df["Product_Name"].unique(), key="trend_prod")
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

    # ── STEP 8: Product Size variations ────────────────────────

    # Add after the profit range filter and before the key metrics section:
    st.header("8️⃣ Product Size Variations")

    # Compute volume
    df["Volume"] = df["Height_cm"] * df["Width_cm"] * df["Depth_cm"]

    # Extract base product names
    product_bases = df["Product_Name"].str.extract(r'(.*?)(?:\s+\d+.*)?$')[0]
    products_with_variations = product_bases[product_bases.duplicated(
        keep=False)].unique()

    if len(products_with_variations) > 0:
        selected_product = st.selectbox(
            "Select product to analyze size variations",
            products_with_variations,
            key="product_variation"
        )

    # Filter for selected product
    variations_df = df[df["Product_Name"].str.contains(
        selected_product, case=False)].copy()

    # Group by weight (assuming each weight variant represents a different size)
    grouped = variations_df.groupby(["Weight_Volume", "Season"]).agg({
        "Sales_Last_30_Days": "sum",
        "Profit_Per_Unit": "mean",
        "Height_cm": "first",
        "Width_cm": "first",
        "Depth_cm": "first",
        "Volume": "first"
    }).reset_index()

    # 📊 Conditional Chart Rendering: Sales by size (weight) and season
    st.subheader("📊 Sales by Size (Weight)")
    # Chart type selection
    chart_type = st.selectbox(
        "Choose Chart Type for Sales by Size and Season:",
        ("Horizontal Bar", "Pie Chart")
    )

    if chart_type == "Horizontal Bar":
        fig_sales = px.bar(
            grouped,
            x="Sales_Last_30_Days",
            y="Weight_Volume",
            color="Season",
            barmode="group",
            text="Sales_Last_30_Days",
            orientation='h',
            hover_data=["Volume", "Profit_Per_Unit",
                        "Height_cm", "Width_cm", "Depth_cm"],
            labels={"Weight_Volume": "Size (Weight)",
                    "Sales_Last_30_Days": "Sales (30d)"},
            title=f"Seasonal Sales Distribution by Size for {selected_product}"
        )
        fig_sales.update_traces(textposition='outside')
        fig_sales.update_layout(
            xaxis_title="Sales in Last 30 Days", yaxis_title="Weight")

    elif chart_type == "Pie Chart":
        pie_data = grouped.groupby("Weight_Volume")[
            "Sales_Last_30_Days"].sum().reset_index()
        fig_sales = px.pie(
            pie_data,
            names="Weight_Volume",
            values="Sales_Last_30_Days",
            title=f"Sales Distribution by Size for {selected_product}"
        )
        fig_sales.update_traces(textinfo='percent+label')

    st.plotly_chart(fig_sales, use_container_width=True)

    # Calculate total profit and round to 2 decimal places
    variations_df["Total_Profit"] = (
        variations_df["Sales_Last_30_Days"] * variations_df["Profit_Per_Unit"]
    ).round(2)

    # Group and round again (in case of float sum inaccuracies)
    profit_by_size = variations_df.groupby(
        "Weight_Volume")["Total_Profit"].sum().round(2).reset_index()

    st.subheader("💰 Total Profit by Size")
    fig_profit = px.bar(
        profit_by_size,
        x="Total_Profit",
        y="Weight_Volume",
        color="Weight_Volume",  # Distinct color for each weight
        orientation='h',
        text="Total_Profit",
        title=f"Total Profit by Size for {selected_product}",
        labels={"Weight_Volume": "Size", "Total_Profit": "Total Profit (₹)"}
    )

    fig_profit.update_traces(textposition='outside', width=0.6)
    fig_profit.update_layout(showlegend=False)
    st.plotly_chart(fig_profit, use_container_width=True)

    # Calculate Days to Expiry
    variations_df["Days_To_Expiry"] = (
        pd.to_datetime(variations_df["Expiry_Date"]) - pd.Timestamp.now()
    ).dt.days

    # Optional: size markers by urgency (soon-to-expire = larger size)
    variations_df["Urgency_Score"] = variations_df["Days_To_Expiry"].apply(
        lambda x: max(1, 100 - x) if x >= 0 else 100)

    # Scatter plot
    st.subheader("📅 Days to Expiry vs Sales")

    fig_expiry = px.scatter(
        variations_df,
        x="Days_To_Expiry",
        y="Sales_Last_30_Days",
        size="Urgency_Score",  # Marker size reflects urgency
        color="Weight_Volume",  # Color by size group
        hover_data={
            "Product_Name": True,
            "Days_To_Expiry": True,
            "Sales_Last_30_Days": True,
            "Weight_Volume": True,
            "Urgency_Score": False,
        },
        title="Days to Expiry vs Sales in Last 30 Days",
        labels={
            "Days_To_Expiry": "🕒 Days Until Expiry",
            "Sales_Last_30_Days": "📈 Sales (Last 30 Days)",
            "Weight_Volume": "📦 Size Variant"
        },
        template="plotly_white",
    )

    fig_expiry.update_traces(
        marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey'))
    )

    fig_expiry.update_layout(
        height=500,
        margin=dict(t=50, b=40, l=30, r=30),
        title_font_size=20,
        title_font_family="Arial Black",
    )

    st.plotly_chart(fig_expiry, use_container_width=True)

    # ── STEP 9: Download filtered data ────────────────────────

    st.header("9️⃣ Download your filtered data")
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
