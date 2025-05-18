import streamlit as st
import pandas as pd

def upload_and_preview_data():
    uploaded_file = st.file_uploader("📁 Upload retail data CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Compute Shelf_Area
        if {'Width_cm', 'Height_cm'}.issubset(df.columns):
            df['Shelf_Area'] = df['Width_cm'] * df['Height_cm']

        # Parse Expiry_Date
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                df['Expiry_Date'] = pd.to_datetime(df['Expiry_Date'], format=fmt)
                break
            except ValueError:
                continue
        df['Expiry_Date'] = pd.to_datetime(df['Expiry_Date'], errors='coerce')
        df = df.dropna(subset=['Expiry_Date'])

        return df
    return None

def filter_by_season(df, current_season):
    if current_season == "Select season":
        return pd.DataFrame()
    if current_season == "All":
        return df
    return df[df['Season'].str.contains(current_season, case=False)]
