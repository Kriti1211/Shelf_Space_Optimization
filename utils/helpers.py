import streamlit as st
import pandas as pd

def upload_and_preview_data():
    uploaded_file = st.file_uploader("📁 Upload retail data CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Shelf_Area'] = df['Width_cm'] * df['Height_cm']
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
        return pd.DataFrame()  # Return empty DataFrame if no season selected
    if current_season == "All":
        return df[df['Season_Used'].str.lower() == "all season"]
    return df[df['Season_Used'].str.contains(current_season, case=False) |
              (df['Season_Used'].str.lower() == "all season")]
