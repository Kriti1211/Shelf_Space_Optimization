import streamlit as st
import pandas as pd


def upload_and_preview_data():
    uploaded_file = st.file_uploader("📁 Upload retail data CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Shelf_Area'] = df['Width_cm'] * df['Height_cm']
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                df['Expiry_Date'] = pd.to_datetime(
                    df['Expiry_Date'], format=fmt)
                break
            except ValueError:
                continue
        df['Expiry_Date'] = pd.to_datetime(df['Expiry_Date'], errors='coerce')
        df = df.dropna(subset=['Expiry_Date'])
        return df
    return None


def filter_by_season(df, current_season):
    if current_season.strip().lower() == "all":
        return df.copy()  # Always return a copy to avoid side effects

    # Clean the 'Season_Used' column for safe string operations
    season_col = df['Season_Used'].fillna("").str.lower()
    current_season = current_season.strip().lower()

    # Return rows where the season matches or it's "all season"
    return df[(season_col.str.contains(current_season)) | (season_col == "all season")]
