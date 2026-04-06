# src/data_preprocessing.py

import pandas as pd
import os

# File paths
RAW_DATA_PATH = "data/raw/raw_data.csv"
PROCESSED_DATA_PATH = "data/processed/clean_data.csv"


def load_data(path):
    """Load dataset"""
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully")
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


def clean_data(df):
    """Perform data cleaning"""

    print("\n🔹 Initial Shape:", df.shape)

    # -----------------------------
    # Standardize column names
    # -----------------------------
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Rainfall (mm)": "rainfall",
        "Temperature (°C)": "temperature",
        "Humidity (%)": "humidity",
        "River Discharge (m³/s)": "river_discharge",
        "Water Level (m)": "water_level",
        "Elevation (m)": "elevation",
        "Land Cover": "land_cover",
        "Soil Type": "soil_type",
        "Population Density": "population_density",
        "Infrastructure": "infrastructure",
        "Historical Floods": "past_disasters",
        "Flood Occurred": "flood_occurred"
    })

    # -----------------------------
    # Remove duplicates
    # -----------------------------
    df = df.drop_duplicates()
    print("🔹 After removing duplicates:", df.shape)

    # -----------------------------
    # Convert numeric columns
    # -----------------------------
    numeric_cols = [
        "rainfall",
        "temperature",
        "humidity",
        "river_discharge",
        "water_level",
        "elevation",
        "population_density",
        "past_disasters",
        "latitude",
        "longitude",
        "flood_occurred"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # -----------------------------
    # Handle missing values
    # -----------------------------
    df = df.dropna()
    print("🔹 After removing missing values:", df.shape)

    # -----------------------------
    # Convert categorical columns
    # -----------------------------
    categorical_cols = ["land_cover", "soil_type"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # -----------------------------
    # Ensure binary columns are int
    # -----------------------------
    binary_cols = ["past_disasters", "flood_occurred"]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print("🔹 Final Shape:", df.shape)

    # -----------------------------
    # Data summary
    # -----------------------------
    print("\nData Summary:")
    print(df.describe())

    return df


def save_data(df, path):
    """Save cleaned dataset"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nClean data saved at: {path}")


def main():
    df = load_data(RAW_DATA_PATH)

    if df is not None:
        df_clean = clean_data(df)
        save_data(df_clean, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()