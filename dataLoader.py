import os
import pandas as pd
import json
import numpy as np

def load_electricity_data(folder_path):
    electricity_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), 'r') as f:
                try:
                    data = json.load(f)["response"]["data"]
                    electricity_data.extend(data)
                except KeyError:
                    print(f"Skipping file {file} due to unexpected format")
    electricity_df = pd.DataFrame(electricity_data)
    electricity_df["period"] = pd.to_datetime(electricity_df["period"], errors='coerce')
    electricity_df["value"] = electricity_df["value"].astype(str).str.extract(r'([0-9]+\.?[0-9]*)')[0]
    electricity_df["value"] = pd.to_numeric(electricity_df["value"], errors='coerce')
    electricity_df = electricity_df.rename(columns={"period": "datetime", "value": "demand_mwh", "subba-name" : "Province"})
    electricity_df["datetime"] = pd.to_datetime(electricity_df["datetime"], errors='coerce', utc=True)
    return electricity_df.dropna()

def load_weather_data(folder_path):
    weather_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            df.rename(columns=lambda x: x.strip(), inplace=True)
            df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True)
            weather_data.append(df)
    weather_df = pd.concat(weather_data, ignore_index=True)
    weather_df = weather_df.rename(columns={"date": "datetime", "temperature_2m": "temperature"})
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], errors='coerce', utc=True)
    return weather_df.dropna()
 
def merge_data(folder_path = "raw"):
    electricity_folder = os.path.join(folder_path, "electricity_raw_data")
    weather_folder = os.path.join(folder_path, "weather_raw_data")

    # Load data
    electricity_df = load_electricity_data(electricity_folder)
    weather_df = load_weather_data(weather_folder)
    
    # Merge data
    data = pd.merge(electricity_df, weather_df, on="datetime", how="inner")
    data["demand_mwh"] = pd.to_numeric(data["demand_mwh"], errors='coerce')
    data.sort_values(by="datetime", inplace=True)
    
    return data
