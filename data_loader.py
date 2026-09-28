"""
Data loader module — loads and caches all datasets.
"""

import pandas as pd
import numpy as np
import json
import glob
import os
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))

# Light theme Plotly config
PLOTLY_THEME = {
    "paper_bgcolor": "rgba(255,255,255,0)",
    "plot_bgcolor":  "rgba(248,250,252,0.6)",
    "font_color":    "#1e293b",
    "gridcolor":     "rgba(148,163,184,0.25)",
    "zerolinecolor": "rgba(148,163,184,0.4)",
    "axis_color":    "#64748b",
}
# Backward compatibility alias
PLOTLY_DARK = PLOTLY_THEME

# colour palette
C_BLUE   = "#2563eb"
C_INDIGO = "#6366f1"
C_VIOLET = "#8b5cf6"
C_TEAL   = "#0d9488"
C_GREEN  = "#16a34a"
C_AMBER  = "#d97706"
C_ROSE   = "#e11d48"


# ──────────────────────────────────────────────
#  LOAD WEATHER
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_weather():
    path = os.path.join(BASE, "weather_nyc.csv")
    df = pd.read_csv(path, skiprows=3)
    df.columns = ["time", "temperature", "precipitation"]
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["month_name"] = df["time"].dt.strftime("%b")
    df["is_rainy"] = df["precipitation"] > 0.1
    return df


# ──────────────────────────────────────────────
#  LOAD STATIONS
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_stations():
    path = os.path.join(BASE, "station_information.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    stations = raw["data"]["stations"]
    df = pd.DataFrame(stations)
    cols = ["station_id", "name", "lon", "lat", "region_id", "capacity",
            "has_kiosk", "station_type", "rental_methods"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["rental_methods"] = df["rental_methods"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0).astype(int)
    return df


# ──────────────────────────────────────────────
#  LOAD TRIPS (sampled / parquet cached)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_trips(max_rows_per_file: int = 20_000):
    parquet_path = os.path.join(BASE, "sampled_trips.parquet")
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass

    all_dfs = []
    month_dirs = {
        "202606": os.path.join(BASE, "202606-citibike-tripdata"),
        "202607": os.path.join(BASE, "202607-citibike-tripdata"),
        "202608": os.path.join(BASE, "202608-citibike-tripdata"),
    }

    dtype_map = {
        "ride_id": "str",
        "rideable_type": "category",
        "start_station_name": "str",
        "start_station_id": "str",
        "end_station_name": "str",
        "end_station_id": "str",
        "member_casual": "category",
    }

    for month_key, month_dir in month_dirs.items():
        csv_files = sorted(glob.glob(os.path.join(month_dir, "*.csv")))
        for fpath in csv_files:
            try:
                chunk = pd.read_csv(
                    fpath,
                    dtype=dtype_map,
                    parse_dates=["started_at", "ended_at"],
                    nrows=max_rows_per_file,
                    on_bad_lines="skip",
                )
                all_dfs.append(chunk)
            except Exception:
                continue

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    df["duration_min"] = (
        (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    )
    df = df[(df["duration_min"] > 1) & (df["duration_min"] < 120)]

    df["hour"]        = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.day_name()
    df["date"]        = df["started_at"].dt.date
    df["month"]       = df["started_at"].dt.month
    df["month_name"]  = df["started_at"].dt.strftime("%b")
    df["is_weekend"]  = df["started_at"].dt.dayofweek >= 5

    df = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
    lat1 = np.radians(df["start_lat"]); lon1 = np.radians(df["start_lng"])
    lat2 = np.radians(df["end_lat"]);   lon2 = np.radians(df["end_lng"])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    df = df[(df["distance_km"] > 0.05) & (df["distance_km"] < 30)]

    df["speed_kmh"] = df["distance_km"] / (df["duration_min"] / 60)
    df = df[df["speed_kmh"] < 45]

    return df


# ──────────────────────────────────────────────
#  MERGE TRIPS + WEATHER
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_trips_with_weather():
    trips   = load_trips()
    weather = load_weather()

    trips["dt_hour"] = trips["started_at"].dt.floor("h")
    weather_hourly   = weather.set_index("time")[["temperature", "precipitation", "is_rainy"]]

    trips = trips.merge(
        weather_hourly,
        left_on="dt_hour",
        right_index=True,
        how="left",
    )
    return trips
