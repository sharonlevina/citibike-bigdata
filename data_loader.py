"""
Shared theme, colors, and data loaders for all page modules.
Works seamlessly both locally and when deployed on Streamlit Community Cloud.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
# Check current directory and parent directory for data assets
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "data_lake")):
    BASE_DIR = CURRENT_DIR
elif os.path.exists(os.path.join(os.path.dirname(CURRENT_DIR), "data_lake")):
    BASE_DIR = os.path.dirname(CURRENT_DIR)
else:
    BASE_DIR = CURRENT_DIR

SQL_DIR    = os.path.join(BASE_DIR, "data_lake", "sql_results")
GOLD_DIR   = os.path.join(BASE_DIR, "data_lake", "gold")
ML_DIR     = os.path.join(BASE_DIR, "ml_results")
GRAPH_DIR  = os.path.join(BASE_DIR, "graph_results")

# ─── Design tokens ───────────────────────────────────────────────────────────
C_BLUE    = "#2563eb"
C_INDIGO  = "#4f46e5"
C_VIOLET  = "#7c3aed"
C_TEAL    = "#0d9488"
C_AMBER   = "#d97706"
C_ROSE    = "#e11d48"
C_GREEN   = "#16a34a"
C_SLATE   = "#475569"

PLOTLY_THEME = dict(
    paper_bgcolor = "#ffffff",
    plot_bgcolor  = "#ffffff",
    font_color    = "#1e293b",
    axis_color    = "#64748b",
    gridcolor     = "#f1f5f9",
    zerolinecolor = "#e2e8f0",
)

# ─── Common axis / layout helpers ────────────────────────────────────────────
def ax(title="", **kw):
    T = PLOTLY_THEME
    return dict(
        title=dict(text=title, font=dict(size=11, color=T["axis_color"])),
        gridcolor=T["gridcolor"],
        zerolinecolor=T["zerolinecolor"],
        color=T["axis_color"],
        showgrid=True,
        **kw,
    )

def base_layout(t=30, h=320, **kw):
    T = PLOTLY_THEME
    default_margin = dict(l=15, r=15, t=t, b=25)
    margin = kw.pop("margin", default_margin)
    d = dict(
        paper_bgcolor = T["paper_bgcolor"],
        plot_bgcolor  = T["plot_bgcolor"],
        font=dict(color=T["font_color"], family="Inter, sans-serif", size=11),
        margin=margin,
        height=h,
    )
    d.update(kw)
    return d

# ─── Data loaders ────────────────────────────────────────────────────────────
def _read_parquet(path):
    """Read a Spark-written parquet directory or single file."""
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def load_rush_hour():
    return _read_parquet(os.path.join(SQL_DIR, "q1_rush_hour"))

def load_weekend():
    return _read_parquet(os.path.join(SQL_DIR, "q2_weekend_weekday"))

def load_top_routes():
    df = _read_parquet(os.path.join(SQL_DIR, "q3_top_routes"))
    if df.empty:
        df = _read_parquet(os.path.join(GOLD_DIR, "gold_top_routes"))
    return df

def load_member_behavior():
    return _read_parquet(os.path.join(SQL_DIR, "q4_member_behavior"))

def load_station_flow():
    return _read_parquet(os.path.join(SQL_DIR, "q5_station_flow"))

def load_monthly_growth():
    return _read_parquet(os.path.join(SQL_DIR, "q6_monthly_growth"))

def load_top_start_stations():
    return _read_parquet(os.path.join(GOLD_DIR, "gold_top_start_stations"))

def load_top_end_stations():
    return _read_parquet(os.path.join(GOLD_DIR, "gold_top_end_stations"))

def load_kmeans_clusters():
    return _read_parquet(os.path.join(ML_DIR, "kmeans_station_clusters"))

def load_kmeans_metrics():
    path = os.path.join(ML_DIR, "kmeans_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_rf_metrics():
    path = os.path.join(ML_DIR, "rf_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_feature_importance():
    path = os.path.join(ML_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_rf_model():
    path = os.path.join(ML_DIR, "random_forest_model.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def load_graph_summary():
    path = os.path.join(GRAPH_DIR, "graph_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_centrality():
    path = os.path.join(GRAPH_DIR, "node_centrality_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()
