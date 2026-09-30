import streamlit as st
import importlib

st.set_page_config(
    page_title="CitiBike NYC — Big Data Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pages_content import overview, demand_analysis, weather_impact, station_analysis, ml_prediction
from pages_content import graph_analytics

# Force fresh reload of all page submodules on every page view
importlib.reload(overview)
importlib.reload(demand_analysis)
importlib.reload(weather_impact)
importlib.reload(station_analysis)
importlib.reload(ml_prediction)
importlib.reload(graph_analytics)

PAGES = {
    "Overview":          overview,
    "Demand Analysis":   demand_analysis,
    "Weather Impact":    weather_impact,
    "Station Analysis":  station_analysis,
    "ML Prediction":     ml_prediction,
    "Graph Analytics":   graph_analytics,
}

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App background ── */
    .stApp { background: #f8fafc; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.04);
    }
    section[data-testid="stSidebar"] .stRadio label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 10px 14px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #475569;
        font-weight: 500;
        font-size: 0.9rem;
        width: 100%;
        display: block;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #eff6ff;
        border-color: #bfdbfe;
        color: #1d4ed8;
        transform: translateX(3px);
    }

    /* ── Column stretch alignment ── */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
    }
    div[data-testid="column"] > div {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }
    div[data-testid="column"] [data-testid="stVerticalBlock"] {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }
    div[data-testid="column"] [data-testid="stElementContainer"] {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }
    div[data-testid="column"] [data-testid="stMarkdownContainer"] {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }

    /* ── Cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 14px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        transition: all 0.25s ease;
        height: 100% !important;
        min-height: 112px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }
    .metric-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 18px rgba(37,99,235,0.12);
        transform: translateY(-2px);
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1d4ed8;
        line-height: 1.15;
        margin-bottom: 6px;
        white-space: nowrap;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        font-weight: 600;
        white-space: nowrap;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        border-left: 4px solid #2563eb;
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    /* ── Insight cards ── */
    .insight-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        box-sizing: border-box !important;
    }
    .insight-card:hover {
        box-shadow: 0 4px 14px rgba(37,99,235,0.1);
        border-left-color: #1d4ed8;
    }
    .insight-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.98rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 10px;
        min-height: 48px;
        display: flex;
        align-items: flex-start;
        line-height: 1.35;
    }
    .insight-body {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.65;
        flex: 1 1 auto;
    }

    /* ── Hero ── */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 28px;
    }

    /* ── Charts ── */
    div[data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        background: #ffffff;
    }

    hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0 22px 0;'>
            <div style='font-family:Space Grotesk,sans-serif; font-size:1.4rem;
                        font-weight:700; color:#1d4ed8;'>CitiBike NYC</div>
            <div style='font-size:0.68rem; color:#94a3b8; text-transform:uppercase;
                        letter-spacing:2px; margin-top:3px;'>Big Data Analytics</div>
        </div>
        <hr style='border-top:1px solid #e2e8f0; margin:0 0 16px 0;'/>
        """, unsafe_allow_html=True)

        page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                    padding:14px; font-size:0.75rem; color:#64748b; line-height:1.8;'>
            <div style='color:#1d4ed8; font-weight:600; font-size:0.75rem; margin-bottom:6px;
                        text-transform:uppercase; letter-spacing:1px;'>Dataset Info</div>
            <div>Period: Jun to Aug 2026</div>
            <div>Trip Records: ~16M rows</div>
            <div>Stations: 2,100+</div>
            <div>Variety: CSV + JSON</div>
            <div>Volume: ~3.3 GB</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.67rem; color:#cbd5e1; text-align:center; padding-bottom:8px;'>
            COMP8035041 — Big Data Analytics<br/>Final Project 2026
        </div>
        """, unsafe_allow_html=True)

    return page


def main():
    load_css()
    page = sidebar()
    PAGES[page].render()

if __name__ == "__main__":
    main()
