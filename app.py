import streamlit as st
import importlib

st.set_page_config(
    page_title="CitiBike NYC — Big Data Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pages_content import overview, demand_analysis, weather_impact, station_analysis, ml_prediction

# Force fresh reload of all page submodules on every page view
importlib.reload(overview)
importlib.reload(demand_analysis)
importlib.reload(weather_impact)
importlib.reload(station_analysis)
importlib.reload(ml_prediction)

PAGES = {
    "Overview":          overview,
    "Demand Analysis":   demand_analysis,
    "Weather Impact":    weather_impact,
    "Station Analysis":  station_analysis,
    "ML Prediction":     ml_prediction,
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

    /* ── Cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 18px rgba(37,99,235,0.12);
        transform: translateY(-2px);
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1d4ed8;
        line-height: 1.1;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
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
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .insight-card:hover {
        box-shadow: 0 4px 14px rgba(37,99,235,0.1);
        border-left-color: #1d4ed8;
    }
    .insight-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #1d4ed8;
        margin-bottom: 7px;
    }
    .insight-body {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.7;
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
