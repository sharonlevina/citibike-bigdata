import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import (
    load_monthly_growth, load_rush_hour, load_station_flow,
    base_layout, ax, C_BLUE, C_INDIGO, C_VIOLET, PLOTLY_THEME
)

T = PLOTLY_THEME


@st.cache_data
def _monthly():   return load_monthly_growth()
@st.cache_data
def _rush():      return load_rush_hour()
@st.cache_data
def _flow():      return load_station_flow()


def render():
    st.markdown("<div class='hero-title'>Citi Bike NYC — Big Data Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Medallion Lakehouse Architecture · Apache Spark · NetworkX Graph Analytics · Machine Learning</div>", unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    kpis = [
        ("22.8M+",   "Total Trip Records"),
        ("2,416",    "Active Stations"),
        ("13.4 min", "Avg Trip Duration"),
        ("0.4162",   "K-Means Silhouette"),
        ("5.85 min", "RF Prediction MAE"),
    ]
    cols = st.columns(5)
    for col, (val, label) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Monthly Volume ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Monthly Trip Volume Trend</div>", unsafe_allow_html=True)
    df_mg = _monthly()
    if not df_mg.empty:
        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        df_mg = df_mg.sort_values("month")
        df_mg["month_name"] = df_mg["month"].map(month_map)

        col1, col2 = st.columns([3, 2])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_mg["month_name"], y=df_mg["member_trips"],
                name="Member", marker_color=C_BLUE, marker_line_width=0,
            ))
            fig.add_trace(go.Bar(
                x=df_mg["month_name"], y=df_mg["casual_trips"],
                name="Casual", marker_color=C_VIOLET, marker_line_width=0,
            ))
            fig.update_layout(
                **base_layout(h=300),
                barmode="stack",
                xaxis=ax("Month"),
                yaxis=ax("Total Trips"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(size=11, color="#475569")),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure(go.Pie(
                labels=["Member", "Casual"],
                values=[df_mg["member_trips"].sum(), df_mg["casual_trips"].sum()],
                hole=0.60,
                marker=dict(colors=[C_BLUE, C_VIOLET], line=dict(color="#fff", width=2)),
                textfont=dict(color="#1e293b"),
            ))
            pct = df_mg["member_trips"].sum() / (df_mg["member_trips"].sum() + df_mg["casual_trips"].sum()) * 100
            fig2.add_annotation(text=f"<b>{pct:.1f}%</b><br>Members",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(size=16, color=C_BLUE, family="Space Grotesk"))
            fig2.update_layout(**base_layout(h=300),
                               legend=dict(font=dict(color="#475569")))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Three Business Insights ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Three Core Business Insights</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    insights = [
        ("Insight 1: Peak-Hour Commute Demand",
         "Trips spike at 08:00 and 17:00–18:00 on weekdays, confirming Citi Bike's role "
         "as a first/last-mile transit option for NYC office commuters. Fleet rebalancing "
         "should prioritise business districts before 07:30 each morning."),
        ("Insight 2: Member vs Casual Behaviour Gap",
         "Members (75%+ share) take shorter, purpose-driven trips averaging 11–13 minutes. "
         "Casual riders average 20+ minutes with leisure patterns peaking on weekends. "
         "Distinct pricing and capacity strategies are needed for each segment."),
        ("Insight 3: Station Flow Imbalance",
         "A subset of stations consistently emit more bikes than they receive (Source), "
         "while others accumulate excess supply (Sink). Graph-based rebalancing routes "
         "derived from PageRank analysis can reduce dead-zones by an estimated 30%."),
    ]
    for col, (title, body) in zip([c1, c2, c3], insights):
        with col:
            st.markdown(f"""
            <div class='insight-card' style='min-height:230px;'>
                <div class='insight-title'>{title}</div>
                <div class='insight-body'>{body}</div>
            </div>""", unsafe_allow_html=True)

    # ── Big Data 5V ───────────────────────────────────────────────────────────
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Big Data Characteristics — 5V Framework</div>", unsafe_allow_html=True)
    rows = [
        ("Volume",   "22.8M+ trip records across multiple months, stored as Parquet Snappy in a Medallion Lakehouse (Bronze / Silver / Gold)."),
        ("Variety",  "Three source types: tabular CSV trips, structured JSON station registry, and time-series weather data from Open-Meteo API."),
        ("Velocity", "Citi Bike publishes live GBFS feeds every 30 seconds; historical batch represents millions of events per month."),
        ("Veracity", "Cleaning pipeline removes sub-60 s ghost rides, trips over 24 h, missing station names, and out-of-bounds GPS coordinates."),
        ("Value",    "ML clustering and graph PageRank enable proactive fleet rebalancing and reduce operational cost through data-driven decisions."),
    ]
    for label, text in rows:
        st.markdown(f"""
        <div style='background:#fff; border:1px solid #e2e8f0; border-left:4px solid {C_BLUE};
                    border-radius:10px; padding:13px 18px; margin-bottom:8px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);'>
            <span style='color:{C_BLUE}; font-weight:600; font-size:0.85rem;'>{label}:</span>
            <span style='color:#475569; font-size:0.85rem; margin-left:6px;'>{text}</span>
        </div>""", unsafe_allow_html=True)
