import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import (
    load_top_start_stations, load_top_end_stations,
    load_station_flow, load_top_routes,
    base_layout, ax, C_BLUE, C_INDIGO, C_VIOLET, C_TEAL, C_AMBER, C_ROSE, PLOTLY_THEME
)

T = PLOTLY_THEME


@st.cache_data
def _top_start(): return load_top_start_stations()
@st.cache_data
def _top_end():   return load_top_end_stations()
@st.cache_data
def _flow():      return load_station_flow()
@st.cache_data
def _routes():    return load_top_routes()


def render():
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Station Network & Flow Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Station-level departure/arrival volumes, net flow bottlenecks, and high-frequency transit corridors</div>", unsafe_allow_html=True)

    # ── Top Departures & Arrivals ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Top Busiest Stations by Volume</div>", unsafe_allow_html=True)
    df_start = _top_start()
    df_end   = _top_end()

    col1, col2 = st.columns(2)

    with col1:
        if not df_start.empty:
            df_s15 = df_start.head(12).sort_values("total_departures", ascending=True)
            fig_start = go.Figure(go.Bar(
                x=df_s15["total_departures"],
                y=df_s15["start_station_name"],
                orientation="h",
                marker=dict(
                    color=df_s15["total_departures"],
                    colorscale=[[0, "#93c5fd"], [1, C_BLUE]],
                    showscale=False
                ),
                text=[f"{v:,}" for v in df_s15["total_departures"]],
                textposition="outside",
                textfont=dict(size=10, color="#334155"),
            ))
            fig_start.update_layout(
                **base_layout(h=380),
                title=dict(text="Top 12 Origin Stations (Departures)", font=dict(size=13, color="#0f172a")),
                xaxis=ax("Total Departures"),
                yaxis=ax(""),
                margin=dict(l=10, r=40, t=35, b=25),
            )
            st.plotly_chart(fig_start, use_container_width=True)

    with col2:
        if not df_end.empty:
            df_e15 = df_end.head(12).sort_values("total_arrivals", ascending=True)
            fig_end = go.Figure(go.Bar(
                x=df_e15["total_arrivals"],
                y=df_e15["end_station_name"],
                orientation="h",
                marker=dict(
                    color=df_e15["total_arrivals"],
                    colorscale=[[0, "#c4b5fd"], [1, C_VIOLET]],
                    showscale=False
                ),
                text=[f"{v:,}" for v in df_e15["total_arrivals"]],
                textposition="outside",
                textfont=dict(size=10, color="#334155"),
            ))
            fig_end.update_layout(
                **base_layout(h=380),
                title=dict(text="Top 12 Destination Stations (Arrivals)", font=dict(size=13, color="#0f172a")),
                xaxis=ax("Total Arrivals"),
                yaxis=ax(""),
                margin=dict(l=10, r=40, t=35, b=25),
            )
            st.plotly_chart(fig_end, use_container_width=True)

    # ── Flow Imbalance Analysis (Source vs Sink) ──────────────────────────────
    st.markdown("<div class='section-header'>Station Flow Imbalance (Supply Deficit vs Accumulation)</div>", unsafe_allow_html=True)
    df_flow = _flow()
    if not df_flow.empty:
        # Sort by net flow
        df_deficit = df_flow.sort_values("net_flow", ascending=True).head(8)
        df_surplus = df_flow.sort_values("net_flow", ascending=False).head(8)

        cf1, cf2 = st.columns(2)
        with cf1:
            fig_def = go.Figure(go.Bar(
                x=df_deficit["net_flow"],
                y=df_deficit["station"],
                orientation="h",
                marker_color=C_ROSE,
                text=[f"{v:,}" for v in df_deficit["net_flow"]],
                textposition="inside",
                textfont=dict(size=10, color="#ffffff"),
            ))
            fig_def.update_layout(
                **base_layout(h=300),
                title=dict(text="Top Deficit Stations (Outflow > Inflow)", font=dict(size=13, color="#991b1b")),
                xaxis=ax("Net Flow (Trips)"),
                yaxis=ax(""),
                margin=dict(l=10, r=20, t=35, b=25),
            )
            st.plotly_chart(fig_def, use_container_width=True)

        with cf2:
            fig_sur = go.Figure(go.Bar(
                x=df_surplus["net_flow"],
                y=df_surplus["station"],
                orientation="h",
                marker_color=C_TEAL,
                text=[f"+{v:,}" for v in df_surplus["net_flow"]],
                textposition="inside",
                textfont=dict(size=10, color="#ffffff"),
            ))
            fig_sur.update_layout(
                **base_layout(h=300),
                title=dict(text="Top Surplus Stations (Inflow > Outflow)", font=dict(size=13, color="#115e59")),
                xaxis=ax("Net Flow (Trips)"),
                yaxis=ax(""),
                margin=dict(l=10, r=20, t=35, b=25),
            )
            st.plotly_chart(fig_sur, use_container_width=True)

    # ── Interactive Station Geospatial Map ────────────────────────────────────
    st.markdown("<div class='section-header'>Geospatial Station Network Map</div>", unsafe_allow_html=True)
    if not df_start.empty and "start_lat" in df_start.columns and "start_lng" in df_start.columns:
        map_df = df_start.dropna(subset=["start_lat", "start_lng"]).head(100).copy()
        
        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=map_df["start_lat"],
            lon=map_df["start_lng"],
            mode="markers",
            marker=dict(
                size=np.clip(map_df["total_departures"] / map_df["total_departures"].max() * 24, 6, 26),
                color=map_df["total_departures"],
                colorscale=[[0, "#60a5fa"], [0.5, "#2563eb"], [1, "#1e3a8a"]],
                showscale=True,
                colorbar=dict(title=dict(text="Departures", font=dict(size=11)), thickness=12, len=0.8),
                opacity=0.85
            ),
            text=map_df["start_station_name"],
            hovertemplate="<b>%{text}</b><br>Departures: %{marker.color:,}<extra></extra>"
        ))
        
        fig_map.update_layout(
            mapbox=dict(
                style="carto-positron",
                zoom=11.2,
                center=dict(lat=40.745, lon=-73.985),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Top Transit Routes ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Top 10 High-Frequency Transit Corridors</div>", unsafe_allow_html=True)
    df_r = _routes()
    if not df_r.empty:
        df_r10 = df_r.head(10).copy()
        df_r10["Route"] = df_r10["start_station_name"] + " → " + df_r10["end_station_name"]
        
        cols_show = ["Route", "total_trips", "avg_duration_min"]
        if "trip_type" in df_r10.columns:
            cols_show.append("trip_type")
            
        display_routes = df_r10[cols_show].copy()
        display_routes.columns = ["Transit Corridor", "Total Trips", "Avg Duration (min)"] + (["Trip Type"] if "trip_type" in df_r10.columns else [])
        display_routes["Total Trips"] = display_routes["Total Trips"].map("{:,.0f}".format)
        display_routes["Avg Duration (min)"] = display_routes["Avg Duration (min)"].map("{:.1f}".format)
        
        st.dataframe(display_routes, use_container_width=True, hide_index=True)

    # ── Insights ─────────────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>Asymmetric Commuter Flows</div>
            <div class='insight-body'>
                High-density residential areas (such as Greenpoint and Williamsburg) experience severe morning <strong>bike depletion</strong> (net negative flow), while commercial hubs around Grand Central and Midtown experience <strong>dock shortages</strong> (net positive flow) as commuters arrive.
            </div>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>Short-Hop High Volume Corridors</div>
            <div class='insight-body'>
                The highest-traffic OD pairs have average durations between 5 and 11 minutes, connecting subway hubs to office complexes. Implementing dedicated shuttle van replenishment between these adjacent node pairs prevents morning gridlock.
            </div>
        </div>""", unsafe_allow_html=True)
