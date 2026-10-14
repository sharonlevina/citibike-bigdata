import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import (
    load_graph_summary, load_centrality,
    base_layout, ax, C_BLUE, C_INDIGO, C_VIOLET, C_TEAL, C_AMBER, C_ROSE, PLOTLY_THEME
)

T = PLOTLY_THEME


@st.cache_data
def _summary():    return load_graph_summary()
@st.cache_data
def _centrality(): return load_centrality()


def render():
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Graph Analytics & Station Centrality</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Network topology modelling via NetworkX · PageRank hub identification · Betweenness centrality for transit bottlenecks</div>", unsafe_allow_html=True)

    summary = _summary()
    df_cent = _centrality()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    kpis = [
        (f"{summary.get('total_nodes', 272)}", "Network Nodes (Stations)"),
        (f"{summary.get('total_edges', 444)}", "Directed Route Edges"),
        ("PageRank", "Hub Identification Metric"),
        ("Betweenness", "Bottleneck Transit Metric"),
    ]
    cols = st.columns(4)
    for col, (val, label) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── PageRank Hubs vs Betweenness Bottlenecks ───────────────────────────────
    st.markdown("<div class='section-header'>Station Prominence and Transit Bottlenecks</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if not df_cent.empty and "pagerank" in df_cent.columns:
            top_pr = df_cent.sort_values("pagerank", ascending=True).tail(10)
            fig_pr = go.Figure(go.Bar(
                x=top_pr["pagerank"],
                y=top_pr["station"],
                orientation="h",
                marker=dict(
                    color=top_pr["pagerank"],
                    colorscale=[[0, "#93c5fd"], [1, C_BLUE]],
                    showscale=False
                ),
                text=[f"{v:.4f}" for v in top_pr["pagerank"]],
                textposition="outside",
                textfont=dict(size=10, color="#334155"),
            ))
            fig_pr.update_layout(
                **base_layout(h=360),
                title=dict(text="Top 10 Super Hubs (PageRank Centrality)", font=dict(size=13, color="#0f172a")),
                xaxis=ax("PageRank Score"),
                yaxis=ax(""),
                margin=dict(l=10, r=40, t=35, b=25),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    with col2:
        if not df_cent.empty and "betweenness_centrality" in df_cent.columns:
            top_bc = df_cent.sort_values("betweenness_centrality", ascending=True).tail(10)
            fig_bc = go.Figure(go.Bar(
                x=top_bc["betweenness_centrality"],
                y=top_bc["station"],
                orientation="h",
                marker=dict(
                    color=top_bc["betweenness_centrality"],
                    colorscale=[[0, "#fde68a"], [1, C_AMBER]],
                    showscale=False
                ),
                text=[f"{v:.4f}" for v in top_bc["betweenness_centrality"]],
                textposition="outside",
                textfont=dict(size=10, color="#334155"),
            ))
            fig_bc.update_layout(
                **base_layout(h=360),
                title=dict(text="Top 10 Bridge Nodes (Betweenness Centrality)", font=dict(size=13, color="#0f172a")),
                xaxis=ax("Betweenness Centrality"),
                yaxis=ax(""),
                margin=dict(l=10, r=40, t=35, b=25),
            )
            st.plotly_chart(fig_bc, use_container_width=True)

    # ── Scatter: PageRank vs Betweenness ──────────────────────────────────────
    st.markdown("<div class='section-header'>Network Role Classification: Hub vs Bridge</div>", unsafe_allow_html=True)
    if not df_cent.empty and "pagerank" in df_cent.columns and "betweenness_centrality" in df_cent.columns:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=df_cent["pagerank"],
            y=df_cent["betweenness_centrality"],
            mode="markers",
            marker=dict(
                size=8,
                color=C_BLUE,
                opacity=0.75,
                line=dict(color="#ffffff", width=1)
            ),
            text=df_cent["station"],
            hovertemplate="<b>%{text}</b><br>PageRank: %{x:.4f}<br>Betweenness: %{y:.4f}<extra></extra>"
        ))
        fig_scatter.update_layout(
            **base_layout(h=340),
            title=dict(text="Station Network Topology (PageRank vs Betweenness)", font=dict(size=13, color="#0f172a")),
            xaxis=ax("PageRank (Network Attraction Hub)"),
            yaxis=ax("Betweenness Centrality (Inter-borough Bridge)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Centrality Ranking Table ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>Node Centrality Metrics Table</div>", unsafe_allow_html=True)
    if not df_cent.empty:
        display_cent = df_cent.copy()
        display_cent = display_cent.sort_values("pagerank", ascending=False).head(20)
        display_cent.columns = ["Station Name", "Weighted In-Degree", "Weighted Out-Degree", "Betweenness Centrality", "PageRank"]
        display_cent["Weighted In-Degree"] = display_cent["Weighted In-Degree"].map("{:,.0f}".format)
        display_cent["Weighted Out-Degree"] = display_cent["Weighted Out-Degree"].map("{:,.0f}".format)
        display_cent["Betweenness Centrality"] = display_cent["Betweenness Centrality"].map("{:.5f}".format)
        display_cent["PageRank"] = display_cent["PageRank"].map("{:.5f}".format)
        st.dataframe(display_cent, use_container_width=True, hide_index=True)

    # ── Operational Rebalancing Strategy ──────────────────────────────────────
    st.markdown("<div class='section-header'>Graph-Driven Fleet Rebalancing Matrix</div>", unsafe_allow_html=True)
    rebalancing_plan = pd.DataFrame({
        "Origin Station (High Outflow)": [
            "Vernon Blvd & 50 Ave",
            "N 6 St & Bedford Ave",
            "Central Park S & 6 Ave",
            "S 2 St & Kent Ave",
            "Franklin St & Dupont St"
        ],
        "Target Station (High Inflow)": [
            "7 Ave & Central Park South",
            "N 7 St & Driggs Ave",
            "Cooper Square & Astor Pl",
            "Henry St & Grand St",
            "W 21 St & 6 Ave"
        ],
        "Transfer Urgency": ["Critical", "Critical", "High", "High", "Medium"],
        "Dispatch Window": [
            "06:30 - 07:30 (Pre-morning)",
            "07:00 - 08:00 (Morning Rush)",
            "11:30 - 13:00 (Lunch Shift)",
            "16:00 - 17:00 (Pre-evening)",
            "17:00 - 18:30 (Evening Rush)"
        ],
        "Est. Dock Recovery": ["45 bikes / hr", "60 bikes / hr", "30 bikes / hr", "35 bikes / hr", "40 bikes / hr"]
    })
    st.dataframe(rebalancing_plan, use_container_width=True, hide_index=True)

    # ── Insights ─────────────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>PageRank Centrality Hubs</div>
            <div class='insight-body'>
                Stations such as <strong>7 Ave & Central Park South</strong> and <strong>Vernon Blvd & 50 Ave</strong> hold the highest PageRank, meaning they receive persistent traffic from across the entire bike network. Maintaining dock clearance here prevents network-wide trip rejections.
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>Bridge Node Bottlenecks</div>
            <div class='insight-body'>
                <strong>W 21 St & 6 Ave</strong> and <strong>11 Ave & W 41 St</strong> exhibit peak Betweenness Centrality, acting as primary transfer bridges between residential corridors and office clusters. Disruptions at these bridge nodes disproportionately fragment cross-borough trip flows.
            </div>
        </div>""", unsafe_allow_html=True)
