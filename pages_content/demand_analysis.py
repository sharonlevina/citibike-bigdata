import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_trips, PLOTLY_THEME, C_BLUE, C_INDIGO, C_VIOLET, C_TEAL, C_AMBER

T = PLOTLY_THEME
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def ax(title="", **kw):
    return dict(
        title=dict(text=title, font=dict(size=11, color=T["axis_color"])),
        gridcolor=T["gridcolor"],
        zerolinecolor=T["zerolinecolor"],
        color=T["axis_color"],
        showgrid=True,
        **kw,
    )


def base_layout(t=30, h=330, **kw):
    d = dict(
        paper_bgcolor=T["paper_bgcolor"],
        plot_bgcolor=T["plot_bgcolor"],
        font=dict(color=T["font_color"], family="Inter, sans-serif", size=11),
        margin=dict(l=15, r=15, t=t, b=25),
        height=h,
    )
    d.update(kw)
    return d


def render():
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Demand Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Hourly, daily, and monthly demand patterns across rider categories and bicycle propulsion types</div>", unsafe_allow_html=True)

    with st.spinner("Loading trip data..."):
        trips = load_trips()

    if trips.empty:
        st.error("No trip data found.")
        return

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander("Filter Data Views", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            rider_filter = st.multiselect("Rider Type", ["member", "casual"], default=["member", "casual"])
        with col2:
            bike_filter = st.multiselect("Bike Type", trips["rideable_type"].unique().tolist(),
                                          default=trips["rideable_type"].unique().tolist())
        with col3:
            month_filter = st.multiselect("Month", ["Jun", "Jul", "Aug"], default=["Jun", "Jul", "Aug"])

    df = trips[
        trips["member_casual"].isin(rider_filter) &
        trips["rideable_type"].isin(bike_filter) &
        trips["month_name"].isin(month_filter)
    ]
    if df.empty:
        st.warning("No data matches current filters.")
        return

    # ── 1. Hourly Pattern ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Hourly Ride Distribution (24-Hour Profile)</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#0f172a; margin-bottom:8px;'>Hourly Volume: Subscribers vs Casual Cyclists</div>", unsafe_allow_html=True)

    hourly = df.groupby(["hour", "member_casual"], observed=False).size().reset_index(name="count")
    colors = {"member": C_BLUE, "casual": C_VIOLET}

    fig = go.Figure()
    for rider in ["member", "casual"]:
        sub = hourly[hourly["member_casual"] == rider]
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["count"],
            name=f"{rider.capitalize()}s",
            mode="lines+markers",
            line=dict(color=colors[rider], width=2.5, shape="spline"),
            marker=dict(size=6, color=colors[rider]),
            fill="tozeroy",
            fillcolor=colors[rider] + "15",
        ))
    fig.update_layout(
        **base_layout(t=35, h=320),
        xaxis=ax("Hour of Day (00:00 - 23:00)", tickvals=list(range(0, 24, 2))),
        yaxis=ax("Sample Rides"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11, color="#475569"),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='insight-card'>
        <div class='insight-title'>Diurnal Commute Dynamics</div>
        <div class='insight-body'>
            <strong>Subscribers (Members):</strong> Exhibit strong twin commute spikes at <strong>08:00</strong> and <strong>17:00–18:00</strong>, indicating daily office commuting along arterial bridges and financial corridors.<br/>
            <strong>Casual Users:</strong> Exhibit a smooth, single-peaked leisure curve swelling between <strong>13:00 and 16:00</strong>, reflecting tourist outings and recreational cycling along Central Park and waterfront greenways.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── 2. Day of Week Patterns ───────────────────────────────────────────
    st.markdown("<div class='section-header'>Day of Week Patterns</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#0f172a; margin-bottom:6px;'>Rides by Day of Week</div>", unsafe_allow_html=True)
        dow = df.groupby(["day_of_week", "member_casual"], observed=False).size().reset_index(name="count")
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=DAY_ORDER, ordered=True)
        dow = dow.sort_values("day_of_week")

        max_dow_count = max(1, dow["count"].max())

        fig2 = go.Figure()
        for rider in ["member", "casual"]:
            sub = dow[dow["member_casual"] == rider]
            fig2.add_trace(go.Bar(
                x=[d[:3] for d in sub["day_of_week"]],
                y=sub["count"],
                name=f"{rider.capitalize()}s",
                marker_color=colors[rider],
                marker_line=dict(width=0),
            ))
        fig2.update_layout(
            **base_layout(t=35, h=320),
            barmode="group",
            xaxis=ax("Day of Week"),
            yaxis=ax("Ride Count", range=[0, max_dow_count * 1.2]),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11, color="#475569"),
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#0f172a; margin-bottom:6px;'>Avg Ride Duration by Day (Minutes)</div>", unsafe_allow_html=True)
        dur_day = df.groupby(["day_of_week", "member_casual"], observed=False)["duration_min"].mean().reset_index()
        dur_day["day_of_week"] = pd.Categorical(dur_day["day_of_week"], categories=DAY_ORDER, ordered=True)
        dur_day = dur_day.sort_values("day_of_week")

        max_dur = max(1.0, dur_day["duration_min"].max())

        fig3 = go.Figure()
        for rider in ["member", "casual"]:
            sub = dur_day[dur_day["member_casual"] == rider]
            fig3.add_trace(go.Bar(
                x=[d[:3] for d in sub["day_of_week"]],
                y=sub["duration_min"].round(1),
                name=f"{rider.capitalize()}s",
                marker_color=colors[rider],
                marker_line=dict(width=0),
                text=[f"{v:.1f}" for v in sub["duration_min"]],
                textposition="outside",
                textfont=dict(size=10, color="#334155"),
            ))
        fig3.update_layout(
            **base_layout(t=35, h=320),
            barmode="group",
            xaxis=ax("Day of Week"),
            yaxis=ax("Avg Duration (min)", range=[0, max_dur * 1.25]),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11, color="#475569"),
            ),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── 3. Bike Type Analysis ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Bike Type Analysis</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    label_map = {
        "classic_bike": "Classic Bike",
        "electric_bike": "Electric Bike",
        "docked_bike": "Docked Bike"
    }

    with col1:
        st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#0f172a; margin-bottom:6px;'>Total Rides by Bike Type</div>", unsafe_allow_html=True)
        bt = df.groupby("rideable_type", observed=False).agg(
            count=("ride_id", "count"),
            avg_duration=("duration_min", "mean"),
            avg_distance=("distance_km", "mean"),
        ).reset_index()

        bt["clean_label"] = bt["rideable_type"].map(lambda x: label_map.get(x, str(x).replace("_", " ").title()))
        max_bt = max(1, bt["count"].max())
        palette = [C_BLUE, C_VIOLET, C_TEAL]

        fig4 = go.Figure(go.Bar(
            x=bt["clean_label"],
            y=bt["count"],
            marker=dict(color=palette[:len(bt)]),
            text=[f"{v:,}" for v in bt["count"]],
            textposition="outside",
            textfont=dict(size=11, color="#1e293b"),
        ))
        fig4.update_layout(
            **base_layout(t=25, h=320),
            xaxis=ax("Bike Type"),
            yaxis=ax("Total Rides (Sample)", range=[0, max_bt * 1.2]),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown("<div style='font-size:0.95rem; font-weight:600; color:#0f172a; margin-bottom:6px;'>Travel Speed by Bike Type (km/h)</div>", unsafe_allow_html=True)
        spd = df.groupby("rideable_type", observed=False)["speed_kmh"].agg(["mean", "median"]).reset_index()
        spd["clean_label"] = spd["rideable_type"].map(lambda x: label_map.get(x, str(x).replace("_", " ").title()))

        max_spd = max(1.0, max(spd["mean"].max(), spd["median"].max()))

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=spd["clean_label"],
            y=spd["mean"].round(1),
            name="Mean Speed",
            marker_color=C_BLUE,
            text=[f"{v:.1f} km/h" for v in spd["mean"]],
            textposition="outside",
            textfont=dict(size=10, color="#1e293b"),
        ))
        fig5.add_trace(go.Bar(
            x=spd["clean_label"],
            y=spd["median"].round(1),
            name="Median Speed",
            marker_color=C_AMBER,
            text=[f"{v:.1f} km/h" for v in spd["median"]],
            textposition="outside",
            textfont=dict(size=10, color="#1e293b"),
        ))
        fig5.update_layout(
            **base_layout(t=35, h=320),
            barmode="group",
            xaxis=ax("Bike Type"),
            yaxis=ax("Speed (km/h)", range=[0, max_spd * 1.28]),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11, color="#475569"),
            ),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── 4. Heatmap ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Demand Heatmap: Hour vs Day of Week</div>", unsafe_allow_html=True)

    hmap = df.groupby(["day_of_week", "hour"], observed=False).size().reset_index(name="count")
    hmap["day_of_week"] = pd.Categorical(hmap["day_of_week"], categories=DAY_ORDER, ordered=True)
    pivot = hmap.pivot(index="day_of_week", columns="hour", values="count").fillna(0)

    fig6 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[d[:3] for d in pivot.index],
        colorscale=[[0, "#eff6ff"], [0.35, "#93c5fd"], [0.7, "#2563eb"], [1, "#1e3a8a"]],
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#475569", size=10),
            title=dict(text="Rides", font=dict(color="#475569", size=11)),
            thickness=14,
        ),
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>Rides: %{z:,}<extra></extra>",
    ))
    fig6.update_layout(
        paper_bgcolor=T["paper_bgcolor"],
        plot_bgcolor=T["plot_bgcolor"],
        font=dict(color=T["font_color"], family="Inter", size=12),
        xaxis=dict(color="#64748b", title="Hour of Day", tickangle=-45),
        yaxis=dict(color="#64748b", title="Day"),
        height=280,
        margin=dict(l=15, r=15, t=15, b=25),
    )
    st.plotly_chart(fig6, use_container_width=True)
