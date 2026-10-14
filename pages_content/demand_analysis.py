import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import (
    load_rush_hour, load_weekend, load_member_behavior,
    base_layout, ax, C_BLUE, C_VIOLET, C_AMBER, PLOTLY_THEME
)

T = PLOTLY_THEME
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_SHORT  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data
def _rush():    return load_rush_hour()
@st.cache_data
def _weekend(): return load_weekend()
@st.cache_data
def _member():  return load_member_behavior()


def render():
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Demand Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Hourly, daily, and membership demand patterns derived from Spark SQL aggregations on the Gold Zone</div>", unsafe_allow_html=True)

    # ── 1. Rush Hour ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Hourly Ride Distribution — 24-Hour Profile</div>", unsafe_allow_html=True)
    df_rh = _rush()
    if not df_rh.empty:
        df_rh = df_rh.sort_values("hour")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_rh["hour"], y=df_rh["total_trips"],
            mode="lines+markers",
            line=dict(color=C_BLUE, width=2.5, shape="spline"),
            marker=dict(size=6, color=C_BLUE),
            fill="tozeroy", fillcolor=C_BLUE + "18",
            name="Total Trips",
        ))
        fig.add_vrect(x0=7.5, x1=9.5,   fillcolor="#fef08a", opacity=0.35, line_width=0)
        fig.add_vrect(x0=16.5, x1=19.0, fillcolor="#fca5a5", opacity=0.30, line_width=0)
        fig.add_annotation(x=8.5, y=df_rh["total_trips"].max() * 0.95,
                           text="Morning Peak", showarrow=False,
                           font=dict(size=10, color="#92400e"), bgcolor="#fef9c3",
                           borderpad=3)
        fig.add_annotation(x=17.7, y=df_rh["total_trips"].max() * 0.95,
                           text="Evening Peak", showarrow=False,
                           font=dict(size=10, color="#991b1b"), bgcolor="#fee2e2",
                           borderpad=3)
        fig.update_layout(
            **base_layout(h=320),
            xaxis=ax("Hour of Day", tickvals=list(range(0, 24, 2))),
            yaxis=ax("Total Trips"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='insight-card'>
            <div class='insight-title'>Diurnal Commute Pattern</div>
            <div class='insight-body'>
                Demand peaks at <strong>08:00</strong> (morning commute) and
                <strong>17:00–18:00</strong> (evening commute), a bimodal pattern
                consistent with NYC office hours. Off-peak hours (00:00–05:00)
                show minimal activity. Fleet rebalancing should complete before 07:30
                each morning to ensure dock availability at high-demand stations.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── 2. Weekday vs Weekend ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Weekday vs Weekend Patterns</div>", unsafe_allow_html=True)
    df_ww = _weekend()
    if not df_ww.empty:
        df_ww = df_ww.sort_values("day_of_week")
        dow_map = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
        df_ww["day_name"] = df_ww["day_of_week"].map(dow_map)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure()
            colors = df_ww["is_weekend"].map({"Weekday": C_BLUE, "Weekend": C_AMBER})
            fig2.add_trace(go.Bar(
                x=df_ww["day_name"], y=df_ww["total_trips"],
                marker_color=colors, marker_line_width=0,
                text=[f"{v/1e6:.2f}M" if v >= 1e6 else f"{v:,}" for v in df_ww["total_trips"]],
                textposition="outside", textfont=dict(size=10, color="#334155"),
            ))
            fig2.update_layout(
                **base_layout(h=300),
                xaxis=ax("Day of Week"),
                yaxis=ax("Total Trips"),
                showlegend=False,
                title=dict(text="Trip Volume by Day", font=dict(size=13, color="#0f172a")),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df_ww["day_name"], y=df_ww["avg_duration"],
                mode="lines+markers",
                line=dict(color=C_VIOLET, width=2.5, shape="spline"),
                marker=dict(size=8, color=C_VIOLET),
                fill="tozeroy", fillcolor=C_VIOLET + "15",
            ))
            fig3.update_layout(
                **base_layout(h=300),
                xaxis=ax("Day of Week"),
                yaxis=ax("Avg Duration (min)"),
                showlegend=False,
                title=dict(text="Avg Trip Duration by Day", font=dict(size=13, color="#0f172a")),
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(f"""
        <div class='insight-card'>
            <div class='insight-title'>Weekday vs Weekend Behaviour</div>
            <div class='insight-body'>
                Weekdays record higher <strong>trip volumes</strong> driven by commuter demand,
                while weekends show longer <strong>average durations</strong> as riders use
                the service recreationally. Saturday and Sunday see casual-rider share increase
                significantly, suggesting leisure use along park paths and waterfront routes.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── 3. Member Behavior ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Rider Segment and Bike Type Breakdown</div>", unsafe_allow_html=True)
    df_mb = _member()
    if not df_mb.empty:
        label_map = {"classic_bike": "Classic", "electric_bike": "Electric", "docked_bike": "Docked"}
        df_mb["bike_label"] = df_mb["rideable_type"].map(label_map).fillna(df_mb["rideable_type"])

        col3, col4 = st.columns(2)
        with col3:
            fig4 = go.Figure()
            for i, rider in enumerate(["member", "casual"]):
                sub = df_mb[df_mb["member_casual"] == rider]
                fig4.add_trace(go.Bar(
                    x=sub["bike_label"], y=sub["total_trips"],
                    name=rider.capitalize(),
                    marker_color=[C_BLUE, C_VIOLET][i],
                    marker_line_width=0,
                ))
            fig4.update_layout(
                **base_layout(h=300),
                barmode="group",
                xaxis=ax("Bike Type"),
                yaxis=ax("Total Trips"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(size=11, color="#475569")),
                title=dict(text="Trips by Bike Type and Rider Segment", font=dict(size=13, color="#0f172a")),
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col4:
            fig5 = go.Figure()
            for i, rider in enumerate(["member", "casual"]):
                sub = df_mb[df_mb["member_casual"] == rider]
                fig5.add_trace(go.Bar(
                    x=sub["bike_label"], y=sub["avg_duration_min"].round(1),
                    name=rider.capitalize(),
                    marker_color=[C_BLUE, C_VIOLET][i],
                    marker_line_width=0,
                    text=[f"{v:.1f}" for v in sub["avg_duration_min"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#334155"),
                ))
            fig5.update_layout(
                **base_layout(h=300),
                barmode="group",
                xaxis=ax("Bike Type"),
                yaxis=ax("Avg Duration (min)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(size=11, color="#475569")),
                title=dict(text="Avg Duration by Bike Type and Rider Segment", font=dict(size=13, color="#0f172a")),
            )
            st.plotly_chart(fig5, use_container_width=True)
