import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import (
    load_kmeans_clusters, load_kmeans_metrics,
    load_rf_metrics, load_feature_importance, load_rf_model,
    base_layout, ax, C_BLUE, C_VIOLET, C_TEAL, C_AMBER, PLOTLY_THEME
)

T = PLOTLY_THEME

CLUSTER_COLORS = {0: C_BLUE, 1: "#ef4444", 2: C_TEAL, 3: C_AMBER}
CLUSTER_LABELS = {
    0: "Residential — Member Dominated",
    1: "Super Hub — Manhattan Core",
    2: "Suburban — High Electric Share",
    3: "Leisure — High Casual Share",
}


@st.cache_data
def _clusters():   return load_kmeans_clusters()
@st.cache_data
def _km_metrics(): return load_kmeans_metrics()
@st.cache_data
def _rf_metrics(): return load_rf_metrics()
@st.cache_data
def _feat_imp():   return load_feature_importance()
@st.cache_resource
def _rf_model():   return load_rf_model()


def render():
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Machine Learning</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Station segmentation via Spark MLlib K-Means · Trip duration prediction via Scikit-Learn Random Forest</div>", unsafe_allow_html=True)

    # ── Model A: K-Means ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model A — Station Clustering (Spark MLlib K-Means)</div>", unsafe_allow_html=True)

    km_m = _km_metrics()
    kpis = [
        ("4", "Optimal Clusters (K)"),
        (f"{km_m.get('silhouette_score', 0):.4f}", "Silhouette Score"),
        ("2,416", "Stations Clustered"),
        ("Spark MLlib", "Library"),
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
    df_km = _clusters()
    if not df_km.empty:
        # Cluster summary table
        summary = df_km.groupby("cluster").agg(
            Stations=("start_station_name", "count"),
            Avg_Trips=("total_trips", "mean"),
            Avg_Duration=("avg_duration", "mean"),
            Pct_Casual=("casual_ratio", lambda x: x.mean() * 100),
            Pct_Electric=("electric_ratio", lambda x: x.mean() * 100),
        ).reset_index()
        summary["Segment"] = summary["cluster"].map(CLUSTER_LABELS)

        import pandas as pd
        display_df = summary[["cluster", "Segment", "Stations", "Avg_Trips", "Avg_Duration", "Pct_Casual", "Pct_Electric"]].copy()
        display_df.columns = ["Cluster", "Segment Description", "Stations", "Avg Trips", "Avg Duration (min)", "% Casual", "% Electric"]
        display_df["Avg Trips"] = display_df["Avg Trips"].map("{:,.0f}".format)
        display_df["Avg Duration (min)"] = display_df["Avg Duration (min)"].map("{:.1f}".format)
        display_df["% Casual"] = display_df["% Casual"].map("{:.1f}%".format)
        display_df["% Electric"] = display_df["% Electric"].map("{:.1f}%".format)
        
        # Display full-width table so all columns fit cleanly without truncation
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        col_pie, col_sc = st.columns([2, 3])

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=[CLUSTER_LABELS[i] for i in summary["cluster"]],
                values=summary["Stations"],
                hole=0.55,
                marker=dict(colors=[CLUSTER_COLORS[i] for i in summary["cluster"]],
                            line=dict(color="#fff", width=2)),
                textfont=dict(size=11, color="#1e293b"),
            ))
            fig_pie.update_layout(
                **base_layout(h=360, margin=dict(l=15, r=15, t=45, b=60)),
                legend=dict(font=dict(color="#475569", size=10), orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
                title=dict(text="Cluster Station Share", font=dict(size=13, color="#0f172a")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_sc:
            fig_sc = go.Figure()
            for c in sorted(df_km["cluster"].unique()):
                sub = df_km[df_km["cluster"] == c]
                short_label = CLUSTER_LABELS[c].split("—")[0].strip()
                fig_sc.add_trace(go.Scatter(
                    x=sub["total_trips"], y=sub["avg_duration"],
                    mode="markers",
                    name=f"C{c}: {short_label}",
                    marker=dict(color=CLUSTER_COLORS[c], size=7, opacity=0.75,
                                line=dict(color="#fff", width=0.5)),
                    hovertemplate="<b>%{text}</b><br>Cluster: " + CLUSTER_LABELS[c] + "<br>Trips: %{x:,}<br>Avg Duration: %{y:.1f} min<extra></extra>",
                    text=sub["start_station_name"],
                ))
            fig_sc.update_layout(
                **base_layout(h=360, margin=dict(l=15, r=15, t=45, b=60)),
                xaxis=ax("Total Trips per Station"),
                yaxis=ax("Avg Trip Duration (min)"),
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
                            font=dict(size=9.5, color="#475569")),
                title=dict(text="Station Characterisation: Trips vs Duration", font=dict(size=13, color="#0f172a")),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    # ── Model B: Random Forest ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model B — Trip Duration Prediction (Scikit-Learn Random Forest)</div>", unsafe_allow_html=True)

    rf_m = _rf_metrics()
    kpis2 = [
        (f"{rf_m.get('mae_minutes', 0):.2f} min", "Mean Absolute Error"),
        (f"{rf_m.get('rmse_minutes', 0):.2f} min", "Root Mean Squared Error"),
        (f"{rf_m.get('r2_score', 0):.4f}", "R² Score"),
        (f"{rf_m.get('train_size', 0):,}", "Training Samples"),
    ]
    cols2 = st.columns(4)
    for col, (val, label) in zip(cols2, kpis2):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_f, col_sim = st.columns([1, 1])

    with col_f:
        st.markdown("<div class='section-header' style='font-size:1rem;'>Feature Importance</div>", unsafe_allow_html=True)
        df_fi = _feat_imp()
        if not df_fi.empty:
            df_fi = df_fi.sort_values("importance", ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=df_fi["importance"], y=df_fi["feature"],
                orientation="h",
                marker=dict(color=df_fi["importance"], colorscale=[[0,"#bfdbfe"],[1,C_BLUE]]),
                text=[f"{v:.3f}" for v in df_fi["importance"]],
                textposition="outside",
                textfont=dict(size=10, color="#334155"),
            ))
            fig_fi.update_layout(
                **base_layout(h=300),
                xaxis=ax("Importance Score"),
                yaxis=ax(""),
                showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    with col_sim:
        st.markdown("<div class='section-header' style='font-size:1rem;'>Duration Estimator</div>", unsafe_allow_html=True)
        rf = _rf_model()
        if rf is not None:
            with st.form("pred_form"):
                c1, c2 = st.columns(2)
                with c1:
                    hour   = st.slider("Departure Hour", 0, 23, 17)
                    dow    = st.selectbox("Day", list(range(1, 8)),
                                          format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])
                    month  = st.selectbox("Month", list(range(1, 13)), index=5)
                with c2:
                    member = st.radio("Rider Type", ["Member", "Casual"])
                    bike   = st.radio("Bike Type", ["Classic", "Electric"])
                submit = st.form_submit_button("Estimate Duration")

            if submit:
                member_enc = 1 if member == "Member" else 0
                bike_enc   = 1 if bike == "Electric" else 0
                X = np.array([[hour, dow, month, member_enc, bike_enc,
                               40.750, -73.990, 40.730, -73.985]])
                pred = rf.predict(X)[0]
                st.markdown(f"""
                <div style='background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px;
                            padding:20px; text-align:center; margin-top:12px;'>
                    <div style='font-size:0.75rem; color:#1d4ed8; font-weight:600;
                                text-transform:uppercase; letter-spacing:1px;'>Estimated Duration</div>
                    <div style='font-family:Space Grotesk,sans-serif; font-size:2.4rem;
                                font-weight:700; color:#1d4ed8; margin:8px 0;'>{pred:.1f} min</div>
                    <div style='font-size:0.8rem; color:#64748b;'>±5.85 min (MAE) · Random Forest (100 trees)</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Model file not found. Run 05_machine_learning.py to generate it.")

    # ── Comparison Table ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Comparison: Spark MLlib vs Scikit-Learn</div>", unsafe_allow_html=True)
    import pandas as pd
    comp = pd.DataFrame({
        "Aspect":    ["Library", "Task", "Scalability", "Silhouette / MAE", "Best For"],
        "Spark MLlib":    ["PySpark MLlib", "K-Means Clustering", "Distributed (scales to TB)", "0.4162", "Big Data environments"],
        "Scikit-Learn":   ["sklearn", "Random Forest Regression", "Single-node (fits in RAM)", "5.85 min MAE", "Moderate-size datasets"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)
