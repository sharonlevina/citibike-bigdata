"""
graph_analytics.py
==================
Standalone Graph Analytics script for CitiBike NYC — COMP8035041 Assignment II
Uses NetworkX to analyse the station-trip network.

Algorithms implemented
----------------------
1. PageRank          — identify most "strategically influential" stations
2. Community Detection (Greedy Modularity) — cluster stations into mobility zones
3. Degree Centrality — most connected hubs
4. Betweenness Centrality — bottleneck bridges between zones
5. Shortest Path (Dijkstra) — optimal redistribution route between two stations

How to run
----------
    python graph_analytics.py

Outputs saved to: graph_output/
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1.  BUILD GRAPH
# ---------------------------------------------------------------------------

def build_graph(top_n_stations: int = 100) -> nx.DiGraph:
    """
    Build a directed weighted graph from the CitiBike sampled trip data.

    Nodes  : Stations (station_name)
    Edges  : Directed trip flows  (start_station -> end_station)
    Weight : Number of trips on that corridor
    """
    print("[1/6] Loading trip data ...")
    parquet_path = os.path.join(BASE, "sampled_trips.parquet")
    df = pd.read_parquet(
        parquet_path,
        columns=["start_station_name", "end_station_name",
                 "start_lat", "start_lng", "end_lat", "end_lng",
                 "duration_min", "distance_km", "member_casual"]
    )
    df = df.dropna(subset=["start_station_name", "end_station_name"])

    # Keep only top-N busiest stations to keep graph tractable
    top_stations = set(
        df["start_station_name"].value_counts().head(top_n_stations).index
    ) | set(
        df["end_station_name"].value_counts().head(top_n_stations).index
    )
    df = df[
        df["start_station_name"].isin(top_stations) &
        df["end_station_name"].isin(top_stations)
    ]
    # Remove self-loops (round trips)
    df = df[df["start_station_name"] != df["end_station_name"]]

    print(f"    -> {len(df):,} trips across {len(top_stations)} stations")

    # Build OD edge weight table
    od = (
        df.groupby(["start_station_name", "end_station_name"])
        .agg(
            weight=("duration_min", "count"),
            avg_duration=("duration_min", "mean"),
            avg_distance=("distance_km", "mean"),
        )
        .reset_index()
    )

    # Station metadata (coordinates)
    station_meta = {}
    for col_s, col_lat, col_lng in [
        ("start_station_name", "start_lat", "start_lng"),
        ("end_station_name",   "end_lat",   "end_lng"),
    ]:
        for _, row in df.dropna(subset=[col_lat, col_lng]).drop_duplicates(col_s).iterrows():
            station_meta[row[col_s]] = {"lat": row[col_lat], "lng": row[col_lng]}

    # Create DiGraph
    print("[2/6] Building directed weighted graph ...")
    G = nx.DiGraph()

    for name, meta in station_meta.items():
        G.add_node(name, lat=meta["lat"], lng=meta["lng"])

    for _, row in od.iterrows():
        G.add_edge(
            row["start_station_name"],
            row["end_station_name"],
            weight=int(row["weight"]),
            avg_duration=round(row["avg_duration"], 2),
            avg_distance=round(row["avg_distance"], 3),
        )

    print(f"    -> Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# 2.  PAGERANK  -- Influential Stations
# ---------------------------------------------------------------------------

def run_pagerank(G: nx.DiGraph, top_k: int = 20) -> pd.DataFrame:
    """
    PageRank identifies stations that are 'important' not just because many
    trips depart from them, but because they are linked to other important
    stations -- analogous to Google's web page ranking.

    In a bike-sharing context, high-PageRank stations are strategic hubs where
    rebalancing investment yields the greatest system-wide return.
    """
    print("[3/6] Computing PageRank ...")
    pr = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=200)
    df = (
        pd.DataFrame.from_dict(pr, orient="index", columns=["pagerank"])
        .reset_index()
        .rename(columns={"index": "station"})
        .sort_values("pagerank", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    df.index += 1  # 1-based rank
    df["pagerank_pct"] = (df["pagerank"] / df["pagerank"].sum() * 100).round(2)
    print(f"    -> Top station: {df.iloc[0]['station']} (score={df.iloc[0]['pagerank']:.5f})")
    return df


# ---------------------------------------------------------------------------
# 3.  COMMUNITY DETECTION  -- Neighbourhood Zones
# ---------------------------------------------------------------------------

def run_community_detection(G: nx.DiGraph) -> dict:
    """
    Community detection groups stations into clusters that have dense internal
    connectivity relative to connections to other clusters.  In a bike-sharing
    network, communities correspond to natural mobility zones (e.g. Midtown,
    Brooklyn, Upper West Side).

    Uses Greedy Modularity (undirected projection).
    """
    print("[4/6] Running community detection (Greedy Modularity) ...")
    UG = G.to_undirected()
    # Keep only the largest connected component
    largest_cc = max(nx.connected_components(UG), key=len)
    UG_sub = UG.subgraph(largest_cc).copy()

    communities = nx.algorithms.community.greedy_modularity_communities(UG_sub, weight="weight")
    community_map = {}
    for cid, nodes in enumerate(communities):
        for node in nodes:
            community_map[node] = cid

    n_communities = len(communities)
    sizes = [len(c) for c in communities]
    print(f"    -> {n_communities} communities detected | sizes: {sorted(sizes, reverse=True)[:5]} ...")

    return community_map


# ---------------------------------------------------------------------------
# 4.  CENTRALITY METRICS  -- Hub & Bottleneck Analysis
# ---------------------------------------------------------------------------

def run_centrality(G: nx.DiGraph, top_k: int = 20) -> pd.DataFrame:
    """
    Degree Centrality  : fraction of nodes the station is directly connected to
    In-Degree          : weighted trips arriving
    Out-Degree         : weighted trips departing
    Betweenness        : how often a station sits on shortest paths (bottleneck)
    """
    print("[5/6] Computing centrality metrics ...")
    deg_c   = nx.degree_centrality(G)
    in_deg  = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    UG = G.to_undirected()
    btw = nx.betweenness_centrality(UG, weight="weight", normalized=True, k=min(50, len(UG)))

    df = pd.DataFrame({
        "station":           list(deg_c.keys()),
        "degree_centrality": [deg_c[s] for s in deg_c],
        "in_degree_weight":  [in_deg.get(s, 0) for s in deg_c],
        "out_degree_weight": [out_deg.get(s, 0) for s in deg_c],
        "betweenness":       [btw.get(s, 0.0) for s in deg_c],
    })
    df["total_flow"] = df["in_degree_weight"] + df["out_degree_weight"]
    df = df.sort_values("betweenness", ascending=False).head(top_k).reset_index(drop=True)
    print(f"    -> Top bottleneck: {df.iloc[0]['station']}")
    return df


# ---------------------------------------------------------------------------
# 5.  SHORTEST PATH  -- Optimal Redistribution Route
# ---------------------------------------------------------------------------

def run_shortest_path(G: nx.DiGraph,
                      source: str = None,
                      target: str = None) -> dict:
    """
    Dijkstra shortest path (weighted by inverse trip count, so high-volume
    corridors are 'shorter') between two stations.

    Operationally this represents the most efficient redistribution route a
    rebalancing van should take to move bikes from an over-stocked station to
    one that is running low.
    """
    print("[6/6] Computing shortest redistribution path ...")
    UG = G.to_undirected()

    node_list = sorted(UG.nodes(), key=lambda n: UG.degree(n, weight="weight"), reverse=True)
    if source is None:
        source = node_list[0]
    if target is None:
        target = node_list[4]

    try:
        for u, v, d in UG.edges(data=True):
            d["inv_weight"] = 1.0 / max(d.get("weight", 1), 1)

        path = nx.shortest_path(UG, source=source, target=target, weight="inv_weight")
        length = nx.shortest_path_length(UG, source=source, target=target, weight="inv_weight")
        print(f"    -> Route: {' -> '.join(path)}")
        return {"source": source, "target": target, "path": path, "path_length": round(length, 4)}
    except nx.NetworkXNoPath:
        print(f"    -> No path found between {source} and {target}")
        return {"source": source, "target": target, "path": [], "path_length": None}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CitiBike NYC -- Graph Analytics  (COMP8035041 Asg II)")
    print("=" * 60)

    G = build_graph(top_n_stations=100)

    pr_df = run_pagerank(G, top_k=20)
    pr_path = os.path.join(OUTPUT_DIR, "pagerank_results.csv")
    pr_df.to_csv(pr_path, index=True)
    print(f"\n  PageRank saved -> {pr_path}")

    community_map = run_community_detection(G)
    cm_df = (
        pd.DataFrame(list(community_map.items()), columns=["station", "community_id"])
        .sort_values(["community_id", "station"])
    )
    cm_path = os.path.join(OUTPUT_DIR, "community_detection.csv")
    cm_df.to_csv(cm_path, index=False)
    print(f"  Communities saved -> {cm_path}")

    cen_df = run_centrality(G, top_k=20)
    cen_path = os.path.join(OUTPUT_DIR, "centrality_metrics.csv")
    cen_df.to_csv(cen_path, index=False)
    print(f"  Centrality saved -> {cen_path}")

    sp_result = run_shortest_path(G)
    sp_path = os.path.join(OUTPUT_DIR, "shortest_path.json")
    with open(sp_path, "w") as f:
        json.dump(sp_result, f, indent=2)
    print(f"  Shortest path saved -> {sp_path}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Nodes (stations)  : {G.number_of_nodes()}")
    print(f"  Edges (corridors) : {G.number_of_edges()}")
    print(f"  Communities       : {len(set(community_map.values()))}")
    print(f"\n  Top PageRank Station:")
    print(f"    {pr_df.iloc[0]['station']}  (score={pr_df.iloc[0]['pagerank']:.5f})")
    print(f"\n  Top Betweenness (Bottleneck):")
    print(f"    {cen_df.iloc[0]['station']}  (btw={cen_df.iloc[0]['betweenness']:.4f})")
    print(f"\n  Redistribution Path:")
    if sp_result['path']:
        print(f"    {' -> '.join(sp_result['path'])}")
    print("=" * 60)
    print("  All outputs saved in: graph_output/")


if __name__ == "__main__":
    main()
