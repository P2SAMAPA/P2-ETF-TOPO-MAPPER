import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from collections import defaultdict

def run_mapper(data, n_intervals=10, overlap=0.5, cluster_method="dbscan",
               dbscan_eps=0.5, dbscan_min_samples=2, kmeans_n_clusters=5):
    """
    Apply Mapper algorithm on ETF return data (points = ETFs, features = returns over rolling window).
    data: numpy array of shape (n_etfs, n_days) – each row is an ETF's return series.
    Returns: graph (networkx Graph), node_assignments (dict: node_id -> list of ETF indices), node_centralities.
    """
    n_etfs = data.shape[0]
    # Filter function: first principal component of the return matrix (across ETFs)
    # But we need a 1‑D filter per point. We'll use the first PC of each ETF's returns? Actually each ETF is a point in R^window.
    # We can use the 1st PC of the entire point cloud (across all ETFs) as filter.
    # Alternatively, use the first PC of each ETF's returns? That gives 1 value per ETF.
    # Standard approach: each ETF is a point in high-dim space (its return series). The filter is a function on that point cloud.
    # We'll compute the first principal component of all points (n_etfs, window) -> filter = projection onto first PC.
    pca = PCA(n_components=1)
    filter_vals = pca.fit_transform(data).flatten()
    # Normalise filter values to [0,1]
    f_min, f_max = filter_vals.min(), filter_vals.max()
    filter_norm = (filter_vals - f_min) / (f_max - f_min + 1e-8)
    # Create intervals
    interval_step = 1.0 / n_intervals
    intervals = []
    for i in range(n_intervals):
        left = i * interval_step
        right = left + interval_step + overlap * interval_step
        intervals.append((left, min(right, 1.0)))
    # For each interval, find points in that filter range
    cover = {}
    for i, (low, high) in enumerate(intervals):
        idx_in = np.where((filter_norm >= low) & (filter_norm <= high))[0]
        if len(idx_in) == 0:
            continue
        # Subset data for this interval
        subset = data[idx_in]
        # Cluster within interval
        if cluster_method == "dbscan":
            if len(subset) >= dbscan_min_samples:
                clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(subset)
                labels = clustering.labels_
            else:
                labels = np.zeros(len(subset))
        else:  # kmeans
            n_clust = min(kmeans_n_clusters, len(subset))
            if n_clust < 1:
                n_clust = 1
            clustering = KMeans(n_clusters=n_clust, random_state=42).fit(subset)
            labels = clustering.labels_
        # For each cluster label, store member indices
        unique_labels = set(labels)
        for lbl in unique_labels:
            if lbl == -1:  # noise in DBSCAN, skip
                continue
            members = idx_in[labels == lbl]
            cover[(i, lbl)] = members.tolist()
    # Build nerve graph: nodes are cover elements, edges if they share any point
    node_ids = list(cover.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    G = nx.Graph()
    for nid in node_ids:
        G.add_node(nid, members=cover[nid])
    # Add edges between nodes that share at least one ETF
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            if set(cover[node_ids[i]]) & set(cover[node_ids[j]]):
                G.add_edge(node_ids[i], node_ids[j])
    # Compute node centralities (degree centrality)
    degree_cent = nx.degree_centrality(G)
    # Also eigenvector centrality (if graph is connected enough)
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigen_cent = {n: 0.0 for n in G.nodes}
    # Combine: for each ETF, its centrality = max degree of nodes it belongs to
    etf_centrality = defaultdict(float)
    for nid, members in cover.items():
        cent = degree_cent[nid]
        for etf_idx in members:
            if cent > etf_centrality[etf_idx]:
                etf_centrality[etf_idx] = cent
    return G, cover, degree_cent, eigen_cent, etf_centrality

def mapper_to_json(G, cover, etf_centrality, tickers):
    """Convert graph to JSON serializable format."""
    nodes = []
    for nid in G.nodes:
        members = cover[nid]
        member_tickers = [tickers[i] for i in members]
        nodes.append({
            "id": str(nid),
            "members": member_tickers,
            "size": len(members),
            "centrality": float(degree_centrality[nid])  # need degree_centrality from calling function
        })
    edges = []
    for u, v in G.edges:
        edges.append({"source": str(u), "target": str(v)})
    etf_ranks = sorted(etf_centrality.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": tickers[i], "centrality": float(etf_centrality[i])} for i, _ in etf_ranks[:config.TOP_N]]
    return {"nodes": nodes, "edges": edges, "top_etfs": top_etfs}
