import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from mapper_engine import run_mapper, mapper_to_json

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Topological Mapper) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.ROLLING_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Use last ROLLING_WINDOW days of returns
        data = returns.iloc[-config.ROLLING_WINDOW:].values.T   # shape (n_etfs, window)
        # Remove any ETF with all NaN
        valid = ~np.isnan(data).any(axis=1)
        data = data[valid]
        valid_tickers = [tickers[i] for i in range(len(tickers)) if valid[i]]
        if len(data) < 3:
            print("  Not enough valid ETFs")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Run Mapper
        G, cover, degree_cent, eigen_cent, etf_centrality = run_mapper(
            data,
            n_intervals=config.N_INTERVALS,
            overlap=config.OVERLAP_PCT,
            cluster_method=config.CLUSTER_METHOD,
            dbscan_eps=config.DBSCAN_EPS,
            dbscan_min_samples=config.DBSCAN_MIN_SAMPLES,
            kmeans_n_clusters=config.KMEANS_N_CLUSTERS
        )

        # Prepare output
        nodes = []
        for nid in G.nodes:
            members = cover[nid]
            member_tickers = [valid_tickers[i] for i in members]
            nodes.append({
                "id": str(nid),
                "members": member_tickers,
                "size": len(members),
                "centrality": float(degree_cent[nid])
            })
        edges = [{"source": str(u), "target": str(v)} for u, v in G.edges]
        # Top ETFs by centrality
        sorted_etfs = sorted(etf_centrality.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        full_scores = {}
        for i, (idx, cent) in enumerate(sorted_etfs[:config.TOP_N]):
            ticker = valid_tickers[idx]
            top_etfs.append({"ticker": ticker, "centrality": float(cent)})
            full_scores[ticker] = float(cent)
        print(f"  Top 3 ETFs by topological centrality: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "graph": {"nodes": nodes, "edges": edges},
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/mapper_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Algebraic Topology Mapper Engine complete ===")

if __name__ == "__main__":
    main()
