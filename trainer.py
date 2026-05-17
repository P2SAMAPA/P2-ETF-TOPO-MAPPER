import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from mapper_engine import run_mapper

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
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            # Use last `win` days of returns
            data = returns.iloc[-win:].values.T   # shape (n_etfs, win)
            # Remove any ETF with all NaN
            valid = ~np.isnan(data).any(axis=1)
            if valid.sum() < 3:
                print(f"    Not enough valid ETFs for window {win}d")
                continue
            data = data[valid]
            valid_tickers = [tickers[i] for i in range(len(tickers)) if valid[i]]
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
            # etf_centrality is already a dict mapping ETF index (in valid_tickers) to centrality
            # We need to map back to ticker names
            etf_scores = {valid_tickers[i]: etf_centrality[i] for i in range(len(valid_tickers))}
            window_results[win] = etf_scores
            for etf, score in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            # Fallback: use historical mean return (positive if >0, else small positive)
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        # Store full scores for all ETFs
        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "centrality": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by topological centrality: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/mapper_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Topological Mapper Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
