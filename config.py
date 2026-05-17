import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-topo-mapper-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

WINDOWS = [63, 252, 504, 1008, 2016]
# Rolling window for returns (days)
ROLLING_WINDOW = 252

# Mapper parameters
N_INTERVALS = 10          # number of intervals covering the filter range
OVERLAP_PCT = 0.5         # overlap between intervals (e.g., 0.5 = 50%)
CLUSTER_METHOD = "dbscan" # or "kmeans"
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 2
KMEANS_N_CLUSTERS = 5

# For ranking: use node degree centrality (higher = more central in topological graph)
TOP_N = 3
