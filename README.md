# Algebraic Topology Mapper Engine

Topological Data Analysis (TDA) Mapper algorithm applied to ETF return spaces.  
Constructs a nerve graph that reveals regime clusters and transition paths invisible to standard clustering.

- **Filter function:** first principal component of returns
- **Covering:** overlapping intervals (10 intervals, 50% overlap)
- **Clustering:** DBSCAN within each interval
- **Output:** Topological graph (nodes = clusters, edges = overlaps), ETF centrality (degree of nodes containing each ETF)
- **Top ETFs:** those with highest topological centrality

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
