import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import networkx as nx
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Topological Mapper", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧬 Algebraic Topology Mapper Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">TDA Mapper | Nerve graph of ETF return space | Nodes = clusters, edges = overlaps</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧬 Topological Mapper")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** PCA filter, overlapping intervals, DBSCAN")
st.sidebar.markdown(f"**Intervals:** {config.N_INTERVALS}, overlap {config.OVERLAP_PCT*100:.0f}%")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'mapper_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Topological Centrality (Node Degree)")

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    
    # Top 3 cards
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.metric(etf['ticker'], f"centrality = {etf['centrality']:.4f}")
    
    # Full ranking table (all ETFs)
    full = uni_data.get("full_scores", {})
    if full:
        df_full = pd.DataFrame(list(full.items()), columns=["ETF", "Topological Centrality"])
        df_full = df_full.sort_values("Topological Centrality", ascending=False)
        with st.expander(f"📋 Full ranking – all ETFs in {universe_name}"):
            st.dataframe(df_full, use_container_width=True, hide_index=True)
        
        # Dropdown to select an ETF and see its details
        etf_options = list(full.keys())
        selected_etf = st.selectbox(f"Select ETF to see detailed information", etf_options, key=f"select_{universe_name}")
        if selected_etf:
            cent = full[selected_etf]
            st.write(f"**Centrality:** {cent:.4f}")
            # Find nodes that contain this ETF
            graph = uni_data.get("graph", {})
            nodes = graph.get("nodes", [])
            containing_nodes = [n for n in nodes if selected_etf in n.get("members", [])]
            if containing_nodes:
                st.write(f"**Appears in {len(containing_nodes)} node(s):**")
                for n in containing_nodes:
                    st.write(f"- Node {n['id']} (size {n['size']}, centrality {n['centrality']:.3f}) – members: {', '.join(n['members'][:5])}{'...' if len(n['members'])>5 else ''}")
            else:
                st.write("Not found in any node? (Should not happen)")
    
    # Plot interactive graph
    graph = uni_data.get("graph", {})
    if graph:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        if nodes and edges:
            G = nx.Graph()
            for n in nodes:
                G.add_node(n["id"])
            for e in edges:
                G.add_edge(e["source"], e["target"])
            pos = nx.spring_layout(G, seed=42)
            edge_x = []
            edge_y = []
            for u, v in G.edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            node_x = [pos[n["id"]][0] for n in nodes]
            node_y = [pos[n["id"]][1] for n in nodes]
            node_cent = [n["centrality"] for n in nodes]
            node_sizes = [n["size"] * 5 for n in nodes]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=1),
                hoverinfo='none', name='edges'
            ))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(size=node_sizes, color=node_cent, colorscale='Viridis',
                            showscale=True, colorbar=dict(title='Centrality')),
                text=[n["id"][:10] for n in nodes], textposition="top center",
                hoverinfo='text', hovertext=[f"Members: {', '.join(n['members'][:5])}" for n in nodes],
                name='nodes'
            ))
            fig.update_layout(
                title=f"Mapper graph – {universe_name}",
                showlegend=False, hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                height=600,
                margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    st.divider()

st.caption("The Mapper algorithm constructs a topological graph of the ETF return space. Nodes represent clusters of ETFs with similar return patterns over the last 252 days. Edges indicate overlapping clusters. ETF centrality is the degree of the most central node containing that ETF. Higher centrality = more central in the topological structure.")
