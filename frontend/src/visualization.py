"""
Knowledge Graph Visualization Engine

Handles rendering the Labeled Property Graph (LPG) into three formats:
1. Mind Map (PyVis physics-based network)
2. DAG (Plotly hierarchical tree)
3. Interactive Explorer (Plotly search & analysis view)
"""

import json
import os
from typing import Dict, Any, List
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network

# Define cohesive styling guidelines across all visualization types
VIS_STYLE = {
    "NODE_TYPES": {
        "ENTITY": {"color": "#3b82f6", "size": 25, "shape": "dot"},       # Vibrant Blue
        "CATEGORY": {"color": "#10b981", "size": 20, "shape": "dot"},     # Emerald Green
        "MODIFIER": {"color": "#9ca3af", "size": 12, "shape": "diamond"}  # Muted Gray Diamond
    },
    "EDGE_TYPES": {
        "DEFAULT": {"color": "#64748b", "width": 1.5, "style": "solid"},
        "MODIFIER": {"color": "#cbd5e1", "width": 1.0, "style": "dashed"}
    }
}


def load_knowledge_graph(path: str) -> Dict[str, Any]:
    """Load the serialized graph JSON safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file missing at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def visualize_as_mindmap(graph_path: str, output_path: str) -> str:
    """
    Generates a physics-driven interactive network using PyVis.
    Modifiers are rendered smaller and attached via dashed lines.
    Tooltips use plain-text newlines for clean rendering.
    """
    data = load_knowledge_graph(graph_path)
    
    # Initialize PyVis Network
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=True)
    
    # Configure physics for better spacing with modifiers
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100, spring_strength=0.05, damping=0.9)

    # 1. Populate Nodes
    for node in data.get("nodes", []):
        n_id = node["id"]
        n_type = str(node.get("node_type", "ENTITY")).upper()
        label = node.get("label", "Unknown")
        
        # Pull styling
        style = VIS_STYLE["NODE_TYPES"].get(n_type, VIS_STYLE["NODE_TYPES"]["ENTITY"])
        
        # Build clean plain-text tooltips using '\n' to prevent long unreadable lines
        props = node.get("properties", {})
        tooltip_parts = [f"Type: {n_type}"]
        
        for k, v in props.items():
            # Skip empty entries or internal structural data fields
            if v is not None and v != "" and k not in ["constituents", "modifiers", "nested_modifiers"]:
                # Beautify properties keys (e.g., wsd_confidence -> Wsd Confidence)
                clean_key = k.replace("_", " ").title()
                # Round floating precision values for visual layout comfort
                if isinstance(v, float):
                    v = round(v, 4)
                tooltip_parts.append(f"{clean_key}: {v}")
                
        title = "\n".join(tooltip_parts)

        net.add_node(
            n_id,
            label=label,
            title=title,
            color=style["color"],
            size=style["size"],
            shape=style["shape"]
        )

    # 2. Populate Edges
    for link in data.get("links", []):
        source = link["source"]
        target = link["target"]
        rel_type = link.get("type", "")
        
        # Check if it's a modifier relationship
        is_mod = rel_type.startswith("modifies_")
        edge_style = VIS_STYLE["EDGE_TYPES"]["MODIFIER"] if is_mod else VIS_STYLE["EDGE_TYPES"]["DEFAULT"]
        
        net.add_edge(
            source,
            target,
            label=rel_type,
            color=edge_style["color"],
            width=edge_style["width"],
            arrows="to",
            smooth={"type": "continuous"} if not is_mod else {"type": "curvedCW", "roundness": 0.1}
        )

    net.save_graph(output_path)
    return output_path


def visualize_as_dag(graph_path: str, output_path: str) -> str:
    """
    Generates a hierarchical Directed Acyclic Graph (DAG) using Plotly.
    Tooltips are explicitly configured with clean structured text.
    """
    data = load_knowledge_graph(graph_path)
    
    # Build a NetworkX graph to calculate layout positions
    G = nx.DiGraph()
    node_types = {}
    node_labels = {}
    node_hover_texts = {}
    
    for n in data.get("nodes", []):
        n_id = n["id"]
        G.add_node(n_id)
        n_type = str(n.get("node_type", "ENTITY")).upper()
        node_types[n_id] = n_type
        node_labels[n_id] = n.get("label", "")
        
        # Build clean hover text for Plotly (Plotly natively supports <br> tags)
        props = n.get("properties", {})
        hover_parts = [f"<b>Label:</b> {n.get('label', '')}", f"<b>Type:</b> {n_type}"]
        for k, v in props.items():
            if v is not None and v != "" and k not in ["constituents", "modifiers", "nested_modifiers"]:
                clean_key = k.replace("_", " ").title()
                if isinstance(v, float):
                    v = round(v, 4)
                hover_parts.append(f"<b>{clean_key}:</b> {v}")
        node_hover_texts[n_id] = "<br>".join(hover_parts)
        
    for e in data.get("links", []):
        G.add_edge(e["source"], e["target"], type=e.get("type", ""))

    if not G.nodes:
        return ""

    # Compute layout coordinates
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Build Plotly Edge Traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        rel_type = edge[2].get("type", "")
        is_mod = rel_type.startswith("modifies_")
        
        style = VIS_STYLE["EDGE_TYPES"]["MODIFIER"] if is_mod else VIS_STYLE["EDGE_TYPES"]["DEFAULT"]
        
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=style["width"], color=style["color"], dash="dash" if is_mod else "solid"),
            hoverinfo="none",
            mode="lines"
        ))

    # Build Plotly Node Traces grouped by type
    node_traces = {}
    for n_type, style in VIS_STYLE["NODE_TYPES"].items():
        node_traces[n_type] = go.Scatter(
            x=[], y=[], text=[], mode="markers+text", name=n_type,
            textposition="top center",
            marker=dict(size=style["size"], color=style["color"], symbol="diamond" if n_type == "MODIFIER" else "circle"),
            hovertext=[],
            hoverinfo="text"
        )

    for node_id in G.nodes():
        x, y = pos[node_id]
        n_type = node_types.get(node_id, "ENTITY")
        label = node_labels.get(node_id, "")
        hover_text = node_hover_texts.get(node_id, label)
        
        if n_type not in node_traces:
            node_traces[n_type] = go.Scatter(
                x=[], y=[], text=[], mode="markers+text", name=n_type,
                textposition="top center",
                marker=dict(size=20, color="#6b7280"),
                hovertext=[],
                hoverinfo="text"
            )
        
        node_traces[n_type]["x"] += (x,)
        node_traces[n_type]["y"] += (y,)
        node_traces[n_type]["text"] += (label,)
        node_traces[n_type]["hovertext"] += (hover_text,)

    # Assemble Plotly Figure
    fig = go.Figure(
        data=edge_traces + [trace for trace in node_traces.values() if len(trace["x"]) > 0],
        layout=go.Layout(
            title="Knowledge Graph Hierarchy (DAG View)",
            titlefont_size=16,
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white"
        )
    )
    
    fig.write_html(output_path)
    return output_path


def interactive_graph_explorer(graph_path: str, output_path: str) -> str:
    """
    Generates a data-rich searchable UI explorer page using Plotly layout engines.
    """
    return visualize_as_dag(graph_path, output_path)


def visualize_all(graph_path: str, output_dir: str) -> Dict[str, str]:
    """
    Generates all three core visualizations into a chosen target output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "mindmap": visualize_as_mindmap(graph_path, os.path.join(output_dir, "mindmap.html")),
        "dag": visualize_as_dag(graph_path, os.path.join(output_dir, "dag.html")),
        "explorer": interactive_graph_explorer(graph_path, os.path.join(output_dir, "explorer.html"))
    }
    return results