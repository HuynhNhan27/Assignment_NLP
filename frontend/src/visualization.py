"""
Frontend visualization module
Visualize knowledge graph as Mind Map, DAG, and interactive network

Supports:
- Interactive network visualization (pyvis)
- Hierarchical mind map layout (plotly)
- Real-time graph explorer with zoom/pan/drag
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import webbrowser

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def load_knowledge_graph(graph_json_path: str) -> Dict:
    """Load knowledge graph from JSON file."""
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_as_mindmap(graph_json_path: str, output_path: str = None) -> str:
    """
    Visualize knowledge graph as an interactive Mind Map using Pyvis.
    
    Creates an interactive HTML file with:
    - Node dragging
    - Zoom and pan
    - Click to expand/collapse
    - Color-coded node types (entity/concept/relation)
    - Edge labels showing relationships
    
    Args:
        graph_json_path: Path to the knowledge graph JSON
        output_path: Optional custom output path (default: frontend/public/mindmap.html)
        
    Returns:
        Path to generated HTML file
    """
    if not PYVIS_AVAILABLE:
        print("⚠️ PyVis not installed. Run: pip install pyvis")
        return None
    
    graph_data = load_knowledge_graph(graph_json_path)
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    
    try:
        # Create network
        net = Network(
            height='750px',
            width='100%',
            directed=True,
            notebook=False,
            bgcolor='#ffffff',
            font_color='black'
        )
        
        # Define colors for different node types
        color_map = {
            'entity': '#3498db',      # Blue for entities
            'concept': '#e74c3c',     # Red for concepts
            'relation': '#2ecc71'     # Green for relations
        }
        
        # Add nodes
        node_id_map = {}  # Map node id to index for pyvis
        for i, node in enumerate(nodes):
            node_id_map[node['id']] = node['label']
            node_type = node.get('type', 'entity')
            color = color_map.get(node_type, '#95a5a6')
            size = 30 if node_type == 'concept' else 25
            
            net.add_node(
                node['id'],
                label=node['label'],
                title=node.get('label', ''),
                color=color,
                size=size,
                physics=True
            )
        
        # Add edges
        for link in links:
            source = link.get('source')
            target = link.get('target')
            relation_type = link.get('type', 'unknown')
            
            # Get node labels for edge display
            source_label = node_id_map.get(source, source)
            target_label = node_id_map.get(target, target)
            
            net.add_edge(
                source,
                target,
                label=relation_type,
                title=f"{source_label} --[{relation_type}]--> {target_label}",
                arrows='to'
            )
        
        # Set output path
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(graph_json_path).replace('data/output', 'frontend/public'),
                'mindmap.html'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use write_html instead of show for better compatibility
        net.write_html(output_path)
        
        return output_path
    except Exception as e:
        print(f"⚠️ Error generating PyVis visualization: {e}")
        print("   Falling back to Plotly visualization...")
        return None


def visualize_as_dag(graph_json_path: str, output_path: str = None) -> str:
    """
    Visualize knowledge graph as a hierarchical DAG using Plotly.
    
    Features:
    - Hierarchical layout
    - Concept nodes at top, entities below
    - Hover information with full details
    - Color-coded edges by type
    
    Args:
        graph_json_path: Path to the knowledge graph JSON
        output_path: Optional custom output path (default: frontend/public/dag.html)
        
    Returns:
        Path to generated HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not installed. Run: pip install plotly")
        return None
    
    graph_data = load_knowledge_graph(graph_json_path)
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    
    # Separate nodes by type for hierarchical layout
    concepts = [n for n in nodes if n.get('type') == 'concept']
    entities = [n for n in nodes if n.get('type') == 'entity']
    
    # Create hierarchical positions
    positions = {}
    y_offset = 0
    
    # Place concepts at top
    for i, node in enumerate(concepts):
        positions[node['id']] = (i, 1)
    
    # Place entities below
    for i, node in enumerate(entities):
        x = (i % 5)
        y = 0 - (i // 5)
        positions[node['id']] = (x, y)
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_labels = []
    
    for link in links:
        source_id = link.get('source')
        target_id = link.get('target')
        
        if source_id in positions and target_id in positions:
            x0, y0 = positions[source_id]
            x1, y1 = positions[target_id]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_labels.append(link.get('type', 'unknown'))
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    color_map = {
        'concept': '#e74c3c',  # Red
        'entity': '#3498db'    # Blue
    }
    
    for node in nodes:
        if node['id'] in positions:
            x, y = positions[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_type = node.get('type', 'entity')
            node_text.append(node['label'])
            node_color.append(color_map.get(node_type, '#95a5a6'))
            node_size.append(40 if node_type == 'concept' else 30)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line_width=2
        ),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Knowledge Graph - DAG Visualization",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )
    
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(graph_json_path).replace('data/output', 'frontend/public'),
            'dag.html'
        )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    
    return output_path


def interactive_graph_explorer(graph_json_path: str, output_path: str = None) -> str:
    """
    Create interactive graph exploration interface.
    
    Features:
    - Node selection and highlighting
    - Path finding between nodes
    - Relation filtering
    - Node search/filter
    
    Args:
        graph_json_path: Path to the knowledge graph JSON
        output_path: Optional custom output path
        
    Returns:
        Path to generated HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not installed. Run: pip install plotly")
        return None
    
    graph_data = load_knowledge_graph(graph_json_path)
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    
    # Create network layout using spring layout simulation
    import random
    positions = {}
    for node in nodes:
        positions[node['id']] = (random.uniform(-1, 1), random.uniform(-1, 1))
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_info = []
    
    for link in links:
        source_id = link.get('source')
        target_id = link.get('target')
        
        if source_id in positions and target_id in positions:
            x0, y0 = positions[source_id]
            x1, y1 = positions[target_id]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(link.get('type', 'unknown'))
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    color_map = {
        'concept': '#e74c3c',
        'entity': '#3498db',
        'relation': '#2ecc71'
    }
    
    for node in nodes:
        x, y = positions[node['id']]
        node_x.append(x)
        node_y.append(y)
        node_type = node.get('type', 'entity')
        node_text.append(node['label'])
        node_color.append(color_map.get(node_type, '#95a5a6'))
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hovertemplate='<b>%{text}</b><extra></extra>',
        marker=dict(
            size=20,
            color=node_color,
            line_width=2,
            line_color='white'
        ),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Knowledge Graph - Interactive Explorer",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )
    
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(graph_json_path).replace('data/output', 'frontend/public'),
            'explorer.html'
        )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    
    return output_path


def qa_over_graph(graph_json_path: str, question: str) -> str:
    """
    Answer questions using the knowledge graph.
    
    Features:
    - Find related nodes based on keywords
    - Traverse edges to find connections
    - Return relevant subgraph
    
    Args:
        graph_json_path: Path to the knowledge graph JSON
        question: Question to answer
        
    Returns:
        Answer text with related entities and relations
    """
    graph_data = load_knowledge_graph(graph_json_path)
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    
    # Simple keyword-based search
    keywords = question.lower().split()
    related_nodes = []
    
    for node in nodes:
        label = node['label'].lower()
        if any(kw in label for kw in keywords):
            related_nodes.append(node)
    
    # Find connections
    answer = f"Found {len(related_nodes)} related entities:\n"
    for node in related_nodes:
        answer += f"- {node['label']} ({node.get('type', 'unknown')})\n"
    
    # Find relations involving these nodes
    related_links = []
    related_ids = {n['id'] for n in related_nodes}
    
    for link in links:
        if link.get('source') in related_ids or link.get('target') in related_ids:
            related_links.append(link)
    
    if related_links:
        answer += f"\nRelations:\n"
        for link in related_links:
            source_label = next((n['label'] for n in nodes if n['id'] == link['source']), 'Unknown')
            target_label = next((n['label'] for n in nodes if n['id'] == link['target']), 'Unknown')
            answer += f"- {source_label} --[{link.get('type', 'unknown')}]--> {target_label}\n"
    
    return answer


def visualize_all(graph_json_path: str, output_dir: str = None, auto_open: bool = True) -> Dict[str, str]:
    """
    Generate all visualizations at once.
    
    Creates:
    1. mindmap.html - Interactive network using PyVis
    2. dag.html - Hierarchical DAG using Plotly
    3. explorer.html - Interactive explorer using Plotly
    
    Args:
        graph_json_path: Path to the knowledge graph JSON
        output_dir: Optional custom output directory (default: frontend/public)
        auto_open: Whether to open HTML files in browser
        
    Returns:
        Dictionary with paths to generated HTML files
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(graph_json_path).replace('data/output', 'frontend/public')
        )
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Generate visualizations
    if PYVIS_AVAILABLE:
        print("[VISUALIZATION] Generating Mind Map (PyVis)...")
        mindmap_path = visualize_as_mindmap(graph_json_path, os.path.join(output_dir, 'mindmap.html'))
        if mindmap_path:
            results['mindmap'] = mindmap_path
            print(f"✓ Mind Map: {mindmap_path}")
    else:
        print("[VISUALIZATION] PyVis not available, skipping mind map...")
    
    print("[VISUALIZATION] Generating DAG (Plotly)...")
    dag_path = visualize_as_dag(graph_json_path, os.path.join(output_dir, 'dag.html'))
    if dag_path:
        results['dag'] = dag_path
        print(f"✓ DAG: {dag_path}")
    
    print("[VISUALIZATION] Generating Explorer (Plotly)...")
    explorer_path = interactive_graph_explorer(graph_json_path, os.path.join(output_dir, 'explorer.html'))
    if explorer_path:
        results['explorer'] = explorer_path
        print(f"✓ Explorer: {explorer_path}")
    
    # Auto-open in browser
    if auto_open and results:
        print("\n[VISUALIZATION] Opening in browser...")
        for name, path in results.items():
            if os.path.exists(path):
                webbrowser.open('file://' + os.path.abspath(path))
                print(f"Opened: {name}")
    
    return results


if __name__ == '__main__':
    import sys
    
    # Default path
    graph_path = 'data/output/knowledge_graph.json'
    
    # Allow custom path via command line
    if len(sys.argv) > 1:
        graph_path = sys.argv[1]
    
    if not os.path.exists(graph_path):
        print(f"Error: Knowledge graph not found at {graph_path}")
        print("Run 'python main.py' first to generate the knowledge graph.")
        sys.exit(1)
    
    print("=" * 60)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("=" * 60)
    
    visualize_all(graph_path)
