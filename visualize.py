#!/usr/bin/env python3
"""
Quick visualization script for knowledge graph

Usage:
    python visualize.py              # Generate all visualizations
    python visualize.py mindmap      # Generate only mind map
    python visualize.py dag          # Generate only DAG
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from frontend.src.visualization import (
    visualize_all,
    load_knowledge_graph,
    visualize_as_mindmap,
    visualize_as_dag,
    interactive_graph_explorer
)


def print_graph_info(graph_data):
    """Print statistics about the knowledge graph."""
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    metadata = graph_data.get('metadata', {})
    
    print("\n📊 Graph Statistics:")
    print(f"   Nodes: {len(nodes)}")
    print(f"   Edges: {len(links)}")
    
    if metadata:
        node_types = metadata.get('node_types', {})
        edge_types = metadata.get('edge_types', {})
        
        if node_types:
            print(f"   Node Types: {node_types}")
        if edge_types:
            print(f"   Edge Types: {edge_types}")
        if 'density' in metadata:
            print(f"   Density: {metadata['density']:.3f}")


if __name__ == '__main__':
    graph_path = 'data/output/knowledge_graph.json'
    vis_type = 'all'  # Default: generate all
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        vis_type = sys.argv[1].lower()
    
    print("\n" + "=" * 75)
    print("  🧠 KNOWLEDGE GRAPH VISUALIZATION")
    print("=" * 75)
    
    # Check if graph exists
    if not os.path.exists(graph_path):
        print(f"\n❌ Error: Knowledge graph not found at '{graph_path}'")
        print("\n📝 Please run the pipeline first:")
        print("   python main.py")
        print("\n💡 Then run this script:")
        print("   python visualize.py")
        sys.exit(1)
    
    # Load and display graph info
    print(f"\n📊 Loading knowledge graph from: {graph_path}")
    try:
        graph_data = load_knowledge_graph(graph_path)
        print_graph_info(graph_data)
    except Exception as e:
        print(f"⚠️ Warning: Could not load graph info: {e}")
    
    # Generate visualizations based on user choice
    output_dir = 'frontend/public'
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print(f"\n🎨 Generating visualizations...")
    
    if vis_type in ['all', 'mindmap']:
        print("   [1/3] Generating Mind Map (PyVis network)...")
        path = visualize_as_mindmap(graph_path, os.path.join(output_dir, 'mindmap.html'))
        if path:
            results['mindmap'] = path
            print(f"       ✓ Saved: {path}")
    
    if vis_type in ['all', 'dag']:
        print("   [2/3] Generating DAG (Plotly hierarchical)...")
        path = visualize_as_dag(graph_path, os.path.join(output_dir, 'dag.html'))
        if path:
            results['dag'] = path
            print(f"       ✓ Saved: {path}")
    
    if vis_type in ['all', 'explorer']:
        print("   [3/3] Generating Explorer (Plotly interactive)...")
        path = interactive_graph_explorer(graph_path, os.path.join(output_dir, 'explorer.html'))
        if path:
            results['explorer'] = path
            print(f"       ✓ Saved: {path}")
    
    # Generate index page if all visualizations created
    if vis_type == 'all' and len(results) == 3:
        index_path = os.path.join(output_dir, 'index.html')
        if os.path.exists(index_path):
            print(f"       ✓ Index: {index_path}")
    
    # Summary
    print("\n" + "=" * 75)
    print("✅ Visualization Complete!")
    print("=" * 75)
    
    print("\n📂 Generated files:")
    for name, path in results.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"   • {name:12} → {path:30} ({size_kb:6.1f} KB)")
    
    print("\n🌐 Opening visualizations in your browser:")
    print("   1️⃣  Open index.html for a menu:")
    print("       file://" + os.path.abspath('frontend/public/index.html'))
    print("\n   2️⃣  Or open visualizations directly:")
    print("       • Mind Map  : " + os.path.abspath('frontend/public/mindmap.html'))
    print("       • DAG       : " + os.path.abspath('frontend/public/dag.html'))
    print("       • Explorer  : " + os.path.abspath('frontend/public/explorer.html'))
    
    print("\n💡 Visualization Types:")
    print("   • mindmap.html  : Interactive network (drag, zoom, physics)")
    print("   • dag.html      : Hierarchical tree layout")
    print("   • explorer.html : Search and explore nodes")
    
    print("\n📖 For detailed guide, see: VISUALIZATION.md")
    print("=" * 75)
    print()