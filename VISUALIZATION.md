# Knowledge Graph Visualization Guide

## Overview

The pipeline generates three interactive HTML visualizations to explore your knowledge graph from different perspectives:

1. **Mind Map** (PyVis) - Interactive network with physics simulation
2. **DAG** (Plotly) - Hierarchical directed acyclic graph
3. **Explorer** (Plotly) - Search and explore interface

---

## Quick Start

### Step 1: Generate the Knowledge Graph
```bash
python main.py
```
This creates `data/output/knowledge_graph.json`

### Step 2: Generate Visualizations
```bash
python visualize.py
```
This creates three HTML files in `frontend/public/`:
- `mindmap.html`
- `dag.html`
- `explorer.html`
- `index.html` (hub page with links to all visualizations)

### Step 3: Open in Browser
Open any of the HTML files in your web browser to explore the knowledge graph.

---

## Visualization Types

### 1. Mind Map (PyVis Network)
**File**: `frontend/public/mindmap.html`

**Features**:
- 🖱️ **Drag nodes** - Click and drag to reposition nodes
- 🔍 **Zoom & Pan** - Scroll to zoom, click and drag background to pan
- 📌 **Fixed nodes** - Double-click to pin/unpin a node
- 🎨 **Color-coded nodes**:
  - Blue: Entities (concrete concepts like "cows", "grass")
  - Red: Concepts (abstract categories like "bovid")
  - Green: Relations (types of relationships)
- 📊 **Physics simulation** - Nodes interact with spring physics for better layout
- 🔗 **Edge labels** - Shows relationship types (consume, raise, is_a, etc.)

**Use Cases**:
- Explore complex relationships organically
- Find connections between distant concepts
- Understand node clustering and proximity
- Interactive graph manipulation

**Controls**:
| Action | Effect |
|--------|--------|
| Click & Drag | Move node around |
| Double Click | Fix/unfix node position |
| Scroll | Zoom in/out |
| Drag Background | Pan the view |
| Hover over node | See node name |
| Hover over edge | See relationship |

---

### 2. DAG Visualization (Plotly Hierarchical)
**File**: `frontend/public/dag.html`

**Features**:
- 📐 **Hierarchical layout** - Concepts at top, entities in rows below
- 🏗️ **Structured relationships** - Clear parent-child connections
- 🎨 **Color-coded by type** - Entities vs concepts visually distinct
- 📊 **Node size** - Concepts slightly larger (importance indicator)
- ⬆️⬇️ **Top-to-bottom flow** - Follows taxonomy direction

**Use Cases**:
- Understanding concept hierarchies
- Analyzing taxonomy and ontology
- Following inheritance chains
- Overview of graph structure

**Layout**:
```
            [Concepts]
                 ↓
         [Entities Layer 1]
                 ↓
         [Entities Layer 2]
                 ↓
         [Entities Layer 3]
```

---

### 3. Graph Explorer (Plotly Interactive)
**File**: `frontend/public/explorer.html`

**Features**:
- 🔍 **Hover tooltips** - See node names on hover
- 📍 **Positioned layout** - Organic spring-based positioning
- 🎨 **Color-coded nodes** - Same color scheme as Mind Map
- 🖱️ **Hover information** - Full details on mouse over
- 📊 **Complete graph view** - All nodes and edges visible

**Use Cases**:
- Quick overview of all nodes
- Search by hovering over nodes
- Identify node clusters
- Lightweight exploration

---

## Example Workflow

### Exploring a Knowledge Graph

1. **Start with Index Page**
   - Open `frontend/public/index.html`
   - Review available visualizations
   - Choose visualization type

2. **Use Mind Map for Deep Exploration**
   - Open `mindmap.html`
   - Drag related nodes close together
   - Identify communities and clusters
   - Fix important nodes to prevent movement

3. **Check DAG for Structure**
   - Open `dag.html`
   - Review concept hierarchy
   - Understand is_a relationships
   - Identify taxonomy levels

4. **Use Explorer for Quick Overview**
   - Open `explorer.html`
   - Hover over nodes to identify key concepts
   - Get quick sense of graph size and density

---

## Graph Statistics

Each visualization displays basic statistics:
- **Number of nodes**: Total entities + concepts
- **Number of edges**: Total relationships
- **Node types**: Breakdown by entity/concept/relation
- **Edge types**: Breakdown by relationship type
- **Graph density**: Ratio of actual edges to possible edges

**Example from Sample Data**:
```
Nodes: 13
├─ Entities: 12 (cows, sheep, grass, etc.)
└─ Concepts: 1 (bovid)

Edges: 4
├─ consume: 1 (cows → plant matter)
├─ raise: 1 (farmers → cattle)
├─ is_a: 2 (sheep → bovid, other hierarchies)
└─ (other types as applicable)

Density: 0.026 (2.6% possible connections realized)
```

---

## Node Color Scheme

### Standard Colors
```
Entity Node (Blue #3498db)
  └─ Concrete, real-world concepts
     Examples: "cows", "grass", "farmer", "meadow"

Concept Node (Red #e74c3c)
  └─ Abstract categories or generalizations
     Examples: "bovid", "herbivore", "plant"

Relation (Green #2ecc71)
  └─ Types of relationships (if shown)
     Examples: "consume", "raise", "is_a"
```

### Hover Information
When you hover over a node:
- **Node name** - Primary label
- **Node type** - Entity / Concept
- **Relationships** - Connected nodes visible

---

## Relationship Types

Common edge types in the knowledge graph:

| Type | Meaning | Example |
|------|---------|---------|
| `consume` | Subject eats/uses object | cows → grass |
| `raise` | Subject cultivates object | farmer → cattle |
| `is_a` | Subject is instance of concept | sheep → bovid |
| `has` | Subject possesses object | tree → leaves |
| `similar_to` | Semantic similarity | grass ≈ hay |
| `related_to` | General association | * → * |

---

## Tips & Tricks

### For Mind Map
1. **Use physics pause** - Click physics button to toggle simulation
2. **Export as PNG** - Right-click → Take Screenshot (in most browsers)
3. **Find isolated nodes** - Drag background to pan entire graph
4. **Create clusters** - Manually arrange related nodes near each other

### For DAG
1. **Identify root concepts** - Look at top row
2. **Trace hierarchies** - Follow vertical connections downward
3. **Count hierarchy depth** - Number of layers indicates complexity
4. **Find leaf nodes** - Concepts at bottom with no children

### For Explorer
1. **Quick scan** - Use for initial graph assessment
2. **Identify hubs** - Nodes with many connections
3. **Find isolated nodes** - Single or weakly connected nodes
4. **Size reference** - Quick sense of graph scale

---

## Python API

You can also generate visualizations programmatically:

```python
from frontend.src.visualization import (
    visualize_as_mindmap,
    visualize_as_dag,
    interactive_graph_explorer,
    visualize_all,
    qa_over_graph
)

# Generate single visualization
mindmap_path = visualize_as_mindmap('data/output/knowledge_graph.json')

# Generate all at once
results = visualize_all('data/output/knowledge_graph.json')
# Returns: {'mindmap': path, 'dag': path, 'explorer': path}

# Answer questions using graph
answer = qa_over_graph('data/output/knowledge_graph.json', 'What do cows eat?')
print(answer)
```

---

## Troubleshooting

### Visualizations Not Opening
- **Check path**: Ensure `data/output/knowledge_graph.json` exists
- **Run pipeline first**: `python main.py`
- **Check browser**: Ensure you have a modern browser (Chrome, Firefox, Safari, Edge)

### PyVis Mind Map Issues
- **May not work in some environments**: Fallback to DAG visualization available
- **Large graphs slow down**: Consider filtering nodes for better performance

### File Not Found Errors
- Ensure you're running from the project root: `/home/nhan/BTL/Assignment_NLP`
- Check that `frontend/public/` directory exists
- Verify `data/output/knowledge_graph.json` was generated

### Browser Compatibility
- Tested: Chrome, Firefox, Safari, Edge
- Requires: JavaScript enabled, modern HTML5 support
- Works offline: All files are self-contained

---

## Performance Notes

| Graph Size | Mind Map | DAG | Explorer |
|-----------|----------|-----|----------|
| Small (< 50 nodes) | ✅ Smooth | ✅ Fast | ✅ Fast |
| Medium (50-200 nodes) | ✅ Good | ✅ Good | ✅ Fast |
| Large (200+ nodes) | ⚠️ Slower | ✅ Good | ✅ Good |

**Optimization Tips**:
1. Use DAG for large graphs (more efficient layout)
2. Pause physics in Mind Map for frozen view
3. Use Explorer for quick overview
4. Consider filtering or sampling nodes for very large graphs

---

## Next Steps

- 📊 **Modify layout**: Edit visualization.py to customize colors, sizes, layouts
- 🔗 **Add custom analysis**: Implement specialized queries using Python API
- 🎨 **Style customization**: Modify HTML/CSS in frontend/public/
- 💾 **Export as image**: Use browser's screenshot or visualization library's export

---

## References

- **PyVis Documentation**: https://pyvis.readthedocs.io/
- **Plotly Documentation**: https://plotly.com/python/
- **NetworkX Documentation**: https://networkx.org/

---

Generated by: Text to Knowledge Graph Pipeline  
Last Updated: 2026-06-09  
