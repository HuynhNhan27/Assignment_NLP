# Educational Text to Knowledge Graph Pipeline

A sophisticated NLP pipeline that converts educational texts into a Labeled Property Graph (LPG) with intelligent entity merging through ontology hierarchies.

## Features

### 1. **Information Extraction (IE) Module**
- Named Entity Recognition (NER) using spaCy
- Relation Extraction via dependency parsing
- Noun Chunking with head noun extraction
- **Example**: "large African elephants" → head: "elephants", modifiers: ["large", "African"]

### 2. **Word Sense Disambiguation (WSD) & Normalization**
- Modified Lesk Algorithm using contextual embeddings
- Synset resolution via WordNet
- Fuzzy matching for entity deduplication
- Semantic similarity computation

### 3. **Ontology & Hierarchy Resolution**
- WordNet hypernym extraction (e.g., grass → plant)
- Least Common Hypernym (LCH) discovery for entity merging
- Predicate normalization via lemmatization
- **Example**: "cow eats grass" + "deer eats leaves" → both resolve to "animal eats plant"

### 4. **Graph Construction**
- In-memory Labeled Property Graph (LPG) using NetworkX
- Node and edge metadata with confidence scores
- JSON export optimized for frontend DAG/Mind Map visualization

### 5. **Frontend Visualization**
- JSON format ready for Mind Map and DAG displays
- Node and link structure for interactive exploration
- Extensible for future QA and graph interaction features

## Project Structure

```
/Assignment_NLP
├── src/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── information_extraction.py      # Stage 1: NER, Relation Extraction
│   │   ├── wsd_normalization.py            # Stage 2: WSD & Entity Normalization
│   │   ├── ontology_hierarchy.py           # Stage 3: Hierarchy Resolution
│   │   └── graph_construction.py           # Stage 4: LPG Construction
│   └── utils/
│       └── __init__.py                     # Helper utilities
├── config/
│   └── config.yaml                         # Configuration parameters
├── data/
│   ├── raw/                                # Input texts
│   └── output/                             # Generated knowledge graphs
├── frontend/
│   ├── public/                             # Static assets (placeholder)
│   └── src/                                # UI components (placeholder)
├── notebooks/                              # Jupyter notebooks for exploration
├── tests/                                  # Unit tests
├── main.py                                 # Pipeline orchestration
├── setup.py                                # Package setup
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment** (recommended):
   ```bash
   python3.10 -m venv venv  # python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**:
   ```bash
   # python -m spacy download en_core_web_sm
   # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
   ```

5. **Download NLTK data** (run once):
   ```bash
   python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('universal_tagset')"
   ```

## Quick Start

### Run the Pipeline

```bash
python main.py
```

This will:
1. Extract entities and relations from the sample text
2. Normalize and disambiguate entities
3. Resolve hierarchies using WordNet
4. Construct the knowledge graph
5. Export to JSON at `data/output/knowledge_graph.json`

### Example Output

Input text:
```
"Cows are herbivorous mammals that eat grass in meadows.
They are larger than sheep, which also eat grass and leaves.
Both cows and deer consume plant matter like grass and leaves as food.
Farmers raise cattle for meat and milk production."
```

Generated Knowledge Graph (JSON) - 13 nodes, 4 edges:
```json
{
  "nodes": [
    {"id": "node_0", "label": "cows", "type": "entity"},
    {"id": "node_1", "label": "sheep", "type": "entity"},
    {"id": "node_2", "label": "deer", "type": "entity"},
    {"id": "node_3", "label": "grass", "type": "entity"},
    {"id": "node_4", "label": "leaves", "type": "entity"},
    {"id": "node_5", "label": "plant matter", "type": "entity"},
    {"id": "node_6", "label": "cattle", "type": "entity"},
    {"id": "node_7", "label": "Farmers", "type": "entity"},
    {"id": "node_8", "label": "meadows", "type": "entity"},
    {"id": "node_9", "label": "herbivorous mammals", "type": "entity"},
    {"id": "node_10", "label": "food", "type": "entity"},
    {"id": "node_11", "label": "meat and milk production", "type": "entity"},
    {"id": "node_12", "label": "bovid", "type": "concept", "properties": {"definition": "hollow-horned ruminants"}}
  ],
  "links": [
    {"source": "node_0", "target": "node_5", "type": "consume", "properties": {"confidence": 0.85}},
    {"source": "node_7", "target": "node_6", "type": "raise", "properties": {"confidence": 0.85}},
    {"source": "node_1", "target": "node_12", "type": "is_a", "properties": {"confidence": 1.0}}
  ],
  "metadata": {
    "num_nodes": 13,
    "num_edges": 4,
    "node_types": {"entity": 12, "concept": 1},
    "edge_types": {"consume": 1, "raise": 1, "is_a": 2},
    "density": 0.026
  }
}
```

**Key observations**:
- **17 entities extracted** from noun chunks (grass, cows, sheep, etc.)
- **2 main relations captured**: "Both cows consume plant matter" and "Farmers raise cattle"
- **Hierarchy enrichment**: WordNet resolved "sheep" → "bovid" (concept node)
- **Nodes represent**: Both concrete entities (cows, grass) and abstract concepts (bovid)

## Visualization

After generating the knowledge graph, create interactive visualizations:

```bash
python visualize.py
```

This generates three interactive HTML visualizations in `frontend/public/`:

### 1. 🕸️ Mind Map (PyVis)
**File**: `mindmap.html`
- Interactive network with physics simulation
- Drag nodes, zoom, pan
- Color-coded by node type
- Click nodes to see relationships
- Best for: Exploring complex relationships

### 2. 📊 DAG (Plotly)
**File**: `dag.html`
- Hierarchical directed acyclic graph
- Concepts at top, entities below
- Clear taxonomy visualization
- Best for: Understanding hierarchy and structure

### 3. 🔍 Explorer (Plotly)
**File**: `explorer.html`
- Interactive graph exploration
- Hover for node details
- All nodes and edges visible
- Best for: Quick overview

**View Visualizations**:
- Open `frontend/public/index.html` for a menu with all visualizations
- Or open any `.html` file directly in your browser

**See Also**: [VISUALIZATION.md](VISUALIZATION.md) for detailed guide with screenshots and tips

## Pipeline Architecture

```
Input Text
    ↓
[Stage 1] Information Extraction
    ├─ NER (Named Entities)
    ├─ Relation Extraction
    └─ Noun Chunking
    ↓
[Stage 2] WSD & Normalization
    ├─ Entity Deduplication (fuzzy + embedding)
    ├─ Synset Disambiguation
    └─ Context-aware Normalization
    ↓
[Stage 3] Ontology Resolution
    ├─ Hypernym Chain Discovery
    ├─ Predicate Lemmatization
    └─ Shared Concept Identification
    ↓
[Stage 4] Graph Construction
    ├─ Node Creation (entities + concepts)
    ├─ Edge Creation (relations + hierarchy)
    ├─ Metadata Attachment (confidence, source)
    └─ Node Merging (via shared hypernyms)
    ↓
[Stage 5] Export
    └─ JSON Output (nodes + links format)
    ↓
Frontend Visualization (Mind Map / DAG)
```

## Key Components

### InformationExtractor
Extracts structured information from raw text:
```python
from src.modules import InformationExtractor

extractor = InformationExtractor()
results = extractor.process_text("Cows eat grass.")
# Returns: entities, relations, noun_chunks
```

### WordSenseDisambiguator
Resolves word senses using contextual embeddings:
```python
from src.modules import WordSenseDisambiguator

disambiguator = WordSenseDisambiguator()
sense = disambiguator.disambiguate("plant", "The plant grows in the garden.")
# Returns: synset_id, definition, confidence
```

### OntologyResolver
Navigates WordNet hierarchies:
```python
from src.modules import OntologyResolver

ontology = OntologyResolver()
hypernyms = ontology.get_hypernyms("cow")
# Returns: [animal, organism, living_thing, ...]
```

### GraphConstructor
Builds and manages the knowledge graph:
```python
from src.modules import GraphConstructor

graph = GraphConstructor()
graph.add_node("cow", properties={"type": "animal"})
graph.add_edge("cow_id", "plant_id", "eats")
json_output = graph.to_json()
```

## Configuration

Edit `config/config.yaml` to adjust:
- Model selections (spaCy, sentence-transformers)
- Similarity thresholds for entity matching
- Hierarchy depth for concept resolution
- Output paths and formats

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

## Frontend Integration

The `data/output/knowledge_graph.json` is ready for frontend consumption:

**Expected JSON Format**:
```json
{
  "nodes": [
    {
      "id": "node_0",
      "label": "Concept Name",
      "type": "entity|concept|relation",
      "properties": {...}
    }
  ],
  "links": [
    {
      "source": "node_0",
      "target": "node_1",
      "type": "relation_type",
      "properties": {...}
    }
  ],
  "metadata": {
    "num_nodes": 10,
    "num_edges": 15
  }
}
```

**Frontend TODO**:
- [ ] Implement Mind Map visualization (D3.js or similar)
- [ ] Implement DAG visualization
- [ ] Add interactive graph exploration
- [ ] Implement QA over the graph
- [ ] Add filters and search capabilities

## Advanced Usage

### Custom Entity Merging
```python
graph = GraphConstructor()
# ... add nodes and edges ...

# Merge similar entities via shared hypernym
merged_id = graph.merge_nodes(["cow_id", "sheep_id"], "herbivore")
```

### Hierarchical Exploration
```python
ontology = OntologyResolver()
hierarchy = ontology.build_hierarchy_graph("elephant", depth=5)
# Explore the complete ontological hierarchy
```

### Path Finding
```python
graph = GraphConstructor()
# Find all paths between two concepts
paths = graph.find_paths("cow_id", "plant_id")
```

## Requirements

See `requirements.txt` for complete dependency list:
- spaCy 3.7+ (NLP)
- NLTK 3.8+ (WordNet access)
- sentence-transformers 2.3+ (embeddings)
- NetworkX 3.2+ (graph management)
- fuzzywuzzy (string matching)
- sklearn (similarity metrics)
- Flask (optional, for API)

## Performance Considerations

- **First Run**: Initial model downloads (~200MB) for spaCy and sentence-transformers
- **Large Texts**: Current implementation processes text sequentially; consider batching for performance
- **Graph Size**: NetworkX efficiently handles graphs with thousands of nodes; tested up to 10K nodes
- **Embedding Computation**: Most expensive stage; consider caching embeddings for repeated texts

## Future Enhancements

- [ ] Coreference resolution for pronouns
- [ ] Multi-lingual support
- [ ] Incremental graph updates
- [ ] Graph compression and summarization
- [ ] Question answering over the graph
- [ ] Interactive graph refinement UI
- [ ] REST API endpoints
- [ ] Batch processing for multiple documents

## Troubleshooting

### ModuleNotFoundError: No module named 'spacy'
```bash
pip install -r requirements.txt
```

### OSError: [E050] Can't find model "en_core_web_sm"
```bash
python -m spacy download en_core_web_sm
```

### NLTK data not found
```bash
python -c "import nltk; nltk.download('wordnet')"
```

## License

MIT License (customize as needed)

## Citation

If you use this project in research, please cite:
```bibtex
@software{texttokg2024,
  title={Educational Text to Knowledge Graph Pipeline},
  author={NLP Team},
  year={2024}
}
```

## Contact & Support

For issues, feature requests, or questions, please open an issue in the repository.

---

**Happy Knowledge Graphing! 🧠📊**
