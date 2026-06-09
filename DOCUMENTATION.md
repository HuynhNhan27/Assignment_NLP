"""
Project Documentation

## Overview
This is a comprehensive Python project for converting educational texts into Knowledge Graphs.
The system uses advanced NLP techniques including Named Entity Recognition, Word Sense 
Disambiguation, and Ontology Resolution to build semantic representations.

## Quick Reference

### Installation
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Download models:
   - `python -m spacy download en_core_web_sm`
   - `python -c "import nltk; nltk.download('wordnet')"`

### Running the Pipeline
```bash
python main.py
```

### Testing
```bash
pytest tests/ -v
```

### API Server
```bash
python api.py
# Then POST to http://localhost:5000/process
```

## Module Descriptions

### src/modules/information_extraction.py
- **InformationExtractor**: Main class for NER, relation extraction, noun chunking
- Uses: spaCy en_core_web_sm
- Key methods:
  - `extract_entities(text)`: Returns Entity objects with head nouns
  - `extract_relations(text)`: Returns SVO relations
  - `extract_noun_chunks(text)`: Returns noun phrases with modifiers
  - `process_text(text)`: Runs full pipeline

### src/modules/wsd_normalization.py
- **WordSenseDisambiguator**: Modified Lesk algorithm using embeddings
- **EntityNormalizer**: Fuzzy matching + semantic deduplication
- Uses: sentence-transformers (all-MiniLM-L6-v2), NLTK WordNet, fuzzywuzzy
- Key methods:
  - `disambiguate(word, context)`: Returns synset with confidence
  - `normalize_entities(entities)`: Returns canonical entity mapping

### src/modules/ontology_hierarchy.py
- **OntologyResolver**: WordNet hierarchy navigation
- Uses: NLTK WordNet, lemmatization
- Key methods:
  - `get_hypernyms(word, depth)`: Walks up hierarchy
  - `get_common_hypernym(word1, word2)`: Finds LCH
  - `normalize_predicate(predicate)`: Lemmatizes relations
  - `find_shared_hypernym(entities)`: Entity merging support

### src/modules/graph_construction.py
- **GraphConstructor**: LPG construction and export
- Uses: NetworkX
- Key methods:
  - `add_node(label, type, properties)`: Creates node
  - `add_edge(source, target, relation)`: Creates relation
  - `merge_nodes(ids, label)`: Consolidates similar entities
  - `to_json()`: Exports for frontend
  - `to_dict()`: Returns Python dict representation

## Data Flow

```
Input Text
    ↓ [IE Module]
Entities + Relations + Noun Chunks
    ↓ [WSD & Normalization]
Canonical Entities + Synsets + Entity Map
    ↓ [Ontology Module]
Hierarchies + Normalized Relations + Shared Concepts
    ↓ [Graph Construction]
LPG: Nodes (entities + concepts) + Edges (relations + hierarchy)
    ↓ [Export]
JSON (nodes + links)
    ↓
Frontend Visualization
```

## Configuration (config/config.yaml)

- `SPACY_MODEL`: spaCy model name (default: en_core_web_sm)
- `EMBEDDING_MODEL`: sentence-transformers model (default: all-MiniLM-L6-v2)
- `FUZZY_THRESHOLD`: String similarity threshold (default: 0.85)
- `EMBEDDING_THRESHOLD`: Semantic similarity threshold (default: 0.8)
- `HIERARCHY_DEPTH`: Max ontology traversal depth (default: 3)

## Example Usage

```python
from src.modules import InformationExtractor, GraphConstructor

# Extract information
extractor = InformationExtractor()
result = extractor.process_text("Dogs eat bones and meat.")

# Build graph
graph = GraphConstructor()
dog_id = graph.add_node("dog", node_type="entity")
bone_id = graph.add_node("bone", node_type="entity")
graph.add_edge(dog_id, bone_id, "eat")

# Export
json_str = graph.to_json()
```

## Performance Notes

- **Memory**: Efficient for texts < 10K words
- **Speed**: ~1-2 seconds per sentence (with model warmup)
- **Bottleneck**: Embedding computation (slow for large batches)
- **Graph Size**: Handles 10K+ nodes smoothly with NetworkX

## Known Limitations

1. No pronoun coreference resolution yet
2. Relation extraction is rule-based (could use neural models)
3. Monolingual (English only)
4. No incremental updates (full recomputation needed)
5. Graph visualization is frontend-only (no built-in UI)

## Future Work

- [ ] Coreference resolution for pronouns
- [ ] Neural relation extraction (transformers)
- [ ] Multi-document graph fusion
- [ ] Incremental graph updates
- [ ] REST API completion
- [ ] Interactive web UI
- [ ] Question answering module
- [ ] Knowledge graph embedding (Node2Vec, TransE)

## Testing

Test suite in `tests/test_modules.py`:
- Entity extraction tests
- Relation extraction tests
- Hypernym chain tests
- Predicate normalization tests

Run: `pytest tests/ -v`

## Contributing

1. Follow PEP 8 style guide
2. Add tests for new features
3. Update documentation
4. Test with provided example texts

## Contact

For questions or improvements, please refer to the README.md.
"""
