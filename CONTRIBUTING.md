# Development Guide

## Project Philosophy

This project uses a modular, stage-based architecture where each stage:
1. Has a clear, single responsibility
2. Takes input from the previous stage
3. Produces output for the next stage
4. Is independently testable

## Code Style

- Follow PEP 8
- Use type hints for all functions
- Add docstrings (Google style) to all public methods
- Keep functions under 30 lines when possible
- Use dataclasses for data structures

## Adding New Features

### Example: Adding Coreference Resolution

1. **Create a new module** or extend existing:
   ```python
   # src/modules/coreference_resolution.py
   class CoreferenceResolver:
       def resolve(self, text: str, entities: List[Entity]) -> Dict[str, str]:
           """Returns mapping of pronouns to entities."""
   ```

2. **Add to pipeline** (main.py):
   ```python
   def stage_1_5_coreference_resolution(self, ...):
       # Between IE and WSD stages
   ```

3. **Write tests** (tests/test_modules.py):
   ```python
   def test_coreference_resolution():
       resolver = CoreferenceResolver()
       # Test case
   ```

4. **Update documentation**
   - Add to README.md feature list
   - Document in DOCUMENTATION.md
   - Update main.py docstring

## Testing Guidelines

Run tests before committing:
```bash
pytest tests/ -v --cov=src
```

Test coverage should be > 80%.

## Performance Optimization Tips

1. **Batch Processing**: Group similar operations
   ```python
   # Instead of disambiguating one word at a time
   disambiguator.batch_disambiguate(words, context)
   ```

2. **Caching**: Cache expensive operations
   ```python
   @functools.lru_cache(maxsize=1000)
   def get_hypernyms(word):
       return ...
   ```

3. **Lazy Loading**: Load models on demand
   ```python
   @property
   def nlp(self):
       if self._nlp is None:
           self._nlp = spacy.load(...)
       return self._nlp
   ```

## Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Intermediate Results
```python
# In main.py, print after each stage
import json
print(json.dumps(stage1_result, indent=2))
```

### Test Individual Modules
```python
from src.modules import InformationExtractor

extractor = InformationExtractor()
result = extractor.process_text("Test text")
print(result)
```

## Extending Graph Capabilities

### Add Custom Node Types
```python
graph.add_node("concept_name", node_type="custom_type", properties={...})
```

### Add Custom Edge Types
```python
graph.add_edge(node1, node2, relation_type="custom_relation", properties={...})
```

### Implement Graph Algorithms
```python
class GraphAnalyzer:
    def __init__(self, graph_constructor):
        self.graph = graph_constructor.graph
    
    def find_communities(self):
        # Use networkx algorithms
        from networkx.algorithms import community
        return community.best_partition(self.graph)
```

## Frontend Integration Checklist

- [ ] Graph is exported to valid JSON
- [ ] Node IDs are unique strings
- [ ] Links reference existing source/target node IDs
- [ ] Properties field contains all metadata
- [ ] Metadata includes graph statistics
- [ ] Test with example JSON in browser

## Release Checklist

Before releasing a new version:

- [ ] Run full test suite (`pytest tests/`)
- [ ] Update version in setup.py
- [ ] Update CHANGELOG.md
- [ ] Update README.md if needed
- [ ] Check for security vulnerabilities
- [ ] Create git tag: `git tag v0.1.0`
- [ ] Build package: `python setup.py sdist bdist_wheel`

## Common Issues & Solutions

### Issue: "No module named 'spacy'"
**Solution**: `pip install -r requirements.txt`

### Issue: "ModuleNotFoundError: main"
**Solution**: Make sure you're in the project root directory

### Issue: Graph has very few nodes
**Solution**: Check entity extraction - print `stage1_result['entities']`

### Issue: Relations not found
**Solution**: Check relation extraction patterns - might need dependency parsing tuning

## Performance Benchmarks

Expected timings on standard hardware (first run includes model loading):

| Task | Time | Notes |
|------|------|-------|
| Model loading | ~5s | One-time startup |
| NER (1 sentence) | ~50ms | spaCy |
| Embedding generation | ~100ms | sentence-transformers |
| Synset disambiguation | ~200ms | WordNet lookup + similarity |
| Entity normalization | ~150ms | Fuzzy + embedding matching |
| Graph construction | ~50ms | NetworkX operations |
| Full pipeline (1 paragraph) | ~1-2s | End-to-end |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run `pytest tests/ -v`
5. Commit: `git commit -am "Add my feature"`
6. Push: `git push origin feature/my-feature`
7. Create Pull Request with description

## Documentation Standards

All public classes and functions should have:

```python
def function_name(param1: str, param2: int) -> Dict[str, Any]:
    """
    One-line summary.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name("test", 42)
        >>> result['key']
        'value'
    """
    # Implementation
    pass
```

## Future Development Areas

### High Priority
- [ ] Coreference resolution
- [ ] Neural relation extraction
- [ ] Frontend visualization
- [ ] Docker support

### Medium Priority
- [ ] Multi-lingual support
- [ ] Incremental updates
- [ ] Graph embeddings
- [ ] REST API completion

### Low Priority
- [ ] Web UI
- [ ] Mobile app
- [ ] Advanced QA
- [ ] Knowledge graph fusion

---

Happy developing! 🚀
