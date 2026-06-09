"""
Unit tests for the pipeline modules
"""

import pytest
from src.modules import InformationExtractor, OntologyResolver


class TestInformationExtractor:
    """Test the Information Extraction module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = InformationExtractor()
    
    def test_extract_entities(self):
        """Test entity extraction."""
        text = "Apple Inc. was founded by Steve Jobs in California."
        entities = self.extractor.extract_entities(text)
        
        assert len(entities) > 0
        assert any(e.label == "ORG" for e in entities)  # Apple Inc.
        assert any(e.label == "PERSON" for e in entities)  # Steve Jobs
    
    def test_extract_relations(self):
        """Test relation extraction."""
        text = "Dogs eat meat and bones."
        relations = self.extractor.extract_relations(text)
        
        assert len(relations) > 0


class TestOntologyResolver:
    """Test the Ontology Resolver module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ontology = OntologyResolver()
    
    def test_get_hypernyms(self):
        """Test hypernym retrieval."""
        hierarchy = self.ontology.get_hypernyms("dog")
        
        assert len(hierarchy) > 0
        assert hierarchy[0].text.lower() == "dog"
    
    def test_normalize_predicate(self):
        """Test predicate normalization."""
        assert self.ontology.normalize_predicate("eats") == "eat"
        assert self.ontology.normalize_predicate("running") == "run"
