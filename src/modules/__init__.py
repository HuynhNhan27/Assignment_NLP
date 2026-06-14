"""
NLP Pipeline Modules

Modular components for:
- Information Extraction (IE)
- Word Sense Disambiguation (WSD) & Normalization
- Ontology & Hierarchy Resolution
- Graph Construction
"""

from .information_extraction import InformationExtractor
from .wsd_normalization import WordSenseDisambiguator, EntityNormalizer
from .ontology_hierarchy import OntologyResolver
from .graph_construction import GraphConstructor

__all__ = [
    'InformationExtractor',
    'WordSenseDisambiguator',
    'EntityNormalizer',
    'OntologyResolver',
    'GraphConstructor',
]
