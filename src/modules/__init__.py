"""
NLP Pipeline Modules

Modular components for:
- Information Extraction (IE)
- Modifier Extraction (with hierarchical support)
- Word Sense Disambiguation (WSD) & Normalization
- Ontology & Hierarchy Resolution
- Graph Construction
"""

from .information_extraction import InformationExtractor
from .modifier_extraction import ModifierExtractor, Modifier, ModifierType
from .wsd_normalization import WordSenseDisambiguator, EntityNormalizer
from .ontology_hierarchy import OntologyResolver
from .graph_construction import GraphConstructor

__all__ = [
    'InformationExtractor',
    'ModifierExtractor',
    'Modifier',
    'ModifierType',
    'WordSenseDisambiguator',
    'EntityNormalizer',
    'OntologyResolver',
    'GraphConstructor',
]
