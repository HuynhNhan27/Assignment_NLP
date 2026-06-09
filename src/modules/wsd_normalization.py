"""
Word Sense Disambiguation (WSD) & Entity Normalization Module

Handles:
- Modified Lesk Algorithm using embeddings
- Synset resolution via WordNet
- Fuzzy matching for entity deduplication
- Semantic similarity computation

Tools: sentence-transformers, NLTK WordNet
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import nltk
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DisambiguatedEntity:
    """Entity with resolved sense (synset)."""
    text: str
    synset_id: str
    definition: str
    confidence: float
    original_text: str


class WordSenseDisambiguator:
    """
    Disambiguate word senses using Modified Lesk Algorithm.
    
    Process:
    1. Generate embedding for word in context (sentence context)
    2. For each possible synset, get the definition
    3. Generate embeddings for synset definitions
    4. Compare embeddings using cosine similarity
    5. Select synset with highest similarity
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize WSD module.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(model_name)
    
    def disambiguate(self, word: str, context: str) -> Optional[DisambiguatedEntity]:
        """
        Disambiguate word sense using Modified Lesk Algorithm.
        
        Args:
            word: Word to disambiguate
            context: Context sentence
            
        Returns:
            DisambiguatedEntity with resolved synset, or None if not found
        """
        # Get all synsets for the word
        synsets = wn.synsets(word)
        if not synsets:
            return None
        
        # Generate embedding for the context
        context_embedding = self.embedding_model.encode(context, convert_to_tensor=False)
        
        best_synset = None
        best_score = -1
        
        for synset in synsets:
            # Get definition (gloss)
            definition = synset.definition()
            
            # Generate embedding for the definition
            definition_embedding = self.embedding_model.encode(definition, convert_to_tensor=False)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                [context_embedding], 
                [definition_embedding]
            )[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_synset = synset
        
        if best_synset is None:
            return None
        
        return DisambiguatedEntity(
            text=best_synset.name(),
            synset_id=best_synset.offset(),
            definition=best_synset.definition(),
            confidence=float(best_score),
            original_text=word
        )
    
    def batch_disambiguate(self, words: List[str], context: str) -> List[DisambiguatedEntity]:
        """
        Disambiguate multiple words in the same context.
        
        Args:
            words: List of words to disambiguate
            context: Context sentence
            
        Returns:
            List of DisambiguatedEntity objects
        """
        results = []
        for word in words:
            result = self.disambiguate(word, context)
            if result:
                results.append(result)
        return results


class EntityNormalizer:
    """
    Normalize and deduplicate entities using fuzzy matching and embeddings.
    
    Features:
    - Fuzzy string matching (Levenshtein distance)
    - Semantic similarity matching via embeddings
    - Merging similar entities
    """
    
    def __init__(self, 
                 fuzzy_threshold: float = 0.85,
                 embedding_threshold: float = 0.8,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Entity Normalizer.
        
        Args:
            fuzzy_threshold: Threshold for fuzzy string matching (0-100)
            embedding_threshold: Threshold for embedding similarity (0-1)
            model_name: Sentence transformer model
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.embedding_model = SentenceTransformer(model_name)
    
    def fuzzy_match(self, entity1: str, entity2: str) -> Tuple[bool, float]:
        """
        Check if two entities match using fuzzy matching.
        
        Args:
            entity1: First entity text
            entity2: Second entity text
            
        Returns:
            Tuple of (is_match: bool, similarity_score: float)
        """
        similarity = fuzz.token_set_ratio(entity1, entity2) / 100.0
        is_match = similarity >= self.fuzzy_threshold / 100.0
        return is_match, similarity
    
    def semantic_match(self, entity1: str, entity2: str) -> Tuple[bool, float]:
        """
        Check if two entities match using semantic similarity.
        
        Args:
            entity1: First entity text
            entity2: Second entity text
            
        Returns:
            Tuple of (is_match: bool, similarity_score: float)
        """
        embeddings = self.embedding_model.encode([entity1, entity2])
        similarity = cosine_similarity(
            [embeddings[0]], 
            [embeddings[1]]
        )[0][0]
        
        is_match = similarity >= self.embedding_threshold
        return is_match, float(similarity)
    
    def normalize_entities(self, entities: List[str]) -> Dict[str, str]:
        """
        Normalize a list of entities, merging similar ones.
        
        Returns a mapping of original entity -> canonical entity
        
        Uses:
        - Fuzzy matching (high threshold for string similarity)
        - Semantic similarity (embeddings)
        
        Args:
            entities: List of entity texts
            
        Returns:
            Dict mapping original entity to canonical form
        """
        entity_map = {}
        canonical_entities = {}
        
        # Sort entities by length (longer/more specific first)
        sorted_entities = sorted(entities, key=len, reverse=True)
        
        for entity in sorted_entities:
            if entity in entity_map:
                continue
            
            # Check against existing canonical entities
            found_match = False
            for canonical, members in canonical_entities.items():
                fuzzy_match, fuzzy_score = self.fuzzy_match(entity, canonical)
                
                # Only merge if fuzzy match is VERY high (>0.95)
                if fuzzy_match and fuzzy_score > 0.95:
                    entity_map[entity] = canonical
                    members.append(entity)
                    found_match = True
                    break
                
                # Try semantic similarity only if fuzzy fails
                if not found_match:
                    semantic_match, semantic_score = self.semantic_match(entity, canonical)
                    if semantic_match and semantic_score > 0.90:
                        # Double-check with fuzzy - if very different strings, skip
                        if fuzzy_score < 0.60:  # Prevent merging totally different entities
                            continue
                        
                        entity_map[entity] = canonical
                        members.append(entity)
                        found_match = True
                        break
            
            # If no match found, this becomes a new canonical entity
            if not found_match:
                entity_map[entity] = entity
                canonical_entities[entity] = [entity]
        
        return entity_map
    
    def get_canonical_form(self, entity: str, entity_map: Dict[str, str]) -> str:
        """
        Get the canonical form of an entity.
        
        Args:
            entity: Entity text
            entity_map: Mapping from entity normalization
            
        Returns:
            Canonical entity text
        """
        return entity_map.get(entity, entity)
