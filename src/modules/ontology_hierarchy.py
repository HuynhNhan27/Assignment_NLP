"""
Ontology & Hierarchy Module

Handles:
- WordNet hierarchy traversal
- Hypernym resolution (e.g., "Grass" -> "Plant")
- Predicate normalization via lemmatization
- Building hierarchy relationships

Tool: NLTK WordNet
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


@dataclass
class HierarchyNode:
    """Represents a node in the ontology hierarchy."""
    text: str
    synset: str
    hypernyms: List[str]
    hyponyms: List[str]
    definition: str
    level: int  # Distance from original entity


class OntologyResolver:
    """
    Resolve entities to their ontology hierarchy using WordNet.
    
    Features:
    - Hypernym chain extraction
    - Finding least common hypernym
    - Building concept hierarchies
    - Predicate lemmatization
    """
    
    def __init__(self):
        """Initialize Ontology Resolver."""
        self.lemmatizer = WordNetLemmatizer()
    
    def get_synset(self, word: str, pos: Optional[str] = None):
        """
        Get the most common synset for a word.
        
        Args:
            word: Word to get synset for
            pos: Part of speech (optional)
            
        Returns:
            WordNet synset or None
        """
        synsets = wn.synsets(word, pos=pos)
        if synsets:
            return synsets[0]  # Return most common
        return None
    
    def get_hypernyms(self, word: str, depth: int = 5) -> List[HierarchyNode]:
        """
        Get hypernym chain for a word (walking up the hierarchy).
        
        Example: "cow" -> "animal" -> "organism" -> "living_thing" -> "entity"
        
        Args:
            word: Word to trace upward
            depth: Maximum depth to traverse
            
        Returns:
            List of HierarchyNode objects in ascending order
        """
        synset = self.get_synset(word)
        if not synset:
            return []
        
        hierarchy = []
        current = synset
        level = 0
        visited = set()
        
        while current and level < depth:
            if current.name() in visited:
                break
            
            visited.add(current.name())
            
            # Get hypernyms
            hypernyms = [h.name() for h in current.hypernyms()]
            hyponyms = [h.name() for h in current.hyponyms()[:5]]  # Limit hyponyms
            
            node = HierarchyNode(
                text=current.name().split('.')[0].replace('_', ' '),
                synset=current.name(),
                hypernyms=hypernyms,
                hyponyms=hyponyms,
                definition=current.definition(),
                level=level
            )
            hierarchy.append(node)
            
            # Move to first hypernym
            hypernym_list = current.hypernyms()
            current = hypernym_list[0] if hypernym_list else None
            level += 1
        
        return hierarchy
    
    def get_common_hypernym(self, word1: str, word2: str) -> Optional[HierarchyNode]:
        """
        Find the least common hypernym (LCH) between two words.
        
        Example: "cow" and "sheep" -> both have "animal" as LCH
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            HierarchyNode representing the common hypernym, or None
        """
        synset1 = self.get_synset(word1)
        synset2 = self.get_synset(word2)
        
        if not synset1 or not synset2:
            return None
        
        # Get lowest common hypernym
        lch = synset1.lowest_common_hypernyms(synset2)
        if not lch:
            return None
        
        lch_synset = lch[0]
        return HierarchyNode(
            text=lch_synset.name().split('.')[0].replace('_', ' '),
            synset=lch_synset.name(),
            hypernyms=[h.name() for h in lch_synset.hypernyms()],
            hyponyms=[h.name() for h in lch_synset.hyponyms()[:5]],
            definition=lch_synset.definition(),
            level=0
        )
    
    def normalize_predicate(self, predicate: str) -> str:
        """
        Normalize predicate (edge label) through lemmatization.
        
        Examples:
        - "eats" -> "eat"
        - "consuming" -> "consume"
        - "feeds on" -> "feed on"
        
        Args:
            predicate: Predicate/relation text
            
        Returns:
            Lemmatized predicate
        """
        # Tokenize and get POS tags
        tokens = word_tokenize(predicate)
        pos_tags = pos_tag(tokens)
        
        lemmatized = []
        for token, pos in pos_tags:
            if pos.startswith('VB'):  # Verb
                lemma = self.lemmatizer.lemmatize(token, pos='v')
            elif pos.startswith('NN'):  # Noun
                lemma = self.lemmatizer.lemmatize(token, pos='n')
            elif pos.startswith('JJ'):  # Adjective
                lemma = self.lemmatizer.lemmatize(token, pos='a')
            else:
                lemma = token
            lemmatized.append(lemma)
        
        return ' '.join(lemmatized)
    
    def build_hierarchy_graph(self, word: str, depth: int = 3) -> Dict[str, Dict]:
        """
        Build a hierarchy graph for a word.
        
        Returns a nested dictionary structure representing the hierarchy.
        
        Args:
            word: Root word
            depth: Maximum depth
            
        Returns:
            Dictionary representing the hierarchy
        """
        hierarchy = self.get_hypernyms(word, depth=depth)
        
        if not hierarchy:
            return {"word": word, "found": False}
        
        graph = {
            "word": word,
            "hierarchy": []
        }
        
        for node in hierarchy:
            graph["hierarchy"].append({
                "text": node.text,
                "synset": node.synset,
                "definition": node.definition,
                "level": node.level,
                "hypernyms": node.hypernyms,
                "hyponyms": node.hyponyms
            })
        
        return graph
    
    def find_shared_hypernym(self, entities: List[str]) -> Optional[str]:
        """
        Find a shared hypernym among multiple entities.
        
        Useful for merging similar entities in the graph.
        
        Example: ["grass", "leaves", "flowers"] -> "plant"
        
        Args:
            entities: List of entity texts
            
        Returns:
            Shared hypernym or None
        """
        if not entities:
            return None
        
        if len(entities) == 1:
            synset = self.get_synset(entities[0])
            return synset.name().split('.')[0].replace('_', ' ') if synset else None
        
        # Get hypernyms for all entities
        all_hypernyms = []
        for entity in entities:
            synset = self.get_synset(entity)
            if synset:
                hypernyms = [h.name().split('.')[0].replace('_', ' ') 
                            for h in synset.hypernyms()]
                all_hypernyms.append(set(hypernyms))
        
        if not all_hypernyms:
            return None
        
        # Find intersection (shared hypernyms)
        shared = set.intersection(*all_hypernyms) if all_hypernyms else set()
        
        return list(shared)[0] if shared else None
