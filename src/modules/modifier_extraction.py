"""
Modifier Extraction Module

Handles comprehensive modifier extraction including:
- Adjectives, Noun Adjuncts, Possessives
- Participles (VBG, VBN forms)
- Quantifiers, Negations
- Temporal expressions
- Prepositional Phrases, Relative Clauses, Appositives
- Hierarchical/Nested Modifier Relationships
- Modifier Chains (e.g., "highly efficient" → "efficient" for "algorithm")

Tool: spaCy dependency parsing + custom extraction logic
"""

import spacy
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum


class ModifierType(Enum):
    """Comprehensive classifier for modifier types"""
    # Basic modifiers
    ADJECTIVE = "adjective"                # blue, happy, efficient
    NOUN_ADJUNCT = "noun_adjunct"          # coffee (in "coffee cup")
    POSSESSIVE = "possessive"              # Sarah's, my
    
    # Participles
    PARTICIPLE_ING = "participle_ing"      # swimming, running
    PARTICIPLE_ED = "participle_ed"        # broken, stolen, implemented
    
    # Quantifiers
    QUANTIFIER = "quantifier"              # two, few, all, some
    
    # Complex phrases
    PREPOSITIONAL_PHRASE = "prep_phrase"   # on the roof, across factories
    RELATIVE_CLAUSE = "relative_clause"    # that you lent me, who had fever
    APPOSITIVE = "appositive"              # scientist, Marie Curie
    ADVERBIAL_PHRASE = "adverbial_phrase"  # very quickly, across all
    
    # NEW TYPES: Negation & Temporal
    NEGATION = "negation"                  # not, no, never, neither
    TEMPORAL = "temporal"                  # was, were, will, shall, V-ed (past)


@dataclass
class Modifier:
    """Structured representation of a single modifier with hierarchical support"""
    text: str                              # Raw text
    type: ModifierType                     # Classification
    head_token: str                        # Core word (lemma)
    position: str                          # "pre" or "post" relative to noun
    dependency: Optional[str]              # spaCy dep_ label
    confidence: float                      # 0.0-1.0
    constituents: List[str] = field(default_factory=list)  # Sub-components
    modifiers: List['Modifier'] = field(default_factory=list)  # ← HIERARCHICAL: modifiers of this modifier
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "type": self.type.value,
            "head_token": self.head_token,
            "position": self.position,
            "dependency": self.dependency,
            "confidence": self.confidence,
            "constituents": self.constituents,
            "modifiers": [m.to_dict() for m in self.modifiers]  # Nested modifiers
        }


class ModifierExtractor:
    """Comprehensive modifier extraction engine with hierarchical support"""
    
    NEGATION_MARKERS = {
        "not", "no", "never", "neither", "nobody", "nothing", 
        "nor", "cannot", "can't", "won't", "shouldn't", "wouldn't"
    }
    
    TEMPORAL_VERBS = {
        # Auxiliary verbs indicating tense/aspect
        "be", "is", "are", "am", "was", "were", "being", "been",
        "have", "has", "had",
        "will", "shall", "would", "should",
        "may", "might", "must", "can", "could"
    }
    
    ADVERBIAL_INTENSIFIERS = {
        "very", "highly", "extremely", "incredibly", "remarkably",
        "quite", "rather", "fairly", "pretty", "somewhat",
        "really", "so", "too", "much", "more", "most",
        "less", "least", "almost", "barely", "hardly"
    }
    
    def __init__(self):
        """Initialize modifier extractor"""
        pass
    
    def extract_modifiers_from_chunk(self, chunk) -> List[Modifier]:
        """
        Extract all modifiers from a noun chunk with hierarchical relationships.
        
        Args:
            chunk: spaCy Span object representing a noun chunk
            
        Returns:
            List of Modifier objects with hierarchical structure
        """
        modifiers = []
        
        # 1. NEGATION MARKERS
        negation_mods = self._extract_negations(chunk)
        modifiers.extend(negation_mods)
        
        # 2. TEMPORAL EXPRESSIONS
        temporal_mods = self._extract_temporal_modifiers(chunk)
        modifiers.extend(temporal_mods)
        
        # 3. QUANTIFIERS (before adjectives to establish hierarchy)
        quantifier_mods = self._extract_quantifiers(chunk)
        quantifier_dict = {m.head_token: m for m in quantifier_mods}
        modifiers.extend(quantifier_mods)
        
        # 4. ADJECTIVES (with hierarchical modifiers attached)
        adjective_mods = self._extract_adjectives(chunk, quantifier_dict, negation_mods)
        modifiers.extend(adjective_mods)
        
        # 5. NOUN ADJUNCTS
        noun_adjunct_mods = self._extract_noun_adjuncts(chunk)
        modifiers.extend(noun_adjunct_mods)
        
        # 6. POSSESSIVES
        possessive_mods = self._extract_possessives(chunk)
        modifiers.extend(possessive_mods)
        
        # 7. PARTICIPLES
        participle_mods = self._extract_participles(chunk)
        modifiers.extend(participle_mods)
        
        # 8. PREPOSITIONAL PHRASES
        prep_mods = self._extract_prepositional_phrases(chunk.root)
        modifiers.extend(prep_mods)
        
        return modifiers
    
    def _extract_negations(self, chunk) -> List[Modifier]:
        """Extract negation markers (not, no, never, etc.)"""
        negations = []
        for token in chunk:
            if token.dep_ == "neg" or token.lemma_.lower() in self.NEGATION_MARKERS:
                negations.append(Modifier(
                    text=token.text,
                    type=ModifierType.NEGATION,
                    head_token=token.lemma_.lower(),
                    position="pre",
                    dependency="neg",
                    confidence=0.98
                ))
        return negations
    
    def _extract_temporal_modifiers(self, chunk) -> List[Modifier]:
        """
        Extract temporal expressions including auxiliary verbs and past participles.
        Examples: "was", "were", "will be", "V-ed" forms
        """
        temporal_mods = []
        seen_indices = set()
        
        for i, token in enumerate(chunk):
            if i in seen_indices:
                continue
            
            # Auxiliary verbs (be, have, will, shall, etc.)
            if token.lemma_.lower() in self.TEMPORAL_VERBS:
                # Build auxiliary chain (e.g., "will have been")
                aux_parts = [token.text]
                j = i + 1
                while j < len(chunk) and chunk[j].pos_ in ["AUX", "VERB"]:
                    aux_parts.append(chunk[j].text)
                    seen_indices.add(j)
                    j += 1
                
                temporal_text = " ".join(aux_parts)
                temporal_mods.append(Modifier(
                    text=temporal_text,
                    type=ModifierType.TEMPORAL,
                    head_token=token.lemma_.lower(),
                    position="pre",
                    dependency=token.dep_,
                    confidence=0.95,
                    constituents=aux_parts
                ))
                seen_indices.add(i)
            
            # Past participles used as temporal markers (e.g., "implemented", "stored")
            elif token.tag_ == "VBN" and token.pos_ == "VERB":
                temporal_mods.append(Modifier(
                    text=token.text,
                    type=ModifierType.TEMPORAL,
                    head_token=token.lemma_.lower(),
                    position="post",
                    dependency=token.dep_,
                    confidence=0.90
                ))
                seen_indices.add(i)
        
        return temporal_mods
    
    def _extract_quantifiers(self, chunk) -> List[Modifier]:
        """Extract quantifiers (one, two, few, all, some, etc.)"""
        quantifiers = {
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "few", "several", "many", "all", "some", "any", "no", "both", "each",
            "every", "either", "neither", "whole", "entire", "single", "multiple"
        }
        quantifier_mods = []
        
        for token in chunk:
            if token.lemma_.lower() in quantifiers or token.pos_ == "NUM":
                quantifier_mods.append(Modifier(
                    text=token.text,
                    type=ModifierType.QUANTIFIER,
                    head_token=token.lemma_.lower(),
                    position="pre",
                    dependency=token.dep_,
                    confidence=0.92
                ))
        
        return quantifier_mods
    
    def _extract_adjectives(self, chunk, quantifier_dict: Dict, negation_mods: List) -> List[Modifier]:
        """
        Extract adjectives with hierarchical modifiers.
        Adjectives can be modified by:
        - Adverbial intensifiers (very, highly, extremely)
        - Negations (not efficient)
        - Quantifiers (somewhat efficient)
        """
        adjective_mods = []
        
        for token in chunk:
            if token.pos_ == "ADJ":
                # Find hierarchical modifiers for this adjective
                nested_modifiers = self._find_modifiers_of_token(token, chunk)
                
                adj_mod = Modifier(
                    text=token.text,
                    type=ModifierType.ADJECTIVE,
                    head_token=token.lemma_.lower(),
                    position="pre",
                    dependency=token.dep_,
                    confidence=0.95,
                    modifiers=nested_modifiers
                )
                adjective_mods.append(adj_mod)
        
        return adjective_mods
    
    def _find_modifiers_of_token(self, token, chunk) -> List[Modifier]:
        """
        Find modifiers that directly modify a given token.
        Examples:
        - "very" modifies "efficient"
        - "not" modifies "efficient"
        - "highly" modifies "efficient"
        
        This creates the hierarchical structure: "highly efficient"
        where "highly" modifies "efficient"
        """
        nested_mods = []
        
        # Check for adverbial modifiers (advmod dependencies)
        for child in token.children:
            if child.dep_ in ["advmod", "amod"]:
                if child.lemma_.lower() in self.ADVERBIAL_INTENSIFIERS:
                    nested_mods.append(Modifier(
                        text=child.text,
                        type=ModifierType.ADVERBIAL_PHRASE,
                        head_token=child.lemma_.lower(),
                        position="pre",
                        dependency="advmod",
                        confidence=0.93
                    ))
        
        # Check for negation modifying this adjective
        for child in token.children:
            if child.dep_ == "neg":
                nested_mods.append(Modifier(
                    text=child.text,
                    type=ModifierType.NEGATION,
                    head_token=child.lemma_.lower(),
                    position="pre",
                    dependency="neg",
                    confidence=0.98
                ))
        
        return nested_mods
    
    def _extract_noun_adjuncts(self, chunk) -> List[Modifier]:
        """Extract noun adjuncts (nouns modifying the head noun)"""
        noun_adjunct_mods = []
        
        for token in chunk:
            if token.pos_ == "NOUN" and token != chunk.root and token.dep_ == "compound":
                noun_adjunct_mods.append(Modifier(
                    text=token.text,
                    type=ModifierType.NOUN_ADJUNCT,
                    head_token=token.lemma_.lower(),
                    position="pre",
                    dependency="compound",
                    confidence=0.90
                ))
        
        return noun_adjunct_mods
    
    def _extract_possessives(self, chunk) -> List[Modifier]:
        """Extract possessive modifiers (Sarah's, my, his, etc.)"""
        possessive_mods = []
        
        for token in chunk:
            if token.dep_ == "poss":
                possessive_mods.append(Modifier(
                    text=token.text,
                    type=ModifierType.POSSESSIVE,
                    head_token=token.lemma_.lower(),
                    position="pre",
                    dependency="poss",
                    confidence=0.95
                ))
        
        return possessive_mods
    
    def _extract_participles(self, chunk) -> List[Modifier]:
        """Extract participles (VBG and VBN forms acting as adjectives)"""
        participle_mods = []
        
        for token in chunk:
            if token.pos_ == "VERB" and token.tag_ in ["VBG", "VBN"]:
                mod_type = (
                    ModifierType.PARTICIPLE_ING if token.tag_ == "VBG"
                    else ModifierType.PARTICIPLE_ED
                )
                participle_mods.append(Modifier(
                    text=token.text,
                    type=mod_type,
                    head_token=token.lemma_.lower(),
                    position="post" if token.i > chunk.root.i else "pre",
                    dependency=token.dep_,
                    confidence=0.88
                ))
        
        return participle_mods
    
    def _extract_prepositional_phrases(self, head_token) -> List[Modifier]:
        """Extract prepositional phrase modifiers attached to head noun"""
        prep_mods = []
        
        for child in head_token.children:
            if child.dep_ == "prep":
                # Collect full prepositional phrase and its object
                prep_parts = [child.text]
                pobj_tokens = []
                
                for subchild in child.children:
                    if subchild.dep_ == "pobj":
                        # Recursively get all tokens in the prepositional object
                        pobj_tokens.extend([t.text for t in subchild.subtree])
                    else:
                        prep_parts.append(subchild.text)
                
                prep_parts.extend(pobj_tokens)
                prep_text = " ".join(prep_parts)
                
                prep_mods.append(Modifier(
                    text=prep_text,
                    type=ModifierType.PREPOSITIONAL_PHRASE,
                    head_token=child.lemma_.lower(),
                    position="post",
                    dependency="prep",
                    confidence=0.90,
                    constituents=prep_parts
                ))
        
        return prep_mods
    
    def extract_clause_modifiers(self, head_token) -> List[Modifier]:
        """Extract relative and appositive clause modifiers"""
        clause_mods = []
        
        # Relative clauses
        for child in head_token.children:
            if child.dep_ == "relcl":
                clause_tokens = [t.text for t in child.subtree]
                clause_text = " ".join(clause_tokens)
                
                clause_mods.append(Modifier(
                    text=clause_text,
                    type=ModifierType.RELATIVE_CLAUSE,
                    head_token=child.lemma_.lower(),
                    position="post",
                    dependency="relcl",
                    confidence=0.85,
                    constituents=clause_tokens
                ))
        
        # Appositive clauses
        for child in head_token.children:
            if child.dep_ == "appos":
                clause_tokens = [t.text for t in child.subtree]
                clause_text = " ".join(clause_tokens)
                
                clause_mods.append(Modifier(
                    text=clause_text,
                    type=ModifierType.APPOSITIVE,
                    head_token=child.lemma_.lower(),
                    position="post",
                    dependency="appos",
                    confidence=0.85,
                    constituents=clause_tokens
                ))
        
        return clause_mods
    
    def extract_all_modifiers(self, chunk, head_token) -> List[Modifier]:
        """Extract all modifiers including clauses"""
        modifiers = self.extract_modifiers_from_chunk(chunk)
        clause_mods = self.extract_clause_modifiers(head_token)
        modifiers.extend(clause_mods)
        return modifiers
    
    def get_modifier_summary(self, modifiers: List[Modifier]) -> Dict[str, int]:
        """Get count of each modifier type"""
        summary = {}
        for mod in modifiers:
            key = mod.type.value
            summary[key] = summary.get(key, 0) + 1
        return summary
