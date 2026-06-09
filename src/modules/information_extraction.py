"""
Information Extraction (IE) Module

Handles:
- Named Entity Recognition (NER)
- Relation Extraction
- Co-reference Resolution
- Noun Chunking with Head Noun Extraction

Tool: spaCy (en_core_web_sm)
"""

import spacy
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    text: str
    label: str
    start_char: int
    end_char: int
    head_noun: str
    modifiers: List[str]
    confidence: float


@dataclass
class Relation:
    """Represents an extracted relation between entities."""
    subject: str
    predicate: str
    obj: str
    source_sentence: str
    confidence: float


class InformationExtractor:
    """
    Extracts entities, relations, and performs noun chunking on educational texts.
    
    Features:
    - NER using spaCy
    - Relation Extraction (rule-based and dependency-based)
    - Noun Chunking with head noun extraction
    - Optional: Coreference resolution
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the Information Extractor.
        
        Args:
            model_name: spaCy model to load (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please run: python -m spacy download {model_name}")
            raise
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from noun chunks (general nouns) + named entities.
        
        Primary source: noun chunks (e.g., "large animals", "green grass")
        Secondary: named entities for proper nouns
        Filters: Remove pronouns, single-character tokens, articles
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects with head nouns and modifiers
        """
        doc = self.nlp(text)
        entities = []
        seen_spans = set()  # Track span positions to avoid duplicates
        
        # Stopwords to filter
        filter_words = {'that', 'which', 'what', 'who', 'this', 'these', 'those', 
                       'these', 'a', 'an', 'the', 'is', 'are', 'am', 'be'}
        
        # First, extract from noun chunks (general nouns)
        for chunk in doc.noun_chunks:
            span_key = (chunk.start, chunk.end)
            
            # Skip if already seen or is very short
            if span_key in seen_spans or len(chunk.text) < 2:
                continue
            
            # Check if any token in chunk is a pronoun
            is_pronoun = any(token.pos_ == "PRON" for token in chunk)
            if is_pronoun:
                continue
            
            # Skip if chunk is mostly stopwords
            chunk_lower = chunk.text.lower()
            if chunk_lower in filter_words:
                continue
            
            head_noun = self._extract_head_noun(chunk)
            
            # Skip if head noun is a pronoun
            if head_noun.lower() in filter_words or len(head_noun) < 2:
                continue
            
            modifiers = self._extract_modifiers(chunk)
            
            entity = Entity(
                text=chunk.text,
                label="NOUN",  # Noun chunk label
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                head_noun=head_noun,
                modifiers=modifiers,
                confidence=0.90
            )
            entities.append(entity)
            seen_spans.add(span_key)
        
        # Then, add named entities that weren't already captured
        for ent in doc.ents:
            span_key = (ent.start, ent.end)
            
            if span_key not in seen_spans and len(ent.text) > 2:
                head_noun = self._extract_head_noun(ent)
                modifiers = self._extract_modifiers(ent)
                
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    head_noun=head_noun,
                    modifiers=modifiers,
                    confidence=0.95
                )
                entities.append(entity)
                seen_spans.add(span_key)
        
        return entities
    
    def extract_relations(self, text: str) -> List[Relation]:
        """
        Extract relations between entities using dependency parsing.
        
        Strategy:
        1. Find all verbs (especially root verbs)
        2. For each verb, find associated subjects and objects
        3. Map them to noun chunks when possible
        4. Only keep relations where both subject and object are proper nouns/noun chunks
        
        Args:
            text: Input text
            
        Returns:
            List of Relation objects
        """
        doc = self.nlp(text)
        relations = []
        
        # Stopwords to filter
        filter_words = {'that', 'which', 'what', 'who', 'this', 'these', 'those', 
                       'these', 'a', 'an', 'the', 'is', 'are', 'am', 'be'}
        
        # Create a map of noun chunks for quick lookup
        noun_chunks_map = {chunk.text.lower(): chunk for chunk in doc.noun_chunks}
        
        # Find all verbs (potential predicates)
        for token in doc:
            if token.pos_ == "VERB":
                predicate = token.lemma_
                
                # Find subject
                subjects = []
                objects = []
                
                # Search through children for nsubj (subject), dobj (object), etc.
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:  # Subject
                        subjects.append(child)
                    elif child.dep_ in ["dobj", "attr", "pobj"]:  # Object
                        objects.append(child)
                
                # If no direct children found, try ancestors/siblings
                if not subjects or not objects:
                    for ancestor in token.ancestors:
                        if ancestor.pos_ == "VERB":
                            for child in ancestor.children:
                                if child.dep_ == "nsubj" and child not in subjects:
                                    subjects.append(child)
                                elif child.dep_ in ["dobj", "attr"] and child not in objects:
                                    objects.append(child)
                
                # Create relations from all subject-verb-object combinations
                if subjects:
                    for subj in subjects:
                        # Get the noun chunk containing the subject
                        subj_text = self._get_noun_chunk_text(subj, doc)
                        
                        # Filter out pronouns and stopwords
                        if subj_text.lower() in filter_words or subj.pos_ == "PRON":
                            continue
                        
                        if objects:
                            for obj in objects:
                                obj_text = self._get_noun_chunk_text(obj, doc)
                                
                                # Filter out pronouns and stopwords
                                if obj_text.lower() in filter_words or obj.pos_ == "PRON":
                                    continue
                                
                                relation = Relation(
                                    subject=subj_text,
                                    predicate=predicate,
                                    obj=obj_text,
                                    source_sentence=text,
                                    confidence=0.80
                                )
                                relations.append(relation)
                        else:
                            # Even without explicit object, create relation
                            # Try to find objects through prepositions
                            for child in token.children:
                                if child.pos_ == "ADP":  # Preposition
                                    for pobj in child.children:
                                        obj_text = self._get_noun_chunk_text(pobj, doc)
                                        
                                        if obj_text.lower() in filter_words or pobj.pos_ == "PRON":
                                            continue
                                        
                                        relation = Relation(
                                            subject=subj_text,
                                            predicate=predicate,
                                            obj=obj_text,
                                            source_sentence=text,
                                            confidence=0.75
                                        )
                                        relations.append(relation)
        
        # Remove duplicates
        unique_relations = {}
        for rel in relations:
            key = (rel.subject.lower(), rel.predicate.lower(), rel.obj.lower())
            if key not in unique_relations:
                unique_relations[key] = rel
        
        return list(unique_relations.values())
    
    def _get_noun_chunk_text(self, token, doc) -> str:
        """
        Get the complete noun chunk text for a token.
        If token is not in a noun chunk, return token text.
        
        Args:
            token: spaCy token
            doc: spaCy doc
            
        Returns:
            Noun chunk text or token text
        """
        for chunk in doc.noun_chunks:
            if chunk.start <= token.i < chunk.end:
                return chunk.text
        return token.text
    
    def extract_noun_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract noun chunks with head noun and modifiers.
        
        Example: "large African elephants" 
        -> head: "elephants", modifiers: ["large", "African"]
        
        Args:
            text: Input text
            
        Returns:
            List of dicts with chunk text, head noun, and modifiers
        """
        doc = self.nlp(text)
        chunks = []
        
        for chunk in doc.noun_chunks:
            head_noun = self._extract_head_noun(chunk)
            modifiers = self._extract_modifiers(chunk)
            
            chunks.append({
                "text": chunk.text,
                "head_noun": head_noun,
                "modifiers": modifiers,
                "start": chunk.start_char,
                "end": chunk.end_char
            })
        
        return chunks
    
    def _extract_head_noun(self, span) -> str:
        """
        Extract head noun from a span (entity or noun chunk).
        The head noun is typically the rightmost noun in an English phrase.
        
        Args:
            span: spaCy Span object
            
        Returns:
            Head noun text
        """
        # Find the rightmost noun
        for token in reversed(list(span)):
            if token.pos_ in ["NOUN", "PROPN"]:
                return token.text
        return span.text  # Fallback to full span if no noun found
    
    def _extract_modifiers(self, span) -> List[str]:
        """
        Extract modifier words (adjectives, adverbs) from a span.
        
        Args:
            span: spaCy Span object
            
        Returns:
            List of modifier tokens
        """
        modifiers = []
        for token in span:
            if token.pos_ in ["ADJ", "ADV"]:
                modifiers.append(token.text)
        return modifiers
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Full information extraction pipeline.
        
        Args:
            text: Input educational text
            
        Returns:
            Dictionary containing extracted entities, relations, and chunks
        """
        entities = self.extract_entities(text)
        relations = self.extract_relations(text)
        chunks = self.extract_noun_chunks(text)
        
        return {
            "text": text,
            "entities": [asdict(e) for e in entities],
            "relations": [asdict(r) for r in relations],
            "noun_chunks": chunks
        }
