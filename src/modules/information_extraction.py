"""
Information Extraction (IE) Module

Handles:
- Named Entity Recognition (NER)
- Relation Extraction
- Co-reference Resolution
- Noun Chunking with Head Noun Extraction
- Comprehensive Modifier Extraction with Hierarchical Support

Tool: spaCy (en_core_web_sm)
"""

import uuid
import spacy
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field

from .modifier_extraction import Modifier, ModifierExtractor, ModifierType


@dataclass
class Entity:
    """Represents an extracted entity with metadata and structured modifiers."""
    text: str
    label: str
    start_char: int
    end_char: int
    head_noun: str
    lemma: str  # Lemma of the head noun
    modifiers: List[Modifier] = field(default_factory=list)
    modifier_summary: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.9
    canonical_id: Optional[str] = None  # ID định danh duy nhất cho KG

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
            self.preprocessor = spacy.load(model_name)
            
            # Try to add fastcoref for coreference resolution (optional)
            try:
                self.preprocessor.add_pipe("fastcoref", config={"model_architecture": "FCoref", "device": "cpu"})
                self.has_coref = True
            except ValueError as e:
                print(f"⚠️  Warning: fastcoref not available - coreference resolution disabled")
                print(f"    To enable, run: pip install spacy-coref")
                self.has_coref = False
                
        except OSError:
            print(f"Model {model_name} not found. Please run: python -m spacy download {model_name}")
            raise
        
        # Initialize modifier extraction engine
        self.modifier_extractor = ModifierExtractor()

    def preprocess(self, text: str):
        """Preprocess text with optional coreference resolution."""
        if self.has_coref:
            try:
                return self.preprocessor(text, component_cfg={"fastcoref": {"resolve_text": True}})
            except Exception as e:
                print(f"⚠️  Coreference resolution failed: {e}. Falling back to plain spaCy parsing.")
        return self.nlp(text)

    def extract_base_entities(self, text: str, doc) -> List[Entity]:
        """Extract entities from NER and noun chunks without coreference aliases."""
        entities = []
        seen_token_indices = set()

        for ent in doc.ents:
            seen_token_indices.update(range(ent.start, ent.end))

            head_noun = self._extract_head_noun(ent)
            modifiers = self._extract_modifiers(ent)
            head_noun_lemma = self._get_lemma(ent.root)
            modifier_summary = self.modifier_extractor.get_modifier_summary(modifiers)
            
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                head_noun=head_noun,
                lemma=head_noun_lemma,
                modifiers=modifiers,
                modifier_summary=modifier_summary,
                confidence=0.95,
                canonical_id=str(uuid.uuid4())
            ))

        for chunk in doc.noun_chunks:
            # FIX: Only skip if the core root noun of the chunk is already captured by an NER entity.
            # This prevents skipping valid noun phrases when an NER entity only covers a prefix modifier or quantifier (e.g., "Three").
            if chunk.root.i in seen_token_indices:
                continue

            if chunk.root.pos_ == "PRON":
                continue

            head_noun = chunk.root.text
            modifiers = self._extract_modifiers(chunk)
            head_noun_lemma = self._get_lemma(chunk.root)
            modifier_summary = self.modifier_extractor.get_modifier_summary(modifiers)
            
            entities.append(Entity(
                text=chunk.text,
                label="NOUN",
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                head_noun=head_noun,
                lemma=head_noun_lemma,
                modifiers=modifiers,
                modifier_summary=modifier_summary,
                confidence=0.90,
                canonical_id=str(uuid.uuid4())
            ))

            seen_token_indices.update(range(chunk.start, chunk.end))

        return entities

    def _resolve_relative_clauses(self, doc, entities: List[Entity]) -> List[str]:
        """
        Với mỗi relative clause, tạo câu mới dạng:
        "<entity.text> <clause_verb_subtree>."
        Subject được lookup từ entities_1 thay vì dùng antecedent.text trực tiếp.
        """
        generated = []
        entity_by_lemma = {e.lemma: e for e in entities}

        for token in doc:
            if token.dep_ != "relcl":
                continue

            antecedent = token.head

            subj_ent = entity_by_lemma.get(antecedent.lemma_.lower())
            if subj_ent is None:
                subj_ent = next((e for e in entities if e.head_noun.lower() == antecedent.lemma_.lower()), None)
            if subj_ent is None:
                continue

            rel_pron = next((c for c in token.children if c.lower_ in {"which", "that", "who", "whom"}), None)
            if rel_pron is None:
                continue

            subtree_tokens = [t for t in sorted(token.subtree, key=lambda t: t.i) if t != rel_pron]
            clause = " ".join(t.text for t in subtree_tokens)
            generated.append(f"{subj_ent.text} {clause}.")

        return generated

    def _resolve_appositions(self, doc, entities: List[Entity]) -> List[str]:
        """
        Với mỗi apposition, tạo câu mới dạng:
        "<head_ent.text> is <appos_ent.text>."
        Cả head và appos đều được lookup từ entities_1.
        """
        generated = []
        entity_by_lemma = {e.lemma: e for e in entities}

        for token in doc:
            if token.dep_ != "appos":
                continue

            head = token.head

            head_ent = entity_by_lemma.get(head.lemma_.lower())
            if head_ent is None:
                head_ent = next((e for e in entities if e.head_noun.lower() == head.lemma_.lower()), None)
            if head_ent is None:
                continue

            appos_ent = entity_by_lemma.get(token.lemma_.lower())
            if appos_ent is None:
                appos_ent = next((e for e in entities if e.head_noun.lower() == token.lemma_.lower()), None)
            if appos_ent is None:
                continue

            if head_ent.lemma == appos_ent.lemma:
                continue

            generated.append(f"{head_ent.text} is {appos_ent.text}.")

        return generated

    def extract_entities(self, text: str, doc=None) -> List[Entity]:
        """Extract entities and optionally add coreference aliases."""
        if doc is None:
            doc = self.preprocess(text)

        base_entities = self.extract_base_entities(text, doc)

        coref_clusters = getattr(getattr(doc, "_", None), "coref_clusters", None)
        if coref_clusters:
            for cluster in coref_clusters:
                rep_span = cluster[0]
                canonical_id = None

                for ent in base_entities:
                    if not (rep_span[1] <= ent.start_char or rep_span[0] >= ent.end_char):
                        canonical_id = ent.canonical_id
                        break

                if not canonical_id:
                    canonical_id = str(uuid.uuid4())

                for mention_span in cluster[1:]:
                    mention_text = text[mention_span[0]:mention_span[1]]
                    alias_entity = Entity(
                        text=mention_text,
                        label="COREF_PRON",
                        start_char=mention_span[0],
                        end_char=mention_span[1],
                        head_noun=mention_text,
                        lemma=mention_text.lower(),
                        modifiers=[],
                        modifier_summary={},
                        confidence=0.90,
                        canonical_id=canonical_id
                    )
                    base_entities.append(alias_entity)

        return base_entities

    def extract_relations(self, text: str, entities: List[Entity] = None, doc=None) -> List[Relation]:
        """
        Trích xuất quan hệ DỰA TRÊN danh sách thực thể đã chốt (Entity-driven).
        Chỉ tạo quan hệ nếu Subject và Object khớp với các Entity hợp lệ.
        """
        if doc is None:
            doc = self.preprocess(text)

        if entities is None:
            entities = self.extract_entities(text, doc)

        relations = []
        
        # BƯỚC 1: XÂY DỰNG TỪ ĐIỂN MAPPING TOKEN -> ENTITY
        token_to_entity = {}
        for ent in entities:
            span = doc.char_span(ent.start_char, ent.end_char)
            if span is not None:
                for token in span:
                    token_to_entity[token.i] = ent

        # BƯỚC 2: QUÉT 1 VÒNG DUY NHẤT LẤY TOÀN BỘ QUAN HỆ
        for token in doc:
            # --- LOẠI 1: QUAN HỆ QUA ĐỘNG TỪ ---
            if token.pos_ == "VERB" or self._get_lemma(token) == "be":
                predicate_parts = []
                
                if token.pos_ == "VERB":
                    neg = next((c for c in token.children if c.dep_ == "neg"), None)
                    if neg: predicate_parts.append(neg.lemma_.lower())
                    
                    predicate_parts.append(token.lemma_.lower())
                    
                    prt = next((c for c in token.children if c.dep_ == "prt"), None)
                    if prt: predicate_parts.append(prt.lemma_.lower())
                    
                    subjects = self._get_subjects(token)
                    objects, prep_lemma = self._get_objects(token)
                    
                    if prep_lemma: predicate_parts.append(prep_lemma.lower())
                else: # token.lemma_ == "be"
                    predicate_parts.append("has_attribute")
                    subjects = self._get_subjects(token)
                    objects = [c for c in token.children if c.dep_ in ("acomp", "attr", "oprd")]
                    
                predicate_text = " ".join(predicate_parts)
                is_passive = any(c.dep_ == "nsubjpass" for c in token.children)
                advmods = [c for c in token.children if c.dep_ == "advmod"]
                
                for subj_token in subjects:
                    subj_ent = token_to_entity.get(subj_token.i)
                    if not subj_ent: continue
                        
                    for obj_token in objects:
                        obj_ent = token_to_entity.get(obj_token.i)
                        
                        if obj_ent:
                            if subj_ent.canonical_id == obj_ent.canonical_id: continue
                            
                            final_subj = obj_ent.canonical_id if is_passive else subj_ent.canonical_id
                            final_obj = subj_ent.canonical_id if is_passive else obj_ent.canonical_id
                                
                            relations.append(Relation(
                                subject=final_subj, predicate=predicate_text, obj=final_obj,
                                source_sentence=text, confidence=0.85
                            ))
                            
                    for adv in advmods:
                        relations.append(Relation(
                            subject=subj_ent.canonical_id, predicate=f"{token.lemma_.lower()}_manner",
                            obj=adv.lemma_.lower(), source_sentence=text, confidence=0.75
                        ))

            # --- LOẠI 2: QUAN HỆ ĐỒNG VỊ (APPOSITION) ---
            if token.dep_ == "appos":
                head = token.head
                subj_ent = token_to_entity.get(head.i)
                appos_ent = token_to_entity.get(token.i)
                
                if subj_ent and appos_ent and subj_ent.canonical_id != appos_ent.canonical_id:
                    relations.append(Relation(
                        subject=subj_ent.canonical_id, predicate="is", obj=appos_ent.canonical_id,
                        source_sentence=text, confidence=0.90
                    ))

        # Lọc trùng lặp Relation
        unique_relations = {(r.subject, r.predicate, r.obj): r for r in relations}
        return list(unique_relations.values())
    
    def _get_lemma(self, token) -> str:
        """
        Trả về lemma chuẩn hóa:
        - PROPN hoặc NER entity: giữ nguyên case gốc của spaCy
        - NOUN thường: lowercase
        """
        if token.pos_ == "PROPN" or token.ent_type_:
            return token.lemma_
        else:
            return token.lemma_.lower()

    def _get_subjects(self, verb_token) -> List[Any]:
        """Tìm các chủ ngữ của một động từ, xử lý cả liên từ và rút gọn chủ ngữ."""
        subjects = []
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
                if child.lower_ in {"that", "which", "who", "whom"} and verb_token.dep_ == "relcl":
                    subjects.append(verb_token.head)
                else:
                    subjects.append(child)
                
                for grandchild in child.children:
                    if grandchild.dep_ == "conj":
                        subjects.append(grandchild)
                        
        if not subjects and verb_token.dep_ == "conj":
            head_verb = verb_token.head
            if head_verb.pos_ == "VERB":
                subjects = self._get_subjects(head_verb)
                
        return subjects

    def _get_objects(self, verb_token) -> Tuple[List[Any], str]:
        """Tìm tân ngữ và giới từ (nếu có) của một động từ."""
        objects = []
        prep_lemma = None
        
        for child in verb_token.children:
            if child.dep_ in ["dobj", "attr", "oprd"]:
                if child.lower_ in {"that", "which", "who", "whom"} and verb_token.dep_ == "relcl":
                    objects.append(verb_token.head)
                else:
                    objects.append(child)
                    
                for grandchild in child.children:
                    if grandchild.dep_ == "conj":
                        objects.append(grandchild)
                        
            elif child.dep_ in ["prep", "agent"]:
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        if pobj.lower_ in {"that", "which", "whom"} and verb_token.dep_ == "relcl":
                            objects.append(verb_token.head)
                        else:
                            objects.append(pobj)
                        
                        if child.dep_ == "prep":
                            prep_lemma = child.lemma_.lower()
                            
                        for grandchild in pobj.children:
                            if grandchild.dep_ == "conj":
                                objects.append(grandchild)
                                
        return objects, prep_lemma
    
    def _get_noun_chunk_text(self, token, doc) -> str:
        """Get the complete noun chunk text for a token."""
        for chunk in doc.noun_chunks:
            if chunk.start <= token.i < chunk.end:
                return chunk.text
        return token.text
    
    def _extract_head_noun(self, span) -> str:
        """Extract head noun from a span (entity or noun chunk)."""
        for token in reversed(list(span)):
            if token.pos_ in ["NOUN", "PROPN"]:
                return token.text
        return span.text
    
    def _extract_modifiers(self, span) -> List[Modifier]:
        """Extract comprehensive modifiers from a span with hierarchical support."""
        return self.modifier_extractor.extract_all_modifiers(span, span.root)

    def _modifier_to_dict(self, modifier: Modifier) -> Dict[str, Any]:
        """Convert a Modifier object into a JSON-serializable dictionary."""
        return modifier.to_dict()

    def _entity_to_dict(self, entity: Entity) -> Dict[str, Any]:
        """Convert an Entity object into a JSON-serializable dictionary."""
        return {
            "text": entity.text,
            "label": entity.label,
            "start_char": entity.start_char,
            "end_char": entity.end_char,
            "head_noun": entity.head_noun,
            "lemma": entity.lemma,
            "modifiers": [self._modifier_to_dict(mod) for mod in entity.modifiers],
            "modifier_summary": entity.modifier_summary,
            "confidence": entity.confidence,
            "canonical_id": entity.canonical_id,
        }
    
    def filter_entities_by_relations(self, entities: List[Entity], relations: List[Relation]) -> List[Entity]:
        related_entity_ids = {
            rel.subject for rel in relations
        } | {
            rel.obj for rel in relations
        }

        filtered_entities = [
            entity
            for entity in entities
            if entity.canonical_id in related_entity_ids
        ]

        return filtered_entities
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Full information extraction pipeline."""
        doc = self.preprocess(text)
        entities = self.extract_entities(text, doc)
        relations = self.extract_relations(text, entities, doc)

        return {
            "text": text,
            "entities": [asdict(e) for e in entities],
            "relations": [asdict(r) for r in relations],
        }