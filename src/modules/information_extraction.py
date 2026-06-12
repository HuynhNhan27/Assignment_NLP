"""
Information Extraction (IE) Module

Handles:
- Named Entity Recognition (NER)
- Relation Extraction
- Co-reference Resolution
- Noun Chunking with Head Noun Extraction

Tool: spaCy (en_core_web_sm)
"""

import uuid
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
    lemma: str # Tạm thời để full lowercase, chưa xét đến tên riêng
    modifiers: List[str]
    confidence: float
    canonical_id: str = None  # ID định danh duy nhất cho KG

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
            self.preprocessor = spacy.load(model_name)
            self.preprocessor.add_pipe("fastcoref", config={"model_architecture": "FCoref", "device": "cpu"})
        except OSError:
            print(f"Model {model_name} not found. Please run: python -m spacy download {model_name}")
            raise

    def preprocess(self, text: str) -> str:
        doc = self.preprocessor(text, component_cfg={"fastcoref": {"resolve_text": True}})
        return doc

    def extract_base_entities(self, text: str, doc) -> List[Entity]:
        """
        Trích xuất thực thể ưu tiên Named Entities (NER) trước,
        sau đó dùng Noun Chunks để bổ sung các danh từ chung chưa được nhận diện.
        """
        # print(f"Entities Extract text:\n{text}")
        # doc = self.nlp(text)
        entities = []
        
        # Dùng tập hợp các index của từng token (chữ) để kiểm tra overlapping triệt để
        seen_token_indices = set()
        
        # BƯỚC 1: Ưu tiên Named Entities (NER)
        for ent in doc.ents:
            # Đánh dấu tất cả các token thuộc NER này là "đã xử lý"
            seen_token_indices.update(range(ent.start, ent.end))
            
            head_noun = self._extract_head_noun(ent)
            modifiers = self._extract_modifiers(ent)
            head_noun_lemma = ent.root.lemma_
            
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                head_noun=head_noun,
                lemma=head_noun_lemma,
                modifiers=modifiers,
                confidence=0.95
            ))
            
        # BƯỚC 2: Quét Noun Chunks (Các cụm danh từ chung)
        for chunk in doc.noun_chunks:
            # Kiểm tra Overlapping: Nếu BẤT KỲ token nào trong chunk này đã nằm trong NER, thì bỏ qua
            if any(i in seen_token_indices for i in range(chunk.start, chunk.end)):
                continue
                
            # Lọc Đại từ: Chỉ bỏ qua nếu DANH TỪ CHÍNH (root) là đại từ (vd: "It", "They").
            # Điều này giúp giữ lại những cụm như "my green car" vì root là "car" (Noun).
            if chunk.root.pos_ == "PRON":
                continue
            
            # Thay vì tự dò _extract_head_noun, spaCy đã cung cấp sẵn chunk.root cực kỳ chính xác
            head_noun = chunk.root.text
            modifiers = self._extract_modifiers(chunk)
            head_noun_lemma = chunk.root.lemma_.lower()
            
            # (Tuỳ chọn bổ sung sau này): Bạn có thể loại bỏ các từ hạn định (a, an, the, my...)
            # ra khỏi text ở đây nếu muốn Knowledge Graph sạch hơn.
            
            entities.append(Entity(
                text=chunk.text,
                label="NOUN", 
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                head_noun=head_noun,
                lemma=head_noun_lemma,
                modifiers=modifiers,
                confidence=0.90
            ))
            
            # Đánh dấu các token của chunk này
            seen_token_indices.update(range(chunk.start, chunk.end))
            
        # # Lấy confidence cao hơn nếu chung lemma (lấy NER)
        # seen_lemmas: Dict[str, Entity] = {}
        # for ent in entities:
        #     existing = seen_lemmas.get(ent.lemma)
        #     if existing is None or ent.confidence > existing.confidence:
        #         seen_lemmas[ent.lemma] = ent

        # return list(seen_lemmas.values())
    
        return entities

    def extract_entities(self, text: str, doc) -> List[Entity]:
        # doc = self.preprocessor(text) # Dùng pipeline có fastcoref
        
        # 1. Chạy NER và Noun Chunks bình thường như code cũ của bạn
        base_entities = self.extract_base_entities(text, doc) 
        
        # Gán UUID mặc định cho mọi entity cơ bản
        for ent in base_entities:
            if ent.canonical_id is None:
                ent.canonical_id = str(uuid.uuid4())
                
        # 2. Xử lý Coref Clusters để tạo Alias Entities
        if doc._.coref_clusters:
            for cluster in doc._.coref_clusters:
                # cluster là list các char span: [(start, end), (start, end)...]
                
                # Tìm entity gốc (Representative) trong list base_entities
                # Thường là span đầu tiên trong cluster hoặc span đã được NER nhận diện
                rep_span = cluster[0]
                canonical_id = None
                
                # Tìm xem rep_span có khớp với entity nào đã extract không
                for ent in base_entities:
                    # Kiểm tra overlap hoặc match start/end char
                    if not (rep_span[1] <= ent.start_char or rep_span[0] >= ent.end_char):
                        canonical_id = ent.canonical_id
                        break
                
                # Nếu cụm coref này không trúng entity nào, tạo ID mới
                if not canonical_id:
                    canonical_id = str(uuid.uuid4())
                    
                # Duyệt qua các mentions còn lại trong cluster (như "he", "it"...)
                for mention_span in cluster[1:]:
                    # Khôi phục span text
                    mention_text = text[mention_span[0]:mention_span[1]]
                    
                    # Bổ sung đại từ này vào danh sách thực thể như một "Alias"
                    # mang chung canonical_id với entity gốc
                    alias_entity = Entity(
                        text=mention_text,
                        label="COREF_PRON",
                        start_char=mention_span[0],
                        end_char=mention_span[1],
                        head_noun=mention_text, # Tạm dùng text
                        lemma=mention_text.lower(),
                        modifiers=[],
                        confidence=0.9,
                        canonical_id=canonical_id # QUAN TRỌNG NHẤT
                    )
                    base_entities.append(alias_entity)
                    
        return base_entities

    def extract_relations(self, text: str, entities: List[Entity], doc=None) -> List[Relation]:
        """
        Trích xuất quan hệ DỰA TRÊN danh sách thực thể đã chốt (Entity-driven).
        Chỉ tạo quan hệ nếu Subject và Object khớp với các Entity hợp lệ.
        """
            
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
            if token.pos_ == "VERB" or token.lemma_.lower() == "be":
                predicate_parts = []
                
                if token.pos_ == "VERB":
                    neg = next((c for c in token.children if c.dep_ == "neg"), None)
                    if neg: predicate_parts.append(neg.lemma_.lower())
                    
                    predicate_parts.append(token.lemma_.lower())
                    
                    prt = next((c for c in token.children if c.dep_ == "prt"), None)
                    if prt: predicate_parts.append(prt.lemma_.lower())
                    
                    # Cây cú pháp xuyên thấu
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
                            # Tránh tự refer (Bây giờ so sánh bằng canonical_id là CHUẨN XÁC NHẤT)
                            if subj_ent.canonical_id == obj_ent.canonical_id: continue
                            
                            final_subj = obj_ent.canonical_id if is_passive else subj_ent.canonical_id
                            final_obj = subj_ent.canonical_id if is_passive else obj_ent.canonical_id
                                
                            relations.append(Relation(
                                subject=final_subj, predicate=predicate_text, obj=final_obj,
                                source_sentence=text, confidence=0.85
                            ))
                            
                    # Xử lý adverbs (manner)
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
                
                # So sánh bằng canonical_id để đảm bảo chúng không trỏ về cùng 1 node
                if subj_ent and appos_ent and subj_ent.canonical_id != appos_ent.canonical_id:
                    relations.append(Relation(
                        subject=subj_ent.canonical_id, predicate="is", obj=appos_ent.canonical_id,
                        source_sentence=text, confidence=0.90
                    ))

        # Lọc trùng lặp Relation
        unique_relations = {(r.subject, r.predicate, r.obj): r for r in relations}
        return list(unique_relations.values())

    def _get_subjects(self, verb_token) -> List[Any]:
        """Tìm các chủ ngữ của một động từ, xử lý cả liên từ và rút gọn chủ ngữ."""
        subjects = []
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
                # TUYỆT CHIÊU: Xuyên thấu đại từ quan hệ
                if child.lower_ in {"that", "which", "who", "whom"} and verb_token.dep_ == "relcl":
                    subjects.append(verb_token.head) # Trả về thẳng danh từ gốc
                else:
                    subjects.append(child)
                
                # Xử lý liên từ (and/or)
                for grandchild in child.children:
                    if grandchild.dep_ == "conj":
                        subjects.append(grandchild)
                        
        # Mượn chủ ngữ nếu tỉnh lược
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
                # Tương tự cho tân ngữ (VD: The book that I read)
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
            if token.pos_ in ["ADJ", "ADV", "NOUN"] and token.text != span.root.text:
                modifiers.append(token.text)
        return modifiers
    
    def filter_entities_by_relations(self, entities: List[Entity], relations: List[Relation]) -> List[Entity]:
        # Lấy tất cả canonical_id xuất hiện trong relation
        related_entity_ids = {
            rel.subject for rel in relations
        } | {
            rel.obj for rel in relations
        }

        # Chỉ giữ entity có canonical_id xuất hiện trong relation
        filtered_entities = [
            entity
            for entity in entities
            if entity.canonical_id in related_entity_ids
        ]

        return filtered_entities
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Full information extraction pipeline.
        
        Args:
            text: Input educational text
            
        Returns:
            Dictionary containing extracted entities, relations, and chunks
        """

        doc = self.preprocessor(text)

        entities = self.extract_entities(text, doc)
        relations = self.extract_relations(text, entities, doc)

        filter_entities = self.filter_entities_by_relations(entities, relations)
        return {
            "text": text,
            "entities": [asdict(e) for e in filter_entities],
            "relations": [asdict(r) for r in relations],
        }
