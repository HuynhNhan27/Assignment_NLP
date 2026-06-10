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
    lemma: str
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
        Trích xuất thực thể ưu tiên Named Entities (NER) trước,
        sau đó dùng Noun Chunks để bổ sung các danh từ chung chưa được nhận diện.
        """
        print(text)
        doc = self.nlp(text)
        entities = []
        
        # Dùng tập hợp các index của từng token (chữ) để kiểm tra overlapping triệt để
        seen_token_indices = set()
        
        # BƯỚC 1: Ưu tiên Named Entities (NER)
        for ent in doc.ents:
            # Đánh dấu tất cả các token thuộc NER này là "đã xử lý"
            seen_token_indices.update(range(ent.start, ent.end))
            
            head_noun = self._extract_head_noun(ent)
            modifiers = self._extract_modifiers(ent)
            head_noun_lemma = ent.root.lemma_.lower()
            
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
            
        return entities
        
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Trích xuất quan hệ DỰA TRÊN danh sách thực thể đã chốt (Entity-driven).
        Chỉ tạo quan hệ nếu Subject và Object khớp với các Entity hợp lệ.
        """
        doc = self.nlp(text)
        relations = []
        
        # 1. BẢN ĐỒ THỰC THỂ (Entity Mapping)
        # Tạo từ điển mapping: vị trí token (index) -> Entity object.
        # Điều này giúp ta biết từ nào trên cây cú pháp thuộc về Entity nào.
        token_to_entity = {}
        for ent in entities:
            # Lấy các token tương ứng với khoảng ký tự của entity
            span = doc.char_span(ent.start_char, ent.end_char)
            if span is not None:
                for token in span:
                    token_to_entity[token.i] = ent
                    
        # 2. TÌM QUAN HỆ QUA ĐỘNG TỪ
        for token in doc:
            if token.pos_ == "VERB":
                # Bước A: Xây dựng Cụm Vị Ngữ (Predicate) hoàn chỉnh
                predicate_parts = []
                
                # Kiểm tra từ phủ định (vd: "NOT eat")
                neg = next((c for c in token.children if c.dep_ == "neg"), None)
                if neg: predicate_parts.append(neg.lemma_.lower())
                
                predicate_parts.append(token.lemma_.lower()) # Động từ chính
                
                # Kiểm tra phrasal verb (vd: turn OFF, give UP)
                prt = next((c for c in token.children if c.dep_ == "prt"), None)
                if prt: predicate_parts.append(prt.lemma_.lower())
                
                # Bước B: Tìm Chủ ngữ và Tân ngữ qua cây cú pháp
                subjects = self._get_subjects(token)
                objects, prep_lemma = self._get_objects(token)
                
                # Nếu có giới từ đi kèm tân ngữ (vd: rely ON), nối vào vị ngữ
                if prep_lemma:
                    predicate_parts.append(prep_lemma.lower())
                    
                predicate_text = " ".join(predicate_parts)
                
                # Bước C: Kiểm tra câu Bị động (Passive Voice)
                is_passive = any(c.dep_ == "nsubjpass" for c in token.children)
                
                # Bước D: Lọc và kết nối các Thực Thể
                for subj_token in subjects:
                    subj_ent = token_to_entity.get(subj_token.i)
                    if not subj_ent: 
                        continue # Bỏ qua nếu chủ ngữ không phải là Entity đã biết
                        
                    for obj_token in objects:
                        obj_ent = token_to_entity.get(obj_token.i)
                        if not obj_ent: 
                            continue # Bỏ qua nếu tân ngữ không phải là Entity đã biết
                            
                        # Chống lỗi vòng lặp (Subject = Object)
                        if getattr(subj_ent, 'lemma', subj_ent.head_noun) == getattr(obj_ent, 'lemma', obj_ent.head_noun):
                            continue
                            
                        # Nếu là câu bị động (Grass is eaten by cow), đảo ngược chiều quan hệ
                        if is_passive:
                            final_subj = getattr(obj_ent, 'lemma', obj_ent.head_noun)
                            final_obj = getattr(subj_ent, 'lemma', subj_ent.head_noun)
                        else:
                            final_subj = getattr(subj_ent, 'lemma', subj_ent.head_noun)
                            final_obj = getattr(obj_ent, 'lemma', obj_ent.head_noun)
                            
                        relations.append(Relation(
                            subject=final_subj,
                            predicate=predicate_text,
                            obj=final_obj,
                            source_sentence=text,
                            confidence=0.85
                        ))
                        
        # Lọc trùng lặp
        unique_relations = { (r.subject, r.predicate, r.obj): r for r in relations }
        return list(unique_relations.values())

    def _get_subjects(self, verb_token) -> List[Any]:
        """Tìm các chủ ngữ của một động từ, xử lý cả liên từ và rút gọn chủ ngữ."""
        subjects = []
        for child in verb_token.children:
            # Chủ ngữ trực tiếp hoặc chủ ngữ bị động
            if child.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
                subjects.append(child)
                # Bắt các chủ ngữ nối nhau bằng "and" (vd: Cow and sheep eat...)
                for grandchild in child.children:
                    if grandchild.dep_ == "conj":
                        subjects.append(grandchild)
                        
        # GIẢI QUYẾT BÀI TOÁN "cow eat grass and leave" (Tỉnh lược chủ ngữ)
        # Nếu động từ này không có chủ ngữ, nhưng nó được nối với một động từ trước đó
        if not subjects and verb_token.dep_ == "conj":
            head_verb = verb_token.head
            if head_verb.pos_ == "VERB":
                # Đệ quy: Mượn chủ ngữ của động từ đứng trước
                subjects = self._get_subjects(head_verb)
                
        return subjects

    def _get_objects(self, verb_token) -> Tuple[List[Any], str]:
        """Tìm tân ngữ và giới từ (nếu có) của một động từ."""
        objects = []
        prep_lemma = None
        
        for child in verb_token.children:
            # Tân ngữ trực tiếp
            if child.dep_ in ["dobj", "attr", "oprd"]:
                objects.append(child)
                # Tân ngữ nối bằng "and" (vd: eat grass and leaves)
                for grandchild in child.children:
                    if grandchild.dep_ == "conj":
                        objects.append(grandchild)
                        
            # Tân ngữ của giới từ / tác nhân bị động (by)
            elif child.dep_ in ["prep", "agent"]:
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        objects.append(pobj)
                        if child.dep_ == "prep":
                            prep_lemma = child.lemma_  # Lưu lại giới từ thông thường (không phải bị động)
                        # Tân ngữ giới từ nối bằng "and"
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
        relations = self.extract_relations(text, entities)
        # chunks = self.extract_noun_chunks(text)
        
        return {
            "text": text,
            "entities": [asdict(e) for e in entities],
            "relations": [asdict(r) for r in relations],
            # "noun_chunks": chunks
        }
