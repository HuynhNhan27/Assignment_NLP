"""
Word Sense Disambiguation (WSD) & Entity Normalization Module

Handles:
- Modified Lesk Algorithm using embeddings
- Synset resolution via WordNet
- Fuzzy matching for entity deduplication
- Semantic similarity computation

Tools: sentence-transformers, NLTK WordNet
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from fuzzywuzzy import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# @dataclass
# class Entity:
#     text: str
#     label: str
#     start_char: int
#     end_char: int
#     head_noun: str
#     lemma: str 
#     modifiers: List[str]
#     confidence: float
#     canonical_id: str = None

@dataclass
class DisambiguatedEntity:
    text: str
    synset_id: str
    definition: str
    confidence: float # Điểm tổng hợp cuối cùng
    semantic_score: float # Lưu lại để tracking/debug
    lexical_score: float  # Lưu lại để tracking/debug
    original_text: str
    canonical_id: str = None

class WordSenseDisambiguator:
    """
    Disambiguate word senses using an Ensemble of Modified Lesk (Semantic) 
    and Original Lesk (Lexical) approaches.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", lexical_weight: float = 0.3):
        """
        Khởi tạo hệ thống WSD.
        
        Args:
            model_name: Tên mô hình Sentence Transformer.
            lexical_weight (alpha): Trọng số cho Lexical Score (0.0 đến 1.0). 
                                  Semantic Score sẽ chiếm (1 - alpha).
                                  Mặc định 0.3 nghĩa là Lexical chiếm 30%, Semantic chiếm 70%.
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.sentence_tokenizer = PunktSentenceTokenizer()
        self.stop_words = set(stopwords.words('english'))
        self.lexical_weight = lexical_weight
        self.semantic_weight = 1.0 - lexical_weight

    def _preprocess_for_lexical(self, text: str) -> Set[str]:
        """Làm sạch và token hóa văn bản để tính toán overlap từ vựng."""
        # Chuyển chữ thường và tokenize
        tokens = word_tokenize(text.lower())
        # Lọc stop words và dấu câu
        cleaned_tokens = {
            t for t in tokens 
            if t not in self.stop_words and t not in string.punctuation
        }
        return cleaned_tokens

    def _calculate_lexical_score(self, context_tokens: Set[str], gloss_tokens: Set[str]) -> float:
        """Tính Jaccard Similarity giữa hai tập hợp từ vựng."""
        if not context_tokens or not gloss_tokens:
            return 0.0
        
        intersection = context_tokens.intersection(gloss_tokens)
        union = context_tokens.union(gloss_tokens)
        
        return len(intersection) / len(union)

    def _get_extended_gloss(self, synset) -> str:
        """Kết hợp định nghĩa và các ví dụ của synset thành một chuỗi duy nhất."""
        definition = synset.definition()
        examples = " ".join(synset.examples())
        
        if examples:
            return f"{definition}. {examples}"
        return definition

    def disambiguate_word(self, word: str, context: str) -> Optional[Dict]:
        """
        Hàm core WSD với Ensemble Scoring.
        """
        synsets = wn.synsets(word)
        if not synsets:
            return None
        
        # Chuẩn bị cho Semantic Score
        context_embedding = self.embedding_model.encode(context, convert_to_tensor=False)
        
        # Chuẩn bị cho Lexical Score
        context_tokens = self._preprocess_for_lexical(context)
        
        best_synset = None
        best_final_score = -1.0
        best_semantic = 0.0
        best_lexical = 0.0
        
        for synset in synsets:
            # 1. Lấy Extended Gloss (Definition + Examples)
            extended_gloss = self._get_extended_gloss(synset)
            
            # 2. Tính Semantic Score (Modified Lesk)
            gloss_embedding = self.embedding_model.encode(extended_gloss, convert_to_tensor=False)
            semantic_score = cosine_similarity([context_embedding], [gloss_embedding])[0][0]
            # Đưa semantic_score về khoảng [0, 1] (từ [-1,1] -> [0, 2] -> [0, 1])
            semantic_score = (semantic_score + 1.0) / 2
            
            # 3. Tính Lexical Score (Original Lesk)
            gloss_tokens = self._preprocess_for_lexical(extended_gloss)
            lexical_score = self._calculate_lexical_score(context_tokens, gloss_tokens)
            
            # 4. Tính Ensemble Score
            final_score = (self.lexical_weight * lexical_score) + (self.semantic_weight * semantic_score)
            
            if final_score > best_final_score:
                best_final_score = final_score
                best_synset = synset
                best_semantic = semantic_score
                best_lexical = lexical_score
                
        if best_synset is None:
            return None
            
        return {
            "text": best_synset.name(),
            "synset_id": best_synset.offset(),
            "definition": best_synset.definition(), # Vẫn trả về definition gốc cho output gọn gàng
            "confidence": float(best_final_score),
            "semantic_score": float(best_semantic),
            "lexical_score": float(best_lexical),
            "original_text": word
        }

    def disambiguate(self, text: str, entities_asdict: List[Dict]) -> List[Dict]:
        """
        Gom nhóm entity theo canonical_id, gom context và disambiguate.
        """
        sentence_spans = list(self.sentence_tokenizer.span_tokenize(text))
        
        grouped_entities = {}
        for ent in entities_asdict:
            c_id = ent.get('canonical_id')
            if not c_id:
                continue
            if c_id not in grouped_entities:
                grouped_entities[c_id] = []
            grouped_entities[c_id].append(ent)
            
        results = []
        
        for c_id, ent_list in grouped_entities.items():
            unique_sentences = set()
            
            for ent in ent_list:
                start_c = ent.get('start_char', 0)
                for s_start, s_end in sentence_spans:
                    if s_start <= start_c < s_end:
                        sentence_text = text[s_start:s_end].strip()
                        unique_sentences.add(sentence_text)
                        break 
            
            combined_context = " ".join(unique_sentences)
            target_word = ent_list[0].get('lemma', '') 
            
            wsd_result = self.disambiguate_word(target_word, combined_context)
            
            if wsd_result:
                final_entity = DisambiguatedEntity(
                    text=wsd_result["text"],
                    synset_id=str(wsd_result["synset_id"]),
                    definition=wsd_result["definition"],
                    confidence=wsd_result["confidence"],
                    semantic_score=wsd_result["semantic_score"],
                    lexical_score=wsd_result["lexical_score"],
                    original_text=wsd_result["original_text"],
                    canonical_id=c_id
                )
                results.append(asdict(final_entity))
                
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
        is_match = similarity >= self.fuzzy_threshold
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
        # print(f"[Norm] entities: {}")
        
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
