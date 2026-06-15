"""
Ontology & Hierarchy Module

- WordNet-based Hypernym extraction (is-a relations) with depth limits.
- OOV (Out of Vocabulary) fallback using syntax-based head nouns.
- Relation deduplication for entity pairs using Sentence Transformers & Cosine Similarity.
- Antonym filtering to prevent merging opposite relations (e.g., increase vs decrease).

Tool: NLTK WordNet
"""

import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

@dataclass
class OntologyNode:
    """Represents a node in the final Knowledge Graph."""
    id: str  # canonical_id
    label: str  # The display name
    node_type: str  # 'ENTITY', 'CATEGORY' (for hypernyms/head nouns)
    properties: Dict[str, Any]

@dataclass
class OntologyRelation:
    """Represents an edge in the final Knowledge Graph."""
    subject: str
    predicate: str
    obj: str
    confidence: float
    source_sentence: str


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
    Builds hierarchical structures and deduplicates relations for an LPG Knowledge Graph.
    """
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", sim_threshold: float = 0.85, max_hypernym_depth: int = 3, max_hypernyms_per_node: int = 5):
        self.embedder = SentenceTransformer(embedding_model)
        self.sim_threshold = sim_threshold
        self.max_hypernym_depth = max_hypernym_depth
        self.max_hypernyms_per_node = max_hypernyms_per_node
        
        # Cache cho categories để tái sử dụng UUID nếu trùng tên category
        self.category_cache: Dict[str, str] = {} 

    def _get_category_id(self, category_name: str) -> str:
        """Lấy hoặc tạo UUID nhất quán cho một danh mục (Category)."""
        if category_name not in self.category_cache:
            self.category_cache[category_name] = str(uuid.uuid4())
        return self.category_cache[category_name]

    def _get_antonyms(self, predicate: str) -> Set[str]:
        """Tìm các từ trái nghĩa của một predicate bằng WordNet."""
        # Chuyển khoảng trắng thành underscore cho WordNet (VD: 'speed up' -> 'speed_up')
        formatted_pred = predicate.replace(" ", "_")
        antonyms = set()
        for syn in wn.synsets(formatted_pred):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    antonyms.add(ant.name().lower().replace("_", " "))
        return antonyms


    def build_hierarchy(self, raw_entities: List[Dict], wsd_entities: List[Dict]) -> Tuple[List[OntologyNode], List[OntologyRelation]]:
        """
        Xây dựng phân cấp Ontology sử dụng Graph Path Compression.
        Chỉ giữ lại các Category là 'điểm giao cắt' (Branching Points) thực sự.
        """
        temp_nodes: Dict[str, OntologyNode] = {}
        adj_up = defaultdict(set)    # node -> các cha của nó
        adj_down = defaultdict(set)  # node -> các con của nó
        
        wsd_map = {ent['canonical_id']: ent for ent in wsd_entities.get('disambiguated') if 'canonical_id' in ent}
        seen_canonical = set()
        entities_ids = set()
        
        # ==========================================
        # BƯỚC 1: XÂY DỰNG ĐỒ THỊ NHÁP (Đầy đủ độ sâu)
        # ==========================================
        for raw_ent in raw_entities:
            c_id = raw_ent['canonical_id']
            if c_id in seen_canonical: continue
            seen_canonical.add(c_id)
            entities_ids.add(c_id)
            
            temp_nodes[c_id] = OntologyNode(
                id=c_id, label=raw_ent['text'], node_type="ENTITY",
                properties={"confidence": raw_ent.get('confidence', 1.0)}
            )
            
            wsd_data = wsd_map.get(c_id)
            
            if wsd_data:
                try:
                    synset = wn.synset(wsd_data['text'])
                    queue = [(synset, c_id, 0)]
                    visited_synsets = set()
                    
                    while queue:
                        curr_syn, curr_id, depth = queue.pop(0)
                        
                        if depth >= self.max_hypernym_depth: continue
                        if curr_syn in visited_synsets: continue
                        visited_synsets.add(curr_syn)
                        
                        selected_hypernyms = curr_syn.hypernyms()[:self.max_hypernyms_per_node]
                        
                        for parent_syn in selected_hypernyms:
                            parent_label = parent_syn.lemmas()[0].name().replace("_", " ")
                            parent_id = self._get_category_id(parent_label)
                            
                            if parent_id not in temp_nodes:
                                temp_nodes[parent_id] = OntologyNode(
                                    id=parent_id, label=parent_label, node_type="CATEGORY", properties={"source": "wordnet"}
                                )
                                
                            # Lưu vào danh sách kề (Adjacency List)
                            adj_up[curr_id].add(parent_id)
                            adj_down[parent_id].add(curr_id)
                            
                            queue.append((parent_syn, parent_id, depth + 1))
                except Exception:
                    pass
            else:
                # OOV Fallback
                head_noun = raw_ent.get('head_noun', '').lower()
                text_lower = raw_ent['text'].lower()
                if head_noun and head_noun != text_lower and head_noun in text_lower:
                    parent_id = self._get_category_id(head_noun)
                    if parent_id not in temp_nodes:
                        temp_nodes[parent_id] = OntologyNode(
                            id=parent_id, label=head_noun, node_type="CATEGORY", properties={"source": "syntax_head_noun"}
                        )
                    adj_up[c_id].add(parent_id)
                    adj_down[parent_id].add(c_id)

        # ==========================================
        # BƯỚC 2: TÌM TẬP HỢP LEAVES (ENTITIES GỐC) CHO TỪNG NODE
        # ==========================================
        leaves = defaultdict(set)
        for ent_id in entities_ids:
            queue = [ent_id]
            visited = set()
            while queue:
                curr = queue.pop(0)
                if curr in visited: continue
                visited.add(curr)
                leaves[curr].add(ent_id) # Node hiện tại cover được Entity gốc này
                queue.extend(adj_up[curr])

        # ==========================================
        # BƯỚC 3: LỌC CÁC ĐIỂM GIAO CẮT (BRANCHING POINTS)
        # ==========================================
        kept_nodes_ids = set(entities_ids) # Luôn giữ Entity
        
        for nid, node in temp_nodes.items():
            if node.node_type == "CATEGORY":
                # Điều kiện 1: Phải chứa từ 2 Entities gốc trở lên
                if len(leaves[nid]) >= 2:
                    has_absorbing_child = False
                    # Điều kiện 2: Kiểm tra xem có node con nào có tập Entities y hệt không
                    for child_id in adj_down[nid]:
                        if leaves[child_id] == leaves[nid]:
                            has_absorbing_child = True
                            break
                            
                    # Chỉ giữ lại nếu nó là điểm gộp nhánh thực sự
                    if not has_absorbing_child:
                        kept_nodes_ids.add(nid)

        # ==========================================
        # BƯỚC 4: PATH COMPRESSION (NỐI TẮT ĐỒ THỊ)
        # ==========================================
        final_nodes = [temp_nodes[nid] for nid in kept_nodes_ids]
        final_relations = []
        
        for start_node in kept_nodes_ids:
            visited = set([start_node])
            queue = list(adj_up[start_node])
            
            while queue:
                curr = queue.pop(0)
                if curr in visited: continue
                visited.add(curr)
                
                # Nếu tìm thấy một node cha (hoặc tổ tiên) được giữ lại
                if curr in kept_nodes_ids:
                    final_relations.append(OntologyRelation(
                        subject=start_node, predicate="is_a", obj=curr, 
                        confidence=1.0, source_sentence="Compressed Hierarchy"
                    ))
                    # Tìm thấy rồi thì DỪNG hướng này, không leo lên trên ông nội nữa 
                    # để giữ đúng phân cấp tầng bậc.
                else:
                    # Nếu node trung gian này bị xóa, tiếp tục leo lên trên để tìm tổ tiên
                    queue.extend(adj_up[curr])

        return final_nodes, final_relations

    def deduplicate_relations(self, extracted_relations: List[Dict]) -> List[OntologyRelation]:
        """
        Gộp các relation trùng lặp về ngữ nghĩa giữa CÙNG một cặp thực thể.
        """
        # Nhóm relation theo cặp (subject, obj)
        grouped_rels: Dict[Tuple[str, str], List[Dict]] = {}
        for rel in extracted_relations:
            pair = (rel['subject'], rel['obj'])
            if pair not in grouped_rels:
                grouped_rels[pair] = []
            grouped_rels[pair].append(rel)
            
        final_relations: List[OntologyRelation] = []
        
        for (subj, obj), rels in grouped_rels.items():
            if len(rels) == 1:
                # Nếu chỉ có 1 relation giữa 2 node, giữ nguyên
                r = rels[0]
                final_relations.append(OntologyRelation(
                    subject=r['subject'], predicate=r['predicate'],
                    obj=r['obj'], confidence=r.get('confidence', 0.8),
                    source_sentence=r.get('source_sentence', '')
                ))
                continue
                
            # Xử lý gộp nếu có > 1 relation
            predicates = [r['predicate'] for r in rels]
            embeddings = self.embedder.encode(predicates)
            sim_matrix = cosine_similarity(embeddings)
            
            merged_indices = set()
            
            for i in range(len(rels)):
                if i in merged_indices:
                    continue
                    
                # i là relation đại diện cho nhóm được gộp
                current_rel = rels[i]
                current_antonyms = self._get_antonyms(current_rel['predicate'])
                
                # Tìm các relation tương đồng với i
                for j in range(i + 1, len(rels)):
                    if j in merged_indices:
                        continue
                        
                    if sim_matrix[i][j] >= self.sim_threshold:
                        target_pred = rels[j]['predicate']
                        
                        # KIỂM TRA TRÁI NGHĨA
                        if target_pred not in current_antonyms:
                            merged_indices.add(j)
                            # Cập nhật confidence (lấy max) nếu có
                            current_conf = current_rel.get('confidence', 0.8)
                            target_conf = rels[j].get('confidence', 0.8)
                            current_rel['confidence'] = max(current_conf, target_conf)
                            
                # Lưu relation đại diện sau khi đã gom
                final_relations.append(OntologyRelation(
                    subject=current_rel['subject'], 
                    predicate=current_rel['predicate'],
                    obj=current_rel['obj'], 
                    confidence=current_rel.get('confidence', 0.8),
                    source_sentence=current_rel.get('source_sentence', '')
                ))
                merged_indices.add(i)
                
        return final_relations

    def resolve(self, raw_entities: List[Dict], wsd_entities: List[Dict], extracted_relations: List[Dict]) -> Dict[str, Any]:
        """
        Hàm Main chạy toàn bộ quy trình Ontology.
        """
        # 1. Xây dựng phân cấp node và lấy relation is_a
        nodes, is_a_relations = self.build_hierarchy(raw_entities, wsd_entities)
        
        # 2. Xử lý và deduplicate các relation do spaCy trích xuất
        deduplicated_extracted_relations = self.deduplicate_relations(extracted_relations)
        
        # 3. Tổng hợp toàn bộ
        all_relations = is_a_relations + deduplicated_extracted_relations
        
        return {
            "nodes": [asdict(n) for n in nodes],
            "relations": [asdict(r) for r in all_relations]
        }
