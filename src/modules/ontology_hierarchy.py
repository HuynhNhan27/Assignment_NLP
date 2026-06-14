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


class OntologyHierarchyResolver:
    """
    Builds hierarchical structures and deduplicates relations for an LPG Knowledge Graph.
    """
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", sim_threshold: float = 0.85, max_hypernym_depth: int = 1, max_hypernyms_per_node: int = 5):
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
        Xây dựng phân cấp Ontology từ WordNet và Head Nouns.
        """
        nodes: Dict[str, OntologyNode] = {}
        relations: List[OntologyRelation] = []
        
        # 1. Map WSD entities theo canonical_id để dễ truy xuất
        wsd_map = {ent['canonical_id']: ent for ent in wsd_entities if 'canonical_id' in ent}
        
        # 2. Xử lý từng entity gốc
        seen_canonical = set()
        
        for raw_ent in raw_entities:
            c_id = raw_ent['canonical_id']
            if c_id in seen_canonical:
                continue
            seen_canonical.add(c_id)
            
            # Tạo Node gốc cho Entity hiện tại
            nodes[c_id] = OntologyNode(
                id=c_id,
                label=raw_ent['text'],
                node_type="ENTITY",
                properties={"confidence": raw_ent.get('confidence', 1.0)}
            )
            
            wsd_data = wsd_map.get(c_id)
            
            if wsd_data:
                # --- TRƯỜNG HỢP CÓ TRONG WORDNET ---
                # Lấy synset từ tên (text của wsd_data đang lưu dạng dog.n.01)
                try:
                    synset = wn.synset(wsd_data['text'])

                    queue = [(synset, c_id, 0)]
                    visited_synsets = set() # Tránh lặp vòng nếu WordNet có cycle (hiếm nhưng an toàn)

                    while queue:
                        curr_syn, curr_id, depth = queue.pop(0)
                        if depth >= self.max_hypernym_depth or curr_syn in visited_synsets:
                            continue

                        all_hypernyms = curr_syn.hypernyms()
                        selected_hypernyms = all_hypernyms[:self.max_hypernyms_per_node]

                        for parent_syn in selected_hypernyms:
                            parent_label = parent_syn.lemmas()[0].name().replace("_", " ")
                            parent_id = self._get_category_id(parent_label)
                            
                            if parent_id not in nodes:
                                nodes[parent_id] = OntologyNode(
                                    id=parent_id, label=parent_label, node_type="CATEGORY", properties={"source": "wordnet"}
                                )
                                
                            relations.append(OntologyRelation(
                                subject=curr_id, predicate="is_a", obj=parent_id, confidence=1.0, source_sentence="WordNet Ontology"
                            ))
                            
                            # Đẩy node cha vào queue để tiếp tục đào sâu (nếu depth chưa max)
                            queue.append((parent_syn, parent_id, depth + 1))
                except Exception:
                    pass # Fallback nếu format synset có vấn đề
            else:
                # --- TRƯỜNG HỢP OOV (OUT OF VOCABULARY) ---
                # Sử dụng head_noun làm Category nhân tạo
                head_noun = raw_ent.get('head_noun', '').lower()
                text_lower = raw_ent['text'].lower()
                
                # Chỉ tạo Category nếu head_noun khác hoàn toàn với text 
                # (VD: "Convolutional Neural Network" -> "network")
                if head_noun and head_noun != text_lower and head_noun in text_lower:
                    parent_id = self._get_category_id(head_noun)
                    
                    if parent_id not in nodes:
                        nodes[parent_id] = OntologyNode(
                            id=parent_id,
                            label=head_noun,
                            node_type="CATEGORY",
                            properties={"source": "syntax_head_noun"}
                        )
                        
                    relations.append(OntologyRelation(
                        subject=c_id,
                        predicate="is_a",
                        obj=parent_id,
                        confidence=0.8, # Thấp hơn WordNet một chút
                        source_sentence="Syntactic Analysis"
                    ))

        # # BƯỚC 2: PRUNING (CẮT TỈA CÁC CATEGORY < 2 CON)
        # # Đếm số lượng node con (subject) trỏ vào mỗi category (obj)
        # category_child_count = {}
        # for rel in relations:
        #     if rel.predicate == "is_a":
        #         category_child_count[rel.obj] = category_child_count.get(rel.obj, 0) + 1

        # # Tìm các Category hợp lệ (>= 2 con)
        # valid_categories = {cat_id for cat_id, count in category_child_count.items() if count >= 2}

        # # Lọc Nodes và Relations
        # final_nodes = [n for nid, n in nodes.items() if n.node_type == "ENTITY" or nid in valid_categories]
        # final_relations = [r for r in relations if r.predicate != "is_a" or r.obj in valid_categories]

        # return final_nodes, final_relations
        return [n for n in nodes.values()], relations

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
