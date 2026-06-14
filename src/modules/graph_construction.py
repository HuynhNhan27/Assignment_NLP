"""
Graph Construction Module (Upgraded)

Handles:
- Building in-memory Labeled Property Graph (LPG) from pipeline outputs
- Node and edge management using Canonical UUIDs
- Metadata enrichment from WSD stage
- JSON serialization for frontend
- Graph statistics and analysis

Tool: NetworkX
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import networkx as nx
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a node in the Knowledge Graph."""
    id: str  # canonical_id (UUID)
    label: str
    node_type: str  # "ENTITY", "CATEGORY", v.v.
    properties: Dict[str, Any]  # definition, confidence, synset_id, v.v.


@dataclass
class GraphEdge:
    """Represents an edge in the Knowledge Graph."""
    source: str      # subject UUID
    target: str      # object UUID
    relation_type: str  # predicate
    properties: Dict[str, Any]  # confidence, source_sentence, v.v.


class GraphConstructor:
    """
    Construct and manage a Labeled Property Graph (LPG) based on pipeline outputs.
    """
    
    def __init__(self):
        """Initialize the graph constructor."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
    
    def add_node(self, node_id: str, label: str, node_type: str, properties: Dict[str, Any] = None) -> str:
        """
        Add a node to the graph using a pre-defined ID.
        """
        if node_id in self.nodes:
            # Cập nhật properties nếu node đã tồn tại (để đảm bảo không mất data)
            if properties:
                self.nodes[node_id].properties.update(properties)
                self.graph.nodes[node_id].update(properties)
            return node_id
        
        if properties is None:
            properties = {}
            
        node = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add an edge between two existing nodes.
        """
        # Bỏ qua nếu 1 trong 2 node không tồn tại trong Graph
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
            
        if properties is None:
            properties = {}
            
        # Tránh thêm edge trùng lặp hoàn toàn
        for existing_edge in self.edges:
            if (existing_edge.source == source_id and 
                existing_edge.target == target_id and 
                existing_edge.relation_type == relation_type):
                return False
                
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            relation_type=relation_type,
            properties=properties
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source_id, target_id, relation_type=relation_type, **properties)
        
        return True

    def build_from_pipeline(self, ontology_data: Dict[str, Any], wsd_data: Optional[List[Dict]] = None) -> None:
        """
        Xây dựng Graph trực tiếp từ output của Stage 3 (Ontology) và Stage 2 (WSD).
        
        Args:
            ontology_data: Output từ OntologyResolver.resolve() 
                           ({"nodes": [...], "relations": [...]})
            wsd_data: Output từ WordSenseDisambiguator.disambiguate()
        """
        # 1. Map WSD data theo canonical_id để tra cứu nhanh (O(1))
        wsd_map = {}
        if wsd_data:
            for wsd_ent in wsd_data:
                c_id = wsd_ent.get('canonical_id')
                if c_id:
                    wsd_map[c_id] = {
                        "definition": wsd_ent.get("definition"),
                        "synset_id": wsd_ent.get("synset_id"),
                        "wsd_confidence": wsd_ent.get("confidence")
                    }

        # 2. Xây dựng Nodes từ Ontology data
        for node_data in ontology_data.get("nodes", []):
            node_id = node_data["id"]
            properties = node_data.get("properties", {}).copy()
            
            # Enrich thêm dữ liệu từ WSD nếu node này là Entity (có trong wsd_map)
            if node_id in wsd_map:
                properties.update(wsd_map[node_id])
                
            self.add_node(
                node_id=node_id,
                label=node_data["label"],
                node_type=node_data["node_type"],
                properties=properties
            )

        # 3. Xây dựng Edges từ Ontology data
        for rel_data in ontology_data.get("relations", []):
            properties = {
                "confidence": rel_data.get("confidence", 1.0),
                "source_sentence": rel_data.get("source_sentence", "")
            }
            
            self.add_edge(
                source_id=rel_data["subject"],
                target_id=rel_data["obj"],
                relation_type=rel_data["predicate"],
                properties=properties
            )
            
    def get_statistics(self) -> Dict[str, Any]:
        """Lấy các chỉ số thống kê của đồ thị."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
            "density": round(nx.density(self.graph), 4) if len(self.nodes) > 1 else 0,
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.node_type] += 1
        return dict(counts)
    
    def _count_edge_types(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for edge in self.edges:
            counts[edge.relation_type] += 1
        return dict(counts)
    
    def to_json(self) -> str:
        """Export graph to JSON format for frontend visualization."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary."""
        nodes_data = [asdict(node) for node in self.nodes.values()]
        links_data = [{
            "source": edge.source,
            "target": edge.target,
            "type": edge.relation_type,
            "properties": edge.properties
        } for edge in self.edges]
        
        return {
            "nodes": nodes_data,
            "links": links_data,
            "metadata": self.get_statistics()
        }
    
    def find_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """Tìm tất cả các đường đi giữa 2 nodes."""
        try:
            return list(nx.all_simple_paths(self.graph, source_id, target_id))
        except nx.NetworkXNoPath:
            return []
    
    def get_neighbors(self, node_id: str, direction: str = "all") -> List[str]:
        """Lấy danh sách các node lân cận."""
        if node_id not in self.graph:
            return []
            
        if direction == "in":
            return list(self.graph.predecessors(node_id))
        elif direction == "out":
            return list(self.graph.successors(node_id))
        else:
            return list(self.graph.neighbors(node_id)) + list(self.graph.predecessors(node_id))