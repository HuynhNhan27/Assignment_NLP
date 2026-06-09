"""
Graph Construction Module

Handles:
- Building in-memory Labeled Property Graph (LPG)
- Node and edge management with metadata
- JSON serialization for frontend
- Graph statistics and analysis

Tool: NetworkX (or plain Python dict)
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import networkx as nx
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a node in the Knowledge Graph."""
    id: str
    label: str
    node_type: str  # "entity", "concept", "relation"
    properties: Dict[str, Any]  # metadata like definition, confidence, source
    hypernyms: List[str] = None
    
    def __post_init__(self):
        if self.hypernyms is None:
            self.hypernyms = []


@dataclass
class GraphEdge:
    """Represents an edge in the Knowledge Graph."""
    source: str
    target: str
    relation_type: str  # e.g., "eats", "is_a", "part_of"
    properties: Dict[str, Any]  # confidence, source_sentence, etc.


class GraphConstructor:
    """
    Construct and manage a Labeled Property Graph (LPG).
    
    Features:
    - Node and edge creation with metadata
    - Automatic deduplication
    - JSON export for frontend visualization
    - Graph statistics
    """
    
    def __init__(self):
        """Initialize the graph constructor."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_counter = 0
    
    def add_node(self, label: str, node_type: str = "entity", 
                 properties: Dict[str, Any] = None, 
                 hypernyms: List[str] = None) -> str:
        """
        Add a node to the graph.
        
        Args:
            label: Node label/name
            node_type: Type of node (entity, concept, relation)
            properties: Dictionary of node metadata
            hypernyms: List of hypernym IDs
            
        Returns:
            Node ID
        """
        # Check if node already exists (by label)
        for node_id, node in self.nodes.items():
            if node.label.lower() == label.lower():
                return node_id
        
        # Create new node
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        if properties is None:
            properties = {}
        
        node = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties,
            hypernyms=hypernyms or []
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, 
                 relation_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation (e.g., "eats", "is_a")
            properties: Edge metadata
            
        Returns:
            True if edge was added, False if already exists
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        # Check if edge already exists
        if self.graph.has_edge(source_id, target_id):
            return False
        
        if properties is None:
            properties = {}
        
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            relation_type=relation_type,
            properties=properties
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source_id, target_id, 
                           relation_type=relation_type, 
                           **properties)
        
        return True
    
    def add_hierarchy_edge(self, child_id: str, parent_id: str,
                          confidence: float = 1.0) -> bool:
        """
        Add a hierarchy edge (is-a relationship).
        
        Args:
            child_id: Child concept ID
            parent_id: Parent concept ID
            confidence: Confidence score
            
        Returns:
            True if edge was added
        """
        return self.add_edge(
            child_id, parent_id,
            relation_type="is_a",
            properties={"confidence": confidence}
        )
    
    def merge_nodes(self, node_ids: List[str], new_label: str) -> Optional[str]:
        """
        Merge multiple nodes into a single node (e.g., shared hypernym).
        
        Redirects all edges from merged nodes to the new node.
        
        Args:
            node_ids: List of node IDs to merge
            new_label: Label for the merged node
            
        Returns:
            ID of new merged node, or None if failed
        """
        if not node_ids:
            return None
        
        # Create new merged node
        merged_id = self.add_node(new_label, node_type="concept")
        
        # Redirect edges
        for node_id in node_ids:
            if node_id not in self.nodes:
                continue
            
            # Redirect incoming edges
            for pred in list(self.graph.predecessors(node_id)):
                if pred != merged_id:
                    self.add_edge(pred, merged_id, 
                                 relation_type="is_a")
            
            # Remove old node
            self.graph.remove_node(node_id)
            del self.nodes[node_id]
        
        return merged_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with graph metrics
        """
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
            "density": nx.density(self.graph),
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.node_type] += 1
        return dict(counts)
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by relation type."""
        counts = defaultdict(int)
        for edge in self.edges:
            counts[edge.relation_type] += 1
        return dict(counts)
    
    def to_json(self) -> str:
        """
        Export graph to JSON format for frontend visualization.
        
        Format:
        {
            "nodes": [{"id", "label", "type", "properties", ...}],
            "links": [{"source", "target", "type", "properties"}],
            "metadata": {...}
        }
        
        Returns:
            JSON string
        """
        nodes_data = []
        for node_id, node in self.nodes.items():
            nodes_data.append({
                "id": node_id,
                "label": node.label,
                "type": node.node_type,
                "properties": node.properties,
                "hypernyms": node.hypernyms
            })
        
        links_data = []
        for edge in self.edges:
            links_data.append({
                "source": edge.source,
                "target": edge.target,
                "type": edge.relation_type,
                "properties": edge.properties
            })
        
        output = {
            "nodes": nodes_data,
            "links": links_data,
            "metadata": self.get_statistics()
        }
        
        return json.dumps(output, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary."""
        nodes_data = []
        for node_id, node in self.nodes.items():
            nodes_data.append({
                "id": node_id,
                "label": node.label,
                "type": node.node_type,
                "properties": node.properties,
                "hypernyms": node.hypernyms
            })
        
        links_data = []
        for edge in self.edges:
            links_data.append({
                "source": edge.source,
                "target": edge.target,
                "type": edge.relation_type,
                "properties": edge.properties
            })
        
        return {
            "nodes": nodes_data,
            "links": links_data,
            "metadata": self.get_statistics()
        }
    
    def find_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """
        Find all paths between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            return list(nx.all_simple_paths(self.graph, source_id, target_id))
        except nx.NetworkXNoPath:
            return []
    
    def get_neighbors(self, node_id: str, direction: str = "all") -> List[str]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: Node ID
            direction: "in", "out", or "all"
            
        Returns:
            List of neighbor node IDs
        """
        if direction == "in":
            return list(self.graph.predecessors(node_id))
        elif direction == "out":
            return list(self.graph.successors(node_id))
        else:
            return list(self.graph.neighbors(node_id)) + list(self.graph.predecessors(node_id))
