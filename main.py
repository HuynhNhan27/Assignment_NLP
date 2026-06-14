"""
Main pipeline script orchestrating the entire Knowledge Graph construction.

Pipeline stages:
1. Information Extraction (NER, Relation Extraction, Noun Chunking)
2. Word Sense Disambiguation & Entity Normalization
3. Ontology & Hierarchy Resolution
4. Graph Construction & Merging
5. JSON Export for Frontend
"""

import sys
import json
import os
from typing import Dict, Any, List, Optional
import spacy

# Import pipeline modules
from src.modules import (
    InformationExtractor,
    WordSenseDisambiguator,
    EntityNormalizer,
    OntologyResolver,
    GraphConstructor
)
from fastcoref import spacy_component
from src.utils import save_json, load_json


class TextToKnowledgeGraphPipeline:
    """
    End-to-end pipeline for converting educational text to Knowledge Graph.
    """
    
    def __init__(self):
        """Initialize all pipeline components."""
        print("[INIT] Initializing TextToKnowledgeGraphPipeline...")
        
        # Stage 1: Information Extraction
        print("[INIT] Loading Information Extractor (spaCy)...")
        self.ie_extractor = InformationExtractor(model_name="en_core_web_sm")
        
        # Stage 2: WSD & Normalization
        print("[INIT] Loading Word Sense Disambiguator (sentence-transformers)...")
        self.wsd = WordSenseDisambiguator()
        
        print("[INIT] Loading Entity Normalizer (fuzzy + embeddings)...")
        self.entity_normalizer = EntityNormalizer()
        
        # Stage 3: Ontology & Hierarchy
        print("[INIT] Loading Ontology Resolver (WordNet)...")
        self.ontology = OntologyResolver()
        
        # Stage 4: Graph Construction
        print("[INIT] Initializing Graph Constructor (NetworkX)...")
        self.graph_constructor = GraphConstructor()

    def _json_safe_modifiers(self, modifiers: List[Any]) -> List[Dict[str, Any]]:
        """Convert nested modifier objects or dicts into plain JSON-safe dictionaries."""
        safe_modifiers = []
        for modifier in modifiers or []:
            if hasattr(modifier, "to_dict"):
                safe_modifiers.append(modifier.to_dict())
            elif isinstance(modifier, dict):
                safe_nested = dict(modifier)
                nested = safe_nested.get("modifiers", [])
                safe_nested["modifiers"] = self._json_safe_modifiers(nested)
                safe_modifiers.append(safe_nested)
        return safe_modifiers
    
    def stage_1_information_extraction(self, text: str) -> Dict[str, Any]:
        """
        Stage 1: Extract entities, relations, and noun chunks.
        
        Args:
            text: Input educational text
            
        Returns:
            Dictionary with extracted information
        """
        print("\n[STAGE 1] Information Extraction")
        
        extraction_result = self.ie_extractor.process_text(text)

        entity_map = {
            ent['canonical_id']: ent['text']
            for ent in extraction_result['entities']
        }
        
        print(f"  - Entities found: {len(extraction_result['entities'])}")
        for entity in extraction_result['entities'][:]:
            modifier_summary = entity.get('modifier_summary', {})
            modifiers = entity.get('modifiers', [])
            print(f"    * {entity['text']} ({entity['label']}) -> head: {entity['head_noun']}")
            print(f"      head_noun: {entity['head_noun']}")
            print(f"      id: {entity['canonical_id']}")
            if modifier_summary:
                print(f"      modifiers: {modifier_summary}")
            if modifiers:
                detailed = []
                for mod in modifiers:
                    text = mod.get('text', '')
                    mod_type = mod.get('type', '')
                    nested = mod.get('modifiers', [])
                    if nested:
                        nested_text = ", ".join(m.get('text', '') for m in nested)
                        detailed.append(f"{text} [{mod_type}] <- {nested_text}")
                    else:
                        detailed.append(f"{text} [{mod_type}]")
                print(f"      detailed: {detailed}")

        print(f"  - Relations found: {len(extraction_result['relations'])}")
        for rel in extraction_result['relations']:
            subject_text = entity_map.get(rel['subject'], rel['subject'])
            object_text = entity_map.get(rel['obj'], rel['obj'])

            print(f"    * {subject_text} --[{rel['predicate']}]--> {object_text}")
            print(f"(ID relation:   * {rel['subject']} --[{rel['predicate']}]--> {rel['obj']})")
        
        return extraction_result
    
    def stage_2_wsd_and_normalization(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: Disambiguate word senses and normalize entities.
        
        Args:
            extraction_result: Result from Stage 1
            text: Original text (for context)
            
        Returns:
            Dictionary with normalized entities and synsets
        """
        print("\n[STAGE 2] Word Sense Disambiguation & Normalization")
        text = extraction_result['text']
        entities = extraction_result['entities']
        
        # Extract unique entities
        entities_text = [e['text'] for e in extraction_result['entities']]
        entity_map = self.entity_normalizer.normalize_entities(entities_text)

        print(f"  - Normalized entities: {len(entity_map)}")
        for original, canonical in list(entity_map.items())[:5]:
            if original != canonical:
                print(f"    * {original} -> {canonical}")
        
        # Disambiguate
        disambiguated_entities = self.wsd.disambiguate(text, entities)
        
        print(f"  - Disambiguated sample: {len(disambiguated_entities)} entities")
        for d in disambiguated_entities:
            print(f"    * {d['text']} -> {d['definition']} (conf: {d['confidence']:.2f})")
            print(f"       ID: {d['canonical_id']}")
        
        return {
            "entity_map": entity_map,
            "disambiguated": disambiguated_entities
        }
    
    def stage_3_ontology_resolution(self, extraction_result: Dict[str, Any], 
                                    normalization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Resolve hierarchy and build ontology.
        
        Args:
            extraction_result: Result from Stage 1
            normalization_result: Result from Stage 2
            
        Returns:
            Dictionary with hierarchy information
        """
        print("\n[STAGE 3] Ontology & Hierarchy Resolution")
        
        # Get relations and normalize predicates
        relations = extraction_result['relations']
        
        print(f"  - Processing {len(relations)} relations")
        normalized_relations = []
        for rel in relations:
            normalized_pred = self.ontology.normalize_predicate(rel['predicate'])
            normalized_relations.append({
                "subject": rel['subject'],
                "predicate": normalized_pred,
                "object": rel['obj']
            })
            print(f"    * {rel['subject']} --[{normalized_pred}]--> {rel['obj']}")
        
        # Build hierarchy for example entities
        hierarchies = {}
        for entity in list(set([e['text'] for e in extraction_result['entities']]))[:2]:
            hierarchy = self.ontology.build_hierarchy_graph(entity, depth=3)
            hierarchies[entity] = hierarchy
            if hierarchy.get('hierarchy'):
                print(f"  - Hierarchy for '{entity}':")
                for h in hierarchy['hierarchy'][:2]:
                    print(f"    * {h['text']} (level {h['level']}): {h['definition'][:50]}...")
        
        return {
            "normalized_relations": normalized_relations,
            "hierarchies": hierarchies
        }
    
    def stage_4_graph_construction(self, 
                                   extraction_result: Dict[str, Any],
                                   normalization_result: Dict[str, Any],
                                   ontology_result: Dict[str, Any]) -> GraphConstructor:
        """
        Stage 4: Build the Knowledge Graph with merging.
        
        Args:
            extraction_result: From Stage 1
            normalization_result: From Stage 2
            ontology_result: From Stage 3
            
        Returns:
            Populated GraphConstructor instance
        """
        print("\n[STAGE 4] Graph Construction")
        
        # Get entities from Stage 1 (now includes noun chunks)
        entities = list(set([e['text'] for e in extraction_result['entities']]))
        entity_map = normalization_result.get('entity_map')
        if entity_map is None:
            entity_map = self.entity_normalizer.normalize_entities(entities)
        
        # If no entities found, extract from relations
        if not entities:
            relations = extraction_result['relations']
            subjects = set([r['subject'] for r in relations])
            objects = set([r['obj'] for r in relations])
            entities = list(subjects | objects)
            entity_map = self.entity_normalizer.normalize_entities(entities)
            print(f"  - No direct entities found; extracted {len(entities)} from relations")
        
        # Add entity nodes to graph
        entity_nodes = {}
        for entity in entities:
            canonical = entity_map.get(entity, entity)
            entity_data = next((e for e in extraction_result['entities'] if e['text'] == entity), None)
            if entity_data is None:
                entity_data = next((e for e in extraction_result['entities'] if e.get('head_noun', '').lower() == entity.lower()), None)
            node_id = self.graph_constructor.add_node(
                label=canonical,
                node_type="entity",
                properties={
                    "original": entity,
                    "canonical_id": entity_data.get('canonical_id') if entity_data else None,
                    "head_noun": entity_data.get('head_noun') if entity_data else None,
                    "lemma": entity_data.get('lemma') if entity_data else None,
                    "label": entity_data.get('label') if entity_data else None,
                    "confidence": entity_data.get('confidence') if entity_data else None,
                    "modifiers": self._json_safe_modifiers(entity_data.get('modifiers') if entity_data else []),
                    "modifier_summary": entity_data.get('modifier_summary') if entity_data else {},
                }
            )
            entity_nodes[canonical] = node_id
            if entity_data:
                if entity_data.get('canonical_id'):
                    entity_nodes[entity_data['canonical_id']] = node_id
                entity_nodes[entity_data.get('text', entity)] = node_id
                if entity_data.get('lemma'):
                    entity_nodes[entity_data['lemma']] = node_id
                if entity_data.get('head_noun'):
                    entity_nodes[entity_data['head_noun']] = node_id

        def resolve_node_id(name: str) -> Optional[str]:
            if not name:
                return None
            if name in entity_nodes:
                return entity_nodes[name]
            lower_name = name.lower()
            if lower_name in entity_nodes:
                return entity_nodes[lower_name]
            return entity_nodes.get(entity_map.get(name, name))
        
        print(f"  - Added {len(entity_nodes)} entity nodes")
        
        # Add relations as edges
        relations = ontology_result['normalized_relations']
        edge_count = 0
        for rel in relations:
            subject_canonical = entity_map.get(rel['subject'], rel['subject'])
            object_canonical = entity_map.get(rel['object'], rel['object'])
            subject_id = resolve_node_id(rel['subject']) or resolve_node_id(subject_canonical)
            object_id = resolve_node_id(rel['object']) or resolve_node_id(object_canonical)

            if subject_id and object_id:
                self.graph_constructor.add_edge(
                    source_id=subject_id,
                    target_id=object_id,
                    relation_type=rel['predicate'],
                    properties={
                        "confidence": 0.85,
                        "source_sentence": extraction_result.get('text', ''),
                        "original_subject": rel['subject'],
                        "original_object": rel['object'],
                    }
                )
                edge_count += 1
        
        print(f"  - Added {edge_count} relation edges")
        
        # Add hierarchy edges (example)
        for entity, hierarchy in ontology_result['hierarchies'].items():
            if hierarchy.get('hierarchy'):
                entity_canonical = entity_map.get(entity, entity)
                if entity_canonical in entity_nodes:
                    for h in hierarchy['hierarchy'][:2]:
                        parent_id = self.graph_constructor.add_node(
                            label=h['text'],
                            node_type="concept",
                            properties={"definition": h['definition']}
                        )
                        self.graph_constructor.add_hierarchy_edge(
                            child_id=entity_nodes[entity_canonical],
                            parent_id=parent_id
                        )
        
        print(f"  - Graph statistics: {self.graph_constructor.get_statistics()}")
        
        return self.graph_constructor
    
    def stage_5_export(self, graph: GraphConstructor, output_path: str) -> None:
        """
        Stage 5: Export graph to JSON for frontend.
        
        Args:
            graph: Constructed graph
            output_path: Output JSON file path
        """
        print("\n[STAGE 5] Export to JSON")
        
        json_output = graph.to_json()
        save_json(json_output, output_path)
        
        print(f"  - Graph exported to: {output_path}")
        
        # Print preview
        graph_data = json.loads(json_output)
        print(f"  - Nodes: {len(graph_data['nodes'])}")
        print(f"  - Links: {len(graph_data['links'])}")
    
    def run(self, text: str, output_path: str = "data/output/knowledge_graph.json") -> None:
        """
        Run the complete pipeline.
        
        Args:
            text: Input educational text
            output_path: Output JSON file path
        """
        print("=" * 80)
        print("TEXT TO KNOWLEDGE GRAPH PIPELINE")
        print("=" * 80)

        print(f"Input text: {text[:100]}...")

        # Tiền xử lý trong IE
        # Execute all stages
        stage1_result = self.stage_1_information_extraction(text)
        stage2_result = self.stage_2_wsd_and_normalization(stage1_result)
        stage3_result = self.stage_3_ontology_resolution(stage1_result, stage2_result)
        graph = self.stage_4_graph_construction(stage1_result, stage2_result, stage3_result)
        self.stage_5_export(graph, output_path)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)


def main():
    """Main entry point."""
    # Example educational text
    sample_text = """
    Cows are herbivorous mammals that eat grass in meadows. 
    They are larger than sheep, which also eat grass and leaves. 
    Both cows and deer consume plant matter like grass and leaves as food.
    Farmers raise cattle such as cows for meat and milk production.
    """

    sample_1 = """
    Cows are mammals.
    Cow eat grass and leaves.
    Farmers raise cow.
    Cows produce milk.
    """

    sample_2 = """
Cattle are large artiodactyls, mammals with cloven hooves, meaning that they walk on two toes, the third and fourth digits. Like all bovid species, they can have horns, which are unbranched and are not shed annually."""
    
    sample_3 = """
    The United States relies on good data.
    The brown cow eats green grass and leaves the farm.
    The cow and sheep eat grass and leaf.
    The green grass is eaten by the brown cow.
    John does not give up the difficult project.
    The cat chases the mouse. It runs quickly.
    """

    sample_test = """
    The newly implemented corporate environmental policy, aiming for zero emissions by 2030, drastically altered production schedules across all domestic factories
    """

    # Cows are herbivorous mammals that eat grass in meadows
    # Elon Musk, who is a billionaire, announced a new model.
    # Two young, talented artists painted a wooden picture frame in the studio.
        # Two không nhận diện được, and và ',' ra kết quả khác nhau.
    # Whispering quietly, the worried mother checked on her sleeping baby who had a fever.
        # Đã phân tích tốt về ngữ nghĩa, nhưng thiếu 

    # Initialize and run pipeline
    pipeline = TextToKnowledgeGraphPipeline()
    
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)
    
    # Run the pipeline
    pipeline.run(sample_test.strip())


if __name__ == "__main__":
    main()
