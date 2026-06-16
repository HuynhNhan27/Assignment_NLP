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
        
        # # Stage 3: Ontology & Hierarchy
        print("[INIT] Loading Ontology Resolver (WordNet)...")
        self.ontology_resolver = OntologyResolver()
        
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
            print(f"      end_char: {entity['end_char']}")

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
        relations = extraction_result['relations']
        
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
        
        wsd_result = self.wsd.normalize_post_wsd(disambiguated_entities, relations)

        print(f"  - Normalize sample: {len(wsd_result['entities'])} entities")
        for d in wsd_result['entities']:
            print(f"    * {d['text']} -> {d['definition']} (conf: {d['confidence']:.2f})")
            print(f"       ID: {d['canonical_id']}")

        return wsd_result
    
    def stage_3_ontology_resolution(self, extraction_result: Dict[str, Any], 
                                    wsd_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Resolve hierarchy and build ontology.
        
        Args:
            extraction_result: Result from Stage 1
            wsd_result: Result from Stage 2
            
        Returns:
            Dictionary with hierarchy information
        """
        print("\n[STAGE 3] Ontology & Hierarchy Resolution")
        final_kg = self.ontology_resolver.resolve(extraction_result["entities"], wsd_result['entities'], wsd_result["relations"])

        print(f"  - Ontology Entities: {len(final_kg['nodes'])} entities")
        for n in final_kg['nodes']:
            print(f"    * {n['id']} : {n['label']}")

        print(f"  - Ontology Relations: {len(final_kg['relations'])} relations")
        for rel in final_kg['relations']:
            print(f"(ID relation:   * {rel['subject']} --[{rel['predicate']}]--> {rel['obj']})")
        
        return final_kg
    
    def stage_4_graph_construction(self, 
                                   extraction_result: Dict[str, Any],
                                   wsd_result: Dict[str, Any],
                                   ontology_result: Dict[str, Any]) -> GraphConstructor:
        """
        Stage 4: Build the Knowledge Graph with merging.
        
        Args:
            extraction_result: From Stage 1
            wsd_result: From Stage 2
            ontology_result: From Stage 3
            
        Returns:
            Populated GraphConstructor instance
        """
        print("\n[STAGE 4] Graph Construction")

        print(f"  - Ontology Entities: {len(ontology_result['nodes'])} entities")
        print(f"  - Ontology Relations: {len(ontology_result['relations'])} relations")

        self.graph_constructor.build_from_pipeline(
            ontology_data=ontology_result, 
            wsd_entities=wsd_result['entities']
        )

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
    Whispering quietly, the worried mother checked on her sleeping baby who had a fever.
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
