"""
Test Module for Hierarchical Modifier Extraction

Tests comprehensive modifier extraction with hierarchical relationships
using test cases from research: Easy, Medium, Hard, Advanced complexity levels.

This demonstrates:
- Basic modifiers (adjectives, quantifiers, etc.)
- Hierarchical modifiers (e.g., "highly efficient" where "highly" modifies "efficient")
- New types: Negation and Temporal expressions
- Option A: Modifier Nodes in knowledge graph
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.information_extraction import InformationExtractor
from src.modules.modifier_extraction import ModifierExtractor, ModifierType
from src.modules.graph_construction import GraphConstructor
import json


class ModifierTestSuite:
    """Test suite for hierarchical modifier extraction"""
    
    def __init__(self):
        self.ie_extractor = InformationExtractor()
        self.modifier_extractor = ModifierExtractor()
        self.graph = GraphConstructor()
    
    def test_easy_level(self):
        """🟢 Easy: One Layer of Description"""
        print("\n" + "="*80)
        print("🟢 EASY LEVEL: One Layer of Description")
        print("="*80)
        
        text = "The hungry cat ate the fresh fish."
        print(f"\nText: {text}")
        
        entities = self.ie_extractor.extract_entities(text)
        
        for ent in entities:
            print(f"\n📌 Entity: '{ent.text}'")
            print(f"   Head Noun: {ent.head_noun}")
            print(f"   Label: {ent.label}")
            print(f"   Modifier Summary: {ent.modifier_summary}")
            print(f"   Modifiers ({len(ent.modifiers)}):")
            
            for mod in ent.modifiers:
                print(f"      - {mod.text:20s} | Type: {mod.type.value:18s} | Pos: {mod.position:4s} | Conf: {mod.confidence:.2f}")
                
                # Show nested modifiers if any
                if mod.modifiers:
                    print(f"         └─ Nested Modifiers:")
                    for nested in mod.modifiers:
                        print(f"            - {nested.text:18s} | Type: {nested.type.value:16s}")
    
    def test_medium_level(self):
        """🟡 Medium: Multiple Modifier Types"""
        print("\n" + "="*80)
        print("🟡 MEDIUM LEVEL: Multiple Modifier Types")
        print("="*80)
        
        text = "Three local police officers investigated the stolen car outside the bank."
        print(f"\nText: {text}")
        
        entities = self.ie_extractor.extract_entities(text)
        
        for ent in entities:
            if ent.head_noun in ["officer", "car"]:  # Focus on main entities
                print(f"\n📌 Entity: '{ent.text}'")
                print(f"   Head Noun: {ent.head_noun}")
                print(f"   Modifier Summary: {ent.modifier_summary}")
                print(f"   Modifiers ({len(ent.modifiers)}):")
                
                for mod in ent.modifiers:
                    print(f"      - {mod.text:20s} | Type: {mod.type.value:18s} | Pos: {mod.position:4s} | Conf: {mod.confidence:.2f}")
                    
                    if mod.modifiers:
                        print(f"         └─ Nested Modifiers:")
                        for nested in mod.modifiers:
                            print(f"            - {nested.text:18s} | Type: {nested.type.value:16s}")
    
    def test_hard_level(self):
        """🟠 Hard: Clauses and Participles"""
        print("\n" + "="*80)
        print("🟠 HARD LEVEL: Clauses and Participles")
        print("="*80)
        
        text = "Whispering quietly, the worried mother checked on her sleeping baby who had a fever."
        print(f"\nText: {text}")
        
        entities = self.ie_extractor.extract_entities(text)
        
        for ent in entities:
            if ent.head_noun in ["mother", "baby"]:
                print(f"\n📌 Entity: '{ent.text}'")
                print(f"   Head Noun: {ent.head_noun}")
                print(f"   Modifier Summary: {ent.modifier_summary}")
                print(f"   Modifiers ({len(ent.modifiers)}):")
                
                for mod in ent.modifiers:
                    print(f"      - {mod.text:25s} | Type: {mod.type.value:18s} | Pos: {mod.position:4s} | Conf: {mod.confidence:.2f}")
                    
                    if mod.modifiers:
                        print(f"         └─ Nested Modifiers:")
                        for nested in mod.modifiers:
                            print(f"            - {nested.text:23s} | Type: {nested.type.value:16s}")
                    
                    if mod.constituents:
                        print(f"         └─ Constituents: {mod.constituents}")
    
    def test_advanced_level(self):
        """🔴 Advanced: Dense Academic/Professional Style with Hierarchical Modifiers"""
        print("\n" + "="*80)
        print("🔴 ADVANCED LEVEL: Dense Academic/Professional Style")
        print("="*80)
        
        text = "The newly implemented corporate environmental policy, aiming for zero emissions by 2030, drastically altered production schedules across all domestic factories."
        print(f"\nText: {text}")
        
        entities = self.ie_extractor.extract_entities(text)
        
        for ent in entities:
            if ent.head_noun in ["policy", "schedules", "factories"]:
                print(f"\n📌 Entity: '{ent.text}'")
                print(f"   Head Noun: {ent.head_noun}")
                print(f"   Modifier Summary: {ent.modifier_summary}")
                print(f"   Modifiers ({len(ent.modifiers)}):")
                
                for mod in ent.modifiers:
                    print(f"      - {mod.text:30s} | Type: {mod.type.value:18s} | Pos: {mod.position:4s} | Conf: {mod.confidence:.2f}")
                    
                    if mod.modifiers:
                        print(f"         └─ Nested Modifiers:")
                        for nested in mod.modifiers:
                            print(f"            - {nested.text:28s} | Type: {nested.type.value:16s}")
                    
                    if mod.constituents and len(mod.constituents) > 1:
                        print(f"         └─ Constituents: {mod.constituents}")
    
    def test_hierarchical_modifier_chain(self):
        """Test hierarchical modifier chains like 'highly efficient algorithm'"""
        print("\n" + "="*80)
        print("🔗 HIERARCHICAL MODIFIER CHAINS")
        print("="*80)
        
        test_cases = [
            ("The highly efficient algorithm processes data.", "Adverb → Adjective → Noun"),
            ("A very efficient algorithm", "Adverb → Adjective → Noun (variant)"),
            ("Some new corporate policies", "Quantifier + Adjective + Adjective → Noun"),
            ("Not available methods", "Negation + Participle → Noun"),
        ]
        
        for text, description in test_cases:
            print(f"\n📝 {description}")
            print(f"   Text: {text}")
            
            entities = self.ie_extractor.extract_entities(text)
            
            for ent in entities:
                if ent.modifiers:
                    print(f"   Modifiers for '{ent.text}':")
                    self._print_modifier_hierarchy(ent.modifiers, indent=6)
    
    def test_negation_and_temporal(self):
        """Test new modifier types: Negation and Temporal"""
        print("\n" + "="*80)
        print("❌/⏰ NEW MODIFIERS: Negation & Temporal")
        print("="*80)
        
        test_cases = [
            ("The policy was not implemented.", "Temporal + Negation"),
            ("The previously existing rules", "Temporal + Participle"),
            ("The completely unavailable resource", "Adverb + Temporal marker"),
            ("The data was stored", "Temporal expression"),
        ]
        
        for text, description in test_cases:
            print(f"\n📝 {description}")
            print(f"   Text: {text}")
            
            entities = self.ie_extractor.extract_entities(text)
            
            for ent in entities:
                if ent.modifiers:
                    print(f"   Modifiers for '{ent.text}':")
                    for mod in ent.modifiers:
                        if mod.type in [ModifierType.NEGATION, ModifierType.TEMPORAL]:
                            print(f"      ✓ {mod.text:20s} | Type: {mod.type.value:15s}")
    
    def test_graph_construction_with_modifiers(self):
        """Test Option A: Creating modifier nodes in the knowledge graph"""
        print("\n" + "="*80)
        print("📊 OPTION A: Modifier Nodes in Knowledge Graph")
        print("="*80)
        
        text = "The highly efficient algorithm processes data."
        print(f"\nText: {text}")
        
        # Extract entities
        entities = self.ie_extractor.extract_entities(text)
        
        for ent in entities:
            # Add entity node
            entity_node_id = self.graph.add_node(
                label=ent.text,
                node_type="entity",
                properties={
                    "head_noun": ent.head_noun,
                    "label": ent.label,
                    "confidence": ent.confidence
                }
            )
            
            # Add modifier nodes (Option A)
            if ent.modifiers:
                self.graph.add_modifier_nodes(entity_node_id, ent.modifiers)
                
                # Retrieve modifiers from graph
                modifiers_by_type = self.graph.get_entity_modifiers(entity_node_id)
                
                print(f"\n📌 Entity Node: '{ent.text}' (ID: {entity_node_id})")
                print(f"   Modifiers in Graph:")
                
                for mod_type, mods in modifiers_by_type.items():
                    print(f"      {mod_type}:")
                    for mod_info in mods:
                        print(f"         - {mod_info['text']:15s} | Conf: {mod_info['confidence']:.2f}")
                        
                        if mod_info['nested_modifiers']:
                            print(f"            └─ Nested Modifiers:")
                            for nested in mod_info['nested_modifiers']:
                                print(f"               - {nested['text']:13s} | Type: {nested['modifier_type']}")
        
        # Print graph statistics
        stats = self.graph.get_statistics()
        print(f"\n📊 Graph Statistics:")
        print(f"   Total Nodes: {stats['num_nodes']}")
        print(f"   Total Edges: {stats['num_edges']}")
        print(f"   Node Types: {stats['node_types']}")
        print(f"   Edge Types: {stats['edge_types']}")
        print(f"   Graph Density: {stats['density']:.3f}")
    
    def _print_modifier_hierarchy(self, modifiers, indent=0):
        """Helper: Print modifier hierarchy with indentation"""
        for mod in modifiers:
            spaces = " " * indent
            print(f"{spaces}- {mod.text:20s} | Type: {mod.type.value:18s}")
            
            if mod.modifiers:
                self._print_modifier_hierarchy(mod.modifiers, indent + 3)


def main():
    """Run all tests"""
    test_suite = ModifierTestSuite()
    
    try:
        test_suite.test_easy_level()
        test_suite.test_medium_level()
        test_suite.test_hard_level()
        test_suite.test_advanced_level()
        test_suite.test_hierarchical_modifier_chain()
        test_suite.test_negation_and_temporal()
        test_suite.test_graph_construction_with_modifiers()
        
        print("\n" + "="*80)
        print("✅ All Tests Completed Successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
