"""
Quick Start: Hierarchical Modifier Extraction System

This example demonstrates how to use the new hierarchical modifier system
in your NLP pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules import InformationExtractor, GraphConstructor
from src.modules.modifier_extraction import ModifierType
import json


def example_1_basic_extraction():
    """Example 1: Extract modifiers from text"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Modifier Extraction")
    print("="*70)
    
    text = "The highly efficient algorithm processes data."
    print(f"\nText: {text}")
    
    # Create extractor
    ie = InformationExtractor()
    
    # Extract entities with modifiers
    entities = ie.extract_entities(text)
    
    for entity in entities:
        print(f"\n📌 Entity: '{entity.text}'")
        print(f"   Head Noun: {entity.head_noun}")
        print(f"   Modifier Count: {len(entity.modifiers)}")
        print(f"   Modifier Summary: {entity.modifier_summary}")
        
        if entity.modifiers:
            print(f"   Modifiers:")
            for mod in entity.modifiers:
                print(f"      - {mod.text:20} | Type: {mod.type.value:20} | Confidence: {mod.confidence:.2f}")
                
                # Show hierarchical modifiers
                if mod.modifiers:
                    print(f"        └─ Nested Modifiers:")
                    for nested in mod.modifiers:
                        print(f"           - {nested.text:18} | Type: {nested.type.value}")


def example_2_hierarchical_structure():
    """Example 2: Demonstrate hierarchical modifier structure"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Hierarchical Modifier Structure")
    print("="*70)
    
    # Compare different modifier chains
    test_cases = [
        "The efficient algorithm",
        "The highly efficient algorithm",
        "The barely efficient algorithm",
    ]
    
    ie = InformationExtractor()
    
    for text in test_cases:
        print(f"\nText: {text}")
        entities = ie.extract_entities(text)
        
        for entity in entities:
            if entity.head_noun == "algorithm":
                # Build modifier chain representation
                chain = f"'{entity.head_noun}'"
                for mod in sorted(entity.modifiers, key=lambda m: m.position):
                    if mod.modifiers:
                        # Hierarchical
                        nested_texts = [m.text for m in mod.modifiers]
                        chain = f"[{', '.join(nested_texts)}] → '{mod.text}' → {chain}"
                    else:
                        chain = f"'{mod.text}' → {chain}"
                
                print(f"  Chain: {chain}")


def example_3_negation_temporal():
    """Example 3: New modifier types - Negation & Temporal"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Negation & Temporal Modifiers (NEW)")
    print("="*70)
    
    test_cases = [
        "The available methods",
        "The not available methods",
        "The data was stored",
        "The data will be stored",
        "The not implemented policy",
    ]
    
    ie = InformationExtractor()
    
    for text in test_cases:
        print(f"\nText: {text}")
        entities = ie.extract_entities(text)
        
        for entity in entities:
            # Look for new modifier types
            negation_mods = [m for m in entity.modifiers if m.type == ModifierType.NEGATION]
            temporal_mods = [m for m in entity.modifiers if m.type == ModifierType.TEMPORAL]
            
            if negation_mods:
                print(f"  ❌ Negation: {[m.text for m in negation_mods]}")
            
            if temporal_mods:
                print(f"  ⏰ Temporal: {[m.text for m in temporal_mods]}")
            
            if not negation_mods and not temporal_mods:
                print(f"  ℹ️ Modifiers: {[m.text for m in entity.modifiers]}")


def example_4_option_a_graph():
    """Example 4: Option A - Modifier Nodes in Knowledge Graph"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Option A - Modifier Nodes in Knowledge Graph")
    print("="*70)
    
    text = "The highly efficient algorithm processes customer data."
    print(f"\nText: {text}\n")
    
    ie = InformationExtractor()
    graph = GraphConstructor()
    
    # Extract entities
    entities = ie.extract_entities(text)
    
    # Add entities and modifiers to graph
    for entity in entities:
        if entity.head_noun in ["algorithm", "data"]:
            # Add entity node
            entity_id = graph.add_node(
                label=entity.text,
                node_type="entity",
                properties={
                    "head_noun": entity.head_noun,
                    "label": entity.label,
                    "confidence": entity.confidence
                }
            )
            
            print(f"✓ Added entity node: '{entity.text}' (ID: {entity_id})")
            
            # Add modifier nodes
            if entity.modifiers:
                graph.add_modifier_nodes(entity_id, entity.modifiers)
                
                # Retrieve and display modifiers
                modifiers = graph.get_entity_modifiers(entity_id)
                
                print(f"  Modifiers in graph:")
                for mod_type, mods in modifiers.items():
                    for mod_info in mods:
                        print(f"    - {mod_info['text']:15} ({mod_type})")
                        
                        if mod_info['nested_modifiers']:
                            for nested in mod_info['nested_modifiers']:
                                print(f"      └─ {nested['text']} ({nested['modifier_type']})")
    
    # Print graph statistics
    stats = graph.get_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   Total Nodes: {stats['num_nodes']}")
    print(f"   Total Edges: {stats['num_edges']}")
    print(f"   Node Types: {stats['node_types']}")
    print(f"   Edge Types: {stats['edge_types']}")
    print(f"   Graph Density: {stats['density']:.3f}")


def example_5_modifier_export():
    """Example 5: Export modifiers to JSON"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Export Modifiers to JSON")
    print("="*70)
    
    text = "Three local police officers"
    print(f"\nText: {text}\n")
    
    ie = InformationExtractor()
    entities = ie.extract_entities(text)
    
    for entity in entities:
        if entity.head_noun == "officer":
            # Convert to JSON-serializable format
            entity_dict = {
                "text": entity.text,
                "head_noun": entity.head_noun,
                "label": entity.label,
                "modifier_summary": entity.modifier_summary,
                "modifiers": [m.to_dict() for m in entity.modifiers]
            }
            
            print("JSON Export:")
            print(json.dumps(entity_dict, indent=2))


def example_6_comparison_flat_vs_hierarchical():
    """Example 6: Comparison - Flat vs Hierarchical Representation"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Flat vs Hierarchical Representation")
    print("="*70)
    
    text = "The very highly efficient algorithm"
    print(f"\nText: {text}\n")
    
    ie = InformationExtractor()
    entities = ie.extract_entities(text)
    
    for entity in entities:
        if entity.head_noun == "algorithm":
            print("❌ OLD APPROACH (Flat List):")
            print(f"   modifiers = ['very', 'highly', 'efficient']")
            print(f"   Problem: Loses hierarchy relationship\n")
            
            print("✅ NEW APPROACH (Hierarchical):")
            print(f"   modifiers = {{")
            
            for i, mod in enumerate(entity.modifiers):
                if mod.modifiers:
                    print(f"     - '{mod.text}' (ADJECTIVE)")
                    for nested in mod.modifiers:
                        print(f"       └─ '{nested.text}' (ADVERBIAL)")
                else:
                    print(f"     - '{mod.text}' ({mod.type.value})")
            
            print(f"   }}")
            print(f"\n   Benefit: Can distinguish:")
            print(f"   - 'highly efficient' vs 'very efficient' vs 'efficient'")
            print(f"   - Different semantic weight and emphasis")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("HIERARCHICAL MODIFIER SYSTEM - QUICK START")
    print("="*70)
    
    try:
        example_1_basic_extraction()
        example_2_hierarchical_structure()
        example_3_negation_temporal()
        example_4_option_a_graph()
        example_5_modifier_export()
        example_6_comparison_flat_vs_hierarchical()
        
        print("\n" + "="*70)
        print("✅ All Examples Completed Successfully!")
        print("="*70)
        print("\nFor more information, see: docs/MODIFIER_SYSTEM.md")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
