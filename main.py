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
            extraction_data=extraction_result,
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
    Read by Richard Adapted by Bertie from a traditional German tale. Edited by Jana Elizabeth. Barking by Minnie the Cockapoo Princess sobs by Jana Dragon hisses by Bertie

This is the story of a shepherd who found himself in jail even though he had committed no crime. He was an honest lad, and his imprisonment was a terrible injustice. How he got there, and how he got out, I shall now tell you.

The shepherd lived with his old father and his sister. When the old man passed to a better world, all he left them was a house and three scraggy sheep. It was not much of an inheritance. The boy said to his sister, “Well, which do you like, the house or the sheep?” and she replied, “I’ll take the house, thank you very much.”

The boy nodded in agreement, and set off into the world chewing on a piece of grass, and driving his three sheep before him. He was determined to make his fortune, and as luck would have it, the next deal he did was rather better than the previous one. On top of a hill, he met a man who had three dogs: a wee little doggie called Salt – she had a white coat you see; a middle sized dog called Pepper – he was black you see; and a great big one called Mustard whose coat was sort of yellow. The master of these three animals said:

“Your three sheep are fattening up nicely.” In truth, they were rather scrawny, but the boy just nodded, and said “aye.”

“I’ll tell you what,” said the man. “I’ll swap yah! If you let me have them sheep of yours, you can take these here dogs.”

The boy thought to himself, “A shepherd without a dog is a bit like a spider without a web. This trade could turn out well for me yet.” And when he had considered everything from all angles, he said:

“Alright then. Let’s swap.”

And so the shepherd continued on his way to he knew not where, and the three dogs followed him with their tails held high.

Late in the evening, the boy sat down on a lump of grass and wiped his brow.

“My hunger’s making me weak,” he said. The little white dog called Salt was looking up at him with her big intelligent eyes. She understood what he said and went running off. Soon she came back with a basket of food. It was filled with the most delicious things: roast partridge, asparagus, and new potatoes with herbs and butter. The boy happily tucked into his meal and did not neglect to share his good fortune with his faithful dogs. Then he felt pleasantly drowsy. It was a warm night, and it was no hardship to lay his head on a soft mossy bank by a bubbling stream.

The following morning, he went on his way again. Around noon, he was leaning on his shepherd’s crook by the side of a crossroads when he saw a horse and carriage coming towards him. The carriage was painted black and gold as for a funeral of somebody rich and important. Up front, sat a coachman, all dressed in black. The boy called up to him:   

“Good afternoon to ya, good coachman. Prey tell me the news. What important man or lady has been and gone and died?”

“No one as yet,” said the coachman. “But I’ve got the fair princess in the back of my coach and I have orders to feed her to the dragon in the swamp.”

“Good gracious,” said the boy, “Why would anyone want you to feed a princess to a dragon?”

“No one wants it,” replied the coachman, “‘part from the dragon that’s greedy for his food. He says if we don’t give him what he wants, he’ll breathe fire and pestilence on the city; and what he wants is the princess. He won’t take no other lesser lass in her place. Very firm he is. Only the princess will do for him, so I’m sorry to say, she’s for it.”

The boy could not see the princess, for the curtains were drawn tight over the windows, but he could hear her sobs. He came over all hot and furious at the injustice. How dare a dragon get above himself and start demanding a princess for his lunch?

“Let me ride along with you,” said the boy, “and I’ll speak to this dragon and tell him how to behave himself better.”

“As you like,” said the coachman. “But I dare say he’ll just breathe fire and pestilence on you.”

“I’ll take that risk,” said the boy.

And so the coach rumbled on. When they reached the edge of the swamp, the boy jumped down and rejoined his faithful dogs who had followed behind. They heard terrible rumbles, as if from the very core of the Earth, but actually it was from the dragon’s tummy. Eventually they saw the beast himself – a horrid ugly, winged serpent he was, with the nastiest breath you ever did smell.

The little dog yapped, the big dog barked, and the middle sized dog, called Pepper, dived into the swamp and went scrambling towards the monster. When he reached the dragon, a terrible fight broke out. From where the boy and the coachman were standing, it looked as if the dragon was sure to win, but Pepper got the swamp-creature by the throat and killed him. When the job was done, he ran back to the coach with one of the dragons’ enormous teeth in his mouth. He dropped the gift at the foot of his master, who rewarded him with no end of praise.

The coachman said, “Well well, the king’s sure to be pleased as punch when he hears how his daughter’s been saved from her terrible fate.”

“I expect he shall be,” said the boy. “And now I’ll be on my way, because I’m impatient to make my fortune.” And with that, he strode off, followed by his faithful dogs.

The coachman drove the princess back to the palace, where he went directly before the king to explain how things had turned out:

“I’m happy to say Sire, that there was no need to feed the princess to the dragon of the swamp because I gave him a good talking to, and when he wouldn’t see no sense, I sorted him out good an proper. He won’t be troubling us no more, nor anyone else for that matter.”
    
“Oh Good Heavens! What surprising and wonderful news!” exclaimed the king, wiping a tear from his eye.

“I expect you’ll be err, wanting to give me a reward,” added the coachman, just in case the king, in his joy, had overlooked what was right and just.

“Yes, yes, you shall have the hand in marriage of my daughter whom you saved so gallantly,” said the king.

The church bells rang out over the city, but the princess did not share in the general rejoicing.

“Marry that brute of a coachman!” she exclaimed. “I’d rather be fed to the dragon!”

The king could not go back on his word to the coachman, but seeing his daughter so distraught, he agreed to postpone the ceremony until she had recovered from the shock of recent events.

Soon after, the shepherd boy came to the city. There he heard from an innkeeper that there was to be a royal wedding. The princess was to marry the coachman who had saved her from the dragon of the swamp. This news sent him flying into a rage. He marched straight down to the palace and demanded to see the king. When he came before his majesty, he said:

“What’s all this about you rewarding that conniving coachman? He’s taking the credit for my dog’s work, he is. It was this here Pepper that did it for that stinking dragon.”

The king was outraged at the idea of a mut saving the life of the princess.

“What would you have me do?” he asked. “Marry my daughter to your dog?”

“I’m not asking for nothing,” said the boy. “Just don’t tell no lies that’s all.”

The king signalled to the guards, who slung the insolent lad into prison. And that is how, as I said at the start of the story, an innocent boy was punished unjustly.

But I’m glad to say that he did not rot in jail for long. Pepper ran off to fetch Salt and Mustard. Just before dawn, the three dogs came to the prison. Salt brought food for his master, and Mustard wrapped his powerful jaws around the bars of the cell before tearing them out. Oh how pleased their master was to see them! And how they happily licked his hands and face!

Now the boy was more angry than ever at his treatment. He marched out of the prison and up to the palace. No guard could prevent him entering because Pepper and Mustard growled at them so fiercely, and Salt yapped around their ankles. Most of the palace was asleep, but the boy was in no mood to wait for anyone to wake up, not even the king. He went directly to the royal bedroom and barged his way in. Pepper jumped onto the chest of the snoring Monarch and woke him up with snarls.

The boy drew the curtains to let in the early morning light, before going up to the bed shaking an enormous tooth in his hand:

“Look at this here fang. It was Pepper what took it from the mouth of the dragon. Now do you believe us?”

“Yes, yes, I believe you,” said the king. “Only get this creature off me.”

“Gladly,” said the boy, who let out a whistle which Pepper instantly obeyed.

Later that morning, the king spoke to his daughter who confirmed that it was true, she had indeed heard a great deal of barking when the dragon was being slayed. Although she had not been able to see anything from within the carriage, for the curtains were drawn and she was hiding her face beneath a pillow, she believed the lad’s story. Now that she laid eyes on her true rescuer for the first time, she saw that he was not at all bad looking; she would far rather marry him, than the ugly coachman.

And so it was settled. The coachman went to jail and the lad married the princess, and he himself became a prince. Some months went by, and when it was almost Christmas, the boy recalled that he had a sister. He sent a carriage to fetch her so that she could share in his good fortune. On Christmas day, Pepper spoke to the boy and said:

“Now you have remembered your family, our work is done.”

And with that, the three dogs turned into birds and flew away to find another honest soul who was in need of their faithful service. And that was the story of ‘The Three Dogs’ read by me, Richard, for Storynory.com."""
    # Cows are herbivorous mammals that eat grass in meadows
    # Elon Musk, who is a billionaire, announced a new model Grok for his company.
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
