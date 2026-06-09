"""
Main orchestration script for knowledge extraction and graph generation

This script demonstrates the complete workflow:
1. Extract facts using the mock knowledge extractor
2. Populate a knowledge graph with the extracted facts
3. Query and display the populated graph
"""

import logging
from knowledge_extractor import create_ollama_extractor
from knowledge_graph import KnowledgeGraph


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main workflow: extract facts and populate knowledge graph."""
    
    print("\n" + "=" * 80)
    print("KNOWLEDGE EXTRACTION AND GRAPH GENERATION")
    print("=" * 80 + "\n")
    
    # Step 1: Create Ollama extractor
    print("[1/4] Creating Ollama knowledge extractor...")
    extractor = create_ollama_extractor(model="llama2")
    print("✓ Ollama extractor created\n")
    
    # Step 2: Extract facts
    print("[2/4] Extracting facts from sample text...")
    sample_text = """
    Nikola Tesla was born on July 10, 1856 in Smiljan, Croatia. He developed the alternating
    current (AC) electrical system and invented the Tesla coil. Marie Curie discovered uranium
    and polonium and was the first woman to win a Nobel Prize. The Eiffel Tower opened in Paris
    in 1889 for the Exposition Universelle. The Earth orbits the Sun once every 365.25 days.
    Python is a high-level programming language created by Guido van Rossum and first released
    in 1991. The Great Wall of China is more than 13,000 miles long.
    """
    
    logger.info(f"LLM Query Input:\n{sample_text}\n")
    print(f"[DEBUG] Sending to LLM:\n{sample_text}\n")
    
    result = extractor.extract_from_text(sample_text)
    
    if result.success:
        print(f"✓ Extraction successful using {result.provider}")
        print(f"✓ Extracted {len(result.facts)} facts:\n")
        logger.info(f"LLM Raw Response:\n{result.raw_response}\n")
        for i, fact in enumerate(result.facts, 1):
            print(f"   {i}. {fact.subject} -> {fact.predicate} -> {fact.object}")
            print(f"      Confidence: {fact.confidence}\n")
    else:
        print(f"✗ Extraction failed: {result.error}")
        return
    
    # Step 3: Initialize knowledge graph and populate it
    print("[3/4] Initializing knowledge graph...")
    try:
        kg = KnowledgeGraph()
        print("✓ Knowledge graph connected to Neo4j\n")
        
        print("[4/4] Populating knowledge graph with extracted facts...")
        kg.populate_from_facts(result.facts)
        print(f"✓ Successfully populated knowledge graph with {len(result.facts)} facts\n")
        
        # Step 4: Query and display the graph
        print("=" * 80)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("=" * 80 + "\n")
        
        # Get all extracted entities
        entities = set()
        for fact in result.facts:
            entities.add(fact.subject)
            entities.add(fact.object)
        
        print(f"Entities in graph: {len(entities)}")
        for entity in sorted(entities):
            print(f"  - {entity}")
        
        print(f"\nTotal facts stored: {len(result.facts)}")
        print("\nFacts by relationship:")
        predicates = {}
        for fact in result.facts:
            pred = fact.predicate
            if pred not in predicates:
                predicates[pred] = []
            predicates[pred].append(f"{fact.subject} -> {fact.object}")
        
        for predicate, relations in sorted(predicates.items()):
            print(f"\n  {predicate}:")
            for relation in relations:
                print(f"    - {relation}")
        
        # Step 5: Extract facts from KG for a claim and prepare prompt
        print("\n" + "=" * 80)
        print("CLAIM VERIFICATION PROMPT PREPARATION")
        print("=" * 80 + "\n")
        
        claim_to_test = "Nikola Tesla invented the alternating current (AC) electrical system and the lightbulb, and was born in Paris."
        print(f"Claim to test: '{claim_to_test}'")
        
        # Extract entities from the claim by simple string matching against known entities
        claim_entities = [e for e in entities if e.lower() in claim_to_test.lower()]
        print(f"Entities identified in claim: {claim_entities}\n")
        
        # Extract facts for these entities
        extracted_facts = []
        for entity in claim_entities:
            extracted_facts.extend(kg.get_entity_facts(entity))
            
        # Deduplicate facts based on subject, predicate, object
        unique_facts = []
        seen = set()
        for fact in extracted_facts:
            fact_tuple = (fact.subject, fact.predicate, fact.object)
            if fact_tuple not in seen:
                seen.add(fact_tuple)
                unique_facts.append(fact)
                
        if unique_facts:
            facts_str = "\n".join([f"- {f.subject} -> {f.predicate} -> {f.object}" for f in unique_facts])
        else:
            facts_str = "- No relevant facts found."
            
        # Prepare the prompt
        prompt = f"""System: You are an expert fact-checker. Please verify the following claim based ONLY on the provided knowledge graph facts.

Claim:
"{claim_to_test}"

Extracted Facts:
{facts_str}

Analyze the claim. State whether it is Supported, Contradicted, or if there is Insufficient Information. Provide your reasoning."""

        print("Generated Prompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80 + "\n")
        print("Knowledge graph has been successfully generated and is ready for:")
        print("  • Fact verification")
        print("  • Contradiction detection")
        print("  • Entity relationship queries")
        print("  • Credibility analysis")
        print()
        
        kg.close()
        
    except Exception as e:
        print(f"✗ Error populating knowledge graph: {e}")
        logger.exception("Knowledge graph population failed")
        return
    
    extractor.close()


if __name__ == "__main__":
    main()
