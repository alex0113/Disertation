"""
Knowledge Graph for Fake News Detection - Proof of Concept

This module demonstrates a knowledge graph that stores factual information
and can be used to detect and verify claims against established facts.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from neo4j import GraphDatabase


class RelationType(Enum):
    """Types of relationships in the knowledge graph."""
    BORN_IN = "born_in"
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    FOUNDED_BY = "founded_by"
    WROTE_ARTICLE = "wrote_article"
    PARTICIPATED_IN = "participated_in"
    LEADS = "leads"
    MEMBER_OF = "member_of"
    PUBLISHED_ON = "published_on"
    CONTRADICTS = "contradicts"


class CredibilityScore(Enum):
    """Credibility levels for facts and sources."""
    VERIFIED = 5
    LIKELY = 4
    NEUTRAL = 3
    QUESTIONABLE = 2
    DISPUTED = 1
    REFUTED = 0


@dataclass
class Source:
    """Represents a source of information."""
    name: str
    credibility: CredibilityScore
    url: str = ""
    publication_date: str = ""
    
    def to_dict(self):
        return {
            "name": self.name,
            "credibility": self.credibility.name,
            "url": self.url,
            "publication_date": self.publication_date
        }


@dataclass
class Fact:
    """Represents a fact in the knowledge graph."""
    subject: str
    predicate: str
    object: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.8
    
    def add_source(self, source: Source):
        """Add a source to this fact."""
        self.sources.append(source)
    
    def get_average_credibility(self) -> float:
        """Calculate average credibility from sources."""
        if not self.sources:
            return self.confidence
        return sum(s.credibility.value for s in self.sources) / len(self.sources)
    
    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence
        }


class KnowledgeGraph:
    """Core knowledge graph for storing facts and verifying claims."""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="hug0nam1"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.claim_history: List[Dict] = []
        self._clear_db()

    def close(self):
        """Close the database connection."""
        self.driver.close()

    def _clear_db(self):
        """Clear the database for the proof of concept."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def add_entity(self, name: str, entity_type: str, properties: Dict[str, str] = None):
        """Add an entity node to the graph."""
        properties = properties or {}
        with self.driver.session() as session:
            session.execute_write(self._create_entity_tx, name, entity_type, properties)
        print(f"✓ Added entity: {name} ({entity_type})")

    @staticmethod
    def _create_entity_tx(tx, name, entity_type, properties):
        query = (
            "MERGE (n:Entity {name: $name}) "
            "SET n.entity_type = $entity_type "
            "SET n += $props"
        )
        tx.run(query, name=name, entity_type=entity_type, props=properties)
    
    def add_fact(self, subject: str, predicate: str, obj: str, 
                 source: Source, confidence: float = 0.8):
        """Add a fact to the graph."""
        try:
            RelationType(predicate)
        except ValueError:
            return

        with self.driver.session() as session:
            session.execute_write(self._create_fact_tx, subject, predicate, obj, source, confidence)
        print(f"✓ Added fact: {subject} {predicate} {obj}")

    @staticmethod
    def _create_fact_tx(tx, subject, predicate, obj, source, confidence):
        source_json = json.dumps(source.to_dict())
        # Using predicate directly in Cypher string is safe here as it's validated against Enum
        query = (
            f"MATCH (s:Entity {{name: $subject}}), (o:Entity {{name: $obj}}) "
            f"MERGE (s)-[r:`{predicate}`]->(o) "
            "ON CREATE SET r.confidence = $confidence, r.sources = [] "
            "SET r.sources = r.sources + $source_json"
        )
        tx.run(query, subject=subject, obj=obj, source_json=source_json, confidence=confidence)
    
    def verify_claim(self, claim: str, subject: str, predicate: str, 
                    obj: str) -> Tuple[CredibilityScore, str]:
        """
        Verify a claim against the knowledge graph.
        Returns (credibility_score, reasoning)
        """
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (s:Entity {{name: $subject}})-[r:`{predicate}`]->(o:Entity {{name: $obj}}) "
                "RETURN r.sources as sources, r.confidence as confidence",
                subject=subject, obj=obj
            ).single()

        if not result:
            return CredibilityScore.DISPUTED, "No supporting facts found in knowledge graph"
        
        sources_data = result["sources"]
        confidence = result["confidence"]
        
        # Reconstruct sources to calculate credibility
        sources = []
        for s_str in sources_data:
            s_dict = json.loads(s_str)
            s_dict["credibility"] = CredibilityScore[s_dict["credibility"]]
            sources.append(Source(**s_dict))
            
        if not sources:
            avg_credibility = confidence
        else:
            avg_credibility = sum(s.credibility.value for s in sources) / len(sources)
            
        num_sources = len(sources)
        
        if avg_credibility >= 4.5:
            score = CredibilityScore.VERIFIED
            reasoning = f"Verified with high confidence ({avg_credibility:.2f}/5) from {num_sources} sources"
        elif avg_credibility >= 3.5:
            score = CredibilityScore.LIKELY
            reasoning = f"Likely accurate ({avg_credibility:.2f}/5) from {num_sources} sources"
        elif avg_credibility >= 2.5:
            score = CredibilityScore.NEUTRAL
            reasoning = f"Neutral credibility ({avg_credibility:.2f}/5) with mixed sources"
        else:
            score = CredibilityScore.QUESTIONABLE
            reasoning = f"Low credibility ({avg_credibility:.2f}/5) from unreliable sources"
        
        return score, reasoning
    
    def check_contradiction(self, subject: str, predicate: str, obj: str) -> Optional[str]:
        """Check if a claimed fact contradicts existing facts."""
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (s:Entity {{name: $subject}})-[r:`{predicate}`]->(o:Entity) "
                "WHERE o.name <> $obj "
                "RETURN o.name as conflicting_obj",
                subject=subject, obj=obj
            ).single()
            
        if result:
            conflicting_obj = result["conflicting_obj"]
            return f"Contradiction found: {subject} {predicate} {obj}, but graph shows {subject} {predicate} {conflicting_obj}"
        
        return None
    
    def get_entity_facts(self, entity: str) -> List[Fact]:
        """Get all facts about a specific entity."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r]->(o:Entity) "
                "WHERE s.name = $entity OR o.name = $entity "
                "RETURN s.name as subject, type(r) as predicate, o.name as object, "
                "r.sources as sources, r.confidence as confidence",
                entity=entity
            )
            
            facts = []
            for record in result:
                sources = []
                for s_str in record["sources"]:
                    s_dict = json.loads(s_str)
                    s_dict["credibility"] = CredibilityScore[s_dict["credibility"]]
                    sources.append(Source(**s_dict))
                
                fact = Fact(
                    subject=record["subject"],
                    predicate=record["predicate"],
                    object=record["object"],
                    sources=sources,
                    confidence=record["confidence"]
                )
                facts.append(fact)
            return facts
    
    def get_related_entities(self, entity: str) -> Set[str]:
        """Get all entities related to a given entity."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Entity {name: $entity})--(related) RETURN related.name as name",
                entity=entity
            )
            return {record["name"] for record in result}
    
    def export_facts(self) -> List[Dict]:
        """Export all facts as dictionaries."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r]->(o:Entity) "
                "RETURN s.name as subject, type(r) as predicate, o.name as object, "
                "r.sources as sources, r.confidence as confidence"
            )
            
            facts_list = []
            for record in result:
                sources = []
                for s_str in record["sources"]:
                    s_dict = json.loads(s_str)
                    s_dict["credibility"] = CredibilityScore[s_dict["credibility"]]
                    sources.append(Source(**s_dict))
                
                fact = Fact(
                    subject=record["subject"],
                    predicate=record["predicate"],
                    object=record["object"],
                    sources=sources,
                    confidence=record["confidence"]
                )
                facts_list.append(fact.to_dict())
            return facts_list
    
    def log_claim_check(self, claim: str, result: CredibilityScore, reasoning: str):
        """Log a claim verification for audit trail."""
        self.claim_history.append({
            "claim": claim,
            "result": result.name,
            "reasoning": reasoning
        })


def demonstrate_knowledge_graph():
    """Demonstrate the knowledge graph with example scenarios."""
    
    print("\n" + "="*70)
    print("Knowledge Graph for Fake News Detection - Proof of Concept")
    print("="*70 + "\n")
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    
    # Define credible sources
    bbc = Source("BBC News", CredibilityScore.VERIFIED, 
                 "https://bbc.com", "2026-03-01")
    nyt = Source("New York Times", CredibilityScore.VERIFIED, 
                 "https://nytimes.com", "2026-03-02")
    wikipedia = Source("Wikipedia", CredibilityScore.LIKELY, 
                       "https://wikipedia.org", "2026-02-28")
    social_media = Source("Social Media", CredibilityScore.QUESTIONABLE, 
                         "", "2026-03-08")
    
    # Add entities to graph
    print("Step 1: Building Knowledge Graph Structure")
    print("-" * 70)
    kg.add_entity("Alice Johnson", "person", {
        "title": "Climate Scientist",
        "institution": "MIT"
    })
    kg.add_entity("MIT", "organization", {
        "type": "University",
        "founded": "1861",
        "location": "Boston, MA"
    })
    kg.add_entity("Climate Change Summit 2025", "event", {
        "date": "2025-12-01",
        "location": "Singapore"
    })
    kg.add_entity("Global Temperature", "metric")
    
    # Add facts to graph
    print("\nStep 2: Populating Graph with Verified Facts")
    print("-" * 70)
    kg.add_fact("Alice Johnson", "works_for", "MIT", nyt)
    kg.add_fact("MIT", "located_in", "Boston, MA", wikipedia)
    kg.add_fact("Alice Johnson", "participated_in", "Climate Change Summit 2025", bbc)
    kg.add_fact("Global Temperature", "increased_by", "1.1°C", bbc)
    
    # Verify claims
    print("\nStep 3: Verifying Claims Against Knowledge Graph")
    print("-" * 70)
    
    # Correct claim
    score, reasoning = kg.verify_claim(
        "Alice Johnson works at MIT",
        "Alice Johnson", "works_for", "MIT"
    )
    kg.log_claim_check("Alice Johnson works at MIT", score, reasoning)
    print(f"\n📋 Claim: Alice Johnson works at MIT")
    print(f"   Status: {score.name} (Level {score.value}/5)")
    print(f"   Reason: {reasoning}")
    
    # Unknown claim
    score, reasoning = kg.verify_claim(
        "Alice Johnson is the CEO of MIT",
        "Alice Johnson", "is_ceo_of", "MIT"
    )
    kg.log_claim_check("Alice Johnson is the CEO of MIT", score, reasoning)
    print(f"\n📋 Claim: Alice Johnson is the CEO of MIT")
    print(f"   Status: {score.name} (Level {score.value}/5)")
    print(f"   Reason: {reasoning}")
    
    # Check for contradictions
    print("\nStep 4: Detecting Contradictions")
    print("-" * 70)
    contradiction = kg.check_contradiction(
        "Alice Johnson", "works_for", "Harvard University"
    )
    if contradiction:
        print(f"⚠️  {contradiction}")
    else:
        print("✓ No contradictions found")
    
    # Show entity relationship network
    print("\nStep 5: Exploring Relationship Network")
    print("-" * 70)
    entity = "Alice Johnson"
    related = kg.get_related_entities(entity)
    print(f"Entities related to '{entity}': {', '.join(related)}")
    print(f"Total facts about '{entity}': {len(kg.get_entity_facts(entity))}")
    
    # Display verification history
    print("\nStep 6: Claim Verification History")
    print("-" * 70)
    for i, record in enumerate(kg.claim_history, 1):
        print(f"{i}. {record['claim']}")
        print(f"   → Result: {record['result']}")
        print(f"   → Reason: {record['reasoning']}\n")
    
    # Export facts
    print("Step 7: Exporting Knowledge Graph Data")
    print("-" * 70)
    facts_export = kg.export_facts()
    print(f"Total facts in graph: {len(facts_export)}")
    print("\nSample fact structure:")
    if facts_export:
        print(json.dumps(facts_export[0], indent=2))
    
    kg.close()


def main():
    """Main entry point."""
    demonstrate_knowledge_graph()
    
    print("\n" + "="*70)
    print("Key Features of This Knowledge Graph PoC:")
    print("="*70)
    print("""
    ✓ Entity-Relationship Model: Stores facts with subjects, predicates, objects
    ✓ Multi-Source Support: Facts can have multiple sources with credibility scores
    ✓ Claim Verification: Matches incoming claims against graph facts
    ✓ Contradiction Detection: Identifies conflicting information
    ✓ Confidence Scoring: Aggregates source credibility for claims
    ✓ Relationship Exploration: Query related entities and fact networks
    ✓ Audit Trail: Logs all claim verifications for transparency
    
    Use Cases for Fake News Detection:
    • Verify factual claims against established knowledge
    • Cross-reference multiple sources for consistency
    • Identify contradictions in narratives
    • Track credibility of information sources
    • Build trust networks based on source reliability
    """)


if __name__ == "__main__":
    main()
