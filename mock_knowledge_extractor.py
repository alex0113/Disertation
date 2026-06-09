"""
Mock Knowledge Extractor for testing without API dependency

This module provides a mock implementation of the KnowledgeExtractor that returns
the same pre-defined facts on every call instead of calling the Gemini API. Useful
for testing and development without incurring API costs or requiring API keys.
"""

import logging
from typing import List, Optional

from knowledge_extractor import KnowledgeExtractor
from llm_base import LLMProvider, ExtractionResult
from knowledge_graph import Fact, Source, CredibilityScore


logger = logging.getLogger(__name__)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that always returns the same pre-defined facts."""

    def __init__(self):
        """Initialize the mock provider with no API key required."""
        self.provider_name = "Mock LLM (Test Data)"
        self.call_count = 0

    def extract_facts(
        self, text: str, context: Optional[str] = None
    ) -> ExtractionResult:
        """
        Return the same mock facts regardless of input.

        Args:
            text: The input text (ignored, always returns same facts)
            context: Optional context (ignored in mock)

        Returns:
            ExtractionResult with pre-defined mock facts
        """
        self.call_count += 1
        logger.info(f"Mock extraction call #{self.call_count}")

        # Create mock source
        mock_source = Source(
            name="Mock Data Source",
            credibility=CredibilityScore.LIKELY,
            url="https://mock.example.com",
            publication_date="2026-04-13",
        )

        # Always return the same facts
        facts = [
            Fact(
                subject="Albert Einstein",
                predicate="born_in",
                object="Ulm, Germany",
                sources=[mock_source],
                confidence=0.95,
            ),
            Fact(
                subject="Albert Einstein",
                predicate="developed",
                object="Theory of Relativity",
                sources=[mock_source],
                confidence=0.98,
            ),
            Fact(
                subject="Marie Curie",
                predicate="discovered",
                object="Radium",
                sources=[mock_source],
                confidence=0.98,
            ),
            Fact(
                subject="Isaac Newton",
                predicate="discovered",
                object="Law of Gravitation",
                sources=[mock_source],
                confidence=0.99,
            ),
        ]

        raw_response = "Mock response with predefined facts"

        return ExtractionResult(
            facts=facts,
            raw_response=raw_response,
            provider=self.provider_name,
            success=True,
        )

    def close(self):
        """Clean up mock resources."""
        logger.info(f"Closed mock provider after {self.call_count} calls")


def create_mock_extractor() -> KnowledgeExtractor:
    """
    Create a knowledge extractor using mock data.

    Returns:
        KnowledgeExtractor instance configured with MockLLMProvider

    Example:
        >>> extractor = create_mock_extractor()
        >>> result = extractor.extract_from_text("Any input text")
        >>> for fact in result.facts:
        ...     print(f"{fact.subject} -> {fact.predicate} -> {fact.object}")
    """
    provider = MockLLMProvider()
    return KnowledgeExtractor(provider)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("Mock Knowledge Extractor - Testing Example")
    print("=" * 70 + "\n")

    # Create mock extractor
    extractor = create_mock_extractor()

    # Test cases - all return the same mock facts
    test_texts = [
        "Tell me about Albert Einstein",
        "What is Marie Curie famous for?",
        "Random text about anything",
    ]

    for test_text in test_texts:
        print(f"\nInput: {test_text}")
        print("-" * 70)

        result = extractor.extract_from_text(test_text)

        if result.success:
            print(f"Provider: {result.provider}")
            print(f"Extracted {len(result.facts)} facts:\n")
            for fact in result.facts:
                print(f"  {fact.subject} -> {fact.predicate} -> {fact.object}")
                print(f"    Confidence: {fact.confidence}")
            print()
        else:
            print(f"Extraction failed: {result.error}\n")

    extractor.close()

    print("\n" + "=" * 70)
    print("Mock Test Complete")
    print("=" * 70)
