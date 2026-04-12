"""
Knowledge Extractor using LLM providers (abstract layer with Gemini implementation)

This module provides an abstract interface for LLM-based knowledge extraction,
with a concrete implementation using Google's Gemini API. The abstraction allows
for easy replacement with other LLM providers in the future.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from dotenv import load_dotenv
from knowledge_graph import Fact, Source, CredibilityScore, RelationType


# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a knowledge extraction operation."""
    facts: List[Fact]
    raw_response: str
    provider: str
    success: bool
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def extract_facts(self, text: str, context: Optional[str] = None) -> ExtractionResult:
        """
        Extract facts from text using the LLM provider.
        
        Args:
            text: The text to extract facts from
            context: Optional context or system prompt to guide extraction
            
        Returns:
            ExtractionResult containing extracted facts and metadata
        """
        pass
    
    @abstractmethod
    def close(self):
        """Clean up any resources used by the provider."""
        pass


class GeminiProvider(LLMProvider):
    """Knowledge extractor using Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY environment variable)
            model: Model name to use (default: gemini-1.5-pro)
        """
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY environment variable not set"
            )
        
        self.genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.provider_name = f"Gemini ({model})"
    
    def extract_facts(self, text: str, context: Optional[str] = None) -> ExtractionResult:
        """
        Extract facts from text using Gemini API.
        
        Args:
            text: The text to extract facts from
            context: Optional system context for extraction guidance
            
        Returns:
            ExtractionResult with extracted Fact objects
        """
        try:
            prompt = self._build_prompt(text, context)
            response = self.model.generate_content(prompt)
            
            raw_response = response.text
            facts = self._parse_response(raw_response, text)
            
            return ExtractionResult(
                facts=facts,
                raw_response=raw_response,
                provider=self.provider_name,
                success=True
            )
        except Exception as e:
            logger.error(f"Error extracting facts with Gemini: {str(e)}")
            return ExtractionResult(
                facts=[],
                raw_response="",
                provider=self.provider_name,
                success=False,
                error=str(e)
            )
    
    def _build_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Build the extraction prompt for Gemini."""
        system_context = context or (
            "You are an expert knowledge extraction system. Your task is to extract "
            "factual information from the provided text in a structured format."
        )
        
        prompt = f"""{system_context}

Extract all factual claims from the following text. For each fact, identify:
1. Subject (the entity or person being described)
2. Predicate/Relationship (the relationship or property)
3. Object (what is being stated about the subject)

Format your response as a JSON array with objects containing:
{{
    "subject": "string",
    "predicate": "string", 
    "object": "string"
}}

Text to analyze:
{text}

Respond ONLY with a valid JSON array, no additional text."""
        
        return prompt
    
    def _parse_response(self, response_text: str, source_text: str) -> List[Fact]:
        """Parse the Gemini response into Fact objects."""
        facts = []
        
        try:
            # Extract JSON from response
            json_str = response_text.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            
            fact_data = json.loads(json_str)
            
            # Convert to Fact objects
            for item in fact_data:
                if not isinstance(item, dict):
                    continue
                
                subject = item.get("subject", "").strip()
                predicate = item.get("predicate", "").strip()
                obj = item.get("object", "").strip()
                
                if subject and predicate and obj:
                    # Create fact with Gemini as source
                    source = Source(
                        name="Gemini LLM",
                        credibility=CredibilityScore.NEUTRAL,
                        url="",
                        publication_date=""
                    )
                    
                    fact = Fact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        sources=[source],
                        confidence=0.7  # Conservative confidence for LLM-extracted facts
                    )
                    facts.append(fact)
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing facts from Gemini response: {str(e)}")
        
        return facts
    
    def close(self):
        """Clean up Gemini resources."""
        pass


class KnowledgeExtractor:
    """Main interface for knowledge extraction."""
    
    def __init__(self, provider: LLMProvider):
        """
        Initialize the knowledge extractor with a provider.
        
        Args:
            provider: An LLMProvider instance (e.g., GeminiProvider)
        """
        self.provider = provider
    
    def extract_from_text(
        self, 
        text: str, 
        context: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract facts from text.
        
        Args:
            text: The text to extract facts from
            context: Optional context/guidance for extraction
            
        Returns:
            ExtractionResult containing extracted facts
        """
        return self.provider.extract_facts(text, context)
    
    def extract_from_file(
        self,
        file_path: str,
        context: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract facts from a file.
        
        Args:
            file_path: Path to the text file
            context: Optional context/guidance for extraction
            
        Returns:
            ExtractionResult containing extracted facts
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.extract_from_text(text, context)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return ExtractionResult(
                facts=[],
                raw_response="",
                provider=self.provider.provider_name if hasattr(self.provider, 'provider_name') else "Unknown",
                success=False,
                error=str(e)
            )
    
    def close(self):
        """Clean up resources."""
        self.provider.close()


# Example usage and helper function
def create_gemini_extractor(api_key: Optional[str] = None) -> KnowledgeExtractor:
    """
    Create a knowledge extractor using Gemini API.
    
    Args:
        api_key: Optional Google API key (falls back to GOOGLE_API_KEY env var)
        
    Returns:
        KnowledgeExtractor instance configured with Gemini
        
    Example:
        >>> extractor = create_gemini_extractor()
        >>> result = extractor.extract_from_text("Albert Einstein was born in Germany.")
        >>> for fact in result.facts:
        ...     print(f"{fact.subject} -> {fact.predicate} -> {fact.object}")
    """
    provider = GeminiProvider(api_key=api_key)
    return KnowledgeExtractor(provider)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create extractor
    extractor = create_gemini_extractor()
    
    # Example text
    sample_text = """
    Albert Einstein was born in Ulm, Germany in 1879. He was a theoretical physicist 
    who developed the theory of relativity. Einstein worked at the Princeton Institute 
    for Advanced Study. He won the Nobel Prize in Physics in 1921.
    """
    
    # Extract facts
    result = extractor.extract_from_text(sample_text)
    
    if result.success:
        print(f"Extraction successful using {result.provider}")
        print(f"Extracted {len(result.facts)} facts:\n")
        for fact in result.facts:
            print(f"  Subject: {fact.subject}")
            print(f"  Predicate: {fact.predicate}")
            print(f"  Object: {fact.object}")
            print(f"  Confidence: {fact.confidence}")
            print()
    else:
        print(f"Extraction failed: {result.error}")
    
    extractor.close()
