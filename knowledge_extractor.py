"""
Knowledge Extractor using LLM providers (abstract layer with pluggable provider modules)

This module provides an abstract interface for LLM-based knowledge extraction,
with support for different provider implementations such as Gemini and Ollama.
The abstraction allows for easy replacement with other LLM providers in the future.
"""

import json
import os
from typing import List, Optional, Any
import logging

from dotenv import load_dotenv
from llm_base import ExtractionResult, LLMProvider
from gemini_provider import GeminiProvider
from ollama_provider import OllamaProvider
from knowledge_graph import KnowledgeGraph


# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

def list_available_models():
    """List all available Gemini models."""
    try:
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("GOOGLE_API_KEY not found in environment variables")
            return
        
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        
        print("Available Gemini Models:")
        print("=" * 50)
        for model in models:
            print(f"- {model.name}")
            if hasattr(model, 'description') and model.description:
                print(f"  Description: {model.description}")
            if hasattr(model, 'version') and model.version:
                print(f"  Version: {model.version}")
            print()
            
    except ImportError:
        print("google-genai package not installed")
    except Exception as e:
        print(f"Error listing models: {e}")


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


def create_ollama_extractor(model: str = "llama2") -> KnowledgeExtractor:
    """
    Create a knowledge extractor using a locally installed Ollama model.

    Args:
        model: Local Ollama model name to use (default: llama2)

    Returns:
        KnowledgeExtractor instance configured with Ollama
    """
    provider = OllamaProvider()
    return KnowledgeExtractor(provider)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # List available Gemini models before running extraction
    list_available_models()
    
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
        
        # Populate knowledge graph with extracted facts
        print("\nPopulating knowledge graph...")
        try:
            kg = KnowledgeGraph()
            kg.populate_from_facts(result.facts)
            print(f"✓ Successfully populated knowledge graph with {len(result.facts)} facts")
            kg.close()
        except Exception as e:
            print(f"Error populating knowledge graph: {e}")
    else:
        print(f"Extraction failed: {result.error}")
    
    extractor.close()
