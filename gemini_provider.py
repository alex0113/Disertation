"""Gemini provider implementation for knowledge extraction."""

import json
import os
import logging
from typing import Optional, Any, List

from dotenv import load_dotenv
from llm_base import LLMProvider, ExtractionResult
from knowledge_graph import Fact, Source, CredibilityScore


load_dotenv()
logger = logging.getLogger(__name__)
GEMINI_MODEL = "gemini-flash-lite-latest"


class GeminiProvider(LLMProvider):
    """Knowledge extractor using Google's Gemini API via google.genai."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        """Initialize the Gemini provider."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY environment variable not set"
            )

        self.model = model
        self.provider_name = f"Gemini ({model})"
        self.client = None

        try:
            from google import genai
            self.genai = genai
            self.client = self._create_client()
        except ImportError:
            raise ImportError(
                "google-genai is not installed. Install it with: pip install google-genai"
            )

    def _create_client(self):
        """Create a google.genai client instance."""
        if hasattr(self.genai, "GenerativeAI"):
            return self.genai.GenerativeAI(api_key=self.api_key)
        if hasattr(self.genai, "Client"):
            return self.genai.Client(api_key=self.api_key)
        if hasattr(self.genai, "GenAI"):
            return self.genai.GenAI(api_key=self.api_key)
        raise AttributeError(
            "Unsupported google.genai client API. "
            "Expected GenerativeAI, Client, or GenAI class."
        )

    def extract_facts(self, text: str, context: Optional[str] = None) -> ExtractionResult:
        """Extract facts from text using Gemini API."""
        try:
            prompt = self._build_prompt(text, context)
            response = self._generate_response(prompt)
            raw_response = self._extract_text_from_response(response)
            facts = self._parse_response(raw_response, text)

            return ExtractionResult(
                facts=facts,
                raw_response=raw_response,
                provider=self.provider_name,
                success=True,
            )
        except Exception as e:
            logger.error(f"Error extracting facts with Gemini: {str(e)}")
            return ExtractionResult(
                facts=[],
                raw_response="",
                provider=self.provider_name,
                success=False,
                error=str(e),
            )

    def _generate_response(self, prompt: str):
        """Generate a response using the google.genai API."""
        if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
            return self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    # Thinking settings can be adjusted here if needed.
                },
            )

        raise AttributeError("The google.genai client does not support a known generation method.")

    def _extract_text_from_response(self, response: Any) -> str:
        """Normalize different Gemini response objects to plain text."""
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if hasattr(response, "text"):
            return getattr(response, "text") or ""

        if hasattr(response, "output_text"):
            return getattr(response, "output_text") or ""

        if hasattr(response, "output"):
            output = getattr(response, "output")
            if isinstance(output, str):
                return output
            if isinstance(output, list) and output:
                return self._extract_text_from_response(output[0])

        if hasattr(response, "candidates"):
            candidates = getattr(response, "candidates")
            if isinstance(candidates, list) and candidates:
                return self._extract_text_from_response(candidates[0])

        if hasattr(response, "content"):
            content = getattr(response, "content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                return self._extract_text_from_response(content[0])

        return str(response)

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
            json_str = response_text.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]

            fact_data = json.loads(json_str)

            for item in fact_data:
                if not isinstance(item, dict):
                    continue

                subject = item.get("subject", "").strip()
                predicate = item.get("predicate", "").strip()
                obj = item.get("object", "").strip()

                if subject and predicate and obj:
                    source = Source(
                        name="Gemini LLM",
                        credibility=CredibilityScore.NEUTRAL,
                        url="",
                        publication_date="",
                    )

                    fact = Fact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        sources=[source],
                        confidence=0.7,
                    )
                    facts.append(fact)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing facts from Gemini response: {str(e)}")

        return facts

    def close(self):
        """Clean up Gemini resources."""
        if self.client and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass
