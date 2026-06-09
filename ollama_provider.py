"""Ollama provider implementation for knowledge extraction."""

import json
import logging
import re
import shutil
import subprocess
from typing import Optional

from llm_base import LLMProvider, ExtractionResult
from knowledge_graph import Fact, Source, CredibilityScore


logger = logging.getLogger(__name__)

# Global Ollama model configuration
OLLAMA_MODEL = "qwen2.5-coder:1.5b"


class OllamaProvider(LLMProvider):
    """Knowledge extractor using a locally installed Ollama model."""

    def __init__(self):
        """Initialize the Ollama provider."""
        if shutil.which("ollama") is None:
            raise EnvironmentError(
                "Ollama CLI not found. Install Ollama and ensure 'ollama' is available on PATH."
            )

        self.provider_name = f"Ollama ({OLLAMA_MODEL})"

    def extract_facts(self, text: str, context: Optional[str] = None) -> ExtractionResult:
        """Extract facts from text using the local Ollama model."""
        try:
            prompt = self._build_prompt(text, context)
            raw_response = self._generate_response(prompt)
            facts = self._parse_response(raw_response, text)

            return ExtractionResult(
                facts=facts,
                raw_response=raw_response,
                provider=self.provider_name,
                success=True,
            )
        except Exception as e:
            logger.error(f"Error extracting facts with Ollama: {str(e)}")
            return ExtractionResult(
                facts=[],
                raw_response="",
                provider=self.provider_name,
                success=False,
                error=str(e),
            )

    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the local Ollama CLI."""
        command = ["ollama", "run", OLLAMA_MODEL]
        try:
            proc = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise EnvironmentError("Ollama CLI not found on PATH") from e

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(
                f"Ollama generation failed (exit {proc.returncode}): {stderr}"
            )

        output = proc.stdout.strip()
        if not output:
            raise ValueError("Ollama returned empty output")

        # Ollama run returns plain text by default, try to parse as JSON if possible
        try:
            # Try to extract JSON if present in the output
            json_start = output.find('[')
            if json_start != -1:
                json_end = output.rfind(']') + 1
                if json_end > json_start:
                    return output[json_start:json_end]
            # Otherwise return the plain text output
            return output
        except Exception:
            return output

    def _build_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Build the extraction prompt for Ollama."""
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
"""
        return prompt

    def _parse_response(self, response_text: str, source_text: str) -> list[Fact]:
        """Parse the Ollama response into Fact objects."""
        facts = []

        try:
            json_str = response_text.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]

            # Extract the first JSON array block from the response if present.
            array_match = re.search(r"\[.*\]", json_str, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)

            # Collapse literal newlines inside quoted strings to keep JSON valid.
            def clean_quoted_strings(match):
                text = match.group(0)
                cleaned = text.replace("\n", " ").replace("\r", " ")
                return cleaned

            # Use a more robust pattern for JSON strings that handles multiline content
            string_pattern = r'"(?:[^"\\]|\\.)*"'
            json_str = re.sub(string_pattern, clean_quoted_strings, json_str, flags=re.DOTALL)

            # Use the cleaned JSON as the raw response

            json_str = json_str.strip()

            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            cleaned_text = ansi_escape.sub('', json_str)
            
            # 2. Extract ONLY the JSON part (ignore the conversational text and markdown)
            # This looks for the first '[' and the last ']' and grabs everything in between
            json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
            if not json_match:
                raise ValueError("Ollama response contained no JSON content")
            
            json_str = json_match.group(0)
            print(f"Attempting to parse JSON from Ollama response: {json_str!r}")

            fact_data = json.loads(json_str)
            print(f"Parsed JSON from Ollama response: {fact_data}")

            for item in fact_data:
                if not isinstance(item, dict):
                    continue

                subject = item.get("subject", "").strip()
                predicate = item.get("predicate", "").strip()
                obj = item.get("object", "").strip()

                if subject and predicate:
                    source = Source(
                        name="Ollama LLM",
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
            logger.warning(
                f"Failed to parse Ollama response as JSON: {str(e)}. "
                f"Raw response: {response_text!r}"
            )
        except Exception as e:
            logger.error(
                f"Error parsing facts from Ollama response: {str(e)}. "
                f"Raw response: {response_text!r}"
            )

        return facts

    def close(self):
        """Clean up Ollama resources."""
        pass
