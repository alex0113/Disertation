"""LLM provider base definitions for knowledge extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExtractionResult:
    """Result of a knowledge extraction operation."""
    facts: List["Fact"]
    raw_response: str
    provider: str
    success: bool
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def extract_facts(self, text: str, context: Optional[str] = None) -> ExtractionResult:
        """Extract facts from the provided text."""
        pass

    @abstractmethod
    def close(self):
        """Clean up any resources used by the provider."""
        pass
