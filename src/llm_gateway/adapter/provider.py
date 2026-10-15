"""LLM provider protocol shared by all provider adapters."""

from collections.abc import AsyncIterator
from typing import Protocol

from llm_gateway.adapter.contracts import ExtractedReview, GenerateResponse


class LLMProvider(Protocol):
    """Common async interface for OpenAI, Claude, and Ollama providers."""

    async def generate(self, prompt: str) -> GenerateResponse:
        """Generate a complete text response."""
        ...

    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated text chunks."""
        ...

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract and validate structured data from a prompt."""
        ...
