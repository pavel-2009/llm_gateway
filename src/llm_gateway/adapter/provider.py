"""LLM Provider protocol for public LLM models eg. OpenAI, Anthropic, etc."""

from typing import Protocol


class LLMProvider(Protocol):
    """LLM Provider protocol"""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        ...

    async def stream(self, prompt: str):
        """Stream text from a prompt"""
        ...

    async def extract(self, prompt: str) -> str:
        """Extract text from a prompt"""
        ...
