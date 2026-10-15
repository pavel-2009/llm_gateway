"""LLM Provider protocol for public LLM models eg. OpenAI, Anthropic, etc."""

from typing import Protocol


class LLMProvider(Protocol):
    """LLM Provider protocol"""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        ...

    def stream(self, prompt: str):
        """Stream text from a prompt"""
        ...

    def extract(self, prompt: str) -> str:
        """Extract text from a prompt"""
        ...
