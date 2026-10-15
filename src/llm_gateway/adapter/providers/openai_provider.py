"""OpenAI model provider."""

from typing import Any, Dict, List, Optional

from llm_gateway.adapter.provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI model provider"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        # Implement the logic to call OpenAI API and return the generated text
        pass

    def stream(self, prompt: str):
        """Stream text from a prompt"""
        # Implement the logic to call OpenAI API and stream the generated text
        pass

    def extract(self, prompt: str) -> str:
        """Extract text from a prompt"""
        # Implement the logic to call OpenAI API and extract specific information
        pass
