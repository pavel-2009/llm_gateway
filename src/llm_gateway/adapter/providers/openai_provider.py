"""OpenAI model provider."""

from typing import Any, Dict, List, Optional

from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import config


class OpenAIProvider(LLMProvider):
    """OpenAI model provider"""

    def __init__(self, 
                api_key: str = config.OPENAI_API_KEY,
                model: str = config.OPENAI_MODEL
                ):
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
