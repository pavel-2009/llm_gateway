"""Ollama model provider."""

from llm_gateway.adapter.provider import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama model provider"""

    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        pass

    def stream(self, prompt: str):
        """Stream text from a prompt"""
        pass

    def extract(self, prompt: str) -> str:
        """Extract text from a prompt"""
        pass
