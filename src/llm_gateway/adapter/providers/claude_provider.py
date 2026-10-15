"""Claude model provider."""

from llm_gateway.adapter.provider import LLMProvider


class ClaudeProvider(LLMProvider):
    """Claude model provider"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        pass

    def stream(self, prompt: str):
        """Stream text from a prompt"""
        pass

    def extract(self, prompt: str) -> str:
        """Extract text from a prompt"""
        pass
