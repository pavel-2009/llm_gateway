"""Claude model provider."""

from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import config


class ClaudeProvider(LLMProvider):
    """Claude model provider"""

    def __init__(self,
                api_key: str = config.CLAUDE_API_KEY,
                model: str = config.CLAUDE_MODEL
                ):
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
