"""Base models and contracts for providers and adapters."""

from llm_gateway.models import GenerateResponse

__all__ = ["GenerateResponse", "Providers"]


class Providers:
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"
