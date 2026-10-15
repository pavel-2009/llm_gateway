"""Base models and contracts for providers and adapters."""

from enum import Enum

from pydantic import BaseModel


class Providers:
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"


class GenerateResponse(BaseModel):
    text: str
    provider: str
    model: str
