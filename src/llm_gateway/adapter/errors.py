"""Custom errors for LLM providers"""


class LLMProviderError(Exception):
    """Base class for LLM provider errors"""


class AIGenerationError(LLMProviderError):
    """Raised when an error occurs during AI generation"""

class AIStreamingError(LLMProviderError):
    """Raised when an error occurs during AI streaming"""

class AIExtractionError(LLMProviderError):
    """Raised when an error occurs during AI extraction"""
