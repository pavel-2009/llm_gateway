"""App configuration."""

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """App configuration."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    CLAUDE_API_KEY: str
    CLAUDE_MODEL: str = "claude-3-5-sonnet-latest"

    # general timeouts
    CONNECT_TIMEOUT: float = 5.0
    REQUEST_TIMEOUT: float = 60.0
    STREAM_TIMEOUT : float = 120.0

    # retry settings
    MAX_RETRIES: int = 2
    RETRY_BASE_DELAY: float = 0.5
    RETRY_MAX_DELAY: float = 4.0
    RETRY_JITTER: float = 0.25

    # rate limiting
    MAX_CONCURRENT_REQUESTS: int = 50

    # Circuit breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = 30.0
    CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS: int = 1

    # Fallback mechanism
    FALLBACK_ENABLED: bool = True

config = AppConfig()
