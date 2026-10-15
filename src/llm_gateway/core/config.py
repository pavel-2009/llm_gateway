"""App configuration."""

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """App configuration."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    OLLAMA_API_KEY: str
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    CLAUDE_API_KEY: str
    CLAUDE_MODEL: str = "claude-3-5-sonnet-latest"
