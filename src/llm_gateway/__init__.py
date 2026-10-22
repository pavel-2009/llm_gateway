"""LLM Gateway package and command-line entry point."""

from llm_gateway.core.config import config


def main() -> None:
    """Start the FastAPI application with the configured HTTP server."""
    import uvicorn

    uvicorn.run(
        "llm_gateway.app:app",
        host=config.HOST,
        port=config.PORT,
    )
