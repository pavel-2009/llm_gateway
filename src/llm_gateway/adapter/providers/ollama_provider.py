"""Ollama model provider."""

import json
from collections.abc import AsyncIterator

import httpx

from llm_gateway.adapter.contracts import (
    EXTRACT_SCHEMA,
    EXTRACT_SYSTEM_PROMPT,
    ExtractedReview,
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.errors import (
    AIExtractionError,
    AIGenerationError,
    AIStreamingError,
)
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import config


ollama_timeout = httpx.Timeout(
    timeout=180.0,
    connect=5.0,
    write=10.0,
    pool=5.0,
)


class OllamaProvider(LLMProvider):
    """Local Ollama model provider."""

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(self, prompt: str) -> GenerateResponse:
        """Generate text from a prompt."""
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()

            return GenerateResponse(
                text=payload["message"]["content"],
                provider=Providers.OLLAMA,
                model=self.model,
            )
        except Exception as exc:
            raise AIGenerationError(
                f"Ollama generation failed: {exc}"
            ) from exc

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        payload = json.loads(line)
                        if payload.get("error"):
                            raise AIStreamingError(
                                f"Ollama streaming failed: {payload['error']}"
                            )

                        text = payload.get("message", {}).get("content", "")
                        if text:
                            yield text
        except AIStreamingError:
            raise
        except Exception as exc:
            raise AIStreamingError(
                f"Ollama streaming failed: {exc}"
            ) from exc

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt."""
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": EXTRACT_SYSTEM_PROMPT,
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "format": EXTRACT_SCHEMA,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()

            return ExtractedReview.model_validate_json(
                payload["message"]["content"]
            )
        except Exception as exc:
            raise AIExtractionError(
                f"Ollama extraction failed: {exc}"
            ) from exc
