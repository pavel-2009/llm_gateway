"""Ollama model provider."""

import json
from collections.abc import AsyncIterator

import httpx

from llm_gateway.adapter.contracts import (
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.providers.errors import (
    raise_provider_error,
)
from llm_gateway.adapter.prompt import build_prompt
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import (
    EXTRACT_SCHEMA,
    ExtractedReview,
    config,
)


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
                        **build_prompt(prompt, None, False, Providers.OLLAMA),
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
            raise_provider_error(Providers.OLLAMA, "generation", exc)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        **build_prompt(prompt, None, True, Providers.OLLAMA),
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        payload = json.loads(line)
                        if payload.get("error"):
                            raise RuntimeError(payload["error"])

                        text = payload.get("message", {}).get("content", "")
                        if text:
                            yield text
        except Exception as exc:
            raise_provider_error(Providers.OLLAMA, "streaming", exc)

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt."""
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        **build_prompt(
                            prompt, EXTRACT_SCHEMA, False, Providers.OLLAMA
                        ),
                    },
                )
                response.raise_for_status()
                payload = response.json()

            return ExtractedReview.model_validate_json(
                payload["message"]["content"]
            )
        except Exception as exc:
            raise_provider_error(Providers.OLLAMA, "extraction", exc)
