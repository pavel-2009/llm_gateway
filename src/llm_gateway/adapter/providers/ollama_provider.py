"""Ollama model provider."""

import json
from collections.abc import AsyncIterator
import logging

import httpx

from llm_gateway.adapter.contracts import (
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.errors import (
    raise_provider_error,
)
from llm_gateway.adapter.prompt import build_prompt
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import (
    EXTRACT_SCHEMA,
    ExtractedReview,
    config,
)
from llm_gateway.core.logging_context import request_id_context
from llm_gateway.telemetry import http_client_span


logger = logging.getLogger(__name__)


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
        request_id = request_id_context.get()
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=generation model=%s",
            request_id,
            Providers.OLLAMA,
            self.model,
        )
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                with http_client_span(
                    Providers.OLLAMA,
                    "generate",
                    self.model,
                    f"{self.base_url}/api/chat",
                ) as span:
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model,
                            **build_prompt(prompt, None, False, Providers.OLLAMA),
                        },
                    )
                    response.raise_for_status()
                    span.set_attribute(
                        "http.response.status_code",
                        response.status_code,
                    )
                payload = response.json()

            result = GenerateResponse(
                text=payload["message"]["content"],
                provider=Providers.OLLAMA,
                model=self.model,
            )
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=generation output_chars=%s",
                request_id,
                Providers.OLLAMA,
                len(result.text),
            )
            return result
        except Exception as exc:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=generation error_type=%s",
                request_id,
                Providers.OLLAMA,
                type(exc).__name__,
            )
            raise_provider_error(Providers.OLLAMA, "generation", exc)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text from Ollama."""
        request_id = request_id_context.get()
        chunks = 0
        output_chars = 0
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=streaming model=%s",
            request_id,
            Providers.OLLAMA,
            self.model,
        )
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                with http_client_span(
                    Providers.OLLAMA,
                    "stream",
                    self.model,
                    f"{self.base_url}/api/chat",
                ):
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
                                chunks += 1
                                output_chars += len(text)
                                yield text
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=streaming chunks=%s output_chars=%s",
                request_id,
                Providers.OLLAMA,
                chunks,
                output_chars,
            )
        except Exception as exc:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=streaming chunks=%s error_type=%s",
                request_id,
                Providers.OLLAMA,
                chunks,
                type(exc).__name__,
            )
            raise_provider_error(Providers.OLLAMA, "streaming", exc)

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt."""
        request_id = request_id_context.get()
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=extraction model=%s",
            request_id,
            Providers.OLLAMA,
            self.model,
        )
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                with http_client_span(
                    Providers.OLLAMA,
                    "extract",
                    self.model,
                    f"{self.base_url}/api/chat",
                ) as span:
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
                    span.set_attribute(
                        "http.response.status_code",
                        response.status_code,
                    )
                payload = response.json()

            result = ExtractedReview.model_validate_json(
                payload["message"]["content"]
            )
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=extraction",
                request_id,
                Providers.OLLAMA,
            )
            return result
        except Exception as exc:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=extraction error_type=%s",
                request_id,
                Providers.OLLAMA,
                type(exc).__name__,
            )
            raise_provider_error(Providers.OLLAMA, "extraction", exc)
