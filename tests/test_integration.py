"""End-to-end tests for the LLM Gateway API and model failure handling."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import cast

from fastapi.testclient import TestClient
import pytest

import llm_gateway.app as gateway_app
from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.errors import (
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.llm_gateway import LLMGateway
from llm_gateway.models import Model
from llm_gateway.resolver.model_registry import ModelRegistry
from llm_gateway.resolver.model_resolver import ModelResolver
from llm_gateway.resolver.resolver_context import ModelTier


ScriptStep = (
    Exception
    | GenerateResponse
    | ExtractedReview
    | tuple[str, ...]
    | Callable[[], object]
)


class ScriptedProvider:
    def __init__(
        self,
        name: str,
        scripts: dict[str, list[ScriptStep]],
        attempts: list[tuple[str, str]],
    ) -> None:
        self.name = name
        self.scripts = scripts
        self.attempts = attempts

    async def _run(self, operation: str) -> object:
        self.attempts.append((self.name, operation))
        steps = self.scripts.get(operation, [])
        if not steps:
            raise AssertionError(f"Unexpected {operation} attempt for {self.name}")

        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            result = step()
            if asyncio.iscoroutine(result):
                return await result
            return result
        return step

    async def generate(self, prompt: str) -> GenerateResponse:
        return cast(GenerateResponse, await self._run("generate"))

    def stream(self, prompt: str) -> AsyncIterator[str]:
        async def chunks() -> AsyncIterator[str]:
            result = cast(tuple[str, ...], await self._run("stream"))
            for chunk in result:
                yield chunk

        return chunks()

    async def extract(self, prompt: str) -> ExtractedReview:
        return cast(ExtractedReview, await self._run("extract"))


class ScriptedRegistry(ModelRegistry):
    def __init__(
        self,
        candidates: tuple[tuple[str, str], ...],
        providers: dict[str, ScriptedProvider],
    ) -> None:
        self.candidates = candidates
        self.providers = providers

    def get_models(
        self,
        tier: ModelTier,
    ) -> tuple[tuple[str, str], ...]:
        return self.candidates

    def create_provider(self, provider: str, model: str) -> ScriptedProvider:
        return self.providers[provider]


def _install_gateway(
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[tuple[str, str], ...],
    scripts: dict[str, dict[str, list[ScriptStep]]],
) -> tuple[TestClient, list[tuple[str, str]]]:
    monkeypatch.setattr(config, "CLAUDE_API_KEY", "test-claude-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-openai-key")

    attempts: list[tuple[str, str]] = []
    providers = {
        name: ScriptedProvider(name, provider_scripts, attempts)
        for name, provider_scripts in scripts.items()
    }
    gateway = LLMGateway()
    gateway.resolver = ModelResolver(ScriptedRegistry(candidates, providers))
    monkeypatch.setattr(gateway_app, "gateway", gateway)
    return TestClient(gateway_app.app, raise_server_exceptions=False), attempts


def _generation(text: str, provider: str, model: str) -> GenerateResponse:
    return GenerateResponse(text=text, provider=provider, model=model)


def test_gateway_starts_and_completes_full_request_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, attempts = _install_gateway(
        monkeypatch,
        (("ollama", "local-model"),),
        {
            "ollama": {
                "generate": [_generation("hello from model", "ollama", "local-model")],
            },
        },
    )

    with client:
        assert client.get("/openapi.json").status_code == 200
        response = client.post(
            "/generate",
            json={"prompt": "hello", "model": Model.LIGHT.value},
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "hello from model",
        "provider": "ollama",
        "model": "local-model",
    }
    assert attempts == [("ollama", "generate")]


def test_extract_endpoint_runs_through_gateway_and_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ExtractedReview(
        sentiment="positive",
        topics=["delivery"],
        rating=5,
    )
    client, attempts = _install_gateway(
        monkeypatch,
        (("ollama", "local-model"),),
        {
            "ollama": {
                "extract": [expected],
            },
        },
    )

    with client:
        response = client.post("/extract", json={"prompt": "review"})

    assert response.status_code == 200
    assert response.json() == {
        "sentiment": "positive",
        "topics": ["delivery"],
        "rating": 5,
    }
    assert attempts == [("ollama", "extract")]


def test_unavailable_model_falls_back_to_next_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    client, attempts = _install_gateway(
        monkeypatch,
        (("claude", "primary"), ("openai", "fallback")),
        {
            "claude": {
                "generate": [
                    ProviderUnavailableError(
                        "provider is offline",
                        provider="claude",
                        retryable=True,
                    ),
                ],
            },
            "openai": {
                "generate": [_generation("fallback answer", "openai", "fallback")],
            },
        },
    )

    with client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["text"] == "fallback answer"
    assert attempts == [("claude", "generate"), ("openai", "generate")]


def test_transient_model_timeout_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 1)
    monkeypatch.setattr(config, "RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(config, "RETRY_MAX_DELAY", 0)
    monkeypatch.setattr(config, "RETRY_JITTER", 0)
    client, attempts = _install_gateway(
        monkeypatch,
        (("openai", "primary"),),
        {
            "openai": {
                "generate": [
                    ProviderTimeoutError(
                        "temporary model timeout",
                        provider="openai",
                        retryable=True,
                    ),
                    _generation("recovered answer", "openai", "primary"),
                ],
            },
        },
    )

    with client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["text"] == "recovered answer"
    assert attempts == [("openai", "generate"), ("openai", "generate")]


def test_hung_model_is_cancelled_when_request_deadline_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 0.03)
    cancellations: list[str] = []

    async def hang_until_cancelled() -> GenerateResponse:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancellations.append("cancelled")
            raise
        return _generation("too late", "ollama", "hung-model")

    client, attempts = _install_gateway(
        monkeypatch,
        (("ollama", "hung-model"),),
        {
            "ollama": {
                "generate": [hang_until_cancelled],
            },
        },
    )

    with client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 500
    assert attempts == [("ollama", "generate")]
    assert cancellations == ["cancelled"]


def test_stream_error_before_first_chunk_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    client, attempts = _install_gateway(
        monkeypatch,
        (("claude", "primary"), ("ollama", "fallback")),
        {
            "claude": {
                "stream": [
                    ProviderTimeoutError(
                        "stream setup timed out",
                        provider="claude",
                        operation="streaming",
                        retryable=True,
                    ),
                ],
            },
            "ollama": {
                "stream": [("fallback ", "stream")],
            },
        },
    )

    with client:
        response = client.post("/stream", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.text == "fallback stream"
    assert attempts == [("claude", "stream"), ("ollama", "stream")]


def test_gateway_reports_error_when_all_models_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    client, attempts = _install_gateway(
        monkeypatch,
        (("claude", "primary"), ("openai", "fallback")),
        {
            "claude": {
                "generate": [
                    ProviderUnavailableError(
                        "first model is offline",
                        provider="claude",
                        retryable=True,
                    ),
                ],
            },
            "openai": {
                "generate": [
                    ProviderUnavailableError(
                        "second model is offline",
                        provider="openai",
                        retryable=True,
                    ),
                ],
            },
        },
    )

    with client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 500
    assert attempts == [("claude", "generate"), ("openai", "generate")]
