"""Integration tests for gateway startup and model failures."""

import asyncio
from collections.abc import AsyncIterator
from typing import cast

from fastapi.testclient import TestClient
import pytest

import llm_gateway.app as gateway_app
from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.errors import ProviderTimeoutError, ProviderUnavailableError
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.llm_gateway import LLMGateway
from llm_gateway.models import Model
from llm_gateway.resolver.model_registry import ModelRegistry
from llm_gateway.resolver.model_resolver import ModelResolver


class FakeProvider:
    def __init__(
        self,
        name: str,
        results: list[object],
        attempts: list[tuple[str, str]],
    ) -> None:
        self.name = name
        self.results = results
        self.attempts = attempts

    def result(self, operation: str) -> object:
        self.attempts.append((self.name, operation))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def generate(self, prompt: str) -> GenerateResponse:
        return cast(GenerateResponse, self.result("generate"))

    def stream(self, prompt: str) -> AsyncIterator[str]:
        async def chunks() -> AsyncIterator[str]:
            for chunk in cast(tuple[str, ...], self.result("stream")):
                yield chunk

        return chunks()

    async def extract(self, prompt: str) -> ExtractedReview:
        return cast(ExtractedReview, self.result("extract"))


def setup_gateway(
    monkeypatch: pytest.MonkeyPatch,
    models: tuple[tuple[str, str], ...],
    results: dict[str, list[object]],
) -> tuple[TestClient, list[tuple[str, str]]]:
    monkeypatch.setattr(config, "CLAUDE_API_KEY", "test-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    attempts: list[tuple[str, str]] = []
    providers = {
        name: FakeProvider(name, provider_results, attempts)
        for name, provider_results in results.items()
    }

    registry = ModelRegistry()
    monkeypatch.setattr(registry, "get_models", lambda tier: models)
    monkeypatch.setattr(
        registry,
        "create_provider",
        lambda name, model: providers[name],
    )
    gateway = LLMGateway()
    gateway.resolver = ModelResolver(registry)
    monkeypatch.setattr(gateway_app, "gateway", gateway)
    client = TestClient(gateway_app.app, raise_server_exceptions=False)
    return client, attempts


def test_gateway_starts_and_completes_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, attempts = setup_gateway(
        monkeypatch,
        (("ollama", "local-model"),),
        {
            "ollama": [
                GenerateResponse(
                    text="hello from model",
                    provider="ollama",
                    model="local-model",
                )
            ]
        },
    )

    with client:
        assert client.get("/openapi.json").status_code == 200
        response = client.post(
            "/generate",
            json={"prompt": "hello", "model": Model.LIGHT.value},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "hello from model"
    assert attempts == [("ollama", "generate")]


def test_extract_runs_through_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, attempts = setup_gateway(
        monkeypatch,
        (("ollama", "local-model"),),
        {
            "ollama": [
                ExtractedReview(
                    sentiment="positive",
                    topics=["delivery"],
                    rating=5,
                )
            ]
        },
    )

    with client:
        response = client.post("/extract", json={"prompt": "review"})

    assert response.status_code == 200
    assert response.json()["rating"] == 5
    assert attempts == [("ollama", "extract")]


def test_unavailable_model_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    client, attempts = setup_gateway(
        monkeypatch,
        (("claude", "primary"), ("openai", "fallback")),
        {
            "claude": [
                ProviderUnavailableError(
                    "offline",
                    provider="claude",
                    retryable=True,
                )
            ],
            "openai": [
                GenerateResponse(
                    text="fallback answer",
                    provider="openai",
                    model="fallback",
                )
            ],
        },
    )

    with client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["text"] == "fallback answer"
    assert attempts == [("claude", "generate"), ("openai", "generate")]


def test_timeout_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 1)
    monkeypatch.setattr(config, "RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(config, "RETRY_MAX_DELAY", 0)
    monkeypatch.setattr(config, "RETRY_JITTER", 0)
    client, attempts = setup_gateway(
        monkeypatch,
        (("openai", "model"),),
        {
            "openai": [
                ProviderTimeoutError(
                    "timed out",
                    provider="openai",
                    retryable=True,
                ),
                GenerateResponse(
                    text="recovered",
                    provider="openai",
                    model="model",
                ),
            ]
        },
    )

    with client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["text"] == "recovered"
    assert attempts == [("openai", "generate")] * 2


def test_hung_model_times_out_and_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 0.03)
    cancelled: list[bool] = []
    attempts: list[tuple[str, str]] = []

    class HungProvider(FakeProvider):
        async def generate(self, prompt: str) -> GenerateResponse:
            self.attempts.append((self.name, "generate"))
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            raise AssertionError("Request should have timed out")

    client, attempts = setup_gateway(
        monkeypatch,
        (("ollama", "hung-model"),),
        {"ollama": []},
    )
    monkeypatch.setattr(
        gateway_app.gateway.resolver.model_registry,
        "create_provider",
        lambda name, model: HungProvider(name, [], attempts),
    )

    with client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 504
    assert cancelled == [True]
    assert attempts == [("ollama", "generate")]


def test_stream_failure_falls_back_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    client, attempts = setup_gateway(
        monkeypatch,
        (("claude", "primary"), ("ollama", "fallback")),
        {
            "claude": [
                ProviderTimeoutError(
                    "stream timed out",
                    provider="claude",
                    operation="streaming",
                    retryable=True,
                )
            ],
            "ollama": [("fallback ", "stream")],
        },
    )

    with client:
        response = client.post("/stream", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.text == (
        'event: chunk\ndata: {"type":"chunk","index":1,"content":"fallback "}\n\n'
        'event: chunk\ndata: {"type":"chunk","index":2,"content":"stream"}\n\n'
        'event: done\ndata: {"type":"done","chunks":2,"output_chars":15}\n\n'
    )
    assert attempts == [("claude", "stream"), ("ollama", "stream")]


def test_stream_failure_after_first_chunk_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    client, attempts = setup_gateway(
        monkeypatch,
        (("claude", "primary"), ("ollama", "fallback")),
        {
            "claude": [],
            "ollama": [("fallback",)],
        },
    )

    class FailingStreamProvider:
        def stream(self, prompt: str) -> AsyncIterator[str]:
            async def chunks() -> AsyncIterator[str]:
                attempts.append(("claude", "stream"))
                yield "first"
                raise ProviderTimeoutError(
                    "private provider failure",
                    provider="claude",
                    operation="streaming",
                    retryable=True,
                )

            return chunks()

    def create_provider(name: str, model: str) -> object:
        if name == "claude":
            return FailingStreamProvider()
        return FakeProvider(name, [("fallback",)], attempts)

    monkeypatch.setattr(
        gateway_app.gateway.resolver.model_registry,
        "create_provider",
        create_provider,
    )

    with client:
        response = client.post("/stream", json={"prompt": "hello"})

    assert response.status_code == 200
    assert (
        'event: chunk\ndata: {"type":"chunk","index":1,"content":"first"}'
        in response.text
    )
    assert '"code":"gateway_timeout"' in response.text
    assert "event: done\n" not in response.text
    assert "private" not in response.text
    assert attempts == [("claude", "stream")]
