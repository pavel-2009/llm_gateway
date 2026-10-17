"""Tests for error handling and retry policies."""

import time
from collections.abc import AsyncIterator

import pytest

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.errors import ProviderTimeoutError
from llm_gateway.policy.base_pipeline import BasePolicyPipeline
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.errors import AllModelsUnavailableError
from llm_gateway.policy.request_context import OperationType, RequestContext
from llm_gateway.policy.retry_policy import RetryPolicy


def test_error_policy_handles_gateway_and_unclassified_errors() -> None:
    policy = ErrorPolicy()
    retryable_error = ProviderTimeoutError("timed out", retryable=True)

    assert policy.should_retry(retryable_error)
    assert not policy.should_retry(ValueError("bad request"))


def test_error_policy_accepts_configured_exception_types() -> None:
    policy = ErrorPolicy(retryable_errors=(ValueError,))

    assert policy.should_retry(ValueError("temporary failure"))


def test_retry_policy_obeys_limit_and_backoff() -> None:
    policy = RetryPolicy(
        max_retries=2,
        base_delay=0.5,
        max_delay=1.0,
        jitter=0,
    )
    error = ProviderTimeoutError("timed out", retryable=True)

    error_policy = ErrorPolicy()
    assert policy.should_retry(error, 0, error_policy=error_policy)
    assert not policy.should_retry(error, 2, error_policy=error_policy)
    assert policy.get_backoff_time(1) == 0.5
    assert policy.get_backoff_time(3) == 1.0


def _context(operation: OperationType) -> RequestContext:
    return RequestContext(
        request_id="test-request",
        operation=operation,
        prompt="test prompt",
        deadline=time.monotonic() + 10,
        primary_model="openai",
    )


@pytest.mark.asyncio
async def test_pipeline_retries_retryable_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.attempts = 0
            self.last_prompt: str | None = None

        async def generate(self, prompt: str) -> GenerateResponse:
            self.attempts += 1
            self.last_prompt = prompt
            if self.attempts == 1:
                raise ProviderTimeoutError("timed out", retryable=True)
            return GenerateResponse(text="ok", provider="openai", model="test")

    provider = FakeProvider()
    monkeypatch.setattr(
        "llm_gateway.policy.base_pipeline.OpenAIProvider",
        lambda: provider,
    )
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(max_retries=1, base_delay=0, max_delay=0, jitter=0),
    )

    result = await pipeline.execute(_context(OperationType.GENERATE))

    assert isinstance(result, GenerateResponse)
    assert result.text == "ok"
    assert provider.attempts == 2
    assert provider.last_prompt == "test prompt"


@pytest.mark.asyncio
async def test_pipeline_raises_when_all_configured_models_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_models: list[str] = []

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

        async def generate(self, prompt: str) -> GenerateResponse:
            assert prompt == "test prompt"
            attempted_models.append(self.model)
            raise ProviderTimeoutError(
                f"{self.model} timed out",
                provider=self.model,
                retryable=True,
            )

    monkeypatch.setattr(
        "llm_gateway.policy.base_pipeline.OpenAIProvider",
        lambda: FakeProvider("openai"),
    )
    monkeypatch.setattr(
        "llm_gateway.policy.base_pipeline.OllamaProvider",
        lambda: FakeProvider("ollama"),
    )
    monkeypatch.setattr(
        "llm_gateway.policy.base_pipeline.ClaudeProvider",
        lambda: FakeProvider("claude"),
    )
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(
            max_retries=3,
            base_delay=0,
            max_delay=0,
            jitter=0,
            fallback_models=("ollama", "claude"),
        ),
    )

    with pytest.raises(AllModelsUnavailableError) as exc_info:
        await pipeline.execute(_context(OperationType.GENERATE))

    assert attempted_models == ["openai", "ollama", "claude"]
    assert set(exc_info.value.failures) == {"openai", "ollama", "claude"}


@pytest.mark.asyncio
async def test_pipeline_retries_stream_error_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.attempts = 0
            self.last_prompt: str | None = None

        def stream(self, prompt: str) -> AsyncIterator[str]:
            self.attempts += 1
            self.last_prompt = prompt

            async def chunks() -> AsyncIterator[str]:
                if self.attempts == 1:
                    raise ProviderTimeoutError("timed out", retryable=True)
                yield "chunk"

            return chunks()

    provider = FakeProvider()
    monkeypatch.setattr(
        "llm_gateway.policy.base_pipeline.OpenAIProvider",
        lambda: provider,
    )
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(max_retries=1, base_delay=0, max_delay=0, jitter=0),
    )
    context = _context(OperationType.STREAM)

    result = await pipeline.execute(context)
    chunks = [chunk async for chunk in result]

    assert chunks == ["chunk"]
    assert context.first_chunk_emitted
    assert provider.attempts == 2
    assert provider.last_prompt == "test prompt"


@pytest.mark.asyncio
async def test_pipeline_does_not_retry_stream_after_emitting_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.attempts = 0
            self.last_prompt: str | None = None

        def stream(self, prompt: str) -> AsyncIterator[str]:
            self.attempts += 1
            self.last_prompt = prompt

            async def chunks() -> AsyncIterator[str]:
                yield "first chunk"
                raise ProviderTimeoutError("timed out", retryable=True)

            return chunks()

    provider = FakeProvider()
    monkeypatch.setattr(
        "llm_gateway.policy.base_pipeline.OpenAIProvider",
        lambda: provider,
    )
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(max_retries=1, base_delay=0, max_delay=0, jitter=0),
    )
    context = _context(OperationType.STREAM)

    result = await pipeline.execute(context)
    chunks: list[str] = []
    with pytest.raises(ProviderTimeoutError):
        async for chunk in result:
            chunks.append(chunk)

    assert chunks == ["first chunk"]
    assert context.first_chunk_emitted
    assert provider.attempts == 1
    assert provider.last_prompt == "test prompt"