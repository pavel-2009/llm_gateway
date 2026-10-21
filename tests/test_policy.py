"""Tests for error handling and retry policies."""

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.errors import ProviderTimeoutError
from llm_gateway.core.config import config
from llm_gateway.policy.base_pipeline import BasePolicyPipeline
from llm_gateway.policy.error_policy import ErrorPolicy
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
    )


@pytest.mark.asyncio
async def test_pipeline_does_not_enforce_overall_operation_deadline() -> None:
    class FakeProvider:
        async def generate(self, prompt: str) -> GenerateResponse:
            await asyncio.sleep(0.12)
            return GenerateResponse(text="ok", provider="openai", model="test")

    pipeline = BasePolicyPipeline(FakeProvider())
    context = _context(OperationType.GENERATE)
    context.deadline = time.monotonic() + 0.1

    result = await pipeline.execute(context)

    assert isinstance(result, GenerateResponse)
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_pipeline_logs_retry_decision(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_RETRIES", 1)
    monkeypatch.setattr(config, "RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(config, "RETRY_MAX_DELAY", 0)
    monkeypatch.setattr(config, "RETRY_JITTER", 0)
    caplog.set_level("INFO")

    class FakeProvider:
        def __init__(self) -> None:
            self.attempts = 0

        async def generate(self, prompt: str) -> GenerateResponse:
            self.attempts += 1
            if self.attempts == 1:
                raise ProviderTimeoutError("timed out", retryable=True)
            return GenerateResponse(text="ok", provider="openai", model="test")

    result = await BasePolicyPipeline(FakeProvider()).execute(
        _context(OperationType.GENERATE)
    )

    assert isinstance(result, GenerateResponse)
    assert result.text == "ok"
    assert "Pipeline retry scheduled" in caplog.text


@pytest.mark.asyncio
async def test_pipeline_retries_retryable_generation_error() -> None:
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
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(max_retries=1, base_delay=0, max_delay=0, jitter=0),
        provider,
    )

    result = await pipeline.execute(_context(OperationType.GENERATE))

    assert isinstance(result, GenerateResponse)
    assert result.text == "ok"
    assert provider.attempts == 2
    assert provider.last_prompt == "test prompt"


@pytest.mark.asyncio
async def test_pipeline_stops_after_retry_limit() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.attempts = 0

        async def generate(self, prompt: str) -> GenerateResponse:
            self.attempts += 1
            raise ProviderTimeoutError("timed out", retryable=True)

    provider = FakeProvider()
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(max_retries=2, base_delay=0, max_delay=0, jitter=0),
        provider,
    )
    context = _context(OperationType.GENERATE)

    with pytest.raises(ProviderTimeoutError):
        await pipeline.execute(context)

    assert provider.attempts == 3
    assert context.retries_made == 2


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_pipeline_retries_stream_error_before_first_chunk() -> None:
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
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(max_retries=1, base_delay=0, max_delay=0, jitter=0),
        provider,
    )
    context = _context(OperationType.STREAM)

    result = await pipeline.execute(context)
    chunks = [chunk async for chunk in result]

    assert chunks == ["chunk"]
    assert context.first_chunk_emitted
    assert provider.attempts == 2
    assert provider.last_prompt == "test prompt"


@pytest.mark.asyncio
async def test_pipeline_does_not_enforce_overall_stream_deadline() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.attempts = 0

        def stream(self, prompt: str) -> AsyncIterator[str]:
            self.attempts += 1

            async def chunks() -> AsyncIterator[str]:
                await asyncio.sleep(0.12)
                yield "chunk"

            return chunks()

    provider = FakeProvider()
    pipeline = BasePolicyPipeline(provider)
    context = _context(OperationType.STREAM)
    context.deadline = time.monotonic() + 0.1

    result = await pipeline.execute(context)

    chunks = [chunk async for chunk in result]

    assert chunks == ["chunk"]
    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_pipeline_does_not_retry_stream_after_emitting_chunk() -> None:
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
    pipeline = BasePolicyPipeline(
        ErrorPolicy(),
        RetryPolicy(max_retries=1, base_delay=0, max_delay=0, jitter=0),
        provider,
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