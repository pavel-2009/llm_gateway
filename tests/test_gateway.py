"""Tests for the public gateway return contracts."""

from collections.abc import AsyncIterable

import pytest

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.llm_gateway import LLMGateway
from llm_gateway.policy.request_context import OperationType
from llm_gateway.resolver.model_resolver import RESOLVER_RESULT
from llm_gateway.resolver.resolver_context import ResolveContext


class FakeResolver:
    def __init__(self, result: RESOLVER_RESULT) -> None:
        self.result = result
        self.context: ResolveContext | None = None

    async def resolve(self, context: ResolveContext) -> RESOLVER_RESULT:
        self.context = context
        return self.result


@pytest.mark.asyncio
async def test_generate_returns_generation_response() -> None:
    expected = GenerateResponse(text="answer", provider="openai", model="test")
    gateway = LLMGateway()
    resolver = FakeResolver(expected)
    gateway.resolver = resolver

    result = await gateway.generate("prompt")

    assert result is expected
    assert resolver.context is not None
    assert resolver.context.operation == OperationType.GENERATE
    assert resolver.context.params.deadline == config.REQUEST_TIMEOUT


@pytest.mark.asyncio
async def test_extract_returns_extracted_review() -> None:
    expected = ExtractedReview(
        sentiment="positive",
        topics=["service"],
        rating=5,
    )
    gateway = LLMGateway()
    resolver = FakeResolver(expected)
    gateway.resolver = resolver

    result = await gateway.extract("review")

    assert result is expected
    assert resolver.context is not None
    assert resolver.context.operation == OperationType.EXTRACT
    assert resolver.context.params.deadline == config.REQUEST_TIMEOUT


@pytest.mark.asyncio
async def test_stream_returns_directly_iterable_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def chunks() -> AsyncIterable[str]:
        yield "first"
        yield "second"

    monkeypatch.setattr(config, "STREAM_TIMEOUT", 90.0)
    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 30.0)
    gateway = LLMGateway()
    resolver = FakeResolver(chunks())
    gateway.resolver = resolver

    result = [chunk async for chunk in gateway.stream("prompt")]

    assert result == ["first", "second"]
    assert resolver.context is not None
    assert resolver.context.operation == OperationType.STREAM
    assert resolver.context.params.deadline == 90.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "wrong_result"),
    [
        (
            "generate",
            ExtractedReview(sentiment="neutral", topics=[], rating=0),
        ),
        (
            "extract",
            GenerateResponse(text="answer", provider="openai", model="test"),
        ),
        (
            "stream",
            GenerateResponse(text="answer", provider="openai", model="test"),
        ),
    ],
)
async def test_gateway_rejects_result_for_wrong_operation(
    method: str,
    wrong_result: RESOLVER_RESULT,
) -> None:
    gateway = LLMGateway()
    gateway.resolver = FakeResolver(wrong_result)
    operation = getattr(gateway, method)

    with pytest.raises(TypeError, match="invalid"):
        result = operation("prompt")
        if method == "stream":
            _ = [chunk async for chunk in result]
        else:
            await result
