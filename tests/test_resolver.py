"""Tests for resolver input validation and pipeline context creation."""

import time

import pytest
from pydantic import ValidationError

from llm_gateway.policy.request_context import OperationType
from llm_gateway.resolver.resolver_context import ResolveContext


def test_resolve_context_accepts_valid_values() -> None:
    context = ResolveContext(
        model="test-model",
        operation=OperationType.GENERATE,
        prompt="test prompt",
        params={"deadline": 5.0},
    )

    assert context.model == "test-model"
    assert context.params.deadline == 5.0


@pytest.mark.parametrize(
    "values",
    [
        {"model": "", "operation": OperationType.GENERATE, "prompt": "hello"},
        {"model": "test", "operation": "unknown", "prompt": "hello"},
        {
            "model": "test",
            "operation": OperationType.GENERATE,
            "prompt": "hello",
            "params": {"deadline": 0},
        },
        {
            "model": "test",
            "operation": OperationType.GENERATE,
            "prompt": "hello",
            "params": {"unexpected": 1},
        },
    ],
)
def test_resolve_context_rejects_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ResolveContext(**values)


@pytest.mark.asyncio
async def test_resolver_converts_timeout_to_monotonic_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    captured_context = None

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def execute(self, context: object) -> str:
            nonlocal captured_context
            captured_context = context
            return "ok"

    class FakeProvider:
        pass

    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)
    monkeypatch.setattr(model_resolver, "OllamaProvider", FakeProvider)
    before = time.monotonic()

    result = await model_resolver.ModelResolver().resolve(
        ResolveContext(
            model="test-model",
            operation=OperationType.GENERATE,
            prompt="test prompt",
            params={"deadline": 5.0},
        )
    )

    assert result == "ok"
    assert captured_context is not None
    assert before + 5.0 <= captured_context.deadline <= time.monotonic() + 5.0