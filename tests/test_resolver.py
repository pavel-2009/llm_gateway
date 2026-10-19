"""Tests for resolver input validation and pipeline context creation."""

import time

import pytest
from pydantic import ValidationError

from llm_gateway.adapter.contracts import Providers
from llm_gateway.core.config import config
from llm_gateway.policy.request_context import OperationType
from llm_gateway.resolver.resolver_context import ModelTier, ResolveContext


def test_resolve_context_accepts_valid_values() -> None:
    context = ResolveContext(
        model="medium",
        operation=OperationType.GENERATE,
        prompt="test prompt",
        params={"deadline": 5.0},
    )

    assert context.model is ModelTier.MEDIUM
    assert context.params.deadline == 5.0


@pytest.mark.parametrize(
    "values",
    [
        {"model": "unknown", "operation": OperationType.GENERATE, "prompt": "hello"},
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
        def __init__(self, model: str) -> None:
            self.model = model

    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)
    monkeypatch.setattr(model_resolver, "OpenAIProvider", FakeProvider)
    before = time.monotonic()

    result = await model_resolver.ModelResolver().resolve(
        ResolveContext(
            model="medium",
            operation=OperationType.GENERATE,
            prompt="test prompt",
            params={"deadline": 5.0},
        )
    )

    assert result == "ok"
    assert captured_context is not None
    assert before + 5.0 <= captured_context.deadline <= time.monotonic() + 5.0


@pytest.mark.parametrize(
    ("tier", "expected_provider", "expected_model"),
    [
        (ModelTier.SMART, Providers.CLAUDE, config.CLAUDE_MODEL),
        (ModelTier.MEDIUM, Providers.OPENAI, config.OPENAI_MODEL),
        (ModelTier.LIGHT, Providers.OLLAMA, config.OLLAMA_MODEL),
    ],
)
def test_resolver_selects_configured_model(
    tier: ModelTier,
    expected_provider: str,
    expected_model: str,
) -> None:
    from llm_gateway.resolver.model_resolver import ModelResolver

    assert ModelResolver().select_model(tier) == (expected_provider, expected_model)


@pytest.mark.parametrize(
    ("tier", "provider_name"),
    [
        (ModelTier.SMART, Providers.CLAUDE),
        (ModelTier.MEDIUM, Providers.OPENAI),
        (ModelTier.LIGHT, Providers.OLLAMA),
    ],
)
def test_resolver_creates_provider_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tier: ModelTier,
    provider_name: str,
) -> None:
    from llm_gateway.resolver import model_resolver

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    provider_attribute = {
        Providers.CLAUDE: "ClaudeProvider",
        Providers.OPENAI: "OpenAIProvider",
        Providers.OLLAMA: "OllamaProvider",
    }[provider_name]
    monkeypatch.setattr(model_resolver, provider_attribute, FakeProvider)

    adapter = model_resolver.ModelResolver().create_provider(
        provider_name,
        "selected-model",
    )

    assert isinstance(adapter, FakeProvider)
    assert adapter.model == "selected-model"