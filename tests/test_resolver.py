"""Tests for resolver input validation and pipeline context creation."""

from collections.abc import AsyncIterable
import asyncio
import time

import pytest
from pydantic import ValidationError

from llm_gateway.adapter.contracts import GenerateResponse, Providers
from llm_gateway.adapter.errors import (
    AllModelsUnavailableError,
    ContentPolicyError,
    ProviderUnavailableError,
)
from llm_gateway.core.config import config
from llm_gateway.policy.request_context import OperationType
from llm_gateway.resolver import model_registry
from llm_gateway.resolver.resolver_context import ModelTier, ResolveContext


def _configure_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CLAUDE_API_KEY", "claude-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-key")


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
    monkeypatch.setattr(model_registry, "OpenAIProvider", FakeProvider)
    _configure_provider_keys(monkeypatch)
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
    assert model_registry.ModelRegistry().get_models(tier)[0] == (
        expected_provider,
        expected_model,
    )


@pytest.mark.parametrize(
    "provider_name",
    [
        Providers.CLAUDE,
        Providers.OPENAI,
        Providers.OLLAMA,
    ],
)
def test_resolver_creates_provider_adapter(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
) -> None:
    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    provider_attribute = {
        Providers.CLAUDE: "ClaudeProvider",
        Providers.OPENAI: "OpenAIProvider",
        Providers.OLLAMA: "OllamaProvider",
    }[provider_name]
    monkeypatch.setattr(model_registry, provider_attribute, FakeProvider)

    adapter = model_registry.ModelRegistry().create_provider(
        provider_name,
        "selected-model",
    )

    assert isinstance(adapter, FakeProvider)
    assert adapter.model == "selected-model"


@pytest.mark.asyncio
async def test_resolver_falls_back_after_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    _configure_provider_keys(monkeypatch)
    monkeypatch.setattr(config, "CLAUDE_MODEL", "smart-model")
    monkeypatch.setattr(config, "OPENAI_MODEL", "medium-model")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "light-model")
    attempted_models: list[str] = []

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakePipeline:
        def __init__(self, provider: FakeProvider, **kwargs: object) -> None:
            self.provider = provider

        async def execute(self, context: object) -> str:
            attempted_models.append(self.provider.model)
            if self.provider.model == "smart-model":
                raise ProviderUnavailableError(
                    "temporarily unavailable",
                    provider="claude",
                    retryable=True,
                )
            return "fallback response"

    monkeypatch.setattr(model_registry, "ClaudeProvider", FakeProvider)
    monkeypatch.setattr(model_registry, "OpenAIProvider", FakeProvider)
    monkeypatch.setattr(model_registry, "OllamaProvider", FakeProvider)
    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)

    result = await model_resolver.ModelResolver().resolve(
        ResolveContext(
            model="smart",
            operation=OperationType.GENERATE,
            prompt="test prompt",
        )
    )

    assert result == "fallback response"
    assert attempted_models == ["smart-model", "medium-model"]


@pytest.mark.asyncio
async def test_resolver_falls_back_when_retry_delay_exceeds_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    _configure_provider_keys(monkeypatch)
    monkeypatch.setattr(config, "CLAUDE_MODEL", "smart-model")
    monkeypatch.setattr(config, "OPENAI_MODEL", "medium-model")
    monkeypatch.setattr(config, "MAX_RETRIES", 1)
    monkeypatch.setattr(config, "RETRY_BASE_DELAY", 1.0)
    monkeypatch.setattr(config, "RETRY_MAX_DELAY", 1.0)
    monkeypatch.setattr(config, "RETRY_JITTER", 0)
    attempted_models: list[str] = []

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

        async def generate(self, prompt: str) -> GenerateResponse:
            attempted_models.append(self.model)
            if self.model == "smart-model":
                raise ProviderUnavailableError(
                    "temporarily unavailable",
                    provider="claude",
                    retryable=True,
                )
            return GenerateResponse(
                text="fallback response",
                provider="openai",
                model=self.model,
            )

    monkeypatch.setattr(model_registry, "ClaudeProvider", FakeProvider)
    monkeypatch.setattr(model_registry, "OpenAIProvider", FakeProvider)

    result = await model_resolver.ModelResolver().resolve(
        ResolveContext(
            model="smart",
            operation=OperationType.GENERATE,
            prompt="test prompt",
            params={"deadline": 0.1},
        )
    )

    assert isinstance(result, GenerateResponse)
    assert result.text == "fallback response"
    assert attempted_models == ["smart-model", "medium-model"]


@pytest.mark.asyncio
async def test_resolver_does_not_fallback_for_content_policy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    _configure_provider_keys(monkeypatch)
    attempted_models: list[str] = []

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakePipeline:
        def __init__(self, provider: FakeProvider, **kwargs: object) -> None:
            self.provider = provider

        async def execute(self, context: object) -> str:
            attempted_models.append(self.provider.model)
            raise ContentPolicyError("request refused", provider="claude")

    monkeypatch.setattr(model_registry, "ClaudeProvider", FakeProvider)
    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)

    with pytest.raises(ContentPolicyError):
        await model_resolver.ModelResolver().resolve(
            ResolveContext(
                model="smart",
                operation=OperationType.GENERATE,
                prompt="test prompt",
            )
        )

    assert len(attempted_models) == 1


@pytest.mark.asyncio
async def test_resolver_aggregates_failures_after_exhausting_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    _configure_provider_keys(monkeypatch)
    attempted_models: list[str] = []

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakePipeline:
        def __init__(self, provider: FakeProvider, **kwargs: object) -> None:
            self.provider = provider

        async def execute(self, context: object) -> str:
            attempted_models.append(self.provider.model)
            raise ProviderUnavailableError(
                "temporarily unavailable",
                provider=self.provider.model,
                retryable=True,
            )

    monkeypatch.setattr(model_registry, "ClaudeProvider", FakeProvider)
    monkeypatch.setattr(model_registry, "OpenAIProvider", FakeProvider)
    monkeypatch.setattr(model_registry, "OllamaProvider", FakeProvider)
    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)

    with pytest.raises(AllModelsUnavailableError) as exc_info:
        await model_resolver.ModelResolver().resolve(
            ResolveContext(
                model="smart",
                operation=OperationType.GENERATE,
                prompt="test prompt",
            )
        )

    assert attempted_models == [
        config.CLAUDE_MODEL,
        config.OPENAI_MODEL,
        config.OLLAMA_MODEL,
    ]
    assert len(exc_info.value.failures) == 3


@pytest.mark.asyncio
async def test_resolver_falls_back_when_cloud_credentials_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    monkeypatch.setattr(config, "CLAUDE_API_KEY", None)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakePipeline:
        def __init__(self, provider: FakeProvider, **kwargs: object) -> None:
            self.provider = provider

        async def execute(self, context: object) -> str:
            return self.provider.model

    monkeypatch.setattr(model_registry, "OllamaProvider", FakeProvider)
    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)

    result = await model_resolver.ModelResolver().resolve(
        ResolveContext(
            model="smart",
            operation=OperationType.GENERATE,
            prompt="test prompt",
        )
    )

    assert result == config.OLLAMA_MODEL


@pytest.mark.asyncio
async def test_resolver_rejects_invalid_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 0)

    with pytest.raises(ValueError, match="positive finite number"):
        await model_resolver.ModelResolver().resolve(
            ResolveContext(
                model="light",
                operation=OperationType.GENERATE,
                prompt="test prompt",
            )
        )


@pytest.mark.asyncio
async def test_resolver_timeout_covers_pipeline_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def execute(self, context: object) -> str:
            await asyncio.Event().wait()
            return "unreachable"

    monkeypatch.setattr(model_registry, "OllamaProvider", FakeProvider)
    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)

    with pytest.raises(TimeoutError):
        await model_resolver.ModelResolver().resolve(
            ResolveContext(
                model="light",
                operation=OperationType.GENERATE,
                prompt="test prompt",
                params={"deadline": 0.01},
            )
        )


@pytest.mark.asyncio
async def test_resolver_timeout_covers_stream_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def execute(self, context: object) -> object:
            async def chunks() -> AsyncIterable[str]:
                await asyncio.Event().wait()
                yield "unreachable"

            return chunks()

    monkeypatch.setattr(model_registry, "OllamaProvider", FakeProvider)
    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)

    result = await model_resolver.ModelResolver().resolve(
        ResolveContext(
            model="light",
            operation=OperationType.STREAM,
            prompt="test prompt",
            params={"deadline": 0.01},
        )
    )

    with pytest.raises(TimeoutError):
        async for _ in result:
            pass


@pytest.mark.asyncio
async def test_stream_does_not_fallback_after_emitting_a_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_gateway.resolver import model_resolver

    _configure_provider_keys(monkeypatch)
    attempted_models: list[str] = []

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakePipeline:
        def __init__(self, provider: FakeProvider, **kwargs: object) -> None:
            self.provider = provider

        async def execute(self, context: object) -> object:
            attempted_models.append(self.provider.model)

            async def chunks() -> AsyncIterable[str]:
                yield "first chunk"
                raise ProviderUnavailableError(
                    "connection lost",
                    provider="claude",
                    retryable=True,
                )

            return chunks()

    monkeypatch.setattr(model_registry, "ClaudeProvider", FakeProvider)
    monkeypatch.setattr(model_registry, "OpenAIProvider", FakeProvider)
    monkeypatch.setattr(model_registry, "OllamaProvider", FakeProvider)
    monkeypatch.setattr(model_resolver, "BasePolicyPipeline", FakePipeline)

    result = await model_resolver.ModelResolver().resolve(
        ResolveContext(
            model="smart",
            operation=OperationType.STREAM,
            prompt="test prompt",
        )
    )
    chunks: list[str] = []
    with pytest.raises(ProviderUnavailableError):
        async for chunk in result:
            chunks.append(chunk)

    assert chunks == ["first chunk"]
    assert len(attempted_models) == 1