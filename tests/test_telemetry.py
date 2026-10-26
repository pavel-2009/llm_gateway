"""Tests for OpenTelemetry spans."""

import asyncio
from collections.abc import AsyncIterator
import time

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import httpx
import pytest

import llm_gateway.app as app_module
import llm_gateway.llm_gateway as gateway_module
import llm_gateway.policy.base_pipeline as pipeline_module
import llm_gateway.resolver.model_resolver as resolver_module
import llm_gateway.telemetry as telemetry_module
from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.errors import (
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from llm_gateway.llm_gateway import LLMGateway
from llm_gateway.policy.base_pipeline import BasePolicyPipeline
from llm_gateway.policy.request_context import OperationType, RequestContext
from llm_gateway.resolver.resolver_context import ResolveContext


def test_configure_tracing_uses_configured_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_module.config, "OTEL_ENABLED", True)
    monkeypatch.setattr(
        telemetry_module.config,
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://tempo.example/v1/traces",
    )
    monkeypatch.setattr(
        telemetry_module.config,
        "OTEL_SERVICE_NAME",
        "test-gateway",
    )
    monkeypatch.setattr(
        telemetry_module.config,
        "OTEL_TRACE_SAMPLE_RATIO",
        0.25,
    )
    exporter = object()
    processor = object()
    configured: dict[str, object] = {}

    class FakeTracerProvider:
        def __init__(self, **kwargs: object) -> None:
            configured["provider_kwargs"] = kwargs

        def add_span_processor(self, span_processor: object) -> None:
            configured["processor"] = span_processor

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(
        telemetry_module,
        "TracerProvider",
        FakeTracerProvider,
    )
    monkeypatch.setattr(
        telemetry_module,
        "OTLPSpanExporter",
        lambda *, endpoint: configured.update(endpoint=endpoint) or exporter,
    )
    monkeypatch.setattr(
        telemetry_module,
        "BatchSpanProcessor",
        lambda value: configured.update(exporter=value) or processor,
    )
    monkeypatch.setattr(
        telemetry_module.trace,
        "set_tracer_provider",
        lambda value: configured.update(tracer_provider=value),
    )
    registered: list[object] = []
    monkeypatch.setattr(
        telemetry_module.atexit,
        "register",
        lambda callback: registered.append(callback),
    )

    telemetry_module._configure_tracing()

    provider_kwargs = configured["provider_kwargs"]
    assert isinstance(provider_kwargs, dict)
    resource = provider_kwargs["resource"]
    assert resource.attributes["service.name"] == "test-gateway"
    assert provider_kwargs["sampler"].__class__.__name__ == "ParentBased"
    assert configured["endpoint"] == "https://tempo.example/v1/traces"
    assert configured["exporter"] is exporter
    assert configured["processor"] is processor
    assert configured["tracer_provider"] is not None
    assert len(registered) == 1


def test_configure_tracing_skips_setup_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_module.config, "OTEL_ENABLED", False)

    def unexpected_setup(*args: object, **kwargs: object) -> None:
        raise AssertionError("tracing setup should be skipped")

    monkeypatch.setattr(telemetry_module, "TracerProvider", unexpected_setup)

    telemetry_module._configure_tracing()


def test_http_and_llm_spans_are_correlated_without_prompt_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(app_module, "tracer", tracer)
    monkeypatch.setattr(gateway_module, "tracer", tracer)

    class FakeResolver:
        async def resolve(self, context: ResolveContext) -> GenerateResponse:
            assert context.prompt == "do not record this prompt"
            return GenerateResponse(
                text="a response",
                provider="ollama",
                model="test-model",
            )

    gateway = LLMGateway()
    gateway.resolver = FakeResolver()
    monkeypatch.setattr(app_module, "gateway", gateway)

    try:
        response = TestClient(app_module.app).post(
            "/generate",
            json={"prompt": "do not record this prompt"},
        )

        assert response.status_code == 200
        spans = exporter.get_finished_spans()
        http_span = next(span for span in spans if span.name == "HTTP POST /generate")
        llm_span = next(span for span in spans if span.name == "llm.generate")
        assert llm_span.parent is not None
        assert llm_span.parent.span_id == http_span.context.span_id
        assert http_span.attributes["http.route"] == "/generate"
        assert llm_span.attributes["llm.provider.name"] == "ollama"
        assert all(
            "do not record this prompt" not in repr(span.attributes)
            for span in spans
        )
    finally:
        provider.shutdown()


def test_resolver_and_pipeline_spans_are_nested_in_gateway_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(app_module, "tracer", tracer)
    monkeypatch.setattr(gateway_module, "tracer", tracer)
    monkeypatch.setattr(resolver_module, "tracer", tracer)
    monkeypatch.setattr(pipeline_module, "tracer", tracer)

    class FakeProvider:
        model = "test-model"

        async def generate(self, prompt: str) -> GenerateResponse:
            return GenerateResponse(
                text="answer",
                provider="ollama",
                model=self.model,
            )

    class FakeRegistry:
        def get_models(self, model: object) -> list[tuple[str, str]]:
            return [("ollama", "test-model")]

        def check_provider_configuration(
            self,
            provider_name: str,
            operation: object,
        ) -> None:
            pass

        def create_provider(self, provider_name: str, model_name: str) -> FakeProvider:
            return FakeProvider()

    from llm_gateway.resolver.model_resolver import ModelResolver

    gateway = LLMGateway()
    gateway.resolver = ModelResolver(FakeRegistry())
    monkeypatch.setattr(app_module, "gateway", gateway)

    try:
        response = TestClient(app_module.app).post(
            "/generate",
            json={"prompt": "private input"},
        )

        assert response.status_code == 200
        spans = exporter.get_finished_spans()
        by_name = {span.name: span for span in spans}
        http_span = by_name["HTTP POST /generate"]
        gateway_span = by_name["llm.generate"]
        resolver_span = by_name["llm.model_resolver"]
        pipeline_span = by_name["llm.policy_pipeline"]
        provider_span = by_name["llm.provider.generate"]

        assert gateway_span.parent is not None
        assert gateway_span.parent.span_id == http_span.context.span_id
        assert resolver_span.parent is not None
        assert resolver_span.parent.span_id == gateway_span.context.span_id
        assert pipeline_span.parent is not None
        assert pipeline_span.parent.span_id == resolver_span.context.span_id
        assert provider_span.parent is not None
        assert provider_span.parent.span_id == pipeline_span.context.span_id
        assert resolver_span.attributes["llm.provider.name"] == "ollama"
        assert pipeline_span.attributes["llm.retry.count"] == 0
        assert all(
            "private input" not in repr(span.attributes)
            for span in spans
        )
    finally:
        provider.shutdown()


def test_http_client_span_records_endpoint_without_request_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        telemetry_module,
        "tracer",
        provider.get_tracer("test"),
    )

    try:
        with telemetry_module.http_client_span(
            "ollama",
            "generate",
            "qwen3:8b",
            "http://ollama:11434/api/chat",
        ):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.name == "HTTP POST /api/chat"
        assert span.attributes["server.address"] == "ollama"
        assert span.attributes["server.port"] == 11434
        assert span.attributes["llm.provider.name"] == "ollama"
        assert span.attributes["llm.outcome"] == "success"
    finally:
        provider.shutdown()


def test_http_client_span_records_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        telemetry_module,
        "tracer",
        provider.get_tracer("test"),
    )
    response = httpx.Response(
        503,
        request=httpx.Request("POST", "https://provider.test"),
    )
    error = httpx.HTTPStatusError(
        "provider unavailable",
        request=response.request,
        response=response,
    )

    try:
        with pytest.raises(httpx.HTTPStatusError, match="provider unavailable"):
            with telemetry_module.http_client_span(
                "openai",
                "generate",
                "test-model",
                "https://provider.test/v1/responses",
            ):
                raise error

        span = exporter.get_finished_spans()[0]
        assert span.attributes["llm.outcome"] == "error"
        assert span.attributes["http.response.status_code"] == 503
    finally:
        provider.shutdown()


def test_http_middleware_continues_when_tracing_span_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenTracer:
        def start_as_current_span(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("tracing backend unavailable")

    class FakeGateway:
        async def generate(
            self,
            prompt: str,
            model: object,
        ) -> GenerateResponse:
            return GenerateResponse(
                text="generated: " + prompt,
                provider="ollama",
                model="test",
            )

    monkeypatch.setattr(app_module, "tracer", BrokenTracer())
    monkeypatch.setattr(app_module, "gateway", FakeGateway())
    caplog.set_level("ERROR")

    response = TestClient(app_module.app).post(
        "/generate",
        json={"prompt": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "generated: hello"
    assert "HTTP tracing span creation failed" in caplog.text
    assert "tracing backend unavailable" in caplog.text


def test_resolver_span_records_fallback_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        resolver_module,
        "tracer",
        provider.get_tracer("test"),
    )

    class FakeRegistry:
        def get_models(self, model: object) -> list[tuple[str, str]]:
            return [("openai", "primary"), ("ollama", "fallback")]

    resolver = resolver_module.ModelResolver(FakeRegistry())

    async def execute(
        context: ResolveContext,
        request_id: str,
        deadline: float,
        provider_name: str,
        model_name: str,
    ) -> GenerateResponse:
        if provider_name == "openai":
            raise ProviderUnavailableError(
                "temporarily unavailable",
                provider=provider_name,
                retryable=True,
            )
        return GenerateResponse(
            text="answer",
            provider="ollama",
            model=model_name,
        )

    monkeypatch.setattr(resolver, "_execute", execute)

    try:
        result = asyncio.run(
            resolver.resolve(
                ResolveContext(
                    model="smart",
                    operation=OperationType.GENERATE,
                    prompt="private input",
                    params={"deadline": 10},
                )
            )
        )

        assert isinstance(result, GenerateResponse)
        span = exporter.get_finished_spans()[0]
        assert span.name == "llm.model_resolver"
        assert span.attributes["llm.provider.name"] == "ollama"
        assert span.attributes["llm.resolver.fallback_count"] == 1
        assert [event.name for event in span.events] == [
            "llm.model_resolver.fallback"
        ]
        assert span.events[0].attributes["llm.provider.name"] == "openai"
        assert "private input" not in repr(span.attributes)
    finally:
        provider.shutdown()


def test_policy_pipeline_traces_retry_check_and_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        pipeline_module,
        "tracer",
        provider.get_tracer("test"),
    )
    monkeypatch.setattr(
        "llm_gateway.core.config.config.MAX_RETRIES",
        1,
    )
    monkeypatch.setattr(
        "llm_gateway.core.config.config.RETRY_BASE_DELAY",
        0,
    )
    monkeypatch.setattr(
        "llm_gateway.core.config.config.RETRY_MAX_DELAY",
        0,
    )
    monkeypatch.setattr(
        "llm_gateway.core.config.config.RETRY_JITTER",
        0,
    )

    class FakeProvider:
        model = "test-model"

        def __init__(self) -> None:
            self.attempts = 0

        async def generate(self, prompt: str) -> GenerateResponse:
            self.attempts += 1
            if self.attempts == 1:
                raise ProviderTimeoutError("temporary", retryable=True)
            return GenerateResponse(
                text="answer",
                provider="ollama",
                model=self.model,
            )

    context = RequestContext(
        request_id="test-request",
        operation=OperationType.GENERATE,
        prompt="private input",
        deadline=time.monotonic() + 10,
    )

    try:
        result = asyncio.run(BasePolicyPipeline(FakeProvider()).execute(context))

        assert isinstance(result, GenerateResponse)
        spans = exporter.get_finished_spans()
        by_name = {span.name: span for span in spans}
        pipeline_span = by_name["llm.policy_pipeline"]
        check_span = by_name["llm.policy.retry_check"]
        wait_span = by_name["llm.policy.retry_wait"]
        assert check_span.parent is not None
        assert check_span.parent.span_id == pipeline_span.context.span_id
        assert check_span.attributes["llm.retry.allowed"] is True
        assert wait_span.parent is not None
        assert wait_span.parent.span_id == pipeline_span.context.span_id
        assert pipeline_span.attributes["llm.retry.count"] == 1
        assert all("private input" not in repr(span.attributes) for span in spans)
    finally:
        provider.shutdown()


def test_stream_span_records_time_to_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(app_module, "tracer", tracer)
    monkeypatch.setattr(gateway_module, "tracer", tracer)
    monkeypatch.setattr(resolver_module, "tracer", tracer)
    monkeypatch.setattr(pipeline_module, "tracer", tracer)

    class FakeProvider:
        model = "test-model"

        async def stream(self, prompt: str) -> AsyncIterator[str]:
            await asyncio.sleep(0.02)
            yield "first"
            yield "second"

    class FakeRegistry:
        def get_models(self, model: object) -> list[tuple[str, str]]:
            return [("ollama", "test-model")]

        def check_provider_configuration(
            self,
            provider_name: str,
            operation: OperationType,
        ) -> None:
            pass

        def create_provider(self, provider_name: str, model_name: str) -> FakeProvider:
            return FakeProvider()

    gateway = LLMGateway()
    gateway.resolver = resolver_module.ModelResolver(FakeRegistry())
    monkeypatch.setattr(app_module, "gateway", gateway)

    try:
        response = TestClient(app_module.app).post(
            "/stream",
            json={"prompt": "private input"},
        )

        assert response.status_code == 200
        assert '"content":"first"' in response.text
        span = next(
            item
            for item in exporter.get_finished_spans()
            if item.name == "llm.stream"
        )
        assert span.attributes["llm.stream.time_to_first_token_ms"] >= 15
        spans = {item.name: item for item in exporter.get_finished_spans()}
        http_span = spans["HTTP POST /stream"]
        resolver_span = spans["llm.model_resolver"]
        pipeline_span = spans["llm.policy_pipeline"]
        provider_span = spans["llm.provider.stream"]
        assert span.parent is not None
        assert span.parent.span_id == http_span.context.span_id
        assert resolver_span.parent is not None
        assert resolver_span.parent.span_id == span.context.span_id
        assert pipeline_span.parent is not None
        assert pipeline_span.parent.span_id == resolver_span.context.span_id
        assert provider_span.parent is not None
        assert provider_span.parent.span_id == pipeline_span.context.span_id
        assert all(
            "private input" not in repr(item.attributes)
            for item in exporter.get_finished_spans()
        )
    finally:
        provider.shutdown()
