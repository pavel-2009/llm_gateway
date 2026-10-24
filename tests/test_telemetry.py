"""Tests for OpenTelemetry spans."""

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import pytest

import llm_gateway.app as app_module
import llm_gateway.llm_gateway as gateway_module
from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.llm_gateway import LLMGateway
from llm_gateway.resolver.resolver_context import ResolveContext


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
