"""OpenTelemetry setup and shared tracer."""

import atexit
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ParentBased,
    TraceIdRatioBased,
)

from llm_gateway.core.config import config


def _configure_tracing() -> None:
    if not config.OTEL_ENABLED:
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": config.OTEL_SERVICE_NAME}),
        sampler=ParentBased(TraceIdRatioBased(config.OTEL_TRACE_SAMPLE_RATIO)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=config.OTEL_EXPORTER_OTLP_ENDPOINT)
        )
    )
    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)


_configure_tracing()
tracer = trace.get_tracer("llm_gateway")


@contextmanager
def http_client_span(
    provider: str,
    operation: str,
    model: str,
    url: str,
) -> Iterator[trace.Span]:
    parsed_url = urlsplit(url)
    path = parsed_url.path or "/"
    attributes: dict[str, str | int] = {
        "http.request.method": "POST",
        "url.scheme": parsed_url.scheme,
        "url.path": path,
        "server.address": parsed_url.hostname or "unknown",
        "llm.provider.name": provider,
        "llm.operation": operation,
        "llm.request.model": model,
    }
    if parsed_url.port is not None:
        attributes["server.port"] = parsed_url.port

    with tracer.start_as_current_span(
        f"HTTP POST {path}",
        attributes=attributes,
    ) as span:
        try:
            yield span
        except Exception as error:
            span.set_attribute("llm.outcome", "error")
            status_code = getattr(error, "status_code", None)
            if status_code is None:
                response = getattr(error, "response", None)
                status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int):
                span.set_attribute("http.response.status_code", status_code)
            raise
        else:
            span.set_attribute("llm.outcome", "success")
