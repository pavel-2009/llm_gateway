"""OpenTelemetry setup and shared tracer."""

import atexit

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
