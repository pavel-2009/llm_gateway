"""Logging context shared by the API and gateway layers."""

from contextvars import ContextVar

from opentelemetry import trace


request_id_context: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)


def current_trace_id() -> str | None:
    """Return the active OpenTelemetry trace ID as a 32-character hex string."""
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"
