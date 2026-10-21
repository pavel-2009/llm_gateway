"""Logging context shared by the API and gateway layers."""

from contextvars import ContextVar


request_id_context: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)
