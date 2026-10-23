"""Prometheus metrics for the LLM Gateway."""

import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


_METRIC_ROUTES = frozenset({"/generate", "/extract", "/stream"})


def _route_label(path: str) -> str:
    return path if path in _METRIC_ROUTES else "/unmatched"


class PrometheusClient:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self.http_requests_total = Counter(
            "llm_gateway_http_requests_total",
            "Total number of completed HTTP requests.",
            labelnames=("method", "route", "status_code"),
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "llm_gateway_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            labelnames=("method", "route"),
            registry=self.registry,
        )
        self.http_requests_in_progress = Gauge(
            "llm_gateway_http_requests_in_progress",
            "Number of HTTP requests currently being processed.",
            labelnames=("method", "route"),
            registry=self.registry,
        )
        self.llm_requests_total = Counter(
            "llm_gateway_llm_requests_total",
            "Total number of completed LLM operations.",
            labelnames=("operation", "model_tier", "outcome"),
            registry=self.registry,
        )
        self.llm_request_duration_seconds = Histogram(
            "llm_gateway_llm_request_duration_seconds",
            "LLM operation duration in seconds.",
            labelnames=("operation", "model_tier"),
            registry=self.registry,
        )
        self.llm_requests_in_progress = Gauge(
            "llm_gateway_llm_requests_in_progress",
            "Number of LLM operations currently being processed.",
            labelnames=("operation", "model_tier"),
            registry=self.registry,
        )
        self.llm_input_characters = Counter(
            "llm_gateway_llm_input_characters_total",
            "Total number of prompt characters submitted to LLM operations.",
            labelnames=("operation", "model_tier"),
            registry=self.registry,
        )
        self.llm_output_characters = Counter(
            "llm_gateway_llm_output_characters_total",
            "Total number of response characters returned by LLM operations.",
            labelnames=("operation", "model_tier"),
            registry=self.registry,
        )
        self.llm_stream_chunks = Counter(
            "llm_gateway_llm_stream_chunks_total",
            "Total number of response chunks returned by streaming operations.",
            labelnames=("model_tier",),
            registry=self.registry,
        )
        self.provider_requests_total = Counter(
            "llm_gateway_provider_requests_total",
            "Total number of provider operation attempts.",
            labelnames=("provider", "operation", "model", "outcome"),
            registry=self.registry,
        )
        self.provider_request_duration_seconds = Histogram(
            "llm_gateway_provider_request_duration_seconds",
            "Provider operation attempt duration in seconds.",
            labelnames=("provider", "operation", "model"),
            registry=self.registry,
        )
        self.provider_requests_in_progress = Gauge(
            "llm_gateway_provider_requests_in_progress",
            "Number of provider operation attempts currently in progress.",
            labelnames=("provider", "operation", "model"),
            registry=self.registry,
        )
        self.provider_retries_total = Counter(
            "llm_gateway_provider_retries_total",
            "Total number of scheduled retries for provider operations.",
            labelnames=("provider", "operation", "model"),
            registry=self.registry,
        )

    def request_started(self, method: str, path: str) -> None:
        route = _route_label(path)
        self.http_requests_in_progress.labels(method, route).inc()

    def request_completed(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        route = _route_label(path)
        self.http_requests_in_progress.labels(method, route).dec()
        self.http_requests_total.labels(method, route, str(status_code)).inc()
        self.http_request_duration_seconds.labels(method, route).observe(
            duration_seconds
        )

    def llm_operation_started(self, operation: str, model_tier: str) -> float:
        self.llm_requests_in_progress.labels(operation, model_tier).inc()
        return time.monotonic()

    def llm_operation_completed(
        self,
        operation: str,
        model_tier: str,
        started_at: float,
        *,
        outcome: str,
        input_characters: int,
        output_characters: int = 0,
        stream_chunks: int = 0,
    ) -> None:
        labels = (operation, model_tier)
        self.llm_requests_in_progress.labels(*labels).dec()
        self.llm_requests_total.labels(*labels, outcome).inc()
        self.llm_request_duration_seconds.labels(*labels).observe(
            time.monotonic() - started_at
        )
        if input_characters:
            self.llm_input_characters.labels(*labels).inc(input_characters)
        if output_characters:
            self.llm_output_characters.labels(*labels).inc(output_characters)
        if stream_chunks:
            self.llm_stream_chunks.labels(model_tier).inc(stream_chunks)

    def provider_operation_started(
        self,
        provider: str,
        operation: str,
        model: str,
    ) -> float:
        self.provider_requests_in_progress.labels(provider, operation, model).inc()
        return time.monotonic()

    def provider_operation_completed(
        self,
        provider: str,
        operation: str,
        model: str,
        started_at: float,
        *,
        outcome: str,
    ) -> None:
        labels = (provider, operation, model)
        self.provider_requests_in_progress.labels(*labels).dec()
        self.provider_requests_total.labels(*labels, outcome).inc()
        self.provider_request_duration_seconds.labels(*labels).observe(
            time.monotonic() - started_at
        )

    def provider_retry_scheduled(
        self,
        provider: str,
        operation: str,
        model: str,
    ) -> None:
        self.provider_retries_total.labels(provider, operation, model).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


prometheus = PrometheusClient()
