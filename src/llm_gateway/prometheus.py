"""Prometheus metrics for the LLM Gateway."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class PrometheusClient:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
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

    def request_started(self, method: str, route: str) -> None:
        self.http_requests_in_progress.labels(method, route).inc()

    def request_finished(
        self,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        self.http_requests_in_progress.labels(method, route).dec()
        self.http_requests_total.labels(method, route, str(status_code)).inc()
        self.http_request_duration_seconds.labels(method, route).observe(
            duration_seconds
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


prometheus = PrometheusClient()
