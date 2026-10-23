"""Tests for Prometheus metrics."""

from prometheus_client import CollectorRegistry

from llm_gateway.prometheus import PrometheusClient


def test_request_metrics_are_recorded() -> None:
    client = PrometheusClient(registry=CollectorRegistry())

    client.request_started("POST", "/generate")
    client.request_finished(
        method="POST",
        route="/generate",
        status_code=200,
        duration_seconds=0.25,
    )

    metrics = client.render().decode()
    assert (
        'llm_gateway_http_requests_total{method="POST",route="/generate",'
        'status_code="200"} 1.0'
    ) in metrics
    assert (
        'llm_gateway_http_request_duration_seconds_count{method="POST",'
        'route="/generate"} 1.0'
    ) in metrics
    assert (
        'llm_gateway_http_requests_in_progress{method="POST",'
        'route="/generate"} 0.0'
    ) in metrics
