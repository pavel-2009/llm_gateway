"""Tests for the FastAPI endpoints."""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
import pytest

import llm_gateway.app as main
from llm_gateway.adapter.errors import (
    AIExtractionError,
    AllModelsUnavailableError,
    InvalidRequestError,
    ProviderTimeoutError,
)
from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview
from llm_gateway.models import Model


class FakeGateway:
    def __init__(self) -> None:
        self.model = Model.SMART

    async def generate(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> GenerateResponse:
        self.model = model
        return GenerateResponse(
            text=f"generated: {prompt}",
            provider="openai",
            model="test",
        )

    async def extract(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> ExtractedReview:
        self.model = model
        return ExtractedReview(sentiment="neutral", topics=[prompt], rating=0)

    def stream(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> AsyncIterator[str]:
        self.model = model

        async def chunks() -> AsyncIterator[str]:
            yield prompt
            yield "!"

        return chunks()


def test_generate_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    fake_gateway = FakeGateway()
    monkeypatch.setattr(main, "gateway", fake_gateway)

    response = TestClient(main.app).post(
        "/generate",
        json={"prompt": "hello", "model": "medium"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "generated: hello",
        "provider": "openai",
        "model": "test",
    }
    assert fake_gateway.model == Model.MEDIUM
    assert "request_id=" in caplog.text
    assert "operation=generate" in caplog.text
    assert "hello" not in caplog.text


def test_extract_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gateway = FakeGateway()
    monkeypatch.setattr(main, "gateway", fake_gateway)

    response = TestClient(main.app).post("/extract", json={"prompt": "review"})

    assert response.status_code == 200
    assert response.json() == {
        "sentiment": "neutral",
        "topics": ["review"],
        "rating": 0,
    }
    assert fake_gateway.model == Model.SMART


def test_stream_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gateway = FakeGateway()
    monkeypatch.setattr(main, "gateway", fake_gateway)

    response = TestClient(main.app).post("/stream", json={"prompt": "hello"})

    assert response.status_code == 200
    events = response.text.strip().split("\n\n")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events == [
        'event: chunk\ndata: {"type":"chunk","index":1,"content":"hello"}',
        'event: chunk\ndata: {"type":"chunk","index":2,"content":"!"}',
        'event: done\ndata: {"type":"done","chunks":2,"output_chars":6}',
    ]
    assert fake_gateway.model == Model.SMART


def test_metrics_endpoint_exposes_http_metrics_without_counting_scrapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "gateway", FakeGateway())
    client = TestClient(main.app)

    initial_response = client.get("/metrics")
    initial_count = _generate_request_count(initial_response.text)
    api_response = client.post("/generate", json={"prompt": "hello"})
    metrics_response = client.get("/metrics")

    assert api_response.status_code == 200
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")
    assert _generate_request_count(metrics_response.text) == initial_count + 1
    assert 'route="/metrics"' not in metrics_response.text


def _generate_request_count(metrics: str, status_code: int = 200) -> float:
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if (
                sample.name == "llm_gateway_http_requests_total"
                and sample.labels
                == {
                    "method": "POST",
                    "route": "/generate",
                    "status_code": str(status_code),
                }
            ):
                return sample.value
    return 0.0


def test_stream_openapi_response_contract() -> None:
    response = main.app.openapi()["paths"]["/stream"]["post"]["responses"]["200"]

    assert "text/event-stream" in response["content"]
    assert "количеством чанков и символов" in response["description"]


def test_empty_stream_reports_zero_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyGateway(FakeGateway):
        def stream(
            self,
            prompt: str,
            model: Model = Model.SMART,
        ) -> AsyncIterator[str]:
            async def chunks() -> AsyncIterator[str]:
                for chunk in ():
                    yield chunk

            return chunks()

    monkeypatch.setattr(main, "gateway", EmptyGateway())
    response = TestClient(main.app).post("/stream", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.text == (
        'event: done\ndata: {"type":"done","chunks":0,"output_chars":0}\n\n'
    )


def test_prompt_is_required() -> None:
    response = TestClient(main.app).post("/generate", json={})

    assert response.status_code == 422


def invalid_structured_response() -> AllModelsUnavailableError:
    error = AIExtractionError("private model output")
    error.category = "invalid_response"
    return AllModelsUnavailableError({"ollama": error})


def test_structured_extraction_error_is_explained_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingExtractionGateway(FakeGateway):
        async def extract(
            self,
            prompt: str,
            model: Model = Model.SMART,
        ) -> ExtractedReview:
            raise invalid_structured_response()

    monkeypatch.setattr(main, "gateway", FailingExtractionGateway())
    response = TestClient(main.app, raise_server_exceptions=False).post(
        "/extract",
        json={"prompt": "review"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The configured model providers returned invalid responses."
    }
    assert "private model output" not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "   "},
        {"prompt": "valid", "model": "unknown"},
        {"prompt": "x" * 10001},
        {"prompt": "valid", "unexpected": True},
    ],
)
def test_invalid_request_contract_returns_sanitized_422(
    payload: dict[str, object],
) -> None:
    response = TestClient(main.app).post("/generate", json=payload)

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (TimeoutError("private timeout"), 504, "The request timed out."),
        (
            AllModelsUnavailableError(
                {"ollama": RuntimeError("private provider secret")}
            ),
            503,
            "All configured model providers are unavailable.",
        ),
        (
            AllModelsUnavailableError(
                {
                    "ollama": ProviderTimeoutError(
                        "private timeout",
                        provider="ollama",
                        retryable=True,
                    )
                }
            ),
            504,
            "The request timed out.",
        ),
        (
            invalid_structured_response(),
            502,
            "The configured model providers returned invalid responses.",
        ),
        (
            InvalidRequestError("private provider error", provider="ollama"),
            400,
            "The model provider rejected the request.",
        ),
        (
            RuntimeError("private internal details"),
            500,
            "An internal server error occurred.",
        ),
    ],
)
def test_gateway_errors_return_safe_http_responses(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    class FailingGateway(FakeGateway):
        async def generate(
            self,
            prompt: str,
            model: Model = Model.SMART,
        ) -> GenerateResponse:
            raise error

    monkeypatch.setattr(main, "gateway", FailingGateway())
    client = TestClient(main.app, raise_server_exceptions=False)
    metrics_before = client.get("/metrics").text
    response = client.post(
        "/generate",
        json={"prompt": "hello"},
    )
    metrics_after = client.get("/metrics").text

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "private" not in response.text
    assert (
        _generate_request_count(metrics_after, status_code)
        == _generate_request_count(metrics_before, status_code) + 1
    )


@pytest.mark.parametrize(
    ("error", "error_code", "message", "chunks"),
    [
        (
            TimeoutError("private timeout"),
            "gateway_timeout",
            "The request timed out.",
            (),
        ),
        (
            RuntimeError("private internal details"),
            "internal_error",
            "An internal server error occurred.",
            ("first",),
        ),
    ],
)
def test_stream_errors_are_sse_events_without_done(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    error_code: str,
    message: str,
    chunks: tuple[str, ...],
) -> None:
    class FailingStreamGateway(FakeGateway):
        def stream(
            self,
            prompt: str,
            model: Model = Model.SMART,
        ) -> AsyncIterator[str]:
            async def failing_chunks() -> AsyncIterator[str]:
                for chunk in chunks:
                    yield chunk
                raise error

            return failing_chunks()

    monkeypatch.setattr(main, "gateway", FailingStreamGateway())
    response = TestClient(main.app).post("/stream", json={"prompt": "hello"})

    assert response.status_code == 200
    assert "event: error\n" in response.text
    assert (
        f'"type":"error","code":"{error_code}","message":"{message}"'
        in response.text
    )
    assert "event: done\n" not in response.text
    assert "private" not in response.text
    if chunks:
        assert (
            'event: chunk\ndata: {"type":"chunk","index":1,'
            '"content":"first"}'
        ) in response.text
