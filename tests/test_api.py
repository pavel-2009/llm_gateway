"""Tests for the FastAPI endpoints."""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
import pytest

import llm_gateway.app as main
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
