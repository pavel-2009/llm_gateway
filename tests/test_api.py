"""Tests for the FastAPI endpoints."""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
import pytest

import main
from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview


class FakeGateway:
    async def generate(self, prompt: str) -> GenerateResponse:
        return GenerateResponse(
            text=f"generated: {prompt}",
            provider="test",
            model="test",
        )

    async def extract(self, prompt: str) -> ExtractedReview:
        return ExtractedReview(sentiment="neutral", topics=[prompt], rating=0)

    def stream(self, prompt: str) -> AsyncIterator[str]:
        async def chunks() -> AsyncIterator[str]:
            yield prompt
            yield "!"

        return chunks()


def test_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "gateway", FakeGateway())

    response = TestClient(main.app).post("/generate", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "text": "generated: hello",
        "provider": "test",
        "model": "test",
    }


def test_extract_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "gateway", FakeGateway())

    response = TestClient(main.app).post("/extract", json={"prompt": "review"})

    assert response.status_code == 200
    assert response.json() == {
        "sentiment": "neutral",
        "topics": ["review"],
        "rating": 0,
    }


def test_stream_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "gateway", FakeGateway())

    response = TestClient(main.app).post("/stream", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.text == "hello!"
    assert response.headers["content-type"].startswith("text/plain")


def test_prompt_is_required() -> None:
    response = TestClient(main.app).post("/generate", json={})

    assert response.status_code == 422
