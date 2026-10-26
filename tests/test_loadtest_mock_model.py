"""Tests for the Ollama-compatible load-test mock."""

import json

from fastapi.testclient import TestClient
import pytest

import loadtest.mock_ollama as mock_ollama


def test_mock_model_returns_ollama_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mock_ollama, "ERROR_RATE", 0)
    monkeypatch.setattr(mock_ollama, "MIN_DELAY_SECONDS", 0)
    monkeypatch.setattr(mock_ollama, "MAX_DELAY_SECONDS", 0)

    response = TestClient(mock_ollama.app).post(
        "/api/chat",
        json={"model": "mock-model", "messages": []},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "mock-model"
    assert response.json()["message"]["role"] == "assistant"
    assert response.json()["message"]["content"]
    assert response.json()["done"] is True


def test_mock_model_can_return_random_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mock_ollama, "ERROR_RATE", 1)
    monkeypatch.setattr(mock_ollama, "MIN_DELAY_SECONDS", 0)
    monkeypatch.setattr(mock_ollama, "MAX_DELAY_SECONDS", 0)
    monkeypatch.setattr(mock_ollama.random, "choice", lambda values: values[-1])

    response = TestClient(mock_ollama.app).post("/api/chat", json={})

    assert response.status_code == 503
    assert response.json()["detail"] == "Random mock model failure."


def test_mock_model_returns_structured_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock_ollama, "ERROR_RATE", 0)
    monkeypatch.setattr(mock_ollama, "MIN_DELAY_SECONDS", 0)
    monkeypatch.setattr(mock_ollama, "MAX_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        mock_ollama.random,
        "choice",
        lambda values: values[0],
    )

    response = TestClient(mock_ollama.app).post(
        "/api/chat",
        json={"model": "mock-model", "format": {}, "messages": []},
    )

    assert response.status_code == 200
    assert response.json()["message"]["content"] == (
        '{"sentiment":"positive","topics":["service","quality"],"rating":5}'
    )


def test_mock_model_streams_ollama_ndjson(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock_ollama, "ERROR_RATE", 0)
    monkeypatch.setattr(mock_ollama, "MIN_DELAY_SECONDS", 0)
    monkeypatch.setattr(mock_ollama, "MAX_DELAY_SECONDS", 0)
    monkeypatch.setattr(mock_ollama, "_generate_response", lambda: "hello world")

    response = TestClient(mock_ollama.app).post(
        "/api/chat",
        json={"model": "mock-model", "stream": True, "messages": []},
    )

    chunks = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [chunk["message"]["content"] for chunk in chunks] == ["hello ", "world", ""]
    assert chunks[-1]["done"] is True
