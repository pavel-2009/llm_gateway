"""Small Ollama-compatible fake model for gateway load tests."""

import asyncio
from collections.abc import AsyncIterator
import json
import math
import os
import random

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse


def _read_probability(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _read_delay(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


ERROR_RATE = _read_probability("MOCK_MODEL_ERROR_RATE", 0.1)
MIN_DELAY_SECONDS = _read_delay("MOCK_MODEL_MIN_DELAY_SECONDS", 0.02)
MAX_DELAY_SECONDS = _read_delay("MOCK_MODEL_MAX_DELAY_SECONDS", 0.15)
if MIN_DELAY_SECONDS > MAX_DELAY_SECONDS:
    raise ValueError(
        "MOCK_MODEL_MIN_DELAY_SECONDS must not exceed "
        "MOCK_MODEL_MAX_DELAY_SECONDS"
    )

_RESPONSES = (
    "A quiet lighthouse guarded the coast through the storm.",
    "Try planting basil, thyme, and mint in a sunny window box.",
    "Leaves change color as chlorophyll breaks down in autumn.",
    "Hello! I hope your day is going wonderfully.",
)
_ERROR_STATUSES = (429, 500, 503)
_EXTRACTION_RESPONSES = (
    '{"sentiment":"positive","topics":["service","quality"],"rating":5}',
    '{"sentiment":"negative","topics":["delivery"],"rating":2}',
    '{"sentiment":"neutral","topics":["product"],"rating":0}',
)

app = FastAPI(title="Mock Ollama model")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=None)
async def chat(
    payload: dict[str, object],
) -> dict[str, object] | StreamingResponse:
    await asyncio.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
    if random.random() < ERROR_RATE:
        status_code = random.choice(_ERROR_STATUSES)
        raise HTTPException(
            status_code=status_code,
            detail="Random mock model failure.",
        )

    model = payload.get("model", "mock-model")
    if payload.get("stream") is True:
        return StreamingResponse(
            _stream_response(model),
            media_type="application/x-ndjson",
        )

    if "format" in payload:
        content = random.choice(_EXTRACTION_RESPONSES)
    else:
        content = _generate_response()

    return {
        "model": model,
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
    }


async def _stream_response(model: object) -> AsyncIterator[str]:
    response = _generate_response()
    chunks = response.split()
    for index, chunk in enumerate(chunks):
        payload = {
            "model": model,
            "message": {
                "role": "assistant",
                "content": chunk + (" " if index < len(chunks) - 1 else ""),
            },
            "done": False,
        }
        yield f"{json.dumps(payload)}\n"
        await asyncio.sleep(0)
    final_payload = {
        "model": model,
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }
    yield f"{json.dumps(final_payload)}\n"


def _generate_response() -> str:
    return (
        f"{random.choice(_RESPONSES)} "
        f"Reference {random.randrange(100_000, 1_000_000)}."
    )
