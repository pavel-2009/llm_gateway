"""FastAPI application and HTTP endpoints."""

from collections.abc import AsyncIterator
import logging
import time
from uuid import uuid4

from fastapi import FastAPI
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.core.logging_context import request_id_context
from llm_gateway.llm_gateway import LLMGateway
from llm_gateway.models import (
    ExtractedReview,
    GenerateResponse,
    Request,
    StreamComplete,
    StreamChunk,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        token = request_id_context.set(request_id)
        method = scope["method"]
        path = scope["path"]
        started_at = time.monotonic()
        status_code = 500
        logger.info(
            "HTTP request started request_id=%s method=%s path=%s",
            request_id,
            method,
            path,
        )

        async def log_response_start(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, log_response_start)
        except Exception as error:
            logger.error(
                "HTTP request failed request_id=%s method=%s path=%s "
                "error_type=%s",
                request_id,
                method,
                path,
                type(error).__name__,
            )
            raise
        finally:
            duration = time.monotonic() - started_at
            logger.info(
                "HTTP request completed request_id=%s method=%s path=%s "
                "status_code=%s duration_ms=%.1f",
                request_id,
                method,
                path,
                status_code,
                duration * 1000,
            )
            request_id_context.reset(token)


app = FastAPI(title="LLM Gateway")
app.add_middleware(RequestLoggingMiddleware)
gateway = LLMGateway()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: Request) -> GenerateResponse:
    logger.info(
        "API operation accepted request_id=%s operation=generate",
        request_id_context.get(),
    )
    return await gateway.generate(request.prompt, request.model)


@app.post("/extract", response_model=ExtractedReview)
async def extract(request: Request) -> ExtractedReview:
    logger.info(
        "API operation accepted request_id=%s operation=extract",
        request_id_context.get(),
    )
    return await gateway.extract(request.prompt, request.model)


@app.post(
    "/stream",
    response_class=StreamingResponse,
    response_model=None,
    responses={
        200: {
            "description": (
                "Server-Sent Events: `chunk` для каждого фрагмента и `done` "
                "с итоговым количеством чанков и символов."
            ),
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": (
                        'event: chunk\ndata: {"type":"chunk","index":1,'
                        '"content":"Hello"}\n\n'
                        'event: done\ndata: {"type":"done","chunks":1,'
                        '"output_chars":5}\n\n'
                    ),
                },
            },
        },
    },
)
async def stream(request: Request) -> StreamingResponse:
    logger.info(
        "API operation accepted request_id=%s operation=stream",
        request_id_context.get(),
    )

    async def chunks() -> AsyncIterator[str]:
        chunk_count = 0
        output_chars = 0
        async for chunk in gateway.stream(request.prompt, request.model):
            chunk_count += 1
            output_chars += len(chunk)
            yield (
                "event: chunk\n"
                f"data: {StreamChunk(index=chunk_count, content=chunk).model_dump_json()}"
                "\n\n"
            )
        completed = StreamComplete(
            chunks=chunk_count,
            output_chars=output_chars,
        )
        yield f"event: done\ndata: {completed.model_dump_json()}\n\n"

    return StreamingResponse(
        chunks(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
