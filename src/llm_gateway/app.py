"""FastAPI application and HTTP endpoints."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request as HTTPRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.propagate import extract as extract_trace_context
from opentelemetry.trace import Status, StatusCode
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.adapter.errors import (
    AllModelsUnavailableError,
    InvalidRequestError,
    LLMProviderError,
    ProviderTimeoutError,
)
from llm_gateway.core.logging_context import request_id_context
from llm_gateway.llm_gateway import LLMGateway
from llm_gateway.prometheus import prometheus
from llm_gateway.models import (
    ExtractedReview,
    GenerateResponse,
    Request,
    StreamComplete,
    StreamChunk,
    StreamError,
)
from llm_gateway.telemetry import tracer


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
        track_metrics = path != "/metrics"
        route = (
            path
            if path in {"/generate", "/extract", "/stream"}
            else "/unmatched"
        )
        if track_metrics:
            prometheus.request_started(method, path)
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
            incoming_context = extract_trace_context(
                {
                    key.decode("latin-1"): value.decode("latin-1")
                    for key, value in scope.get("headers", [])
                }
            )
            span_context = (
                tracer.start_as_current_span(
                    f"HTTP {method} {route}",
                    context=incoming_context,
                    record_exception=False,
                    set_status_on_exception=False,
                    attributes={
                        "http.request.method": method,
                        "http.route": route,
                        "request.id": request_id,
                    },
                )
                if track_metrics
                else nullcontext(trace.get_current_span())
            )
            with span_context as span:
                try:
                    await self.app(scope, receive, log_response_start)
                except Exception as error:
                    span.record_exception(error)
                    span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                    status_code = _error_details(error)[0]
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
                    span.set_attribute("http.response.status_code", status_code)
                    if status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR))
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
            if track_metrics:
                prometheus.request_completed(method, path, status_code, duration)
            request_id_context.reset(token)


app = FastAPI(title="LLM Gateway")
app.add_middleware(RequestLoggingMiddleware)
gateway = LLMGateway()


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(
        content=prometheus.render(),
        media_type=CONTENT_TYPE_LATEST,
    )


def _error_details(error: Exception) -> tuple[int, str, str]:
    if isinstance(error, (asyncio.TimeoutError, ProviderTimeoutError)):
        return 504, "gateway_timeout", "The request timed out."
    if isinstance(error, AllModelsUnavailableError):
        if error.failures and all(
            isinstance(failure, ProviderTimeoutError)
            for failure in error.failures.values()
        ):
            return 504, "gateway_timeout", "The request timed out."
        if error.failures and all(
            isinstance(failure, LLMProviderError)
            and failure.category == "invalid_response"
            for failure in error.failures.values()
        ):
            return (
                502,
                "upstream_error",
                "The configured model providers returned invalid responses.",
            )
        return (
            503,
            "provider_unavailable",
            "All configured model providers are unavailable.",
        )
    if isinstance(error, InvalidRequestError):
        return 400, "invalid_request", "The model provider rejected the request."
    if isinstance(error, LLMProviderError):
        return (
            502,
            "upstream_error",
            "A model provider failed to process the request.",
        )
    return 500, "internal_error", "An internal server error occurred."


@app.exception_handler(Exception)
async def handle_unexpected_error(
    request: HTTPRequest,
    error: Exception,
) -> JSONResponse:
    status_code, _, message = _error_details(error)
    logger.error(
        "API request failed request_id=%s path=%s error_type=%s",
        request_id_context.get(),
        request.url.path,
        type(error).__name__,
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": message},
    )


@app.exception_handler(RequestValidationError)
async def handle_invalid_request(
    request: HTTPRequest,
    _error: RequestValidationError,
) -> JSONResponse:
    logger.info(
        "API request validation failed request_id=%s path=%s error_count=%s",
        request_id_context.get(),
        request.url.path,
        len(_error.errors()),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request."},
    )


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
                "Server-Sent Events: `chunk` для каждого фрагмента; `done` "
                "с итоговым количеством чанков и символов при успехе; `error` "
                "при ошибке."
            ),
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": (
                        'event: chunk\ndata: {"type":"chunk","index":1,'
                        '"content":"Hello"}\n\n'
                        'event: done\ndata: {"type":"done","chunks":1,'
                        '"output_chars":5}\n\n'
                        'event: error\ndata: {"type":"error",'
                        '"code":"gateway_timeout",'
                        '"message":"The request timed out."}\n\n'
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
        try:
            async for chunk in gateway.stream(request.prompt, request.model):
                chunk_count += 1
                output_chars += len(chunk)
                event = StreamChunk(
                    index=chunk_count,
                    content=chunk,
                ).model_dump_json()
                yield (
                    "event: chunk\n"
                    f"data: {event}"
                    "\n\n"
                )
        except asyncio.TimeoutError as exception:
            logger.error(
                "API operation timed out request_id=%s operation=stream",
                request_id_context.get(),
            )
            _, code, message = _error_details(exception)
            error = StreamError(code=code, message=message)
            yield f"event: error\ndata: {error.model_dump_json()}\n\n"
            return
        except Exception as exception:
            logger.error(
                "API stream failed request_id=%s error_type=%s",
                request_id_context.get(),
                type(exception).__name__,
            )
            _, code, message = _error_details(exception)
            error = StreamError(code=code, message=message)
            yield f"event: error\ndata: {error.model_dump_json()}\n\n"
            return
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
