from collections.abc import AsyncIterator
import logging
import time
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview
from llm_gateway.core.logging_context import request_id_context
from llm_gateway.llm_gateway import LLMGateway


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


class PromptRequest(BaseModel):
    prompt: str


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


@app.post("/generate")
async def generate(request: PromptRequest) -> GenerateResponse:
    logger.info(
        "API operation accepted request_id=%s operation=generate",
        request_id_context.get(),
    )
    return await gateway.generate(request.prompt)


@app.post("/extract")
async def extract(request: PromptRequest) -> ExtractedReview:
    logger.info(
        "API operation accepted request_id=%s operation=extract",
        request_id_context.get(),
    )
    return await gateway.extract(request.prompt)


@app.post("/stream")
async def stream(request: PromptRequest) -> StreamingResponse:
    logger.info(
        "API operation accepted request_id=%s operation=stream",
        request_id_context.get(),
    )

    async def chunks() -> AsyncIterator[str]:
        async for chunk in gateway.stream(request.prompt):
            yield chunk

    return StreamingResponse(chunks(), media_type="text/plain")
