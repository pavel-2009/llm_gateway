"""LLM gateway class."""

from collections.abc import AsyncIterable, AsyncIterator
import logging

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.core.logging_context import request_id_context
from llm_gateway.models import Model
from llm_gateway.policy.request_context import OperationType
from llm_gateway.resolver.model_registry import ModelRegistry
from llm_gateway.resolver.model_resolver import ModelResolver
from llm_gateway.resolver.resolver_context import (
    ResolveContext,
    ResolveParams,
)

logger = logging.getLogger(__name__)


class LLMGateway:
    """LLM gateway."""

    def __init__(self) -> None:
        self.resolver = ModelResolver(ModelRegistry())

    async def generate(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> GenerateResponse:
        request_id = request_id_context.get()
        logger.info(
            "Gateway operation started request_id=%s operation=%s",
            request_id,
            OperationType.GENERATE.value,
        )
        result = await self.resolver.resolve(
            self._build_context(prompt, OperationType.GENERATE, model)
        )
        if not isinstance(result, GenerateResponse):
            raise TypeError("resolver returned an invalid generation result")
        logger.info(
            "Gateway operation completed request_id=%s operation=%s "
            "provider=%s model=%s output_chars=%s",
            request_id,
            OperationType.GENERATE.value,
            result.provider,
            result.model,
            len(result.text),
        )
        return result

    async def extract(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> ExtractedReview:
        request_id = request_id_context.get()
        logger.info(
            "Gateway operation started request_id=%s operation=%s",
            request_id,
            OperationType.EXTRACT.value,
        )
        result = await self.resolver.resolve(
            self._build_context(prompt, OperationType.EXTRACT, model)
        )
        if not isinstance(result, ExtractedReview):
            raise TypeError("resolver returned an invalid extraction result")
        logger.info(
            "Gateway operation completed request_id=%s operation=%s "
            "sentiment=%s topics_count=%s rating=%s",
            request_id,
            OperationType.EXTRACT.value,
            result.sentiment,
            len(result.topics),
            result.rating,
        )
        return result

    def stream(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> AsyncIterator[str]:
        async def iterate() -> AsyncIterator[str]:
            request_id = request_id_context.get()
            chunks = 0
            output_chars = 0
            logger.info(
                "Gateway operation started request_id=%s operation=%s",
                request_id,
                OperationType.STREAM.value,
            )
            result = await self.resolver.resolve(
                self._build_context(prompt, OperationType.STREAM, model)
            )
            if not isinstance(result, AsyncIterable):
                raise TypeError("resolver returned an invalid stream result")
            try:
                async for chunk in result:
                    chunks += 1
                    output_chars += len(chunk)
                    yield chunk
            except Exception as error:
                logger.error(
                    "Gateway operation failed request_id=%s operation=%s "
                    "chunks=%s error_type=%s",
                    request_id,
                    OperationType.STREAM.value,
                    chunks,
                    type(error).__name__,
                )
                raise
            logger.info(
                "Gateway operation completed request_id=%s operation=%s "
                "chunks=%s output_chars=%s",
                request_id,
                OperationType.STREAM.value,
                chunks,
                output_chars,
            )

        return iterate()

    @staticmethod
    def _build_context(
        prompt: str,
        operation: OperationType,
        model: Model,
    ) -> ResolveContext:
        timeout = (
            config.STREAM_TIMEOUT
            if operation == OperationType.STREAM
            else config.REQUEST_TIMEOUT
        )
        return ResolveContext(
            model=model,
            operation=operation,
            prompt=prompt,
            params=ResolveParams(deadline=timeout),
        )
