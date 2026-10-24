"""LLM gateway class."""

from collections.abc import AsyncIterable, AsyncIterator
import logging

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.core.logging_context import request_id_context
from llm_gateway.models import Model
from llm_gateway.policy.request_context import OperationType
from llm_gateway.prometheus import prometheus
from llm_gateway.resolver.model_registry import ModelRegistry
from llm_gateway.resolver.model_resolver import ModelResolver
from llm_gateway.resolver.resolver_context import (
    ResolveContext,
    ResolveParams,
)
from llm_gateway.telemetry import tracer

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
        with tracer.start_as_current_span(
            "llm.generate",
            attributes={
                "llm.operation": OperationType.GENERATE.value,
                "llm.model_tier": model.value,
                "llm.input.characters": len(prompt),
            },
        ) as span:
            started_at = prometheus.llm_operation_started(
                OperationType.GENERATE.value, model.value
            )
            succeeded = False
            request_id = request_id_context.get()
            logger.info(
                "Gateway operation started request_id=%s operation=%s",
                request_id,
                OperationType.GENERATE.value,
            )
            output_characters = 0
            try:
                result = await self.resolver.resolve(
                    self._build_context(prompt, OperationType.GENERATE, model)
                )
                if not isinstance(result, GenerateResponse):
                    raise TypeError("resolver returned an invalid generation result")
                output_characters = len(result.text)
                span.set_attribute("llm.provider.name", result.provider)
                span.set_attribute("llm.response.model", result.model)
                span.set_attribute("llm.output.characters", output_characters)
                logger.info(
                    "Gateway operation completed request_id=%s operation=%s "
                    "provider=%s model=%s output_chars=%s",
                    request_id,
                    OperationType.GENERATE.value,
                    result.provider,
                    result.model,
                    output_characters,
                )
                succeeded = True
                return result
            finally:
                span.set_attribute(
                    "llm.outcome",
                    "success" if succeeded else "error",
                )
                prometheus.llm_operation_completed(
                    OperationType.GENERATE.value,
                    model.value,
                    started_at,
                    outcome="success" if succeeded else "error",
                    input_characters=len(prompt),
                    output_characters=output_characters,
                )

    async def extract(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> ExtractedReview:
        with tracer.start_as_current_span(
            "llm.extract",
            attributes={
                "llm.operation": OperationType.EXTRACT.value,
                "llm.model_tier": model.value,
                "llm.input.characters": len(prompt),
            },
        ) as span:
            started_at = prometheus.llm_operation_started(
                OperationType.EXTRACT.value, model.value
            )
            succeeded = False
            request_id = request_id_context.get()
            logger.info(
                "Gateway operation started request_id=%s operation=%s",
                request_id,
                OperationType.EXTRACT.value,
            )
            output_characters = 0
            try:
                result = await self.resolver.resolve(
                    self._build_context(prompt, OperationType.EXTRACT, model)
                )
                if not isinstance(result, ExtractedReview):
                    raise TypeError("resolver returned an invalid extraction result")
                output_characters = len(result.model_dump_json())
                span.set_attribute("llm.output.characters", output_characters)
                logger.info(
                    "Gateway operation completed request_id=%s operation=%s "
                    "sentiment=%s topics_count=%s rating=%s",
                    request_id,
                    OperationType.EXTRACT.value,
                    result.sentiment,
                    len(result.topics),
                    result.rating,
                )
                succeeded = True
                return result
            finally:
                span.set_attribute(
                    "llm.outcome",
                    "success" if succeeded else "error",
                )
                prometheus.llm_operation_completed(
                    OperationType.EXTRACT.value,
                    model.value,
                    started_at,
                    outcome="success" if succeeded else "error",
                    input_characters=len(prompt),
                    output_characters=output_characters,
                )

    def stream(
        self,
        prompt: str,
        model: Model = Model.SMART,
    ) -> AsyncIterator[str]:
        async def iterate() -> AsyncIterator[str]:
            with tracer.start_as_current_span(
                "llm.stream",
                attributes={
                    "llm.operation": OperationType.STREAM.value,
                    "llm.model_tier": model.value,
                    "llm.input.characters": len(prompt),
                },
            ) as span:
                started_at = prometheus.llm_operation_started(
                    OperationType.STREAM.value, model.value
                )
                succeeded = False
                request_id = request_id_context.get()
                chunks = 0
                output_chars = 0
                logger.info(
                    "Gateway operation started request_id=%s operation=%s",
                    request_id,
                    OperationType.STREAM.value,
                )
                try:
                    result = await self.resolver.resolve(
                        self._build_context(prompt, OperationType.STREAM, model)
                    )
                    if not isinstance(result, AsyncIterable):
                        raise TypeError("resolver returned an invalid stream result")
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
                else:
                    logger.info(
                        "Gateway operation completed request_id=%s operation=%s "
                        "chunks=%s output_chars=%s",
                        request_id,
                        OperationType.STREAM.value,
                        chunks,
                        output_chars,
                    )
                    succeeded = True
                finally:
                    span.set_attribute(
                        "llm.output.characters",
                        output_chars,
                    )
                    span.set_attribute("llm.output.chunks", chunks)
                    span.set_attribute(
                        "llm.outcome",
                        "success" if succeeded else "error",
                    )
                    prometheus.llm_operation_completed(
                        OperationType.STREAM.value,
                        model.value,
                        started_at,
                        outcome="success" if succeeded else "error",
                        input_characters=len(prompt),
                        output_characters=output_chars,
                        stream_chunks=chunks,
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
