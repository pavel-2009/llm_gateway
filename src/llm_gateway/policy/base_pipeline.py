"""Base handling pipeline for LLM Gateway."""

import asyncio
from collections.abc import AsyncIterable
import logging
from typing import cast

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import ExtractedReview
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.request_context import OperationType, RequestContext
from llm_gateway.policy.retry_policy import RetryPolicy


PipelineResult = GenerateResponse | ExtractedReview | AsyncIterable[str]
logger = logging.getLogger(__name__)

class BasePolicyPipeline:
    """Execute provider operations with error and retry policies."""

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self.error_policy = ErrorPolicy()
        self.retry_policy = RetryPolicy()
        self.provider = provider

    async def execute(
        self,
        context: RequestContext,
    ) -> PipelineResult:
        """Execute the operation with error and retry policies."""
        if context.is_expired:
            logger.warning(
                "Pipeline deadline expired request_id=%s operation=%s",
                context.request_id,
                context.operation.value,
            )
            raise TimeoutError("Request deadline exceeded")

        logger.info(
            "Pipeline started request_id=%s operation=%s adapter=%s model=%s",
            context.request_id,
            context.operation.value,
            type(self.provider).__name__,
            getattr(self.provider, "model", "unknown"),
        )
        if context.operation == OperationType.STREAM:
            return self._execute_stream(context)

        while True:
            try:
                result = await self._execute_operation(context)
                logger.info(
                    "Pipeline operation completed request_id=%s operation=%s "
                    "retries=%s",
                    context.request_id,
                    context.operation.value,
                    context.retries_made,
                )
                return result
            except Exception as error:
                await self._handle_error(error, context)

    async def _execute_stream(self, context: RequestContext) -> AsyncIterable[str]:
        """Retry stream setup/failures until the first chunk is emitted."""
        while True:
            if context.is_expired:
                logger.warning(
                    "Pipeline stream deadline expired request_id=%s retries=%s",
                    context.request_id,
                    context.retries_made,
                )
                raise TimeoutError("Request deadline exceeded")
            try:
                result = await self._execute_operation(context)
                stream = cast(AsyncIterable[str], result)
                chunks = 0
                async for chunk in stream:
                    chunks += 1
                    context.first_chunk_emitted = True
                    yield chunk
                logger.info(
                    "Pipeline stream completed request_id=%s chunks=%s "
                    "retries=%s",
                    context.request_id,
                    chunks,
                    context.retries_made,
                )
                return
            except Exception as error:
                if context.first_chunk_emitted:
                    logger.error(
                        "Pipeline stream failed after first chunk "
                        "request_id=%s error_type=%s",
                        context.request_id,
                        type(error).__name__,
                    )
                    raise
                await self._handle_error(error, context)

    async def _handle_error(
        self,
        error: Exception,
        context: RequestContext,
    ) -> None:
        should_retry = self.retry_policy.should_retry(
            error,
            context.retries_made,
            error_policy=self.error_policy,
        )

        if not should_retry:
            logger.error(
                "Pipeline operation failed request_id=%s operation=%s "
                "error_type=%s retry=false retries=%s",
                context.request_id,
                context.operation.value,
                type(error).__name__,
                context.retries_made,
            )
            raise error

        context.retries_made += 1
        delay = self.retry_policy.get_backoff_time(context.retries_made)

        if delay >= context.remaining_time:
            logger.error(
                "Pipeline retry skipped request_id=%s operation=%s "
                "error_type=%s reason=deadline retry=%s delay_seconds=%.3f",
                context.request_id,
                context.operation.value,
                type(error).__name__,
                context.retries_made,
                delay,
            )
            raise error

        logger.warning(
            "Pipeline retry scheduled request_id=%s operation=%s "
            "error_type=%s retry=%s delay_seconds=%.3f",
            context.request_id,
            context.operation.value,
            type(error).__name__,
            context.retries_made,
            delay,
        )
        await asyncio.sleep(delay)

    async def _execute_operation(
        self,
        context: RequestContext,
    ) -> PipelineResult:
        """Execute the operation with the given prompt and model."""

        if context.operation == OperationType.GENERATE:
            return await self.provider.generate(context.prompt)
        if context.operation == OperationType.STREAM:
            return self.provider.stream(context.prompt)
        if context.operation == OperationType.EXTRACT:
            return await self.provider.extract(context.prompt)
        raise ValueError(f"Unsupported operation: {context.operation}")
