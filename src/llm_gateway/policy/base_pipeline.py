"""Base handling pipeline for LLM Gateway."""

import asyncio
from collections.abc import AsyncIterable
from enum import Enum
from typing import cast

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.errors import AllModelsUnavailableError
from llm_gateway.adapter.providers import ClaudeProvider, OllamaProvider, OpenAIProvider
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import ExtractedReview
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.request_context import OperationType, RequestContext
from llm_gateway.policy.retry_policy import RetryPolicy


class ModelType(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"


PipelineResult = GenerateResponse | ExtractedReview | AsyncIterable[str]

class BasePolicyPipeline:
    """Execute provider operations with error and retry policies."""

    def __init__(
        self,
        error_policy: ErrorPolicy,
        retry_policy: RetryPolicy,
        provider: LLMProvider,
    ) -> None:
        self.error_policy = error_policy
        self.retry_policy = retry_policy
        self.provider = provider

    async def execute(
        self,
        context: RequestContext,
    ) -> PipelineResult:
        """Execute the operation with error and retry policies."""
        if context.is_expired:
            raise TimeoutError("Request deadline exceeded")

        context.primary_attempts += 1
        context.current_model = context.current_model or context.primary_model

        if context.operation == OperationType.STREAM:
            return self._execute_stream(context)

        async with asyncio.timeout(context.remaining_time):
            while True:
                try:
                    return await self._execute_operation(context)
                except Exception as error:
                    await self._handle_error(error, context)

    async def _execute_stream(self, context: RequestContext) -> AsyncIterable[str]:
        """Retry stream setup/failures until the first chunk is emitted."""
        while True:
            if context.is_expired:
                raise TimeoutError("Request deadline exceeded")
            try:
                result = await self._execute_operation(context)
                stream = cast(AsyncIterable[str], result)
                async for chunk in stream:
                    context.first_chunk_emitted = True
                    yield chunk
                return
            except Exception as error:
                if context.first_chunk_emitted:
                    raise
                await self._handle_error(error, context)

    async def _handle_error(
        self,
        error: Exception,
        context: RequestContext,
    ) -> None:
        model = context.current_model or "unknown"
        context.model_errors[model] = error

        models = self.retry_policy.get_models(context.primary_model)
        all_models_failed = bool(models) and all(
            model in context.model_errors for model in models
        )
        retryable = self.error_policy.should_retry(error)
        should_retry = self.retry_policy.should_retry(
            error,
            context.fallback_attempts,
            error_policy=self.error_policy,
        )

        if all_models_failed and (len(models) > 1 or (retryable and not should_retry)):
            raise AllModelsUnavailableError(context.model_errors) from error
        if not should_retry:
            raise error

        context.fallback_attempts += 1
        delay = self.retry_policy.get_backoff_time(context.fallback_attempts)
        if delay >= context.remaining_time:
            if context.remaining_time > 0:
                await asyncio.sleep(context.remaining_time)
            raise TimeoutError("Request deadline exceeded") from error

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
