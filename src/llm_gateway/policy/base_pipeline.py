"""Base handling pipeline for LLM Gateway."""

import asyncio
from collections.abc import AsyncIterable
from enum import Enum
from typing import cast

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.providers import ClaudeProvider, OllamaProvider, OpenAIProvider
from llm_gateway.core.config import ExtractedReview
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.errors import AllModelsUnavailableError
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
    ) -> None:
        self.error_policy = error_policy
        self.retry_policy = retry_policy

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

        while True:
            if context.is_expired:
                raise TimeoutError("Request deadline exceeded")
            try:
                self._record_model_attempt(context)
                return await self._execute_operation(context)
            except Exception as error:
                context.model_errors[context.current_model or "unknown"] = error
                should_retry = self._should_retry(error, context)
                if self._should_raise_all_models_error(error, context, should_retry):
                    raise AllModelsUnavailableError(context.model_errors) from error
                if not should_retry:
                    raise
                await self._prepare_retry(error, context)

    async def _execute_stream(self, context: RequestContext) -> AsyncIterable[str]:
        """Retry stream setup/failures until the first chunk is emitted."""
        while True:
            if context.is_expired:
                raise TimeoutError("Request deadline exceeded")
            try:
                self._record_model_attempt(context)
                result = await self._execute_operation(context)
                stream = cast(AsyncIterable[str], result)
                async for chunk in stream:
                    context.first_chunk_emitted = True
                    yield chunk
                return
            except Exception as error:
                if context.first_chunk_emitted:
                    raise
                context.model_errors[context.current_model or "unknown"] = error
                should_retry = self._should_retry(error, context)
                if self._should_raise_all_models_error(error, context, should_retry):
                    raise AllModelsUnavailableError(context.model_errors) from error
                if not should_retry:
                    raise
                await self._prepare_retry(error, context)

    @staticmethod
    def _record_model_attempt(context: RequestContext) -> None:
        if context.current_model is not None:
            context.attempted_models.add(context.current_model)

    def _should_raise_all_models_error(
        self,
        error: Exception,
        context: RequestContext,
        should_retry: bool,
    ) -> bool:
        if not self.retry_policy.all_models_attempted(
            context.primary_model,
            context.attempted_models,
        ):
            return False
        models = self.retry_policy.get_models(context.primary_model)
        if len(models) > 1:
            return True
        return not should_retry and self.error_policy.should_retry(error)

    def _should_retry(self, error: Exception, context: RequestContext) -> bool:
        return self.retry_policy.should_retry(
            error,
            context.fallback_attempts,
            error_policy=self.error_policy,
        )

    async def _prepare_retry(
        self,
        error: Exception,
        context: RequestContext,
    ) -> None:
        context.fallback_attempts += 1
        delay = self.retry_policy.get_backoff_time(context.fallback_attempts)
        remaining_time = context.remaining_time
        if remaining_time <= 0:
            raise TimeoutError("Request deadline exceeded") from error
        if delay >= remaining_time:
            await asyncio.sleep(remaining_time)
            raise TimeoutError("Request deadline exceeded") from error

        await asyncio.sleep(delay)
        context.current_model = self.retry_policy.get_next_model(
            context.current_model,
            context.primary_model,
            context.attempted_models,
        )

    async def _execute_operation(
        self,
        context: RequestContext,
    ) -> PipelineResult:
        """Execute the operation with the given prompt and model."""
        model = ModelType(context.current_model)
        if model == ModelType.OPENAI:
            provider = OpenAIProvider()
        elif model == ModelType.OLLAMA:
            provider = OllamaProvider()
        elif model == ModelType.CLAUDE:
            provider = ClaudeProvider()
        else:
            raise ValueError(f"Unsupported model: {model}")

        if context.operation == OperationType.GENERATE:
            return await provider.generate(context.prompt)
        if context.operation == OperationType.STREAM:
            return provider.stream(context.prompt)
        if context.operation == OperationType.EXTRACT:
            return await provider.extract(context.prompt)
        raise ValueError(f"Unsupported operation: {context.operation}")
