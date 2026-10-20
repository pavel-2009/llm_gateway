"""Model resolver"""

from collections.abc import AsyncIterable
import asyncio
from math import isfinite
import time
from typing import cast
from uuid import uuid4

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.adapter.errors import (
    AllModelsUnavailableError,
    ContentPolicyError,
    InvalidRequestError,
    LLMProviderError,
)
from llm_gateway.policy.base_pipeline import BasePolicyPipeline
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.retry_policy import RetryPolicy
from llm_gateway.policy.request_context import OperationType, RequestContext
from llm_gateway.resolver.model_registry import ModelRegistry
from llm_gateway.resolver.resolver_context import ResolveContext
from llm_gateway.core.config import ExtractedReview, config


RESOLVER_RESULT = GenerateResponse | ExtractedReview | AsyncIterable[str]


class ModelResolver:
    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()

    async def resolve(self, context: ResolveContext) -> RESOLVER_RESULT:
        if not isinstance(context, ResolveContext):
            raise TypeError("context must be a ResolveContext")

        request_id = uuid4().hex
        timeout = (
            context.params.deadline
            if context.params.deadline is not None
            else config.REQUEST_TIMEOUT
        )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("request timeout must be a positive finite number")
        deadline = time.monotonic() + timeout

        if context.operation == OperationType.STREAM:
            return self._resolve_stream(context, request_id, deadline)

        async with asyncio.timeout_at(deadline):
            failures: dict[str, Exception] = {}
            for provider_name, model_name in self.model_registry.get_models(
                context.model
            ):
                key = f"{provider_name}:{model_name}"
                try:
                    return await self._execute(
                        context,
                        request_id,
                        deadline,
                        provider_name,
                        model_name,
                    )
                except LLMProviderError as error:
                    if not self._can_fallback(error):
                        raise
                    failures[key] = error

            raise AllModelsUnavailableError(failures)

    async def _execute(
        self,
        context: ResolveContext,
        request_id: str,
        deadline: float,
        provider_name: str,
        model_name: str,
    ) -> RESOLVER_RESULT:
        self.model_registry.check_provider_configuration(
            provider_name,
            context.operation,
        )
        provider = self.model_registry.create_provider(provider_name, model_name)
        pipeline = BasePolicyPipeline(
            provider=provider,
        )
        pipeline_context = RequestContext(
            request_id=request_id,
            operation=context.operation,
            prompt=context.prompt,
            deadline=deadline,
        )
        return await pipeline.execute(pipeline_context)

    def _resolve_stream(
        self,
        context: ResolveContext,
        request_id: str,
        deadline: float,
    ) -> RESOLVER_RESULT:
        async def stream() -> AsyncIterable[str]:
            async with asyncio.timeout_at(deadline):
                failures: dict[str, Exception] = {}
                for provider_name, model_name in self.model_registry.get_models(
                    context.model
                ):
                    key = f"{provider_name}:{model_name}"
                    emitted = False
                    try:
                        result = await self._execute(
                            context,
                            request_id,
                            deadline,
                            provider_name,
                            model_name,
                        )
                        async for chunk in cast(AsyncIterable[str], result):
                            emitted = True
                            yield chunk
                        return
                    except LLMProviderError as error:
                        if emitted or not self._can_fallback(error):
                            raise
                        failures[key] = error
                raise AllModelsUnavailableError(failures)

        return stream()

    @staticmethod
    def _can_fallback(error: LLMProviderError) -> bool:
        return not isinstance(error, (InvalidRequestError, ContentPolicyError))
