"""Model resolver"""

from collections.abc import AsyncIterable
import asyncio
import logging
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
from llm_gateway.policy.request_context import OperationType, RequestContext
from llm_gateway.resolver.model_registry import ModelRegistry
from llm_gateway.resolver.resolver_context import ResolveContext
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.core.logging_context import request_id_context


RESOLVER_RESULT = GenerateResponse | ExtractedReview | AsyncIterable[str]
logger = logging.getLogger(__name__)


class ModelResolver:
    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()

    async def resolve(self, context: ResolveContext) -> RESOLVER_RESULT:
        if not isinstance(context, ResolveContext):
            raise TypeError("context must be a ResolveContext")

        request_id = request_id_context.get() or uuid4().hex
        timeout = (
            context.params.deadline
            if context.params.deadline is not None
            else (
                config.STREAM_TIMEOUT
                if context.operation == OperationType.STREAM
                else config.REQUEST_TIMEOUT
            )
        )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("request timeout must be a positive finite number")
        deadline = time.monotonic() + timeout
        models = self.model_registry.get_models(context.model)
        logger.info(
            "Resolver started request_id=%s operation=%s tier=%s "
            "candidate_count=%s timeout_seconds=%.1f",
            request_id,
            context.operation.value,
            context.model.value,
            len(models),
            timeout,
        )

        if context.operation == OperationType.STREAM:
            return self._resolve_stream(context, request_id, deadline)

        try:
            async with asyncio.timeout_at(deadline):
                failures: dict[str, Exception] = {}
                for provider_name, model_name in models:
                    key = f"{provider_name}:{model_name}"
                    try:
                        result = await self._execute(
                            context,
                            request_id,
                            deadline,
                            provider_name,
                            model_name,
                        )
                        logger.info(
                            "Resolver selected request_id=%s provider=%s model=%s",
                            request_id,
                            provider_name,
                            model_name,
                        )
                        return result
                    except LLMProviderError as error:
                        if not self._can_fallback(error):
                            logger.error(
                                "Resolver stopped request_id=%s provider=%s "
                                "model=%s error_type=%s fallback=false",
                                request_id,
                                provider_name,
                                model_name,
                                type(error).__name__,
                            )
                            raise
                        logger.warning(
                            "Resolver provider failed request_id=%s provider=%s "
                            "model=%s error_type=%s fallback=true",
                            request_id,
                            provider_name,
                            model_name,
                            type(error).__name__,
                        )
                        failures[key] = error

                logger.error(
                    "Resolver exhausted candidates request_id=%s "
                    "failed_candidate_count=%s",
                    request_id,
                    len(failures),
                )
                raise AllModelsUnavailableError(failures)

        except asyncio.TimeoutError:
            logger.error(
                "Resolver timed out request_id=%s operation=%s tier=%s "
                "candidate_count=%s timeout_seconds=%.1f",
                request_id,
                context.operation.value,
                context.model.value,
                len(models),
                timeout,
            )
            raise

    async def _execute(
        self,
        context: ResolveContext,
        request_id: str,
        deadline: float,
        provider_name: str,
        model_name: str,
    ) -> RESOLVER_RESULT:
        logger.info(
            "Resolver executing request_id=%s operation=%s provider=%s model=%s",
            request_id,
            context.operation.value,
            provider_name,
            model_name,
        )
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
                models = self.model_registry.get_models(context.model)
                for provider_name, model_name in models:
                    key = f"{provider_name}:{model_name}"
                    emitted = False
                    try:
                        logger.info(
                            "Resolver stream candidate started request_id=%s "
                            "provider=%s model=%s",
                            request_id,
                            provider_name,
                            model_name,
                        )
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
                        logger.info(
                            "Resolver stream candidate completed "
                            "request_id=%s provider=%s model=%s",
                            request_id,
                            provider_name,
                            model_name,
                        )
                        return
                    except LLMProviderError as error:
                        if emitted or not self._can_fallback(error):
                            logger.error(
                                "Resolver stream stopped request_id=%s "
                                "provider=%s model=%s error_type=%s "
                                "fallback=false emitted=%s",
                                request_id,
                                provider_name,
                                model_name,
                                type(error).__name__,
                                emitted,
                            )
                            raise
                        logger.warning(
                            "Resolver stream provider failed request_id=%s "
                            "provider=%s model=%s error_type=%s fallback=true",
                            request_id,
                            provider_name,
                            model_name,
                            type(error).__name__,
                        )
                        failures[key] = error
                logger.error(
                    "Resolver stream exhausted candidates request_id=%s "
                    "failed_candidate_count=%s",
                    request_id,
                    len(failures),
                )
                raise AllModelsUnavailableError(failures)

        return stream()

    @staticmethod
    def _can_fallback(error: LLMProviderError) -> bool:
        return not isinstance(error, (InvalidRequestError, ContentPolicyError))
