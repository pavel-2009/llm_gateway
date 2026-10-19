"""Model resolver"""

from collections.abc import AsyncIterable
from math import isfinite
import time
from typing import cast
from uuid import uuid4

from llm_gateway.adapter.contracts import GenerateResponse, Providers
from llm_gateway.adapter.errors import (
    AllModelsUnavailableError,
    AuthenticationError,
    ContentPolicyError,
    InvalidRequestError,
    LLMProviderError,
)
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.adapter.providers.claude_provider import ClaudeProvider
from llm_gateway.adapter.providers.openai_provider import OpenAIProvider
from llm_gateway.adapter.providers.ollama_provider import OllamaProvider
from llm_gateway.policy.base_pipeline import BasePolicyPipeline
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.retry_policy import RetryPolicy
from llm_gateway.policy.request_context import OperationType, RequestContext
from llm_gateway.resolver.resolver_context import ModelTier, ResolveContext
from llm_gateway.core.config import ExtractedReview, config


RESOLVER_RESULT = GenerateResponse | ExtractedReview | AsyncIterable[str]


class ModelResolver:
    def select_model(self, tier: ModelTier) -> tuple[str, str]:
        """Return the provider and configured model for a semantic tier."""
        models = {
            ModelTier.SMART: (Providers.CLAUDE, config.CLAUDE_MODEL),
            ModelTier.MEDIUM: (Providers.OPENAI, config.OPENAI_MODEL),
            ModelTier.LIGHT: (Providers.OLLAMA, config.OLLAMA_MODEL),
        }
        try:
            return models[tier]
        except KeyError:
            raise ValueError(f"Unsupported model tier: {tier}") from None

    def create_provider(self, provider: str, model: str) -> LLMProvider:
        """Create the provider adapter for a selected model."""
        providers = {
            Providers.CLAUDE: ClaudeProvider,
            Providers.OPENAI: OpenAIProvider,
            Providers.OLLAMA: OllamaProvider,
        }
        try:
            provider_type = providers[provider]
        except KeyError:
            raise ValueError(f"Unsupported provider: {provider}") from None
        return provider_type(model=model)

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

        failures: dict[str, Exception] = {}
        for tier in self._fallback_tiers(context.model):
            provider_name, model_name = self.select_model(tier)
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
        self._check_provider_configuration(provider_name, context.operation)
        provider = self.create_provider(provider_name, model_name)
        pipeline = BasePolicyPipeline(
            error_policy=ErrorPolicy(),
            retry_policy=RetryPolicy(),
            provider=provider,
        )
        pipeline_context = RequestContext(
            request_id=request_id,
            operation=context.operation,
            prompt=context.prompt,
            deadline=deadline,
        )
        return await pipeline.execute(pipeline_context)

    def _check_provider_configuration(
        self,
        provider_name: str,
        operation: OperationType,
    ) -> None:
        credentials = {
            Providers.CLAUDE: (config.CLAUDE_API_KEY, "CLAUDE_API_KEY"),
            Providers.OPENAI: (config.OPENAI_API_KEY, "OPENAI_API_KEY"),
        }.get(provider_name)
        if credentials is None or credentials[0]:
            return

        operation_name = {
            OperationType.GENERATE: "generation",
            OperationType.STREAM: "streaming",
            OperationType.EXTRACT: "extraction",
        }[operation]
        raise AuthenticationError(
            f"{credentials[1]} is required to use {provider_name}",
            provider=provider_name,
            operation=operation_name,
        )

    def _resolve_stream(
        self,
        context: ResolveContext,
        request_id: str,
        deadline: float,
    ) -> RESOLVER_RESULT:
        async def stream() -> AsyncIterable[str]:
            failures: dict[str, Exception] = {}
            for tier in self._fallback_tiers(context.model):
                provider_name, model_name = self.select_model(tier)
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
    def _fallback_tiers(tier: ModelTier) -> tuple[ModelTier, ...]:
        fallbacks = {
            ModelTier.SMART: (ModelTier.SMART, ModelTier.MEDIUM, ModelTier.LIGHT),
            ModelTier.MEDIUM: (ModelTier.MEDIUM, ModelTier.LIGHT),
            ModelTier.LIGHT: (ModelTier.LIGHT,),
        }
        return fallbacks[tier]

    @staticmethod
    def _can_fallback(error: LLMProviderError) -> bool:
        return not isinstance(error, (InvalidRequestError, ContentPolicyError))
