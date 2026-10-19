"""Model resolver"""

from collections.abc import AsyncIterable
import time
from uuid import uuid4

from llm_gateway.adapter.contracts import GenerateResponse, Providers
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.adapter.providers.claude_provider import ClaudeProvider
from llm_gateway.adapter.providers.openai_provider import OpenAIProvider
from llm_gateway.adapter.providers.ollama_provider import OllamaProvider
from llm_gateway.policy.base_pipeline import BasePolicyPipeline
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.retry_policy import RetryPolicy
from llm_gateway.policy.request_context import RequestContext
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
        return models[tier]

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
        request_id = uuid4().hex
        operation = context.operation
        prompt = context.prompt
        timeout = context.params.deadline or config.REQUEST_TIMEOUT
        provider_name, model_name = self.select_model(context.model)
        provider = self.create_provider(provider_name, model_name)

        pipeline_context = RequestContext(
            request_id=request_id,
            operation=operation,
            prompt=prompt,
            deadline=time.monotonic() + timeout,
        )

        error_policy = ErrorPolicy()
        retry_policy = RetryPolicy()

        pipeline = BasePolicyPipeline(
            error_policy=error_policy,
            retry_policy=retry_policy,
            provider=provider,
        )

        return await pipeline.execute(pipeline_context)
