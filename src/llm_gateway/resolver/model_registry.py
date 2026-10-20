"""Registry for configured models and their provider adapters."""

from llm_gateway.adapter.contracts import Providers
from llm_gateway.adapter.errors import AuthenticationError
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.adapter.providers.claude_provider import ClaudeProvider
from llm_gateway.adapter.providers.ollama_provider import OllamaProvider
from llm_gateway.adapter.providers.openai_provider import OpenAIProvider
from llm_gateway.core.config import config
from llm_gateway.policy.request_context import OperationType
from llm_gateway.resolver.resolver_context import ModelTier


class ModelRegistry:
    """Maps model tiers to configured models and provider adapters."""

    def get_models(self, tier: ModelTier) -> tuple[tuple[str, str], ...]:
        """Return the configured model for this tier and its fallbacks."""
        models = {
            ModelTier.SMART: (
                (Providers.CLAUDE, config.CLAUDE_MODEL),
                (Providers.OPENAI, config.OPENAI_MODEL),
                (Providers.OLLAMA, config.OLLAMA_MODEL),
            ),
            ModelTier.MEDIUM: (
                (Providers.OPENAI, config.OPENAI_MODEL),
                (Providers.OLLAMA, config.OLLAMA_MODEL),
            ),
            ModelTier.LIGHT: ((Providers.OLLAMA, config.OLLAMA_MODEL),),
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

    def check_provider_configuration(
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
