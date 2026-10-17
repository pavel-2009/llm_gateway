"""Claude model provider."""

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from llm_gateway.adapter.contracts import (
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.providers.errors import (
    raise_provider_error,
)
from llm_gateway.adapter.prompt import build_prompt
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import (
    EXTRACT_SCHEMA,
    ExtractedReview,
    config,
)


class ClaudeProvider(LLMProvider):
    """Claude model provider."""

    def __init__(
        self,
        api_key: str | None = config.CLAUDE_API_KEY,
        model: str = config.CLAUDE_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("CLAUDE_API_KEY is required to use ClaudeProvider")
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str) -> GenerateResponse:
        """Generate text from a prompt."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                **build_prompt(prompt, None, False, Providers.CLAUDE),
            )

            text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            )
            return GenerateResponse(
                text=text,
                provider=Providers.CLAUDE,
                model=self.model,
            )
        except Exception as exc:
            raise_provider_error(Providers.CLAUDE, "generation", exc)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text from Claude."""
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                **build_prompt(prompt, None, True, Providers.CLAUDE),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise_provider_error(Providers.CLAUDE, "streaming", exc)

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                **build_prompt(prompt, EXTRACT_SCHEMA, False, Providers.CLAUDE),
            )

            text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            )
            return ExtractedReview.model_validate_json(text)
        except Exception as exc:
            raise_provider_error(Providers.CLAUDE, "extraction", exc)
