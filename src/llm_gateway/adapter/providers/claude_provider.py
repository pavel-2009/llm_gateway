"""Claude model provider."""

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from llm_gateway.adapter.contracts import (
    EXTRACT_SCHEMA,
    EXTRACT_SYSTEM_PROMPT,
    ExtractedReview,
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.errors import (
    AIExtractionError,
    AIGenerationError,
    AIStreamingError,
)
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import config


class ClaudeProvider(LLMProvider):
    """Claude model provider."""

    def __init__(
        self,
        api_key: str = config.CLAUDE_API_KEY,
        model: str = config.CLAUDE_MODEL,
    ) -> None:
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str) -> GenerateResponse:
        """Generate text from a prompt."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
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
            raise AIGenerationError(
                f"Claude generation failed: {exc}"
            ) from exc

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text from Claude."""
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise AIStreamingError(
                f"Claude streaming failed: {exc}"
            ) from exc

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                system=EXTRACT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": EXTRACT_SCHEMA,
                    }
                },
            )

            text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            )
            return ExtractedReview.model_validate_json(text)
        except Exception as exc:
            raise AIExtractionError(
                f"Claude extraction failed: {exc}"
            ) from exc
