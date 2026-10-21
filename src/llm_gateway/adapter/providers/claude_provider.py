"""Claude model provider."""

from collections.abc import AsyncIterator
import logging

from anthropic import AsyncAnthropic

from llm_gateway.adapter.contracts import (
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.errors import (
    raise_provider_error,
)
from llm_gateway.adapter.prompt import build_prompt
from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import (
    EXTRACT_SCHEMA,
    ExtractedReview,
    config,
)
from llm_gateway.core.logging_context import request_id_context


logger = logging.getLogger(__name__)


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
        request_id = request_id_context.get()
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=generation model=%s",
            request_id,
            Providers.CLAUDE,
            self.model,
        )
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
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=generation output_chars=%s",
                request_id,
                Providers.CLAUDE,
                len(text),
            )
            return GenerateResponse(
                text=text,
                provider=Providers.CLAUDE,
                model=self.model,
            )
        except Exception as exc:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=generation error_type=%s",
                request_id,
                Providers.CLAUDE,
                type(exc).__name__,
            )
            raise_provider_error(Providers.CLAUDE, "generation", exc)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text from Claude."""
        request_id = request_id_context.get()
        chunks = 0
        output_chars = 0
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=streaming model=%s",
            request_id,
            Providers.CLAUDE,
            self.model,
        )
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                **build_prompt(prompt, None, True, Providers.CLAUDE),
            ) as stream:
                async for text in stream.text_stream:
                    chunks += 1
                    output_chars += len(text)
                    yield text
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=streaming chunks=%s output_chars=%s",
                request_id,
                Providers.CLAUDE,
                chunks,
                output_chars,
            )
        except Exception as exc:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=streaming chunks=%s error_type=%s",
                request_id,
                Providers.CLAUDE,
                chunks,
                type(exc).__name__,
            )
            raise_provider_error(Providers.CLAUDE, "streaming", exc)

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt."""
        request_id = request_id_context.get()
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=extraction model=%s",
            request_id,
            Providers.CLAUDE,
            self.model,
        )
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
            result = ExtractedReview.model_validate_json(text)
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=extraction",
                request_id,
                Providers.CLAUDE,
            )
            return result
        except Exception as exc:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=extraction error_type=%s",
                request_id,
                Providers.CLAUDE,
                type(exc).__name__,
            )
            raise_provider_error(Providers.CLAUDE, "extraction", exc)
