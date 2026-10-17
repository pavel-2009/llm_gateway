"""OpenAI model provider."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.adapter.contracts import (
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.providers.errors import (
    ProviderReportedError,
    raise_provider_error,
)
from llm_gateway.adapter.prompt import build_prompt
from llm_gateway.core.config import (
    EXTRACT_SCHEMA,
    ExtractedReview,
    config,
)


class OpenAIProvider(LLMProvider):
    """OpenAI model provider"""

    def __init__(self, 
                api_key: str | None = config.OPENAI_API_KEY,
                model: str = config.OPENAI_MODEL
                ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to use OpenAIProvider")
        self.api_key = api_key
        self.model = model

        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str) -> GenerateResponse:
        """Generate text from a prompt"""
        try:
            response = await self.client.responses.create(
                model=self.model,
                **build_prompt(prompt, None, False, Providers.OPENAI),
            )
    
            return GenerateResponse(
                text=response.output_text,
                provider=Providers.OPENAI,
                model=self.model
            )

        except Exception as e:
            raise_provider_error(Providers.OPENAI, "generation", e)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream text from a prompt"""
        try:
            stream = await self.client.responses.create(
                model=self.model,
                **build_prompt(prompt, None, True, Providers.OPENAI),
            )

            async for chunk in stream:
                if chunk.type == "response.output_text.delta":
                    yield chunk.delta

                elif chunk.type == "response.failed":
                    error = chunk.response.error
                    raise ProviderReportedError(
                        error.message if error else "OpenAI returned a failed response",
                        code=error.code if error else None,
                    )
        except Exception as e:
            raise_provider_error(Providers.OPENAI, "streaming", e)

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt"""
        try:
            response = await self.client.responses.create(
                model=self.model,
                **build_prompt(prompt, EXTRACT_SCHEMA, False, Providers.OPENAI),
            )

            return ExtractedReview.model_validate_json(
                response.output_text
            )

        except Exception as e:
            raise_provider_error(Providers.OPENAI, "extraction", e)
