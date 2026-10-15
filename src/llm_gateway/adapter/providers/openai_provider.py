"""OpenAI model provider."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import config
from llm_gateway.adapter.contracts import (
    EXTRACT_SCHEMA,
    EXTRACT_SYSTEM_PROMPT,
    ExtractedReview,
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.errors import (
    ProviderReportedError,
    raise_provider_error,
)


class OpenAIProvider(LLMProvider):
    """OpenAI model provider"""

    def __init__(self, 
                api_key: str = config.OPENAI_API_KEY,
                model: str = config.OPENAI_MODEL
                ):
        self.api_key = api_key
        self.model = model

        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str) -> GenerateResponse:
        """Generate text from a prompt"""
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=prompt,
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
                input=prompt,
                stream=True
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
                instructions=EXTRACT_SYSTEM_PROMPT,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "extracted_review",
                        "strict": True,
                        "schema": EXTRACT_SCHEMA,
                    }
                }
            )

            return ExtractedReview.model_validate_json(
                response.output_text
            )

        except Exception as e:
            raise_provider_error(Providers.OPENAI, "extraction", e)
