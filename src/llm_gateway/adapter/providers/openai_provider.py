"""OpenAI model provider."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.core.config import config
from llm_gateway.adapter.contracts import GenerateResponse, ExtractedReview, Providers, EXTRACT_SYSTEM_PROMPT, EXTRACT_SCHEMA


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
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return GenerateResponse(
            text=response.output_text,
            provider=Providers.OPENAI,
            model=self.model
        )

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream text from a prompt"""
        stream = await self.client.responses.create(
            model=self.model,
            input=prompt,
            stream=True
        )

        async for chunk in stream:
            if chunk.type == "response.output_text.delta":
                yield chunk.delta

            elif chunk.type == "response.failed":
                raise RuntimeError("OpenAI generation failed")

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt"""
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
            },
        )

        return ExtractedReview.model_validate_json(
            response.output_text
        )
