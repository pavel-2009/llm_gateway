"""OpenAI model provider."""

from collections.abc import AsyncIterator
import logging

from openai import AsyncOpenAI

from llm_gateway.adapter.provider import LLMProvider
from llm_gateway.adapter.contracts import (
    GenerateResponse,
    Providers,
)
from llm_gateway.adapter.errors import (
    ProviderReportedError,
    raise_provider_error,
)
from llm_gateway.adapter.prompt import build_prompt
from llm_gateway.core.config import (
    EXTRACT_SCHEMA,
    ExtractedReview,
    config,
)
from llm_gateway.core.logging_context import request_id_context
from llm_gateway.telemetry import http_client_span


logger = logging.getLogger(__name__)


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
        request_id = request_id_context.get()
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=generation model=%s",
            request_id,
            Providers.OPENAI,
            self.model,
        )
        try:
            with http_client_span(
                Providers.OPENAI,
                "generate",
                self.model,
                "https://api.openai.com/v1/responses",
            ):
                response = await self.client.responses.create(
                    model=self.model,
                    **build_prompt(prompt, None, False, Providers.OPENAI),
                )
    
            result = GenerateResponse(
                text=response.output_text,
                provider=Providers.OPENAI,
                model=self.model
            )
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=generation output_chars=%s",
                request_id,
                Providers.OPENAI,
                len(result.text),
            )
            return result

        except Exception as e:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=generation error_type=%s",
                request_id,
                Providers.OPENAI,
                type(e).__name__,
            )
            raise_provider_error(Providers.OPENAI, "generation", e)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream text from a prompt"""
        request_id = request_id_context.get()
        chunks = 0
        output_chars = 0
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=streaming model=%s",
            request_id,
            Providers.OPENAI,
            self.model,
        )
        try:
            with http_client_span(
                Providers.OPENAI,
                "stream",
                self.model,
                "https://api.openai.com/v1/responses",
            ):
                stream = await self.client.responses.create(
                    model=self.model,
                    **build_prompt(prompt, None, True, Providers.OPENAI),
                )

                async for chunk in stream:
                    if chunk.type == "response.output_text.delta":
                        chunks += 1
                        output_chars += len(chunk.delta)
                        yield chunk.delta

                    elif chunk.type == "response.failed":
                        error = chunk.response.error
                        raise ProviderReportedError(
                            error.message
                            if error
                            else "OpenAI returned a failed response",
                            code=error.code if error else None,
                        )
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=streaming chunks=%s output_chars=%s",
                request_id,
                Providers.OPENAI,
                chunks,
                output_chars,
            )
        except Exception as e:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=streaming chunks=%s error_type=%s",
                request_id,
                Providers.OPENAI,
                chunks,
                type(e).__name__,
            )
            raise_provider_error(Providers.OPENAI, "streaming", e)

    async def extract(self, prompt: str) -> ExtractedReview:
        """Extract structured data from a prompt"""
        request_id = request_id_context.get()
        logger.info(
            "Provider request started request_id=%s provider=%s "
            "operation=extraction model=%s",
            request_id,
            Providers.OPENAI,
            self.model,
        )
        try:
            with http_client_span(
                Providers.OPENAI,
                "extract",
                self.model,
                "https://api.openai.com/v1/responses",
            ):
                response = await self.client.responses.create(
                    model=self.model,
                    **build_prompt(prompt, EXTRACT_SCHEMA, False, Providers.OPENAI),
                )

            result = ExtractedReview.model_validate_json(
                response.output_text
            )
            logger.info(
                "Provider request completed request_id=%s provider=%s "
                "operation=extraction",
                request_id,
                Providers.OPENAI,
            )
            return result

        except Exception as e:
            logger.error(
                "Provider request failed request_id=%s provider=%s "
                "operation=extraction error_type=%s",
                request_id,
                Providers.OPENAI,
                type(e).__name__,
            )
            raise_provider_error(Providers.OPENAI, "extraction", e)
