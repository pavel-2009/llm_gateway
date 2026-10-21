"""LLM gateway class."""

from collections.abc import AsyncIterable, AsyncIterator

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.policy.request_context import OperationType
from llm_gateway.resolver.model_registry import ModelRegistry
from llm_gateway.resolver.model_resolver import ModelResolver
from llm_gateway.resolver.resolver_context import (
    ModelTier,
    ResolveContext,
    ResolveParams,
)


class LLMGateway:
    """LLM gateway."""

    def __init__(self) -> None:
        self.resolver = ModelResolver(ModelRegistry())

    async def generate(self, prompt: str) -> GenerateResponse:
        result = await self.resolver.resolve(
            self._build_context(prompt, OperationType.GENERATE)
        )
        if not isinstance(result, GenerateResponse):
            raise TypeError("resolver returned an invalid generation result")
        return result

    async def extract(self, prompt: str) -> ExtractedReview:
        result = await self.resolver.resolve(
            self._build_context(prompt, OperationType.EXTRACT)
        )
        if not isinstance(result, ExtractedReview):
            raise TypeError("resolver returned an invalid extraction result")
        return result

    def stream(self, prompt: str) -> AsyncIterator[str]:
        async def iterate() -> AsyncIterator[str]:
            result = await self.resolver.resolve(
                self._build_context(prompt, OperationType.STREAM)
            )
            if not isinstance(result, AsyncIterable):
                raise TypeError("resolver returned an invalid stream result")
            async for chunk in result:
                yield chunk

        return iterate()

    @staticmethod
    def _build_context(prompt: str, operation: OperationType) -> ResolveContext:
        timeout = (
            config.STREAM_TIMEOUT
            if operation == OperationType.STREAM
            else config.REQUEST_TIMEOUT
        )
        return ResolveContext(
            model=ModelTier.SMART,
            operation=operation,
            prompt=prompt,
            params=ResolveParams(deadline=timeout),
        )
