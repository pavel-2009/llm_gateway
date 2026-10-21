"""LLM gateqway class"""

from collections.abc import AsyncIterable

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.resolver.resolver_context import ResolveContext
from llm_gateway.core.config import ExtractedReview, config


class LLMGateway:
    """LLM gateway"""

    async def generate(self, prompt: str) -> GenerateResponse:
        ...

    async def extract(self, prompt: str) -> ExtractedReview:
        ...

    async def stream(self, prompt: str) -> AsyncIterable[str]:
        ...
