"""LLM gateqway class"""

from collections.abc import AsyncIterable

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.resolver.resolver_context import ResolveContext, ModelTier, ResolveParams
from llm_gateway.core.config import ExtractedReview, config
from llm_gateway.policy.request_context import OperationType


class LLMGateway:
    """LLM gateway"""

    async def generate(self, prompt: str) -> GenerateResponse:
        model = ModelTier.SMART
        operation = OperationType.GENERATE
        params = ResolveParams(deadline=config.REQUEST_TIMEOUT)

        context = ResolveContext(
            model=model,
            operation=operation,
            prompt=prompt,
            params=params
        )

    async def extract(self, prompt: str) -> ExtractedReview:
        ...

    async def stream(self, prompt: str) -> AsyncIterable[str]:
        ...
