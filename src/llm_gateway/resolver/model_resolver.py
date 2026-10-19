"""Model resolver"""

from collections.abc import AsyncIterable
from uuid import uuid4

from llm_gateway.policy.base_pipeline import BasePolicyPipeline
from llm_gateway.policy.error_policy import ErrorPolicy
from llm_gateway.policy.retry_policy import RetryPolicy
from llm_gateway.policy.request_context import RequestContext
from llm_gateway.resolver.resolver_context import ResolveContext
from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview
from llm_gateway.core.config import config
from llm_gateway.adapter.providers.ollama_provider import OllamaProvider


RESOLVER_RESULT = GenerateResponse | ExtractedReview | AsyncIterable[str]


class ModelResolver:

    async def resolve(self, context: ResolveContext):
        request_id = uuid4().hex
        operation = context.operation
        prompt = context.prompt
        deadline = context.params.get("deadline", config.REQUEST_TIMEOUT)

        pipeline_context = RequestContext(
            request_id=request_id,
            operation=operation,
            prompt=prompt,
            deadline=deadline
        )

        error_policy = ErrorPolicy()
        retry_policy = RetryPolicy()

        llm_provider = OllamaProvider()

        pipeline = BasePolicyPipeline(
            error_policy=error_policy,
            retry_policy=retry_policy,
            llm_provider=llm_provider
        )

        try:
            result = await pipeline.execute(pipeline_context)
            return result

        except Exception as e:
            raise e
