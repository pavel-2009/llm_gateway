"""Base handling pipeline for LLM Gateway."""

from typing import Callable, Awaitable, TypeVar
import asyncio

from src.llm_gateway.adapter.errors import
from src.llm_gateway.core.config import config
from src.llm_gateway.policy.request_context import RequestContext


T = TypeVar('T')

class BasePolicyPipeline:
    """policy pipeline"""

    def __init__(
            self, 
            error_policy: ErrorPolicy,
            retry_policy: RetryPolicy
        ) -> None:
        self.error_policy = error_policy
        self.retry_policy = retry_policy

    async def execute(
            self,
            operation: Callable[[], Awaitable[T]],
            context: RequestContext
        ) -> T:
        """Execute the operation with error and retry policies."""

        if context.is_expired:
            raise TimeoutError("Request deadline exceeded")

        context.primary_attempts += 1

        async with asyncio.timeout(context.remaining_time):
            return await operation()
        