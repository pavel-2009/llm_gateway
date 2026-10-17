"""Base handling pipeline for LLM Gateway."""

from typing import Callable, Awaitable, TypeVar
import asyncio
from enum import Enum
from xml.parsers.expat import model

from src.llm_gateway.adapter.errors import
from src.llm_gateway.core.config import config
from src.llm_gateway.policy.request_context import OperationType, RequestContext
from src.llm_gateway.adapter.providers import OllamaProvider, ClaudeProvider, OpenAIProvider


class ModelType(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"


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
            context: RequestContext
        ) -> T:
        """Execute the operation with error and retry policies."""

        if context.is_expired:
            raise TimeoutError("Request deadline exceeded")

        context.primary_attempts += 1

        prompt = context.prompt
        operation = context.operation
        primary_model = context.primary_model
        current_model = context.current_model

        while True:
            try:
                result = await self._execute_operation(prompt, operation, current_model)

                return result

            except Exception as e:
                if not self.error_policy.should_retry(e):
                    raise e

                context.fallback_attempts += 1

                if context.is_expired:
                    raise TimeoutError("Request deadline exceeded")

                current_model = self.retry_policy.get_next_model(current_model, primary_model)
                await asyncio.sleep(self.retry_policy.get_backoff_time(context.fallback_attempts))

    async def _execute_operation(
            self,
            prompt: str,
            operation: OperationType,
            model: ModelType
        ) -> T:
        """Execute the operation with the given prompt and model."""

        if model == ModelType.OPENAI:
            provider = OpenAIProvider()
        elif model == ModelType.OLLAMA:
            provider = OllamaProvider()
        elif model == ModelType.CLAUDE:
            provider = ClaudeProvider()
        else:
            raise ValueError(f"Unsupported model: {model}")

        try:

            if operation == OperationType.GENERATE:
                return await provider.generate(prompt)
            elif operation == OperationType.STREAM:
                return await provider.stream(prompt)
            elif operation == OperationType.EXTRACT:
                return await provider.extract(prompt)

        except Exception as e:
            raise e
        