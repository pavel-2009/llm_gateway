"""Retry limits, backoff, and optional provider fallback selection."""

import random

from llm_gateway.core.config import config
from llm_gateway.policy.error_policy import ErrorPolicy


class RetryPolicy:
    """Control retry count, delay, and optional fallback models."""

    def __init__(
        self,
        *,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        jitter: float | None = None,
        fallback_models: tuple[str, ...] = (),
    ) -> None:
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        self.base_delay = (
            config.RETRY_BASE_DELAY if base_delay is None else base_delay
        )
        self.max_delay = config.RETRY_MAX_DELAY if max_delay is None else max_delay
        self.jitter = config.RETRY_JITTER if jitter is None else jitter
        self.fallback_models = fallback_models

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("Retry delays must be non-negative")

    def should_retry(
        self,
        error: Exception,
        retries_made: int,
        *,
        error_policy: ErrorPolicy,
    ) -> bool:
        """Combine error eligibility with the configured retry limit."""
        return (
            error_policy.should_retry(error)
            and retries_made < self.max_retries
        )

    def get_backoff_time(self, retry_number: int) -> float:
        """Return capped exponential backoff with additive jitter."""
        exponential_delay = self.base_delay * 2 ** max(0, retry_number - 1)
        delay = exponential_delay + random.uniform(0, self.jitter)
        return min(self.max_delay, delay)

    def get_next_model(
        self,
        current_model: str | None,
        primary_model: str | None,
        failed_models: set[str] | None = None,
    ) -> str | None:
        """Select an untried model, or retry the only configured model."""
        candidates = self.get_models(primary_model)
        if not candidates:
            return current_model
        for model in candidates:
            if model not in (failed_models or set()):
                return model
        if len(candidates) == 1:
            return candidates[0]
        return current_model

    def get_models(self, primary_model: str | None) -> tuple[str, ...]:
        """Return configured models in primary-then-fallback order."""
        return tuple(dict.fromkeys(
            model
            for model in (primary_model, *self.fallback_models)
            if model is not None
        ))
