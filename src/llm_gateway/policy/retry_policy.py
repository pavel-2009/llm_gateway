"""Retry limits and backoff."""

import random

from llm_gateway.core.config import config
from llm_gateway.policy.error_policy import ErrorPolicy


class RetryPolicy:
    """Control retry count and delay."""

    def __init__(
        self,
        *,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        jitter: float | None = None,
    ) -> None:
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        self.base_delay = (
            config.RETRY_BASE_DELAY if base_delay is None else base_delay
        )
        self.max_delay = config.RETRY_MAX_DELAY if max_delay is None else max_delay
        self.jitter = config.RETRY_JITTER if jitter is None else jitter
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
