"""Error classification used by the policy pipeline."""


class ErrorPolicy:
    """Decide whether an exception is eligible for retry."""

    def __init__(
        self,
        retryable_errors: tuple[type[Exception], ...] = (),
    ) -> None:
        self.retryable_errors = retryable_errors

    def should_retry(self, error: Exception) -> bool:
        """Return whether the error is marked retryable."""
        if isinstance(error, self.retryable_errors):
            return True
        return bool(getattr(error, "retryable", False))