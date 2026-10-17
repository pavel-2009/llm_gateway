"""Errors raised by the policy pipeline."""


class AllModelsUnavailableError(Exception):
    """All configured models failed to handle the request."""

    def __init__(self, failures: dict[str, Exception]) -> None:
        self.failures = dict(failures)
        details = "; ".join(
            f"{model}: {error}" for model, error in self.failures.items()
        )
        super().__init__(f"All configured models are unavailable: {details}")