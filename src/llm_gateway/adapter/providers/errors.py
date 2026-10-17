"""Provider errors and SDK error translation."""

from typing import Literal, NoReturn

import anthropic
import httpx
import openai


Operation = Literal["generation", "streaming", "extraction"]


class LLMProviderError(Exception):
    """Base error exposed by an LLM provider."""

    category = "provider_error"
    operation: Operation = "generation"

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        operation: Operation | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.operation = operation or type(self).operation
        self.status_code = status_code
        self.retryable = retryable


class AIGenerationError(LLMProviderError):
    """Base class for errors during complete-response generation."""

    operation: Operation = "generation"


class AIStreamingError(LLMProviderError):
    """Base class for errors during streaming."""

    operation: Operation = "streaming"


class AIExtractionError(LLMProviderError):
    """Base class for errors during structured extraction."""

    operation: Operation = "extraction"


class InvalidRequestError(LLMProviderError):
    """The provider rejected the request as invalid."""

    category = "invalid_request"


class AuthenticationError(LLMProviderError):
    """The provider rejected the supplied credentials."""

    category = "authentication"


class PermissionDeniedError(LLMProviderError):
    """The credentials do not have permission to use the requested resource."""

    category = "permission_denied"


class ModelNotFoundError(LLMProviderError):
    """The requested model is unavailable or does not exist."""

    category = "model_not_found"


class RateLimitError(LLMProviderError):
    """The provider rejected the request because of a rate or quota limit."""

    category = "rate_limit"


class ProviderTimeoutError(LLMProviderError):
    """The provider request timed out."""

    category = "timeout"


class ProviderConnectionError(LLMProviderError):
    """A connection to the provider could not be established."""

    category = "connection"


class ProviderUnavailableError(LLMProviderError):
    """The provider is temporarily unable to serve the request."""

    category = "provider_unavailable"


class ProviderResponseError(LLMProviderError):
    """The provider returned an invalid or unexpected response."""

    category = "invalid_response"


class ContentPolicyError(LLMProviderError):
    """The provider refused the request because of a content policy."""

    category = "content_policy"


class ProviderReportedError(Exception):
    """A structured failure event returned in a successful provider response."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class AllModelsUnavailableError(Exception):
    """All configured models failed to handle the request."""

    def __init__(self, failures: dict[str, Exception]) -> None:
        self.failures = dict(failures)
        details = "; ".join(
            f"{model}: {error}" for model, error in self.failures.items()
        )
        super().__init__(f"All configured models are unavailable: {details}")


_OPERATION_ERRORS: dict[Operation, type[LLMProviderError]] = {
    "generation": AIGenerationError,
    "streaming": AIStreamingError,
    "extraction": AIExtractionError,
}

_CATEGORY_ERRORS: dict[str, type[LLMProviderError]] = {
    error.category: error
    for error in (
        InvalidRequestError,
        AuthenticationError,
        PermissionDeniedError,
        ModelNotFoundError,
        RateLimitError,
        ProviderTimeoutError,
        ProviderConnectionError,
        ProviderUnavailableError,
        ProviderResponseError,
        ContentPolicyError,
    )
}


def _typed_error_class(
    operation: Operation,
    category_error: type[LLMProviderError],
) -> type[LLMProviderError]:
    operation_name = operation.title().replace("_", "")
    class_name = f"{operation_name}{category_error.__name__}"
    error_class = type(
        class_name,
        (category_error, _OPERATION_ERRORS[operation]),
        {"__module__": __name__},
    )
    globals()[class_name] = error_class
    return error_class


_TYPED_ERRORS: dict[tuple[Operation, str], type[LLMProviderError]] = {
    (operation, category): _typed_error_class(operation, error_class)
    for operation in _OPERATION_ERRORS
    for category, error_class in _CATEGORY_ERRORS.items()
}


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text.strip()
        if body:
            return f"{exc} ({body})"
    return str(exc)


def _category_for(
    provider: str,
    exc: Exception,
    status_code: int | None,
) -> tuple[str, bool]:
    code = getattr(exc, "code", None)
    if code == "rate_limit_exceeded":
        return "rate_limit", True
    if code == "vector_store_timeout":
        return "timeout", True
    if code == "server_error":
        return "provider_unavailable", True
    if code in {
        "bio_policy",
        "misalignment_policy_violation",
        "image_content_policy_violation",
    }:
        return "content_policy", False
    if code in {
        "invalid_prompt",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
    }:
        return "invalid_request", False

    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return "authentication", False
    if isinstance(exc, (openai.PermissionDeniedError, anthropic.PermissionDeniedError)):
        return "permission_denied", False
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return "rate_limit", True
    if isinstance(
        exc,
        (openai.APITimeoutError, anthropic.APITimeoutError, httpx.TimeoutException),
    ):
        return "timeout", True
    if isinstance(
        exc,
        (openai.APIConnectionError, anthropic.APIConnectionError, httpx.TransportError),
    ):
        return "connection", True

    detail = _error_detail(exc).lower()
    if provider == "ollama" and (
        "model not found" in detail or "model does not exist" in detail
    ):
        return "model_not_found", False

    if status_code == 401:
        return "authentication", False
    if status_code == 403:
        return "permission_denied", False
    if status_code == 404:
        return "model_not_found", False
    if status_code == 429:
        return "rate_limit", True
    if status_code in (408, 504):
        return "timeout", True
    if status_code in (400, 413, 422):
        return "invalid_request", False
    if status_code is not None and status_code >= 500:
        return "provider_unavailable", True
    if isinstance(exc, (openai.BadRequestError, anthropic.BadRequestError)):
        return "invalid_request", False
    if isinstance(exc, (openai.NotFoundError, anthropic.NotFoundError)):
        return "model_not_found", False
    return "invalid_response", False


def raise_provider_error(
    provider: str,
    operation: Operation,
    exc: Exception,
) -> NoReturn:
    """Translate a provider SDK/HTTP exception into a typed gateway error."""
    if isinstance(exc, LLMProviderError):
        raise exc

    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code

    category, retryable = _category_for(provider, exc, status_code)
    error_class = _TYPED_ERRORS[(operation, category)]
    raise error_class(
        f"{provider} {operation} failed: {_error_detail(exc)}",
        provider=provider,
        operation=operation,
        status_code=status_code,
        retryable=retryable,
    ) from exc


__all__ = [
    "Operation",
    "LLMProviderError",
    "AIGenerationError",
    "AIStreamingError",
    "AIExtractionError",
    "InvalidRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "ModelNotFoundError",
    "RateLimitError",
    "ProviderTimeoutError",
    "ProviderConnectionError",
    "ProviderUnavailableError",
    "ProviderResponseError",
    "ContentPolicyError",
    "ProviderReportedError",
    "AllModelsUnavailableError",
    "raise_provider_error",
    *(
        f"{operation.title().replace('_', '')}{error.__name__}"
        for operation in _OPERATION_ERRORS
        for error in _CATEGORY_ERRORS.values()
    ),
]