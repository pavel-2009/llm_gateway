"""Provider errors and provider SDK error translation."""

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
	"""The credentials lack permission to use the requested resource."""

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

_CODE_ERRORS: tuple[tuple[set[str], type[LLMProviderError], bool], ...] = (
	({"rate_limit_exceeded"}, RateLimitError, True),
	({"vector_store_timeout"}, ProviderTimeoutError, True),
	({"server_error"}, ProviderUnavailableError, True),
	(
		{"bio_policy", "misalignment_policy_violation", "image_content_policy_violation"},
		ContentPolicyError,
		False,
	),
	(
		{
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
		},
		InvalidRequestError,
		False,
	),
)

_SDK_ERRORS: tuple[
	tuple[tuple[type[Exception], ...], type[LLMProviderError], bool], ...
] = (
	((openai.AuthenticationError, anthropic.AuthenticationError), AuthenticationError, False),
	((openai.PermissionDeniedError, anthropic.PermissionDeniedError), PermissionDeniedError, False),
	((openai.RateLimitError, anthropic.RateLimitError), RateLimitError, True),
	(
		(openai.APITimeoutError, anthropic.APITimeoutError, httpx.TimeoutException),
		ProviderTimeoutError,
		True,
	),
	(
		(openai.APIConnectionError, anthropic.APIConnectionError, httpx.TransportError),
		ProviderConnectionError,
		True,
	),
	((openai.BadRequestError, anthropic.BadRequestError), InvalidRequestError, False),
	((openai.NotFoundError, anthropic.NotFoundError), ModelNotFoundError, False),
)

_STATUS_ERRORS: dict[int, tuple[type[LLMProviderError], bool]] = {
	400: (InvalidRequestError, False),
	401: (AuthenticationError, False),
	403: (PermissionDeniedError, False),
	404: (ModelNotFoundError, False),
	408: (ProviderTimeoutError, True),
	413: (InvalidRequestError, False),
	422: (InvalidRequestError, False),
	429: (RateLimitError, True),
	504: (ProviderTimeoutError, True),
}


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

	code = getattr(exc, "code", None)
	category_error = None
	retryable = False

	for codes, error_type, can_retry in _CODE_ERRORS:
		if code in codes:
			category_error, retryable = error_type, can_retry
			break

	if category_error is None:
		for error_types, error_type, can_retry in _SDK_ERRORS:
			if isinstance(exc, error_types):
				category_error, retryable = error_type, can_retry
				break

	detail = str(exc)
	if isinstance(exc, httpx.HTTPStatusError) and exc.response.text.strip():
		detail = f"{exc} ({exc.response.text.strip()})"

	if category_error is None and provider == "ollama":
		lowered_detail = detail.lower()
		if "model not found" in lowered_detail or "model does not exist" in lowered_detail:
			category_error = ModelNotFoundError

	if category_error is None and status_code in _STATUS_ERRORS:
		category_error, retryable = _STATUS_ERRORS[status_code]
	if category_error is None and status_code is not None and status_code >= 500:
		category_error, retryable = ProviderUnavailableError, True

	category = (
		category_error.category if category_error else "invalid_response"
	)
	error_type = category_error or _OPERATION_ERRORS[operation]
	error = error_type(
		f"{provider} {operation} failed: {detail}",
		provider=provider,
		operation=operation,
		status_code=status_code,
		retryable=retryable,
	)
	error.category = category
	raise error from exc


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
]
