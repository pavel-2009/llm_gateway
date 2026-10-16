import httpx
import pytest

from llm_gateway.adapter.errors import (
	AIExtractionError,
	AIGenerationError,
	AIStreamingError,
	AuthenticationError,
	ContentPolicyError,
	InvalidRequestError,
	LLMProviderError,
	ModelNotFoundError,
	PermissionDeniedError,
	ProviderConnectionError,
	ProviderUnavailableError,
	ProviderTimeoutError,
	RateLimitError,
	raise_provider_error,
)


@pytest.mark.parametrize(
	"status_code,error_type,retryable",
	[
		(400, InvalidRequestError, False),
		(401, AuthenticationError, False),
		(403, PermissionDeniedError, False),
		(404, ModelNotFoundError, False),
		(408, ProviderTimeoutError, True),
		(429, RateLimitError, True),
		(503, ProviderUnavailableError, True),
	],
)
def test_http_errors_are_mapped_to_provider_errors(
	status_code: int,
	error_type: type[LLMProviderError],
	retryable: bool,
) -> None:
	response = httpx.Response(
		status_code,
		text="provider error",
		request=httpx.Request("POST", "https://provider.test"),
	)
	error = httpx.HTTPStatusError(
		"request failed",
		request=response.request,
		response=response,
	)

	with pytest.raises(error_type) as exc_info:
		raise_provider_error("test-provider", "generation", error)

	assert exc_info.value.provider == "test-provider"
	assert exc_info.value.operation == "generation"
	assert exc_info.value.status_code == status_code
	assert exc_info.value.retryable is retryable
	assert "provider error" in str(exc_info.value)


class _CodedError(Exception):
	def __init__(self, code: str) -> None:
		super().__init__(code)
		self.code = code


@pytest.mark.parametrize(
	"code,error_type,retryable",
	[
		("rate_limit_exceeded", RateLimitError, True),
		("vector_store_timeout", ProviderTimeoutError, True),
		("server_error", ProviderUnavailableError, True),
		("bio_policy", ContentPolicyError, False),
		("invalid_prompt", InvalidRequestError, False),
	],
)
def test_provider_error_codes_are_mapped(
	code: str,
	error_type: type[LLMProviderError],
	retryable: bool,
) -> None:
	with pytest.raises(error_type) as exc_info:
		raise_provider_error("test-provider", "generation", _CodedError(code))

	assert exc_info.value.retryable is retryable


def test_connection_errors_are_retryable() -> None:
	request = httpx.Request("POST", "https://provider.test")
	error = httpx.ConnectError("connection failed", request=request)

	with pytest.raises(ProviderConnectionError) as exc_info:
		raise_provider_error("test-provider", "generation", error)

	assert exc_info.value.retryable is True


@pytest.mark.parametrize(
	"operation,base_error",
	[
		("generation", AIGenerationError),
		("streaming", AIStreamingError),
		("extraction", AIExtractionError),
	],
)
def test_unknown_errors_keep_operation_specific_type(
	operation: str,
	base_error: type[LLMProviderError],
) -> None:
	with pytest.raises(base_error) as exc_info:
		raise_provider_error("test-provider", operation, RuntimeError("unexpected"))

	assert exc_info.value.operation == operation
	assert exc_info.value.category == "invalid_response"
	assert exc_info.value.retryable is False
