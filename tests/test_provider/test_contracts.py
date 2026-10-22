import json

import pytest
from pydantic import ValidationError

from llm_gateway.adapter.contracts import GenerateResponse, Providers
from llm_gateway.core.config import ExtractedReview
from llm_gateway.models import Model, Request


def test_generate_response_accepts_valid_data() -> None:
	response = GenerateResponse(
		text="Generated text",
		provider=Providers.OPENAI,
		model="test-model",
	)

	assert response.text == "Generated text"
	assert response.provider == Providers.OPENAI
	assert response.model == "test-model"


@pytest.mark.parametrize(
	"payload",
	[
		{"prompt": ""},
		{"prompt": "prompt", "model": "unknown"},
		{"prompt": "prompt", "unexpected": True},
	],
)
def test_request_rejects_invalid_data(payload: dict[str, object]) -> None:
	with pytest.raises(ValidationError):
		Request.model_validate(payload)


def test_request_defaults_to_smart_model() -> None:
	request = Request(prompt="prompt")

	assert request.model is Model.SMART


@pytest.mark.parametrize(
	"payload",
	[
		{"text": "", "provider": "openai", "model": "test-model"},
		{"text": "answer", "provider": "unknown", "model": "test-model"},
		{"text": "answer", "provider": "openai", "model": ""},
		{
			"text": "answer",
			"provider": "openai",
			"model": "test-model",
			"unexpected": True,
		},
	],
)
def test_generate_response_rejects_invalid_data(
	payload: dict[str, object],
) -> None:
	with pytest.raises(ValidationError):
		GenerateResponse.model_validate(payload)


def test_extracted_review_accepts_valid_data(valid_review: ExtractedReview) -> None:
	assert valid_review.sentiment == "positive"
	assert valid_review.topics == ["delivery", "quality"]
	assert valid_review.rating == 5


@pytest.mark.parametrize(
	"field,value",
	[
		("sentiment", "unknown"),
		("rating", 6),
		("rating", -1),
	],
)
def test_extracted_review_rejects_invalid_values(
	valid_review_json: str,
	field: str,
	value: str | int,
) -> None:
	payload = json.loads(valid_review_json)
	payload[field] = value

	with pytest.raises(ValidationError):
		ExtractedReview.model_validate(payload)


def test_extracted_review_rejects_extra_fields(valid_review_json: str) -> None:
	payload = json.loads(valid_review_json)
	payload["unexpected"] = True

	with pytest.raises(ValidationError):
		ExtractedReview.model_validate(payload)
