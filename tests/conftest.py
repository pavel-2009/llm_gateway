"""Configuration for pytest tests."""

import json

import pytest

from llm_gateway.core.config import ExtractedReview


@pytest.fixture
def valid_review_json() -> str:
	return json.dumps(
		{
			"sentiment": "positive",
			"topics": ["delivery", "quality"],
			"rating": 5,
		}
	)


@pytest.fixture
def valid_review(valid_review_json: str) -> ExtractedReview:
	return ExtractedReview.model_validate_json(valid_review_json)

