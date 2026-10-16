"""Basic tests for provider implementations."""

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from llm_gateway.adapter.contracts import Providers
from llm_gateway.adapter.prompt import build_prompt
from llm_gateway.adapter.providers import claude_provider, ollama_provider, openai_provider
from llm_gateway.adapter.providers.claude_provider import ClaudeProvider
from llm_gateway.adapter.providers.ollama_provider import OllamaProvider
from llm_gateway.adapter.providers.openai_provider import OpenAIProvider


async def _chunks(*items: SimpleNamespace) -> AsyncIterator[SimpleNamespace]:
	for item in items:
		yield item


async def _string_chunks(*items: str) -> AsyncIterator[str]:
	for item in items:
		yield item


def test_build_prompt_formats_schema_and_stream_by_provider() -> None:
	schema = {"type": "object", "properties": {"rating": {"type": "integer"}}}

	openai_request = build_prompt("review", schema, False, Providers.OPENAI)
	assert openai_request["input"] == "review"
	assert openai_request["text"]["format"]["schema"] == schema
	assert openai_request["text"]["format"]["strict"] is True

	claude_request = build_prompt("review", schema, True, Providers.CLAUDE)
	assert claude_request["messages"] == [
		{"role": "user", "content": "review"}
	]
	assert claude_request["output_config"]["format"]["schema"] == schema
	assert "stream" not in claude_request

	ollama_request = build_prompt("review", schema, True, Providers.OLLAMA)
	assert ollama_request["messages"][-1] == {
		"role": "user",
		"content": "review",
	}
	assert ollama_request["format"] == schema
	assert ollama_request["stream"] is True


class _ClaudeStream:
	def __init__(self, text_stream: AsyncIterator[str]) -> None:
		self.text_stream = text_stream

	async def __aenter__(self) -> "_ClaudeStream":
		return self

	async def __aexit__(self, *args: object) -> None:
		return None


@pytest.mark.asyncio
async def test_openai_provider_generate_stream_and_extract(
	monkeypatch: pytest.MonkeyPatch,
	valid_review_json: str,
) -> None:
	create = AsyncMock(
		side_effect=[
			SimpleNamespace(output_text="generated"),
			_chunks(
				SimpleNamespace(type="response.output_text.delta", delta="chunk-1"),
				SimpleNamespace(type="response.output_text.delta", delta="chunk-2"),
			),
			SimpleNamespace(output_text=valid_review_json),
		]
	)
	monkeypatch.setattr(
		openai_provider,
		"AsyncOpenAI",
		lambda **kwargs: SimpleNamespace(
			responses=SimpleNamespace(create=create)
		),
	)
	provider = OpenAIProvider(api_key="test-key", model="test-model")

	generated = await provider.generate("prompt")
	streamed = [chunk async for chunk in provider.stream("prompt")]
	extracted = await provider.extract("review")

	assert (generated.text, generated.provider, generated.model) == (
		"generated",
		Providers.OPENAI,
		"test-model",
	)
	assert streamed == ["chunk-1", "chunk-2"]
	assert extracted.rating == 5
	assert create.await_count == 3


@pytest.mark.asyncio
async def test_claude_provider_generate_stream_and_extract(
	monkeypatch: pytest.MonkeyPatch,
	valid_review_json: str,
) -> None:
	create = AsyncMock(
		side_effect=[
			SimpleNamespace(content=[SimpleNamespace(type="text", text="generated")]),
			SimpleNamespace(content=[SimpleNamespace(type="text", text=valid_review_json)]),
		]
	)
	stream = _ClaudeStream(_string_chunks("chunk-1", "chunk-2"))
	monkeypatch.setattr(
		claude_provider,
		"AsyncAnthropic",
		lambda **kwargs: SimpleNamespace(
			messages=SimpleNamespace(
				create=create,
				stream=lambda **stream_kwargs: stream,
			)
		),
	)
	provider = ClaudeProvider(api_key="test-key", model="test-model")

	generated = await provider.generate("prompt")
	streamed = [chunk async for chunk in provider.stream("prompt")]
	extracted = await provider.extract("review")

	assert (generated.text, generated.provider, generated.model) == (
		"generated",
		Providers.CLAUDE,
		"test-model",
	)
	assert streamed == ["chunk-1", "chunk-2"]
	assert extracted.rating == 5
	assert create.await_count == 2


@pytest.mark.asyncio
async def test_ollama_provider_generate_stream_and_extract(
	monkeypatch: pytest.MonkeyPatch,
	valid_review_json: str,
) -> None:
	review_payload = json.dumps({"message": {"content": valid_review_json}})
	stream_payload = "\n".join(
		[
			json.dumps({"message": {"content": "chunk-1"}, "done": False}),
			json.dumps({"message": {"content": "chunk-2"}, "done": True}),
		]
	)

	def handler(request: httpx.Request) -> httpx.Response:
		payload = json.loads(request.content)
		if payload.get("stream"):
			return httpx.Response(200, text=stream_payload)
		if "format" in payload:
			return httpx.Response(200, text=review_payload)
		return httpx.Response(
			200,
			json={"message": {"content": "generated"}},
		)

	real_async_client = httpx.AsyncClient
	transport = httpx.MockTransport(handler)
	monkeypatch.setattr(
		ollama_provider.httpx,
		"AsyncClient",
		lambda **kwargs: real_async_client(transport=transport, **kwargs),
	)
	provider = OllamaProvider(model="test-model", base_url="http://ollama.test/")

	generated = await provider.generate("prompt")
	streamed = [chunk async for chunk in provider.stream("prompt")]
	extracted = await provider.extract("review")

	assert (generated.text, generated.provider, generated.model) == (
		"generated",
		Providers.OLLAMA,
		"test-model",
	)
	assert streamed == ["chunk-1", "chunk-2"]
	assert extracted.rating == 5


@pytest.mark.parametrize("provider_type", [OpenAIProvider, ClaudeProvider])
def test_remote_providers_require_api_keys(provider_type: type) -> None:
	with pytest.raises(ValueError, match="API_KEY is required"):
		provider_type(api_key=None)


