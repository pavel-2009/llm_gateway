"""Build provider-specific prompt arguments."""

from typing import Any

from llm_gateway.adapter.contracts import Providers
from llm_gateway.core.config import EXTRACT_SYSTEM_PROMPT


def build_prompt(
    prompt: str,
    schema: dict[str, Any] | None,
    stream: bool,
    provider: str,
) -> dict[str, Any]:
    """Build provider-specific request arguments for a prompt."""
    
    if provider == Providers.OPENAI:
        request: dict[str, Any] = {"input": prompt}
        if stream:
            request["stream"] = True
        if schema is not None:
            request["instructions"] = EXTRACT_SYSTEM_PROMPT
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "extracted_review",
                    "strict": True,
                    "schema": schema,
                }
            }
        return request

    if provider == Providers.CLAUDE:
        request = {"messages": [{"role": "user", "content": prompt}]}
        if schema is not None:
            request["system"] = EXTRACT_SYSTEM_PROMPT
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }
        return request

    if provider == Providers.OLLAMA:
        messages = [{"role": "user", "content": prompt}]
        request = {"messages": messages, "stream": stream}
        if schema is not None:
            request["messages"] = [
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                *messages,
            ]
            request["format"] = schema
        return request

    raise ValueError(f"Unsupported provider: {provider}")