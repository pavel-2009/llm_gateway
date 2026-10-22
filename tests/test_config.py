"""Tests for application settings and the console entry point."""

import pytest
from pydantic import ValidationError

from llm_gateway import config, main
from llm_gateway.core.config import AppConfig


def test_provider_keys_are_optional_and_anthropic_alias_is_supported() -> None:
    settings = AppConfig(
        _env_file=None,
        OPENAI_API_KEY=" ",
        ANTHROPIC_API_KEY="  anthropic-key  ",
    )

    assert settings.OPENAI_API_KEY is None
    assert settings.CLAUDE_API_KEY == "anthropic-key"


def test_settings_load_from_container_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HOST",
        "PORT",
        "CLAUDE_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    settings = AppConfig(_env_file=None)

    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 8123
    assert settings.CLAUDE_API_KEY == "anthropic-key"


@pytest.mark.parametrize(
    "values",
    [
        {"PORT": 0},
        {"PORT": 65536},
        {"HOST": " "},
        {"OLLAMA_BASE_URL": "ollama:11434"},
        {"REQUEST_TIMEOUT": 0},
        {"STREAM_TIMEOUT": float("inf")},
        {"MAX_CONCURRENT_REQUESTS": 0},
        {"CLAUDE_MAX_TOKENS": 0},
    ],
)
def test_invalid_critical_settings_are_rejected(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None, **values)


def test_console_entry_point_runs_configured_http_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    called_with: dict[str, object] = {}

    def fake_run(app: str, *, host: str, port: int) -> None:
        called_with.update(app=app, host=host, port=port)

    monkeypatch.setattr(config, "HOST", "127.0.0.1")
    monkeypatch.setattr(config, "PORT", 8123)
    monkeypatch.setattr(uvicorn, "run", fake_run)

    main()

    assert called_with == {
        "app": "llm_gateway.app:app",
        "host": "127.0.0.1",
        "port": 8123,
    }
