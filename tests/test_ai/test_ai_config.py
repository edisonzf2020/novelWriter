"""Tests for the AI configuration helper."""
from __future__ import annotations

from configparser import ConfigParser
import httpx
import pytest

from novelwriter.ai.config import AIConfig
from novelwriter.ai.errors import NWAiConfigError
from novelwriter.ai.providers import OpenAICompatibleProvider
from novelwriter.common import NWConfigParser


def test_ai_config_defaults() -> None:
    cfg = AIConfig()

    assert cfg.enabled is False
    assert cfg.provider == "openai"
    assert cfg.openai_base_url.startswith("https://")
    assert cfg.api_key == ""
    assert cfg.api_key_from_env is False
    assert cfg.timeout == 30
    assert cfg.max_tokens == 2048
    assert cfg.dry_run_default is True
    assert cfg.ask_before_apply is True


def test_ai_config_load_and_save_roundtrip() -> None:
    parser = NWConfigParser()
    parser[AIConfig.SECTION] = {
        "enabled": "yes",
        "provider": "custom",
        "model": "gpt-4o-mini",
        "openai_base_url": "https://example.com/v1",
        "timeout": "45",
        "max_tokens": "4096",
        "dry_run_default": "no",
        "ask_before_apply": "no",
        "api_key": "stored-key",
    }

    cfg = AIConfig()
    cfg.load_from_main_config(parser)

    assert cfg.enabled is True
    assert cfg.provider == "custom"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.openai_base_url == "https://example.com/v1"
    assert cfg.timeout == 45
    assert cfg.max_tokens == 4096
    assert cfg.dry_run_default is False
    assert cfg.ask_before_apply is False
    assert cfg.api_key == "stored-key"
    assert cfg.api_key_from_env is False

    cfg.api_key = "updated-key"

    out = ConfigParser()
    cfg.save_to_main_config(out)

    assert out.has_section("AI")
    assert out.get("AI", "enabled") == "True"
    assert out.get("AI", "provider") == "custom"
    assert out.get("AI", "model") == "gpt-4o-mini"
    assert out.get("AI", "openai_base_url") == "https://example.com/v1"
    assert out.get("AI", "timeout") == "45"
    assert out.get("AI", "max_tokens") == "4096"
    assert out.get("AI", "dry_run_default") == "False"
    assert out.get("AI", "ask_before_apply") == "False"
    assert out.get("AI", "api_key") == "updated-key"


def test_ai_config_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = NWConfigParser()
    parser[AIConfig.SECTION] = {"api_key": "stored-key"}

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    cfg = AIConfig()
    cfg.load_from_main_config(parser)

    assert cfg.api_key == "env-key"
    assert cfg.api_key_from_env is True

    out = ConfigParser()
    cfg.save_to_main_config(out)

    assert out.has_section("AI")
    assert not out.has_option("AI", "api_key")


def test_ai_config_build_provider_settings_includes_optional_fields() -> None:
    cfg = AIConfig()
    cfg.enabled = True
    cfg.api_key = "token"
    cfg.model = "test-model"
    cfg.timeout = 45
    cfg.openai_organisation = "org-123"
    cfg.user_agent = "custom-UA"
    cfg.extra_headers = {"X-Trace": "1"}

    settings = cfg.build_provider_settings()

    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.organisation == "org-123"
    assert settings.extra_headers == {"X-Trace": "1"}
    assert settings.user_agent == "custom-UA"
    assert settings.timeout == pytest.approx(45.0)


def test_ai_config_build_provider_settings_requires_credentials() -> None:
    cfg = AIConfig()
    cfg.model = "test"
    cfg.api_key = ""

    with pytest.raises(NWAiConfigError):
        cfg.build_provider_settings()


def test_ai_config_create_provider_uses_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter: dict[str, int] = {"responses": 0, "chat": 0, "model": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/responses":
            call_counter["responses"] += 1
            return httpx.Response(200, json={"id": "resp", "usage": {"output_tokens": 4}})
        if path == "/v1/chat/completions":
            call_counter["chat"] += 1
            return httpx.Response(200, json={"id": "chat", "usage": {"completion_tokens": 2}})
        if path == "/v1/models/test-model":
            call_counter["model"] += 1
            return httpx.Response(200, json={"id": "test-model", "output_token_limit": 1024})
        raise AssertionError(f"Unexpected path {path}")

    transport = httpx.MockTransport(handler)

    cfg = AIConfig()
    cfg.enabled = True
    cfg.api_key = "token"
    cfg.model = "test-model"

    provider = cfg.create_provider(transport=transport)

    assert isinstance(provider, OpenAICompatibleProvider)
    caps = provider.ensure_capabilities()
    assert caps.supports_responses is True
    assert call_counter["responses"] == 1
    assert call_counter["model"] == 1


def test_ai_config_set_availability_reason_trims_input() -> None:
    cfg = AIConfig()
    cfg.set_availability_reason("  Something went wrong  ")
    assert cfg.availability_reason == "Something went wrong"
    cfg.set_availability_reason(None)
    assert cfg.availability_reason is None


def test_ai_config_create_provider_unknown_provider_raises() -> None:
    cfg = AIConfig()
    cfg.enabled = True
    cfg.provider = "does-not-exist"
    cfg.api_key = "token"
    cfg.model = "test-model"

    with pytest.raises(NWAiConfigError):
        cfg.create_provider()
