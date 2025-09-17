"""Tests for the AI configuration helper."""
from __future__ import annotations

from configparser import ConfigParser

import pytest

from novelwriter.ai.config import AIConfig
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
