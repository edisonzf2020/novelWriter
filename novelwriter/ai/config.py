"""Configuration helpers for the novelWriter AI Copilot."""

from __future__ import annotations

import logging
import os
from configparser import ConfigParser
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)


class _ConfigReader(Protocol):
    """Protocol describing the subset of config readers we rely on."""

    def has_option(self, section: str, option: str) -> bool:  # pragma: no cover - typing aid
        ...

    def has_section(self, section: str) -> bool:  # pragma: no cover - typing aid
        ...

    def get(self, section: str, option: str, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        ...

    def getboolean(self, section: str, option: str, *args: Any, **kwargs: Any) -> bool:  # pragma: no cover
        ...

    def getint(self, section: str, option: str, *args: Any, **kwargs: Any) -> int:  # pragma: no cover
        ...


_DEF_BASE_URL = "https://api.openai.com/v1"
_ENV_API_KEY = "OPENAI_API_KEY"


class AIConfig:
    """Encapsulates persistent configuration for AI Copilot providers."""

    __slots__ = (
        "enabled",
        "provider",
        "model",
        "openai_base_url",
        "api_key",
        "timeout",
        "max_tokens",
        "dry_run_default",
        "ask_before_apply",
        "_api_key_from_env",
    )

    SECTION = "AI"

    def __init__(self) -> None:
        self.enabled: bool = False
        self.provider: str = "openai"
        self.model: str = "gpt-4"
        self.openai_base_url: str = _DEF_BASE_URL
        self.api_key: str = ""
        self.timeout: int = 30
        self.max_tokens: int = 2048
        self.dry_run_default: bool = True
        self.ask_before_apply: bool = True
        self._api_key_from_env: bool = False

    @property
    def api_key_from_env(self) -> bool:
        """Return ``True`` when the API key originates from an env var."""

        return self._api_key_from_env

    def is_default_state(self) -> bool:
        """Return ``True`` when no user-specific AI settings are active."""

        return (
            not self.enabled
            and self.provider == "openai"
            and self.model == "gpt-4"
            and self.openai_base_url == _DEF_BASE_URL
            and not self.api_key
            and self.timeout == 30
            and self.max_tokens == 2048
            and self.dry_run_default
            and self.ask_before_apply
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def load_from_main_config(self, conf: _ConfigReader) -> None:
        """Populate the AI settings from the main configuration object."""

        reader = _ReaderFacade(conf)
        section = self.SECTION

        self.enabled = reader.get_bool(section, "enabled", self.enabled)
        self.provider = reader.get_str(section, "provider", self.provider)
        self.model = reader.get_str(section, "model", self.model)
        self.openai_base_url = reader.get_str(section, "openai_base_url", self.openai_base_url)
        self.timeout = reader.get_int(section, "timeout", self.timeout)
        self.max_tokens = reader.get_int(section, "max_tokens", self.max_tokens)
        self.dry_run_default = reader.get_bool(section, "dry_run_default", self.dry_run_default)
        self.ask_before_apply = reader.get_bool(section, "ask_before_apply", self.ask_before_apply)

        stored_key = reader.get_str(section, "api_key", "")
        env_key = os.environ.get(_ENV_API_KEY, "").strip()

        if env_key:
            self.api_key = env_key
            self._api_key_from_env = True
            if stored_key:
                logger.debug("Ignoring stored AI API key due to %s override", _ENV_API_KEY)
        else:
            self.api_key = stored_key
            self._api_key_from_env = False

    def save_to_main_config(self, conf: ConfigParser) -> None:
        """Persist current AI configuration into the main config parser."""

        if self.is_default_state() and not self._api_key_from_env:
            return

        section = self.SECTION
        if not conf.has_section(section):
            conf[section] = {}

        conf[section]["enabled"] = str(self.enabled)
        conf[section]["provider"] = str(self.provider)
        conf[section]["model"] = str(self.model)
        conf[section]["openai_base_url"] = str(self.openai_base_url)
        conf[section]["timeout"] = str(self.timeout)
        conf[section]["max_tokens"] = str(self.max_tokens)
        conf[section]["dry_run_default"] = str(self.dry_run_default)
        conf[section]["ask_before_apply"] = str(self.ask_before_apply)

        if self._api_key_from_env:
            if conf.has_option(section, "api_key"):
                conf.remove_option(section, "api_key")
        else:
            conf[section]["api_key"] = str(self.api_key)

    def reset_api_key(self) -> None:
        """Clear any cached API key information and disable env overrides."""

        self.api_key = ""
        self._api_key_from_env = False


class _ReaderFacade:
    """Compatibility wrapper around :class:`ConfigParser` variants."""

    __slots__ = ("_conf",)

    def __init__(self, conf: _ConfigReader) -> None:
        self._conf = conf

    def get_str(self, section: str, option: str, default: str) -> str:
        reader = getattr(self._conf, "rdStr", None)
        if callable(reader):
            return cast(str, reader(section, option, default))
        return self._conf.get(section, option, fallback=default)

    def get_bool(self, section: str, option: str, default: bool) -> bool:
        reader = getattr(self._conf, "rdBool", None)
        if callable(reader):
            return cast(bool, reader(section, option, default))
        try:
            return self._conf.getboolean(section, option, fallback=default)
        except ValueError:
            logger.warning("Invalid boolean for '%s:%s' in AI config", section, option)
            return default

    def get_int(self, section: str, option: str, default: int) -> int:
        reader = getattr(self._conf, "rdInt", None)
        if callable(reader):
            return cast(int, reader(section, option, default))
        try:
            return self._conf.getint(section, option, fallback=default)
        except ValueError:
            logger.warning("Invalid integer for '%s:%s' in AI config", section, option)
            return default

    def get_str_list(self, section: str, option: str, default: list[str]) -> list[str]:  # pragma: no cover
        reader = getattr(self._conf, "rdStrList", None)
        if callable(reader):
            return cast(list[str], reader(section, option, default))
        if self._conf.has_option(section, option):
            raw = self._conf.get(section, option, fallback="")
            return [item.strip() for item in raw.split(",") if item.strip()]
        return default.copy()
