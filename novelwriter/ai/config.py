"""Configuration helpers for the novelWriter AI Copilot."""

from __future__ import annotations

import json
import logging
import os
from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from novelwriter.ai.errors import NWAiConfigError, NWAiProviderError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx
    from novelwriter.ai.providers.base import BaseProvider, ProviderSettings

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

_PROVIDER_SYNONYMS: dict[str, str] = {
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "openai-sdk": "openai-sdk",
    "openai_sdk": "openai-sdk",
}


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
        "temperature",
        "dry_run_default",
        "ask_before_apply",
        "proofreading_enabled",
        "openai_organisation",
        "extra_headers",
        "user_agent",
        "availability_reason",
        "default_model_metadata",
        "metrics_enabled",
        "metrics_log_path",
        "metrics_max_samples",
        "cache_enabled",
        "cache_max_entries",
        "cache_ttl_seconds",
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
        self.temperature: float = 0.7
        self.dry_run_default: bool = True
        self.ask_before_apply: bool = True
        self.proofreading_enabled: bool = False
        self.openai_organisation: str | None = None
        self.extra_headers: dict[str, str] | None = None
        self.user_agent: str | None = None
        self.availability_reason: str | None = None
        self.default_model_metadata: dict[str, Any] | None = None
        self.metrics_enabled: bool = True
        self.metrics_log_path: str | None = None
        self.metrics_max_samples: int = 200
        self.cache_enabled: bool = True
        self.cache_max_entries: int = 256
        self.cache_ttl_seconds: float = 300.0
        self._api_key_from_env: bool = False

        self._sync_instrumentation()

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
            and not self.proofreading_enabled
            and self.openai_organisation is None
            and (self.extra_headers is None or not self.extra_headers)
            and self.user_agent is None
            and self.metrics_enabled
            and self.metrics_log_path is None
            and self.metrics_max_samples == 200
            and self.cache_enabled
            and self.cache_max_entries == 256
            and self.cache_ttl_seconds == 300.0
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def load_from_main_config(self, conf: _ConfigReader) -> None:
        """Populate the AI settings from the main configuration object."""

        reader = _ReaderFacade(conf)
        section = self.SECTION

        self.enabled = reader.get_bool(section, "enabled", self.enabled)
        self.provider = self._normalise_provider_id(
            reader.get_str(section, "provider", self.provider)
        )
        self.model = reader.get_str(section, "model", self.model)
        self.openai_base_url = reader.get_str(section, "openai_base_url", self.openai_base_url)
        self.timeout = reader.get_int(section, "timeout", self.timeout)
        self.max_tokens = reader.get_int(section, "max_tokens", self.max_tokens)
        self.temperature = reader.get_float(section, "temperature", self.temperature)
        self.dry_run_default = reader.get_bool(section, "dry_run_default", self.dry_run_default)
        self.ask_before_apply = reader.get_bool(section, "ask_before_apply", self.ask_before_apply)
        self.proofreading_enabled = reader.get_bool(section, "proofreading_enabled", self.proofreading_enabled)
        self.openai_organisation = self._normalise_optional(
            reader.get_str(section, "openai_organisation", self.openai_organisation or "")
        )
        self.user_agent = self._normalise_optional(
            reader.get_str(section, "user_agent", self.user_agent or "")
        )
        self.extra_headers = self._parse_header_entries(
            reader.get_str(section, "extra_headers", "")
        )
        self.default_model_metadata = self._parse_model_metadata(
            reader.get_str(section, "default_model_metadata", "")
        )
        self.metrics_enabled = reader.get_bool(section, "metrics_enabled", self.metrics_enabled)
        raw_log_path = reader.get_str(section, "metrics_log_path", self.metrics_log_path or "")
        self.metrics_log_path = self._normalise_optional(raw_log_path)
        self.metrics_max_samples = reader.get_int(section, "metrics_max_samples", self.metrics_max_samples)
        self.cache_enabled = reader.get_bool(section, "cache_enabled", self.cache_enabled)
        self.cache_max_entries = reader.get_int(section, "cache_max_entries", self.cache_max_entries)
        self.cache_ttl_seconds = reader.get_float(section, "cache_ttl_seconds", self.cache_ttl_seconds)
        self.availability_reason = None

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

        self._sync_instrumentation()

    def save_to_main_config(self, conf: ConfigParser) -> None:
        """Persist current AI configuration into the main config parser."""

        if self.is_default_state() and not self._api_key_from_env:
            return

        section = self.SECTION
        if not conf.has_section(section):
            conf[section] = {}

        conf[section]["enabled"] = str(self.enabled)
        conf[section]["provider"] = self._normalise_provider_id(self.provider)
        conf[section]["model"] = str(self.model)
        conf[section]["openai_base_url"] = str(self.openai_base_url)
        conf[section]["timeout"] = str(self.timeout)
        conf[section]["max_tokens"] = str(self.max_tokens)
        conf[section]["temperature"] = str(self.temperature)
        conf[section]["dry_run_default"] = str(self.dry_run_default)
        conf[section]["ask_before_apply"] = str(self.ask_before_apply)
        conf[section]["proofreading_enabled"] = str(self.proofreading_enabled)
        conf[section]["metrics_enabled"] = str(self.metrics_enabled)
        conf[section]["metrics_max_samples"] = str(self.metrics_max_samples)
        conf[section]["cache_enabled"] = str(self.cache_enabled)
        conf[section]["cache_max_entries"] = str(self.cache_max_entries)
        conf[section]["cache_ttl_seconds"] = str(self.cache_ttl_seconds)

        if self.openai_organisation:
            conf[section]["openai_organisation"] = str(self.openai_organisation)
        elif conf.has_option(section, "openai_organisation"):
            conf.remove_option(section, "openai_organisation")

        if self.user_agent:
            conf[section]["user_agent"] = str(self.user_agent)
        elif conf.has_option(section, "user_agent"):
            conf.remove_option(section, "user_agent")

        if self.metrics_log_path:
            conf[section]["metrics_log_path"] = str(self.metrics_log_path)
        elif conf.has_option(section, "metrics_log_path"):
            conf.remove_option(section, "metrics_log_path")

        headers_serialised = self._serialise_header_entries(self.extra_headers)
        if headers_serialised:
            conf[section]["extra_headers"] = headers_serialised
        elif conf.has_option(section, "extra_headers"):
            conf.remove_option(section, "extra_headers")

        metadata_serialised = self._serialise_model_metadata(self.default_model_metadata)
        if metadata_serialised:
            conf[section]["default_model_metadata"] = metadata_serialised
        elif conf.has_option(section, "default_model_metadata"):
            conf.remove_option(section, "default_model_metadata")

        if self._api_key_from_env:
            if conf.has_option(section, "api_key"):
                conf.remove_option(section, "api_key")
        else:
            conf[section]["api_key"] = str(self.api_key)

    def _sync_instrumentation(self) -> None:
        """Apply current telemetry configuration to the performance tracker."""

        try:
            from novelwriter.ai.performance import get_tracker
        except Exception:  # pragma: no cover - defensive import guard
            return

        log_path = self.metrics_log_path
        resolved_path = Path(log_path).expanduser() if log_path else Path(".ai") / "debug-log.md"
        get_tracker().configure(
            enabled=self.metrics_enabled,
            log_path=resolved_path,
            max_samples=max(1, int(self.metrics_max_samples or 200)),
        )

    def reset_api_key(self) -> None:
        """Clear any cached API key information and disable env overrides."""

        self.api_key = ""
        self._api_key_from_env = False

    def build_provider_settings(
        self,
        *,
        transport: "httpx.BaseTransport" | None = None,
    ) -> "ProviderSettings":
        """Translate configuration values into :class:`ProviderSettings`."""

        from novelwriter.ai.providers.base import ProviderSettings  # Local import to avoid cycles

        base_url = (self.openai_base_url or "").strip()
        if not base_url:
            raise NWAiConfigError("OpenAI base URL is not configured.")

        api_key = (self.api_key or "").strip()
        if not api_key and not self.api_key_from_env:
            raise NWAiConfigError("OpenAI API key is not configured.")

        model = (self.model or "").strip()
        if not model:
            raise NWAiConfigError("Model name must be configured for the AI provider.")

        return ProviderSettings(
            base_url=base_url,
            api_key=api_key or self.api_key,
            model=model,
            timeout=float(self.timeout),
            organisation=self.openai_organisation,
            extra_headers=self.extra_headers,
            user_agent=self.user_agent,
            transport=transport,
        )

    def create_provider(
        self,
        *,
        transport: "httpx.BaseTransport" | None = None,
    ) -> "BaseProvider":
        """Instantiate the configured provider implementation."""

        from novelwriter.ai.providers.factory import provider_from_config

        try:
            provider = provider_from_config(self, transport=transport)
        except NWAiProviderError as exc:
            message = str(exc) or exc.__class__.__name__
            self.set_availability_reason(message)
            raise NWAiConfigError(message) from exc

        self.set_availability_reason(None)
        return provider

    def set_availability_reason(self, message: str | None) -> None:
        """Update availability diagnostics for UI surfaces."""

        self.availability_reason = self._normalise_optional(message)

    def _normalise_provider_id(self, provider: str) -> str:
        key = (provider or "openai").strip().lower()
        return _PROVIDER_SYNONYMS.get(key, key)

    @staticmethod
    def _normalise_optional(value: str | None) -> str | None:
        cleaned = (value or "").strip()
        return cleaned or None

    @staticmethod
    def _parse_header_entries(raw: str) -> dict[str, str] | None:
        if not raw:
            return None
        entries = [chunk.strip() for chunk in raw.split(";") if chunk.strip()]
        if not entries:
            return None
        headers: dict[str, str] = {}
        for entry in entries:
            if ":" not in entry:
                continue
            key, value = entry.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key:
                headers[key] = value
        return headers or None

    @staticmethod
    def _serialise_header_entries(headers: dict[str, str] | None) -> str:
        if not headers:
            return ""
        return "; ".join(f"{key}: {value}" for key, value in headers.items())

    @staticmethod
    def _parse_model_metadata(raw: str) -> dict[str, Any] | None:
        if not raw.strip():
            return None
        try:
            metadata = json.loads(raw)
            if isinstance(metadata, dict):
                return metadata
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in model metadata configuration")
        return None

    @staticmethod
    def _serialise_model_metadata(metadata: dict[str, Any] | None) -> str:
        if not metadata:
            return ""
        try:
            return json.dumps(metadata, separators=(',', ':'))
        except (TypeError, ValueError):
            logger.warning("Failed to serialize model metadata")
            return ""


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

    def get_float(self, section: str, option: str, default: float) -> float:
        reader = getattr(self._conf, "rdFloat", None)
        if callable(reader):
            return cast(float, reader(section, option, default))
        try:
            # Check if the config object has getfloat method
            conf_obj = getattr(self._conf, 'getfloat', None)
            if conf_obj and callable(conf_obj):
                return conf_obj(section, option, fallback=default)
            else:
                # Fallback for configs that don't have getfloat
                raw_value = self._conf.get(section, option, fallback=str(default))
                return float(raw_value)
        except (ValueError, TypeError):
            logger.warning("Invalid float for '%s:%s' in AI config", section, option)
            return default

    def get_str_list(self, section: str, option: str, default: list[str]) -> list[str]:  # pragma: no cover
        reader = getattr(self._conf, "rdStrList", None)
        if callable(reader):
            return cast(list[str], reader(section, option, default))
        if self._conf.has_option(section, option):
            raw = self._conf.get(section, option, fallback="")
            return [item.strip() for item in raw.split(",") if item.strip()]
        return default.copy()
