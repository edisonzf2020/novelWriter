"""Provider factory helpers for the novelWriter AI Copilot."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Mapping

from novelwriter.ai.errors import NWAiConfigError

from .base import BaseProvider, ProviderSettings
from .openai_compatible import OpenAICompatibleProvider
from .openai_sdk import OpenAISDKProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx
    from novelwriter.ai.config import AIConfig

logger = logging.getLogger(__name__)


_PROVIDER_REGISTRY: Mapping[str, Callable[[ProviderSettings], BaseProvider]] = {
    "openai": OpenAICompatibleProvider,
    "openai-compatible": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "openai-sdk": OpenAISDKProvider,
    "openai_sdk": OpenAISDKProvider,
}


def create_provider(provider_id: str, settings: ProviderSettings) -> BaseProvider:
    """Instantiate a provider by identifier using the registered factories."""

    normalised = (provider_id or "openai").strip().lower()
    try:
        factory = _PROVIDER_REGISTRY[normalised]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise NWAiConfigError(f"Unsupported AI provider '{provider_id}'.") from exc

    provider = factory(settings)
    logger.debug("Created AI provider '%s' with base URL '%s'", normalised, settings.base_url)
    return provider


def provider_from_config(ai_config: "AIConfig", *, transport: "httpx.BaseTransport" | None = None) -> BaseProvider:
    """Create a provider instance based on an :class:`AIConfig` object."""

    base_url = (ai_config.openai_base_url or "").strip()
    if not base_url:
        raise NWAiConfigError("OpenAI base URL is not configured.")

    api_key = (ai_config.api_key or "").strip()
    if not api_key and not ai_config.api_key_from_env:
        raise NWAiConfigError("OpenAI API key is not configured.")

    model = (ai_config.model or "").strip()
    if not model:
        raise NWAiConfigError("Model name must be configured for the AI provider.")

    settings = ProviderSettings(
        base_url=base_url,
        api_key=api_key or ai_config.api_key,
        model=model,
        timeout=float(ai_config.timeout),
        organisation=getattr(ai_config, "openai_organisation", None),
        extra_headers=getattr(ai_config, "extra_headers", None),
        user_agent=getattr(ai_config, "user_agent", None),
        transport=transport,
    )

    return create_provider(ai_config.provider, settings)
