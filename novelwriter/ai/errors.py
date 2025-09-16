"""Error hierarchy shared by the AI Copilot domain layer."""

from __future__ import annotations

__all__ = [
    "NWAiError",
    "NWAiProviderError",
    "NWAiApiError",
    "NWAiConfigError",
]


class NWAiError(Exception):
    """Base error for all AI Copilot domain failures."""


class NWAiProviderError(NWAiError):
    """Raised when a provider backend fails or behaves unexpectedly."""


class NWAiApiError(NWAiError):
    """Raised when the AI domain API detects invalid usage or state."""


class NWAiConfigError(NWAiError):
    """Raised when AI-specific configuration values are missing or invalid."""
