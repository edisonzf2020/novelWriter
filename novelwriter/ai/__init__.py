"""Core AI domain package for the novelWriter AI Copilot."""

from .api import NWAiApi
from .errors import (
    NWAiApiError,
    NWAiConfigError,
    NWAiError,
    NWAiProviderError,
)
from .models import BuildResult, DocumentRef, Suggestion, TextRange

__all__ = [
    "NWAiApi",
    "NWAiError",
    "NWAiApiError",
    "NWAiProviderError",
    "NWAiConfigError",
    "DocumentRef",
    "TextRange",
    "Suggestion",
    "BuildResult",
]
