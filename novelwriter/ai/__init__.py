"""Core AI domain package for the novelWriter AI Copilot."""

from .api import NWAiApi
from .config import AIConfig
from .errors import (
    NWAiApiError,
    NWAiConfigError,
    NWAiError,
    NWAiProviderError,
)
from .models import BuildResult, DocumentRef, ModelInfo, ProofreadResult, Suggestion, TextRange

__all__ = [
    "NWAiApi",
    "AIConfig",
    "NWAiError",
    "NWAiApiError",
    "NWAiProviderError",
    "NWAiConfigError",
    "DocumentRef",
    "TextRange",
    "Suggestion",
    "ProofreadResult",
    "ModelInfo",
    "BuildResult",
]
