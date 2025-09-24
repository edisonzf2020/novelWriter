"""Core AI domain package for the novelWriter AI Copilot."""

from .api import NWAiApi  # Keep for backward compatibility
from .ai_core import AICoreService  # New AI core service
from .config import AIConfig
from .errors import (
    NWAiApiError,
    NWAiConfigError,
    NWAiError,
    NWAiProviderError,
)
from .models import BuildResult, DocumentRef, ModelInfo, ProofreadResult, Suggestion, TextRange

__all__ = [
    "NWAiApi",  # Keep for backward compatibility
    "AICoreService",  # New AI core service
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
