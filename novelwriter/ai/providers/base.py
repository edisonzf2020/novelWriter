"""Base provider abstractions shared by AI Copilot integrations."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

from novelwriter.ai.errors import NWAiProviderError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Describes the remote endpoint abilities discovered at runtime."""

    preferred_endpoint: str
    supports_responses: bool
    supports_chat_completions: bool
    supports_stream: bool
    supports_tool_calls: bool
    max_output_tokens: int | None
    detected_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the capability snapshot."""

        payload: dict[str, Any] = {
            "preferred_endpoint": self.preferred_endpoint,
            "supports_responses": self.supports_responses,
            "supports_chat_completions": self.supports_chat_completions,
            "supports_stream": self.supports_stream,
            "supports_tool_calls": self.supports_tool_calls,
            "max_output_tokens": self.max_output_tokens,
            "detected_at": self.detected_at.isoformat(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class ProviderSettings:
    """Immutable-like configuration holder for provider instances."""

    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0
    organisation: str | None = None
    extra_headers: Mapping[str, str] | None = None
    user_agent: str | None = None
    transport: "httpx.BaseTransport" | None = None


@dataclass(slots=True)
class ProviderSessionState:
    """Mutable session bookkeeping shared across provider operations."""

    detections: int = 0
    last_error: str | None = None
    last_detection_started: datetime | None = None
    last_detection_completed: datetime | None = None


class BaseProvider(ABC):
    """Base class for AI providers with lazy capability detection."""

    def __init__(self, settings: ProviderSettings) -> None:
        self._settings = settings
        self._capabilities: ProviderCapabilities | None = None
        self._session_state = ProviderSessionState()
        self._lock = threading.RLock()

    @property
    def settings(self) -> ProviderSettings:
        """Return the provider settings."""

        return self._settings

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return cached capabilities, forcing detection if necessary."""

        result = self.ensure_capabilities()
        if result is None:  # pragma: no cover - defensive
            raise NWAiProviderError("Capability detection did not return a result.")
        return result

    def ensure_capabilities(self, *, force: bool = False) -> ProviderCapabilities:
        """Return cached capabilities, optionally forcing a detection pass."""

        with self._lock:
            if self._capabilities is not None and not force:
                return self._capabilities

            try:
                self._session_state.detections += 1
                self._session_state.last_detection_started = datetime.now(timezone.utc)
                snapshot = self._detect_capabilities()
                if not isinstance(snapshot, ProviderCapabilities):
                    raise TypeError("Capability detector returned unexpected payload")
                metadata: MutableMapping[str, Any] = dict(snapshot.metadata)
                metadata.setdefault("base_url", self._settings.base_url)
                metadata.setdefault("model", self._settings.model)
                snapshot = ProviderCapabilities(
                    preferred_endpoint=snapshot.preferred_endpoint,
                    supports_responses=snapshot.supports_responses,
                    supports_chat_completions=snapshot.supports_chat_completions,
                    supports_stream=snapshot.supports_stream,
                    supports_tool_calls=snapshot.supports_tool_calls,
                    max_output_tokens=snapshot.max_output_tokens,
                    detected_at=snapshot.detected_at,
                    metadata=MappingProxyType(metadata),
                )
                self._capabilities = snapshot
                self._session_state.last_error = None
                logger.debug(
                    "AI provider capabilities detected: %s",
                    snapshot.as_dict(),
                )
                return snapshot
            except Exception as exc:  # noqa: BLE001 - propagate as provider error
                message = str(exc) or exc.__class__.__name__
                self._session_state.last_error = message
                logger.warning("Capability detection failed: %s", message)
                raise
            finally:
                self._session_state.last_detection_completed = datetime.now(timezone.utc)

    def refresh_capabilities(self) -> ProviderCapabilities:
        """Force a new capability detection cycle."""

        return self.ensure_capabilities(force=True)

    @abstractmethod
    def _detect_capabilities(self) -> ProviderCapabilities:
        """Implement provider-specific capability probing."""

    @abstractmethod
    def generate(
        self,
        messages: list[Mapping[str, Any]],
        *,
        stream: bool = False,
        tools: list[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a generation request against the remote provider."""

    def list_models(self, *, force: bool = False) -> list[Mapping[str, Any]]:
        """Return a cached catalogue of provider models, refreshing when requested."""

        raise NWAiProviderError("Model listing is not supported for this provider.")

    def get_model_metadata(self, model_id: str, *, force: bool = False) -> Mapping[str, Any] | None:
        """Return metadata for a specific provider model."""

        raise NWAiProviderError("Model metadata lookup is not supported for this provider.")

    def close(self) -> None:
        """Release any resources held by the provider instance."""

    def __enter__(self) -> "BaseProvider":  # pragma: no cover - context mgr sugar
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context mgr sugar
        self.close()
