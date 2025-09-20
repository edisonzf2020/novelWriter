"""Caching helpers for AI provider query responses."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import threading
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

from .performance import current_span, get_tracker

__all__ = [
    "CacheConfig",
    "CachedResult",
    "ProviderCache",
    "ProviderQueryCache",
    "build_cache_key",
    "config_from_ai",
]


@dataclass(frozen=True)
class CacheConfig:
    """Runtime configuration determining cache behaviour."""

    enabled: bool
    max_entries: int
    ttl_seconds: float


@dataclass(frozen=True)
class CachedResult:
    """Encapsulates cached response chunks and metadata."""

    key: str
    chunks: tuple[str, ...]
    created_at: float


class ProviderCache:
    """Simple LRU cache with TTL semantics for provider responses."""

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._entries: "OrderedDict[str, CachedResult]" = OrderedDict()
        self._lock = threading.RLock()

    @property
    def config(self) -> CacheConfig:
        return self._config

    def fetch(self, key: str) -> Optional[CachedResult]:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                get_tracker().record_cache_miss()
                span = current_span()
                if span is not None:
                    span.mark_cache_miss("miss")
                return None
            if self._config.ttl_seconds > 0 and now - entry.created_at >= self._config.ttl_seconds:
                self._entries.pop(key, None)
                get_tracker().record_cache_miss()
                span = current_span()
                if span is not None:
                    span.mark_cache_miss("expired")
                return None
            # Re-queue entry for LRU
            self._entries.move_to_end(key)

        get_tracker().record_cache_hit()
        span = current_span()
        if span is not None:
            span.mark_cache_hit(
                ttl=self._config.ttl_seconds,
                age=now - entry.created_at,
            )
        return entry

    def store(self, key: str, chunks: Sequence[str]) -> CachedResult:
        entry = CachedResult(key=key, chunks=tuple(chunks), created_at=time.monotonic())
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._config.max_entries:
                self._entries.popitem(last=False)
        return entry


class ProviderQueryCache:
    """Maintain per-provider caches keyed by provider signature."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._caches: dict[str, ProviderCache] = {}

    def acquire(self, signature: str, config: CacheConfig) -> Optional[ProviderCache]:
        if not config.enabled or config.max_entries <= 0:
            return None
        with self._lock:
            cache = self._caches.get(signature)
            if cache is None or cache.config != config:
                cache = ProviderCache(config)
                self._caches[signature] = cache
            return cache

    def clear(self) -> None:
        with self._lock:
            self._caches.clear()


def build_cache_key(
    *,
    provider: str,
    base_url: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Create a stable cache key for a provider query."""

    payload: dict[str, Any] = {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "messages": _normalise_messages(messages),
    }
    if tools:
        payload["tools"] = _normalise_seq(tools)
    if extra:
        payload["extra"] = _normalise_mapping(extra)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def config_from_ai(ai_config: Any) -> CacheConfig:
    """Translate ``AIConfig`` values into a cache configuration."""

    enabled = bool(getattr(ai_config, "cache_enabled", True))
    max_entries = int(getattr(ai_config, "cache_max_entries", 128) or 0)
    ttl_seconds = float(getattr(ai_config, "cache_ttl_seconds", 120.0) or 0.0)
    if max_entries < 1:
        enabled = False
        max_entries = 0
    if ttl_seconds < 0:
        ttl_seconds = 0.0
    return CacheConfig(enabled=enabled, max_entries=max_entries, ttl_seconds=ttl_seconds)


def _normalise_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for item in messages:
        entry = {
            "role": item.get("role"),
            "content": item.get("content"),
        }
        # Preserve additional keys in sorted order for stability
        for key in sorted(k for k in item.keys() if k not in entry):
            entry[key] = item[key]
        normalised.append(entry)
    return normalised


def _normalise_seq(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_normalise_mapping(item) for item in items]


def _normalise_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: data[key] for key in sorted(data.keys())}
