"""Performance instrumentation for the novelWriter AI pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional
from uuid import uuid4

__all__ = [
    "PerformanceTracker",
    "PerformanceSpan",
    "current_span",
    "get_tracker",
]


@dataclass(slots=True)
class PerformanceSample:
    """Immutable snapshot describing a completed provider request."""

    request_id: str
    provider_id: str
    status: str
    duration: float
    output_chars: int
    stream: bool
    cache_status: str
    endpoint: Optional[str]
    degraded_from: Optional[str]
    timeout: Optional[float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceSpan:
    """Context manager capturing metrics for a single provider call."""

    __slots__ = (
        "_tracker",
        "request_id",
        "provider_id",
        "stream",
        "timeout",
        "_token",
        "_started_perf",
        "_started_wall",
        "_finished",
        "output_chars",
        "cache_status",
        "endpoint",
        "degraded_from",
        "metadata",
        "error",
        "status",
    )

    def __init__(
        self,
        tracker: "PerformanceTracker",
        provider_id: str,
        *,
        stream: bool,
        timeout: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tracker = tracker
        self.request_id = uuid4().hex
        self.provider_id = provider_id
        self.stream = stream
        self.timeout = timeout
        self._token = None
        self._started_perf = time.perf_counter()
        self._started_wall = time.time()
        self._finished = False
        self.output_chars = 0
        self.cache_status = "miss"
        self.endpoint: Optional[str] = None
        self.degraded_from: Optional[str] = None
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.error: Optional[str] = None
        self.status: str = "pending"

    def __enter__(self) -> "PerformanceSpan":
        self._token = _CURRENT_SPAN.set(self)
        self._tracker._register_span(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _CURRENT_SPAN.reset(self._token)
        if exc_type is not None and not self._finished:
            self.fail(str(exc) or exc_type.__name__)
        return False

    # ------------------------------------------------------------------
    # Span mutations used by the streaming pipeline
    # ------------------------------------------------------------------
    def add_output(self, chars: int) -> None:
        self.output_chars += max(0, int(chars))

    def mark_cache_hit(self, *, ttl: float, age: float) -> None:
        self.cache_status = "hit"
        self.metadata.setdefault("cache_age_s", round(age, 3))
        self.metadata.setdefault("cache_ttl_s", round(ttl, 3))

    def mark_cache_miss(self, reason: str) -> None:
        self.cache_status = f"miss:{reason}" if reason else "miss"

    def set_endpoint(self, endpoint: Optional[str]) -> None:
        if endpoint:
            self.endpoint = endpoint

    def mark_degraded(self, original: str, fallback: str) -> None:
        self.degraded_from = f"{original}->{fallback}"

    def finish(self, status: str = "success") -> None:
        if self._finished:
            return
        self.status = status
        self._finished = True
        self._tracker._finalise_span(self)

    def cancel(self, reason: str = "cancelled") -> None:
        if self._finished:
            return
        self.status = reason
        self._finished = True
        self._tracker._finalise_span(self)

    def fail(self, message: str) -> None:
        if self._finished:
            return
        self.status = "error"
        self.error = message
        self._finished = True
        self._tracker._finalise_span(self)


class PerformanceTracker:
    """Central hub that aggregates metrics and writes debug logs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = False
        self._log_path = Path(".ai") / "debug-log.md"
        self._max_samples = 200
        self._samples: list[PerformanceSample] = []
        self._active: dict[str, PerformanceSpan] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    # ------------------------------------------------------------------
    # Configuration & lifecycle
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        enabled: bool,
        log_path: Optional[Path] = None,
        max_samples: int = 200,
    ) -> None:
        with self._lock:
            self._enabled = enabled
            if log_path is not None:
                self._log_path = log_path
            self._max_samples = max(1, max_samples)
            self._samples.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._active.clear()

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._active.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    # ------------------------------------------------------------------
    # Span helpers
    # ------------------------------------------------------------------
    def start_request(
        self,
        provider_id: str,
        *,
        stream: bool,
        timeout: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceSpan:
        return PerformanceSpan(
            self,
            provider_id,
            stream=stream,
            timeout=timeout,
            metadata=metadata,
        )

    def _register_span(self, span: PerformanceSpan) -> None:
        with self._lock:
            self._active[span.request_id] = span

    def _finalise_span(self, span: PerformanceSpan) -> None:
        with self._lock:
            self._active.pop(span.request_id, None)
            if not self._enabled:
                return
            duration = time.perf_counter() - span._started_perf
            sample = PerformanceSample(
                request_id=span.request_id,
                provider_id=span.provider_id,
                status="error" if span.status == "error" else span.status,
                duration=duration,
                output_chars=span.output_chars,
                stream=span.stream,
                cache_status=span.cache_status,
                endpoint=span.endpoint,
                degraded_from=span.degraded_from,
                timeout=span.timeout,
                timestamp=span._started_wall,
                metadata=self._build_metadata(span),
            )
            self._samples.append(sample)
            if len(self._samples) > self._max_samples:
                self._samples.pop(0)
            self._write_log_entry(sample)

    def _build_metadata(self, span: PerformanceSpan) -> Dict[str, Any]:
        payload = dict(span.metadata)
        if span.error:
            payload["error"] = span.error
        return payload

    # ------------------------------------------------------------------
    # Cache accounting
    # ------------------------------------------------------------------
    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "entries": [sample for sample in self._samples],
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _write_log_entry(self, sample: PerformanceSample) -> None:
        try:
            path = self._log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.fromtimestamp(sample.timestamp, tz=timezone.utc).isoformat()
            payload = {
                "provider": sample.provider_id,
                "status": sample.status,
                "endpoint": sample.endpoint,
                "degraded": sample.degraded_from,
                "duration_ms": round(sample.duration * 1000, 3),
                "output_chars": sample.output_chars,
                "cache": sample.cache_status,
                "timeout": sample.timeout,
                "stream": sample.stream,
                "metadata": sample.metadata,
            }
            line = f"- {timestamp} | " + json.dumps(payload, ensure_ascii=True, sort_keys=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - best effort logging
            # Logging must never interfere with runtime behaviour.
            pass


_CURRENT_SPAN: ContextVar[Optional[PerformanceSpan]] = ContextVar(
    "novelwriter_ai_current_span",
    default=None,
)

_tracker_lock = threading.Lock()
_tracker_instance: Optional[PerformanceTracker] = None


def current_span() -> Optional[PerformanceSpan]:
    """Return the active performance span for the current context."""

    return _CURRENT_SPAN.get()


def get_tracker() -> PerformanceTracker:
    """Return a lazily instantiated tracker singleton."""

    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = PerformanceTracker()
        return _tracker_instance
