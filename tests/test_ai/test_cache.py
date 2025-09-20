"""Tests for the AI provider cache helpers."""

from __future__ import annotations

from typing import Any

import pytest

from novelwriter.ai.cache import CacheConfig, ProviderQueryCache, build_cache_key, config_from_ai
from novelwriter.ai.performance import get_tracker


@pytest.fixture(autouse=True)
def reset_tracker() -> None:
    tracker = get_tracker()
    tracker.reset()
    tracker.configure(enabled=False, log_path=None, max_samples=10)


def _make_key(message: str, extra: dict[str, Any] | None = None) -> str:
    return build_cache_key(
        provider="openai",
        base_url="https://api.example.com",
        model="demo-model",
        messages=[{"role": "user", "content": message}],
        tools=None,
        extra=extra or {},
    )


def test_cache_hit_and_expiry(monkeypatch) -> None:
    values = {"t": 0.0}

    def fake_monotonic() -> float:
        return values["t"]

    monkeypatch.setattr("novelwriter.ai.cache.time.monotonic", fake_monotonic)

    registry = ProviderQueryCache()
    cache = registry.acquire("sig", CacheConfig(enabled=True, max_entries=3, ttl_seconds=10.0))
    assert cache is not None

    tracker = get_tracker()

    key = _make_key("hello")
    with tracker.start_request("provider", stream=True, timeout=None) as span:
        miss = cache.fetch(key)
        span.finish("miss")
    assert miss is None

    cache.store(key, ["Hello there!"])

    values["t"] = 5.0
    with tracker.start_request("provider", stream=True, timeout=None) as span:
        hit = cache.fetch(key)
        assert hit is not None
        span.finish()
        assert span.cache_status == "hit"

    values["t"] = 15.0
    with tracker.start_request("provider", stream=True, timeout=None) as span:
        expired = cache.fetch(key)
        span.finish("expired")
    assert expired is None


def test_cache_lru_eviction(monkeypatch) -> None:
    tick = {"value": 0.0}

    def fake_monotonic() -> float:
        return tick["value"]

    monkeypatch.setattr("novelwriter.ai.cache.time.monotonic", fake_monotonic)

    registry = ProviderQueryCache()
    cache = registry.acquire("sig", CacheConfig(enabled=True, max_entries=2, ttl_seconds=100.0))
    assert cache is not None

    key_a = _make_key("A")
    key_b = _make_key("B")
    key_c = _make_key("C")

    cache.store(key_a, ["A"])
    tick["value"] = 1.0
    cache.store(key_b, ["B"])
    tick["value"] = 2.0
    cache.store(key_c, ["C"])

    # Oldest entry (key_a) should be evicted when inserting key_c
    assert cache.fetch(key_a) is None
    assert cache.fetch(key_b) is not None
    assert cache.fetch(key_c) is not None


def test_config_from_ai_defaults() -> None:
    class Dummy:
        cache_enabled = False
        cache_max_entries = 0
        cache_ttl_seconds = -1

    conf = config_from_ai(Dummy())
    assert conf.enabled is False
    assert conf.max_entries == 0
    assert conf.ttl_seconds == 0.0
