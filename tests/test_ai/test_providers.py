"""Tests for AI provider capability detection and behaviour."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import httpx
import pytest

from novelwriter.ai.providers import (
    OpenAICompatibleProvider,
    ProviderCapabilities,
    ProviderSettings,
)


def _make_transport(responses: dict[str, list[httpx.Response]]) -> httpx.MockTransport:
    counter = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counter[path] += 1
        bucket = responses.get(path)
        if not bucket:
            raise AssertionError(f"Unexpected path {path}")
        if path == "/v1/responses":
            payload = json.loads(request.content.decode())
            metadata = payload.get("metadata")
            if metadata:
                assert metadata.get("origin") == "novelwriter-capability-probe"
        return bucket[min(counter[path] - 1, len(bucket) - 1)]

    transport = httpx.MockTransport(handler)
    transport.call_count = counter  # type: ignore[attr-defined]
    return transport


@pytest.mark.parametrize("cached", [True, False])
def test_openai_provider_detects_responses_endpoint_and_caches_result(cached: bool) -> None:
    responses = {
        "/v1/responses": [
            httpx.Response(
                200,
                json={"id": "resp_1", "object": "realtime.response", "usage": {}},
                headers={"x-openai-limit-max-output-tokens": "2048"},
            )
        ],
        "/v1/chat/completions": [
            httpx.Response(200, json={"id": "chat_1", "usage": {"completion_tokens": 12}}),
        ],
        "/v1/models/test-model": [
            httpx.Response(200, json={"id": "test-model", "output_token_limit": 8192}),
        ],
    }
    transport = _make_transport(responses)

    settings = ProviderSettings(
        base_url="https://mock.local",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    provider = OpenAICompatibleProvider(settings)

    capabilities = provider.ensure_capabilities()
    _assert_responses_capabilities(capabilities)

    if cached:
        repeated = provider.ensure_capabilities()
        assert repeated is capabilities

    counts = transport.call_count  # type: ignore[attr-defined]
    expected = 3 if not cached else 3
    assert counts["/v1/responses"] == 1
    assert counts["/v1/chat/completions"] <= 1
    assert counts["/v1/models/test-model"] == 1


def test_openai_provider_falls_back_to_chat_completions_when_responses_missing() -> None:
    responses = {
        "/v1/responses": [
            httpx.Response(404, json={"error": {"message": "not found"}})
        ],
        "/v1/chat/completions": [
            httpx.Response(
                200,
                json={
                    "id": "chat_1",
                    "usage": {"completion_tokens": 16},
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                        }
                    ],
                },
                headers={"x-openai-limit-max-output-tokens": "3072"},
            ),
        ],
        "/v1/models/test-model": [
            httpx.Response(200, json={"id": "test-model", "output_token_limit": 4096}),
        ],
    }
    transport = _make_transport(responses)

    settings = ProviderSettings(
        base_url="https://mock.local",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    provider = OpenAICompatibleProvider(settings)

    capabilities = provider.ensure_capabilities()

    assert capabilities.preferred_endpoint == "chat_completions"
    assert capabilities.supports_responses is False
    assert capabilities.supports_chat_completions is True
    assert capabilities.supports_stream is True
    assert capabilities.max_output_tokens == 3072

    counts = transport.call_count  # type: ignore[attr-defined]
    assert counts["/v1/responses"] == 1
    assert counts["/v1/chat/completions"] == 1


def test_openai_provider_refresh_forces_new_detection() -> None:
    first_round = {
        "/v1/responses": [
            httpx.Response(500, json={"error": {"message": "server error"}})
        ],
        "/v1/chat/completions": [
            httpx.Response(200, json={"id": "chat_a", "usage": {"completion_tokens": 8}}),
        ],
        "/v1/models/test-model": [
            httpx.Response(200, json={"id": "test-model", "output_token_limit": 2048}),
        ],
    }
    second_round = {
        "/v1/responses": [
            httpx.Response(
                200,
                json={"id": "resp_b", "usage": {"output_tokens": 24}},
                headers={"x-openai-limit-max-output-tokens": "4096"},
            )
        ],
        "/v1/chat/completions": [
            httpx.Response(200, json={"id": "chat_b", "usage": {"completion_tokens": 4}}),
        ],
        "/v1/models/test-model": [
            httpx.Response(200, json={"id": "test-model", "output_token_limit": 4096}),
        ],
    }

    combined: dict[str, list[httpx.Response]] = {}
    for key in first_round:
        combined[key] = first_round[key] + second_round.get(key, [])

    transport = _make_transport(combined)

    settings = ProviderSettings(
        base_url="https://mock.local",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    provider = OpenAICompatibleProvider(settings)

    first_caps = provider.ensure_capabilities()
    assert first_caps.preferred_endpoint == "chat_completions"

    refreshed = provider.refresh_capabilities()
    assert refreshed.preferred_endpoint == "responses"
    assert refreshed.supports_responses is True
    assert refreshed.max_output_tokens == 4096

    counts = transport.call_count  # type: ignore[attr-defined]
    assert counts["/v1/responses"] == 2
    assert counts["/v1/chat/completions"] >= 2
    assert counts["/v1/models/test-model"] >= 2


def test_openai_provider_responses_fallbacks_to_string_input() -> None:
    responses = {
        "/v1/responses": [
            httpx.Response(200, json={"id": "resp_probe", "usage": {"output_tokens": 1}}),
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Invalid type for 'input': expected string, but got array.",
                        "code": "invalid_type",
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "resp_success",
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "pong"},
                            ]
                        }
                    ],
                },
            ),
        ],
        "/v1/chat/completions": [
            httpx.Response(200, json={"id": "chat_meta", "usage": {"completion_tokens": 1}}),
        ],
        "/v1/models/test-model": [
            httpx.Response(200, json={"id": "test-model", "output_token_limit": 2048}),
        ],
    }

    transport = _make_transport(responses)
    settings = ProviderSettings(
        base_url="https://mock.local",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    provider = OpenAICompatibleProvider(settings)
    provider.ensure_capabilities()

    response = provider.generate([
        {"role": "user", "content": "Say hello"},
    ])

    assert response.status_code == 200
    assert provider._responses_input_mode == "string"
    counts = transport.call_count  # type: ignore[attr-defined]
    assert counts["/v1/responses"] == 3


def _assert_responses_capabilities(capabilities: ProviderCapabilities) -> None:
    assert capabilities.preferred_endpoint == "responses"
    assert capabilities.supports_responses is True
    assert capabilities.supports_stream is True
    assert capabilities.supports_tool_calls is True
    assert capabilities.max_output_tokens == 2048
    assert "responses_probe" in capabilities.metadata
