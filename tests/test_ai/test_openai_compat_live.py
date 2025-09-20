"""Optional integration test against an OpenAI-compatible endpoint."""
from __future__ import annotations

import os

import pytest

from novelwriter.ai.providers import OpenAICompatibleProvider, ProviderSettings


def _get_env(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        value = value.strip()
    return value or None


@pytest.mark.compat
def test_openai_compat_responses_endpoint_live() -> None:
    """Exercise the Responses endpoint against a user-provided compatibility server."""

    base_url = _get_env("NOVELWRITER_COMPAT_BASE")
    api_key = _get_env("NOVELWRITER_COMPAT_KEY")
    model = _get_env("NOVELWRITER_COMPAT_MODEL")

    if not (base_url and api_key and model):
        pytest.skip("compatibility endpoint credentials are not configured")

    settings = ProviderSettings(base_url=base_url, api_key=api_key, model=model)
    provider = OpenAICompatibleProvider(settings)

    capabilities = provider.ensure_capabilities()
    if not capabilities.supports_responses:
        pytest.skip("remote endpoint does not report responses support")

    response = provider.generate(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ],
        stream=False,
    )

    assert response.status_code == 200
