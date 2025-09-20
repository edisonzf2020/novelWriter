"""Tests for model catalogue integration in the AI Copilot stack."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from novelwriter import CONFIG
from novelwriter.ai import AIConfig, NWAiApi, NWAiApiError
from novelwriter.ai.providers import OpenAICompatibleProvider, ProviderSettings
from novelwriter.core.project import NWProject

from tests.tools import buildTestProject


@pytest.fixture()
def project_fixture(projPath, mockRnd, mockGUIwithTheme) -> NWProject:
    project = NWProject()
    buildTestProject(project, projPath)
    return project


def _make_transport(model_list: dict[str, Any], model_detail: dict[str, Any]) -> tuple[httpx.MockTransport, dict[str, int]]:
    call_counter = {"list": 0, "detail": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            call_counter["list"] += 1
            return httpx.Response(200, json=model_list)
        if request.url.path.startswith("/v1/models/"):
            call_counter["detail"] += 1
            return httpx.Response(200, json=model_detail)
        raise AssertionError(f"Unexpected path: {request.url.path}")

    return httpx.MockTransport(handler), call_counter


def _new_provider(transport: httpx.MockTransport) -> OpenAICompatibleProvider:
    settings = ProviderSettings(
        base_url="https://mock.api",
        api_key="token",
        model="gpt-test",
        transport=transport,
    )
    return OpenAICompatibleProvider(settings)


def test_openai_provider_lists_models_with_cache() -> None:
    model_list = {
        "data": [
            {
                "id": "gpt-alpha",
                "display_name": "GPT Alpha",
                "description": "Alpha preview model",
                "input_token_limit": 8000,
                "output_token_limit": 2000,
                "owned_by": "openai",
            },
            {
                "id": "gpt-beta",
                "description": "Beta model",
                "context_length": 16000,
                "max_output_tokens": 4000,
                "owned_by": "openai:beta",
            },
        ]
    }
    detail = {"id": "gpt-test", "output_token_limit": 8192}
    transport, counter = _make_transport(model_list, detail)
    provider = _new_provider(transport)

    first = provider.list_models()

    assert [item["id"] for item in first] == ["gpt-alpha", "gpt-beta"]
    assert first[0]["metadata"]["id"] == "gpt-alpha"
    assert counter["list"] == 1

    cached = provider.list_models()
    assert cached == first
    assert cached is not first
    assert counter["list"] == 1

    refreshed = provider.list_models(force=True)
    assert refreshed == first
    assert counter["list"] == 2


def test_openai_provider_get_model_metadata_uses_cache() -> None:
    model_list = {"data": []}
    detail = {
        "id": "gpt-test",
        "display_name": "GPT Test",
        "description": "Testing",
        "output_token_limit": 2048,
    }
    transport, counter = _make_transport(model_list, detail)
    provider = _new_provider(transport)

    metadata = provider.get_model_metadata("gpt-test")
    assert metadata is not None
    assert metadata["id"] == "gpt-test"
    assert metadata["output_token_limit"] == 2048
    assert counter["detail"] == 1

    cached = provider.get_model_metadata("gpt-test")
    assert cached == metadata
    assert counter["detail"] == 1

    forced = provider.get_model_metadata("gpt-test", force=True)
    assert forced == metadata
    assert counter["detail"] == 2


def test_nwaiapi_list_available_models_returns_model_info(monkeypatch, project_fixture) -> None:
    model_list = {
        "data": [
            {
                "id": "gpt-alpha",
                "display_name": "GPT Alpha",
                "description": "Alpha preview model",
                "input_token_limit": 8000,
                "output_token_limit": 2000,
            }
        ]
    }
    detail = {"id": "gpt-test", "output_token_limit": 8192}
    transport, _ = _make_transport(model_list, detail)
    provider = _new_provider(transport)

    ai_config = AIConfig()
    ai_config.enabled = True
    ai_config.api_key = "token"
    ai_config.model = "gpt-test"
    monkeypatch.setattr(CONFIG, "_ai_config", ai_config, raising=False)

    api = NWAiApi(project_fixture)
    monkeypatch.setattr(api, "_ensure_provider", lambda: provider, raising=False)

    models = api.listAvailableModels()

    assert len(models) == 1
    info = models[0]
    assert info.id == "gpt-alpha"
    assert info.display_name == "GPT Alpha"
    assert info.output_token_limit == 2000
    assert "provider.models.listed" in {entry["operation"] for entry in api.get_audit_log()}


def test_nwaiapi_get_model_metadata_returns_model_info(monkeypatch, project_fixture) -> None:
    model_list = {"data": []}
    detail = {
        "id": "gpt-test",
        "display_name": "GPT Test",
        "description": "Primary runtime model",
        "input_token_limit": 16000,
        "output_token_limit": 4000,
    }
    transport, counter = _make_transport(model_list, detail)
    provider = _new_provider(transport)

    ai_config = AIConfig()
    ai_config.enabled = True
    ai_config.api_key = "token"
    ai_config.model = "gpt-test"
    monkeypatch.setattr(CONFIG, "_ai_config", ai_config, raising=False)

    api = NWAiApi(project_fixture)
    monkeypatch.setattr(api, "_ensure_provider", lambda: provider, raising=False)

    info = api.getModelMetadata("gpt-test")

    assert info is not None
    assert info.id == "gpt-test"
    assert info.display_name == "GPT Test"
    assert info.output_token_limit == 4000
    assert counter["detail"] == 1
    assert "provider.models.lookup" in {entry["operation"] for entry in api.get_audit_log()}


def test_nwaiapi_list_available_models_failure_sets_reason(monkeypatch, project_fixture) -> None:
    ai_config = AIConfig()
    ai_config.enabled = True
    ai_config.api_key = "token"
    ai_config.model = "gpt-test"
    monkeypatch.setattr(CONFIG, "_ai_config", ai_config, raising=False)

    class FailingProvider(SimpleNamespace):
        def list_models(self, *, force: bool = False):  # noqa: ARG002
            raise Exception("boom")

    api = NWAiApi(project_fixture)
    monkeypatch.setattr(api, "_ensure_provider", lambda: FailingProvider(), raising=False)

    with pytest.raises(NWAiApiError):
        api.listAvailableModels()

    assert ai_config.availability_reason == "boom"
