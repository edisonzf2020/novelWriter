"""Tests for the read-only surface of the AI domain API."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import cast

import httpx
import pytest

from novelwriter import CONFIG
from novelwriter.ai import DocumentRef, NWAiApi, NWAiApiError
from novelwriter.ai.config import AIConfig
from novelwriter.ai.providers import OpenAISDKProvider, ProviderSettings
from novelwriter.core.project import NWProject
from novelwriter.enum import nwItemClass

from tests.tools import buildTestProject


@pytest.fixture()
def patched_ai_config():
    original = getattr(CONFIG, "_ai_config", None)
    cfg = AIConfig()
    CONFIG._ai_config = cfg
    try:
        yield cfg
    finally:
        if original is None:
            if hasattr(CONFIG, "_ai_config"):
                delattr(CONFIG, "_ai_config")
        else:
            CONFIG._ai_config = original


@pytest.fixture()
def api_with_project(projPath, mockRnd, mockGUIwithTheme):
    """Build a minimal project and return it together with an API facade."""

    project = NWProject()
    buildTestProject(project, projPath)

    tree = project.tree
    storage = project.storage

    world_root = tree.findRoot(nwItemClass.WORLD)
    plot_root = tree.findRoot(nwItemClass.PLOT)
    assert world_root is not None
    assert plot_root is not None

    note_handle = project.newFile("World Note", world_root)
    assert note_handle is not None
    storage.getDocument(note_handle).writeDocument("# World Note\n\nSome note text.\n")
    project.index.reIndexHandle(note_handle)

    outline_handle = project.newFile("Plot Thread", plot_root)
    assert outline_handle is not None
    storage.getDocument(outline_handle).writeDocument("## Plot Thread\n\nOutline entry.\n")
    project.index.reIndexHandle(outline_handle)

    project.updateCounts()

    api = NWAiApi(project)
    return project, api, {"note": note_handle, "outline": outline_handle}


def test_get_project_meta_returns_read_only_mapping(api_with_project) -> None:
    """The metadata snapshot should be immutable and contain key fields."""

    project, api, _ = api_with_project
    meta = api.getProjectMeta()

    assert isinstance(meta, MappingProxyType)
    assert meta["uuid"] == project.data.uuid
    assert meta["name"] == project.data.name
    assert meta["projectState"] == project.state.name
    assert meta["totalWords"] == meta["novelWords"] + meta["noteWords"]
    assert meta["totalCharacters"] == meta["novelCharacters"] + meta["noteCharacters"]
    assert meta["lastHandles"] is not project.data.lastHandle

    with pytest.raises(TypeError):
        meta["uuid"] = "override"  # type: ignore[misc]



def test_list_documents_filters_scope(api_with_project) -> None:
    """Document listing honours the requested scope filters."""

    project, api, handles = api_with_project

    all_docs = api.listDocuments()
    assert all_docs, "Expected at least one document in the default scope"

    novel_docs = api.listDocuments("novel")
    assert all(isinstance(ref, DocumentRef) for ref in novel_docs)
    assert {ref.handle for ref in novel_docs}.issubset({ref.handle for ref in all_docs})

    note_docs = api.listDocuments("note")
    note_handles = {ref.handle for ref in note_docs}
    assert handles["note"] in note_handles
    assert all(project.tree[ref.handle].isNoteLayout() for ref in note_docs)

    outline_docs = api.listDocuments("outline")
    outline_handles = {ref.handle for ref in outline_docs}
    assert handles["outline"] in outline_handles
    expected_outline = {
        item.itemHandle
        for item in project.tree
        if item.isFileType() and item.itemClass in {nwItemClass.PLOT, nwItemClass.TIMELINE}
    }
    assert outline_handles.issubset(expected_outline)

    with pytest.raises(NWAiApiError):
        api.listDocuments("unknown")


def test_list_documents_skips_inactive_items(api_with_project) -> None:
    """Inactive documents are never exposed in the listing."""

    project, api, handles = api_with_project
    project.tree[handles["note"]].setActive(False)

    all_handles = {ref.handle for ref in api.listDocuments()}
    assert handles["note"] not in all_handles

    note_scope_handles = {ref.handle for ref in api.listDocuments("note")}
    assert handles["note"] not in note_scope_handles


def test_get_current_document_returns_ref(api_with_project) -> None:
    """Current document lookup returns a valid DocumentRef when set."""

    project, api, handles = api_with_project
    project.data.setLastHandle(handles["note"], "editor")

    current = api.getCurrentDocument()
    assert isinstance(current, DocumentRef)
    assert current.handle == handles["note"]

    project.data.setLastHandle("deadbeefdeadbe", "editor")
    assert api.getCurrentDocument() is None


def test_get_doc_text_returns_document_content(api_with_project) -> None:
    """Retrieving document text should round-trip the stored content."""

    _, api, handles = api_with_project

    text = api.getDocText(handles["note"])
    assert "World Note" in text


def test_get_doc_text_raises_for_invalid_or_inactive_handles(api_with_project) -> None:
    """Invalid handles and inactive documents should raise API errors."""

    project, api, handles = api_with_project

    with pytest.raises(NWAiApiError):
        api.getDocText("does-not-exist")

    project.tree[handles["note"]].setActive(False)
    with pytest.raises(NWAiApiError):
        api.getDocText(handles["note"])


def test_get_provider_capabilities_returns_snapshot(
    patched_ai_config: AIConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patched_ai_config.enabled = True
    patched_ai_config.api_key = "token"
    patched_ai_config.model = "test-model"

    call_counter: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        call_counter[path] += 1
        if path == "/responses":
            return httpx.Response(200, json={"id": "resp", "usage": {"output_tokens": 2}}, headers={"x-openai-limit-max-output-tokens": "2048"})
        if path == "/chat/completions":
            return httpx.Response(200, json={"id": "chat", "usage": {"completion_tokens": 1}}, headers={"x-openai-limit-max-output-tokens": "1024"})
        if path == "/models/test-model":
            return httpx.Response(200, json={"id": "test-model", "output_token_limit": 512})
        raise AssertionError(f"Unexpected path {path}")

    transport = httpx.MockTransport(handler)
    settings = ProviderSettings(
        base_url="https://mock.local",
        api_key="token",
        model="test-model",
        transport=transport,
    )
    provider = OpenAISDKProvider(settings)

    def _fake_create_provider(self, *, transport=None):
        return provider

    monkeypatch.setattr(AIConfig, "create_provider", _fake_create_provider, raising=True)


    patched_ai_config.set_availability_reason("stale")

    api = NWAiApi(cast("NWProject", object()))

    snapshot = api.getProviderCapabilities()
    assert snapshot.supports_responses is True
    assert patched_ai_config.availability_reason is None

    summary = api.getProviderCapabilitiesSummary()
    assert summary["preferred_endpoint"] == "responses"

    refreshed = api.getProviderCapabilities(refresh=True)
    assert refreshed.supports_responses is True
    assert call_counter["/responses"] == 2

    api.resetProvider()
    assert provider._client is None  # type: ignore[attr-defined]


def test_get_provider_capabilities_disabled_sets_reason(
    patched_ai_config: AIConfig,
) -> None:
    patched_ai_config.enabled = False

    api = NWAiApi(cast("NWProject", object()))

    with pytest.raises(NWAiApiError):
        api.getProviderCapabilities()

    assert patched_ai_config.availability_reason is not None
