"""Tests for flexible context selection and conversation memory."""
from __future__ import annotations

import pytest

from novelwriter.ai import NWAiApi, NWAiApiError
from novelwriter.core.project import NWProject
from novelwriter.enum import nwItemClass

from tests.tools import buildTestProject


@pytest.fixture()
def api_with_project(projPath, mockRnd, mockGUIwithTheme):
    """Build a small project with sample documents for context tests."""

    project = NWProject()
    buildTestProject(project, projPath)

    storage = project.storage
    tree = project.tree

    # Capture handles for different scopes
    novel_root = tree.findRoot(nwItemClass.NOVEL)
    plot_root = tree.findRoot(nwItemClass.PLOT)
    assert novel_root is not None
    assert plot_root is not None

    novel_handle = project.newFile("Chapter 1", novel_root)
    assert novel_handle is not None
    storage.getDocument(novel_handle).writeDocument("# Chapter 1\n\nOnce upon a time...")

    outline_handle = project.newFile("Outline Entry", plot_root)
    assert outline_handle is not None
    storage.getDocument(outline_handle).writeDocument("Scene summary line one.\nMore details here.")

    project.updateCounts()

    api = NWAiApi(project)
    handles = {
        "novel": novel_handle,
        "outline": outline_handle,
    }
    return project, api, handles


def test_collect_context_from_selection(api_with_project) -> None:
    """Selection context should reflect user provided snippets."""

    _, api, _ = api_with_project

    assert api.collectContext("selection", selection_text="") == "No text is currently selected in the editor."
    assert api.collectContext("selection", selection_text="  Focus text  ") == "Focus text"


def test_collect_context_from_current_document(api_with_project) -> None:
    """Current document scope returns the active document content."""

    project, api, handles = api_with_project
    project.data.setLastHandle(handles["novel"], "editor")

    context = api.collectContext("current_document")
    assert context.startswith("# Document: Chapter 1")
    assert "Once upon a time" in context


def test_collect_outline_context(api_with_project) -> None:
    """Outline scope returns a structured summary of outline items."""

    _, api, handles = api_with_project
    context = api.collectContext("outline")

    assert "# Project Outline" in context
    assert "Outline Entry" in context
    assert "Scene summary" in context


def test_collect_project_context_applies_length_limit(api_with_project) -> None:
    """Project scope honours explicit length limits and truncates output."""

    _, api, _ = api_with_project

    context = api.collectContext("project", max_length=80)
    assert "# Complete Project Context" in context
    assert "Project context truncated" in context


def test_collect_context_defaults_to_current_document(api_with_project) -> None:
    """Empty scope values should fall back to the current document scope."""

    project, api, handles = api_with_project
    project.data.setLastHandle(handles["novel"], "editor")

    context = api.collectContext("")
    assert context.startswith("# Document: Chapter 1")


def test_collect_context_with_memory_inclusion(api_with_project) -> None:
    """Conversation memory can be appended to collected context."""

    project, api, handles = api_with_project
    project.data.setLastHandle(handles["novel"], "editor")

    api.logConversationTurn(
        "What happened in the last chapter?",
        "The hero discovered the hidden map.",
        context_scope="current_document",
        context_summary="Discussing recent chapter content",
    )

    context = api.collectContext("current_document", include_memory=True, memory_turns=2)
    assert "# Conversation Memory" in context
    assert "What happened in the last chapter?" in context
    assert "The hero discovered the hidden map." in context


def test_get_conversation_history_returns_recent_turns(api_with_project) -> None:
    """Retrieve recent conversation turns filtered by scope."""

    _, api, _ = api_with_project

    api.logConversationTurn("First question", "First answer", context_scope="outline")
    api.logConversationTurn("Second question", "Second answer", context_scope="outline")
    api.logConversationTurn("Cross scope", "Response", context_scope="project")

    history = api.getConversationHistory("outline", max_turns=2)
    assert len(history) == 2
    assert history[0]["user_input"] == "Second question"
    assert history[1]["user_input"] == "First question"


@pytest.mark.parametrize("invalid_scope", ["unknown", "document"])
def test_collect_context_rejects_invalid_scope(api_with_project, invalid_scope: str) -> None:
    """Invalid scopes raise an API error."""

    _, api, _ = api_with_project
    with pytest.raises(NWAiApiError):
        api.collectContext(invalid_scope)
