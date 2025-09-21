"""Tests for AI history timeline and rollback features."""
from __future__ import annotations

import pytest

from novelwriter import CONFIG
from novelwriter.ai import NWAiApi, NWAiApiError
from novelwriter.core.project import NWProject
from novelwriter.enum import nwItemClass

from tests.tools import buildTestProject


@pytest.fixture()
def api_with_project(projPath, mockRnd, mockGUIwithTheme):
    """Create a minimal project and associated API facade."""

    project = NWProject()
    buildTestProject(project, projPath)

    world_root = project.tree.findRoot(nwItemClass.WORLD)
    assert world_root is not None

    note_handle = project.newFile("History Note", world_root)
    assert note_handle is not None
    document = project.storage.getDocument(note_handle)
    assert document is not None
    document.writeDocument("# History Note\n\nOriginal content line.\n")
    project.index.reIndexHandle(note_handle)
    project.updateCounts()

    api = NWAiApi(project)
    return project, api, note_handle


def _enable_ai(monkeypatch) -> None:
    monkeypatch.setattr(CONFIG.ai, "enabled", True, raising=False)
    monkeypatch.setattr(CONFIG.ai, "api_key", "test-ai-key", raising=False)
    monkeypatch.setattr(CONFIG.ai, "_api_key_from_env", False, raising=False)
    monkeypatch.setattr(CONFIG.ai, "proofreading_enabled", True, raising=False)


class TestHistoryTimeline:
    """Validate history snapshot aggregation and rollback operations."""

    def test_snapshot_records_suggestion_metadata(self, api_with_project, monkeypatch) -> None:
        project, api, handle = api_with_project

        new_text = "# History Note\n\nChanged content.\n"

        _enable_ai(monkeypatch)

        def fake_completion(self, messages, *, stream=True, tools=None, extra=None):  # noqa: ARG002
            return iter([new_text])

        monkeypatch.setattr(NWAiApi, "streamChatCompletion", fake_completion, raising=False)

        result = api.proofreadDocument(handle)
        suggestion = result.suggestion
        transaction_id = result.transaction_id
        assert suggestion.diff is not None
        applied = api.applySuggestion(suggestion.id)
        assert applied is True
        api.commit_transaction(transaction_id)

        snapshot = api.getHistorySnapshot(transaction_limit=5)
        transactions = snapshot.get("transactions", [])
        assert transactions

        target = next(
            (txn for txn in transactions if txn.get("transaction_id") == transaction_id),
            None,
        )
        assert target is not None
        assert target.get("status") == "committed"
        assert target.get("rollback_available") is True

        operations = target.get("operations", [])
        assert operations, "Expected committed operations in timeline"
        op_metadata = operations[0].get("metadata", {})
        assert "diff_stats" in op_metadata
        diff_stats = op_metadata["diff_stats"]
        assert diff_stats["additions"] >= 1

        preview_event = next(
            (
                event
                for event in target.get("events", [])
                if event.get("operation") == "suggestion.applied"
            ),
            None,
        )
        assert preview_event is not None
        applied_metadata = preview_event.get("metadata", {})
        assert applied_metadata.get("suggestion_id") == suggestion.id
        assert "diff" in applied_metadata

        assert api.getDocText(handle).rstrip() == new_text.rstrip()

    def test_manual_rollback_restores_content(self, api_with_project, monkeypatch) -> None:
        project, api, handle = api_with_project

        # Store original content before making changes
        original_text = """# History Note

Original content line.
"""
        new_text = """# Rolled Back

Content to revert.
"""

        _enable_ai(monkeypatch)

        def fake_completion(self, messages, *, stream=True, tools=None, extra=None):  # noqa: ARG002
            return iter([new_text])

        monkeypatch.setattr(NWAiApi, "streamChatCompletion", fake_completion, raising=False)

        result = api.proofreadDocument(handle)
        transaction_id = result.transaction_id
        api.applySuggestion(result.suggestion.id)
        api.commit_transaction(transaction_id)

        assert api.getDocText(handle).rstrip() == new_text.rstrip()

        api.rollbackHistoryTransaction(transaction_id)
        assert api.getDocText(handle).rstrip() == original_text.rstrip()

        snapshot = api.getHistorySnapshot(transaction_limit=5)
        transactions = snapshot.get("transactions", [])
        rolled = next(
            (txn for txn in transactions if txn.get("transaction_id") == transaction_id),
            None,
        )
        assert rolled is not None
        assert rolled.get("status") == "rolled_back"
        assert rolled.get("rollback_available") is False
        assert rolled.get("operations"), "Operations summary should remain after rollback"

    def test_rollback_without_metadata_raises(self, api_with_project) -> None:
        project, api, handle = api_with_project

        with pytest.raises(NWAiApiError):
            api.rollbackHistoryTransaction("unknown-tx")
