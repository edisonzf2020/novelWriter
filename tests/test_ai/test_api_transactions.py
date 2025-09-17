"""Tests for the transactional capabilities of the AI domain API."""

from __future__ import annotations

import pytest

from novelwriter.ai import NWAiApi, NWAiApiError
from novelwriter.core.project import NWProject
from novelwriter.enum import nwItemClass
from tests.tools import buildTestProject


@pytest.fixture()
def api_with_transaction_fixture(projPath, mockRnd, mockGUIwithTheme):
    """Build a project instance together with an AI facade for testing."""

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
    handles = {"note": note_handle, "outline": outline_handle}
    return project, api, handles


def test_begin_transaction_records_audit(api_with_transaction_fixture, monkeypatch):
    _, api, _ = api_with_transaction_fixture

    class _StaticUUID:
        hex = "txn-static"

    monkeypatch.setattr("novelwriter.ai.api.uuid4", lambda: _StaticUUID())

    transaction_id = api.begin_transaction()
    assert transaction_id == "txn-static"

    audit = api.get_audit_log()
    assert audit[-1]["operation"] == "transaction.begin"
    assert audit[-1]["summary"] == "depth=1"


def test_commit_requires_active_transaction(api_with_transaction_fixture):
    _, api, _ = api_with_transaction_fixture

    with pytest.raises(NWAiApiError):
        api.commit_transaction("nope")

    audit = api.get_audit_log()
    assert audit[-1]["operation"].startswith("transaction.commit.invalid")


def test_nested_commit_merges_pending_operations(api_with_transaction_fixture, monkeypatch):
    _, api, handles = api_with_transaction_fixture

    class _StaticUUID:
        hex = "txn-merge"

    monkeypatch.setattr("novelwriter.ai.api.uuid4", lambda: _StaticUUID())

    transaction_id = api.begin_transaction()
    api._queue_pending_operation("setDocText", handles["note"], "update note")

    # Begin a nested transaction; should reuse the outer identifier.
    nested_id = api.begin_transaction()
    assert nested_id == transaction_id
    api._queue_pending_operation("applySuggestion", handles["outline"], "apply outline")

    # Commit inner frame first.
    api.commit_transaction(transaction_id)
    audit = api.get_audit_log()
    assert audit[-1]["operation"] == "transaction.commit.nested"

    # Commit outer frame and ensure operations are recorded.
    api.commit_transaction(transaction_id)
    audit = api.get_audit_log()
    committed_ops = [entry for entry in audit if entry["operation"] == "transaction.operation.committed"]
    assert len(committed_ops) == 2
    assert {entry["target"] for entry in committed_ops} == {
        handles["note"],
        handles["outline"],
    }


def test_write_operations_require_transaction(api_with_transaction_fixture):
    _, api, handles = api_with_transaction_fixture

    with pytest.raises(NWAiApiError):
        api.setDocText(handles["note"], "text")

    with pytest.raises(NWAiApiError):
        api.applySuggestion("suggestion-id")


def test_rollback_invokes_undo_callbacks(api_with_transaction_fixture, monkeypatch):
    _, api, handles = api_with_transaction_fixture

    class _StaticUUID:
        hex = "txn-rollback"

    monkeypatch.setattr("novelwriter.ai.api.uuid4", lambda: _StaticUUID())

    counter = {"calls": 0}

    def undo() -> None:
        counter["calls"] += 1

    transaction_id = api.begin_transaction()
    api._queue_pending_operation(
        "setDocText",
        handles["note"],
        "update note",
        undo=undo,
    )

    api.rollback_transaction(transaction_id)

    assert counter["calls"] == 1
    audit = api.get_audit_log()
    assert any(entry["operation"] == "transaction.operation.rolled_back" for entry in audit)


def test_audit_log_returns_sorted_snapshots(api_with_transaction_fixture, monkeypatch):
    _, api, _ = api_with_transaction_fixture

    class _StaticUUID:
        hex = "txn-audit"

    monkeypatch.setattr("novelwriter.ai.api.uuid4", lambda: _StaticUUID())

    transaction_id = api.begin_transaction()
    api.commit_transaction(transaction_id)

    first_snapshot = api.get_audit_log()
    first_snapshot.append({"operation": "mutated"})

    second_snapshot = api.get_audit_log()
    assert all("timestamp" in entry for entry in second_snapshot)
    assert second_snapshot[-1]["operation"] != "mutated"
