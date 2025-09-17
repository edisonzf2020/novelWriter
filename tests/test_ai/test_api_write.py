"""Test suite for NWAiApi write operations (setDocText, suggestions)."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from novelwriter import CONFIG
from novelwriter.ai import NWAiApi, NWAiApiError, Suggestion, TextRange
from novelwriter.core.project import NWProject
from novelwriter.enum import nwItemClass

from tests.tools import buildTestProject


@pytest.fixture()
def api_with_project(projPath, mockRnd, mockGUIwithTheme):
    """Build a minimal project and return it together with an API facade."""

    project = NWProject()
    buildTestProject(project, projPath)

    tree = project.tree
    storage = project.storage

    world_root = tree.findRoot(nwItemClass.WORLD)
    assert world_root is not None

    note_handle = project.newFile("Test Note", world_root)
    assert note_handle is not None
    storage.getDocument(note_handle).writeDocument("# Test Note\n\nOriginal content.\nSecond line.\n")
    project.index.reIndexHandle(note_handle)
    project.updateCounts()

    api = NWAiApi(project)
    return project, api, note_handle


class TestSetDocText:
    """Test document text setting functionality."""

    def test_setDocText_requires_active_transaction(self, api_with_project) -> None:
        """setDocText should fail without an active transaction."""
        project, api, handle = api_with_project
        
        with pytest.raises(NWAiApiError, match="A transaction must be active"):
            api.setDocText(handle, "New content", apply=True)

    def test_setDocText_dry_run_mode(self, api_with_project) -> None:
        """setDocText with apply=False should return False and not modify document."""
        project, api, handle = api_with_project
        
        original_text = api.getDocText(handle)
        new_text = "# Modified Note\n\nNew content here.\n"
        
        tx_id = api.begin_transaction()
        result = api.setDocText(handle, new_text, apply=False)
        api.commit_transaction(tx_id)
            
        assert result is False  # Dry run mode
        assert api.getDocText(handle) == original_text  # No changes

    def test_setDocText_apply_mode(self, api_with_project) -> None:
        """setDocText with apply=True should modify document and return True."""
        project, api, handle = api_with_project
        
        original_text = api.getDocText(handle)
        new_text = "# Modified Note\n\nNew content here.\n"
        
        tx_id = api.begin_transaction()
        result = api.setDocText(handle, new_text, apply=True)
        api.commit_transaction(tx_id)
            
        assert result is True
        assert api.getDocText(handle) == new_text
        assert api.getDocText(handle) != original_text

    def test_setDocText_rollback_restores_original(self, api_with_project) -> None:
        """Transaction rollback should restore original document content."""
        project, api, handle = api_with_project
        
        original_text = api.getDocText(handle)
        new_text = "# Rollback Test\n\nThis should be rolled back.\n"
        
        tx_id = api.begin_transaction()
        api.setDocText(handle, new_text, apply=True)
        assert api.getDocText(handle) == new_text  # Modified during transaction
        api.rollback_transaction(tx_id)
            
        # After rollback, should be restored
        assert api.getDocText(handle) == original_text


class TestSuggestions:
    """Test suggestion preview and application functionality."""

    def test_previewSuggestion_creates_valid_suggestion(self, api_with_project) -> None:
        """previewSuggestion should create a valid suggestion with diff."""
        project, api, handle = api_with_project
        
        original_text = api.getDocText(handle)
        text_range = TextRange(start=0, end=12)  # "# Test Note\n"
        new_text = "# Updated Note\n"
        
        tx_id = api.begin_transaction()
        suggestion = api.previewSuggestion(handle, text_range, new_text)
        api.commit_transaction(tx_id)
            
        assert isinstance(suggestion, Suggestion)
        assert suggestion.handle == handle
        assert suggestion.id is not None
        assert suggestion.diff is not None
        assert new_text in suggestion.preview
        assert "Updated Note" in suggestion.preview

    def test_applySuggestion_applies_cached_suggestion(self, api_with_project) -> None:
        """applySuggestion should apply a cached suggestion."""
        project, api, handle = api_with_project
        
        original_text = api.getDocText(handle)
        text_range = TextRange(start=0, end=12)  # "# Test Note\n"
        new_text = "# Applied Note\n"
        
        tx_id = api.begin_transaction()
        suggestion = api.previewSuggestion(handle, text_range, new_text)
        result = api.applySuggestion(suggestion.id)
        api.commit_transaction(tx_id)
            
        assert result is True
        modified_text = api.getDocText(handle)
        assert "Applied Note" in modified_text
        assert modified_text != original_text

    def test_applySuggestion_unknown_id_raises_error(self, api_with_project) -> None:
        """applySuggestion with unknown ID should raise NWAiApiError."""
        project, api, handle = api_with_project
        
        tx_id = api.begin_transaction()
        try:
            with pytest.raises(NWAiApiError, match="Unknown or expired suggestion"):
                api.applySuggestion("unknown-suggestion-id")
        finally:
            api.rollback_transaction(tx_id)

    def test_suggestion_requires_active_transaction(self, api_with_project) -> None:
        """Suggestion operations should require active transaction."""
        project, api, handle = api_with_project
        
        text_range = TextRange(start=0, end=5)
        
        with pytest.raises(NWAiApiError, match="A transaction must be active"):
            api.previewSuggestion(handle, text_range, "New text")


class TestSimpleIntegration:
    """Test basic integration scenarios."""

    def test_write_operations_create_audit_entries(self, api_with_project) -> None:
        """Write operations should create appropriate audit entries."""
        project, api, handle = api_with_project
        
        tx_id = api.begin_transaction()
        api.setDocText(handle, "# Audit Test\n\nNew content.", apply=True)
        api.commit_transaction(tx_id)
            
        audit_log = api.get_audit_log()
        
        # Should have transaction begin and commit entries
        assert len(audit_log) >= 2
        event_types = [entry["operation"] for entry in audit_log]
        assert "transaction.begin" in event_types
        assert "transaction.operation.committed" in event_types

    def test_setDocText_and_suggestions_together(self, api_with_project) -> None:
        """setDocText and suggestions can be used together in same transaction."""
        project, api, handle = api_with_project
        
        tx_id = api.begin_transaction()
        # Direct text setting
        api.setDocText(handle, "# Mixed Test\n\nDirect content.\n", apply=True)
        
        # Create and apply suggestion on the new content
        current_text = api.getDocText(handle)
        suggestion = api.previewSuggestion(handle, TextRange(0, 13), "# Combined Test\n")
        api.applySuggestion(suggestion.id)
        api.commit_transaction(tx_id)
            
        final_text = api.getDocText(handle)
        assert "Combined Test" in final_text