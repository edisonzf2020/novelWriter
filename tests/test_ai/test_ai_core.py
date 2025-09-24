"""
novelWriter – AI Core Service Tests
====================================

This file is a part of novelWriter
Copyright (C) 2025 Veronica Berglyd Olsen and novelWriter contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch

from novelwriter.ai import AICoreService, NWAiApiError
from novelwriter.ai.models import TextRange, Suggestion, ProofreadResult, ModelInfo
from novelwriter.ai.providers import ProviderCapabilities
from novelwriter.api import NovelWriterAPI


class TestAICoreService:
    """Test the AICoreService class."""

    @pytest.fixture
    def mock_api(self):
        """Create a mock NovelWriterAPI instance."""
        api = Mock()  # Don't use spec to allow adding methods
        api.getProjectMeta.return_value = {
            "name": "Test Project",
            "author": "Test Author",
            "language": "en_US",
            "totalWords": 1000,
            "novelWords": 800,
        }
        api.listDocuments.return_value = [
            {"handle": "doc1", "name": "Chapter 1"},
            {"handle": "doc2", "name": "Chapter 2"},
        ]
        api.getDocText.return_value = "This is test document content."
        api.getCurrentDocument.return_value = {"handle": "doc1", "name": "Chapter 1"}
        return api

    @pytest.fixture
    def ai_core(self, mock_api):
        """Create an AICoreService instance with mock API."""
        return AICoreService(mock_api)

    def test_init(self, mock_api):
        """Test AICoreService initialization."""
        ai_core = AICoreService(mock_api)
        assert ai_core._api is mock_api
        assert ai_core._transaction_stack == []
        assert ai_core._pending_suggestions == {}
        assert ai_core._provider is None

    def test_collect_context_current_document(self, ai_core, mock_api):
        """Test collecting context for current document."""
        context = ai_core.collectContext(scope="current_document", limit=100)
        assert "This is test document content." in context
        mock_api.getCurrentDocument.assert_called_once()
        mock_api.getDocText.assert_called_once_with("doc1")

    def test_collect_context_project(self, ai_core, mock_api):
        """Test collecting context for entire project."""
        context = ai_core.collectContext(scope="project", limit=1000)
        assert "Test Project" in context
        assert "Test Author" in context
        assert "Chapter 1" in context
        mock_api.getProjectMeta.assert_called_once()
        mock_api.listDocuments.assert_called()

    def test_collect_context_invalid_scope(self, ai_core):
        """Test collecting context with invalid scope."""
        with pytest.raises(NWAiApiError) as exc:
            ai_core.collectContext(scope="invalid_scope")
        assert "Invalid context scope" in str(exc.value)

    def test_preview_suggestion(self, ai_core, mock_api):
        """Test generating a preview suggestion."""
        # Start a transaction first
        txn_id = ai_core.begin_transaction()
        
        # Create a suggestion
        rng = TextRange(start=5, end=7)
        suggestion = ai_core.previewSuggestion("doc1", rng, "was")
        
        assert isinstance(suggestion, Suggestion)
        assert suggestion.handle == "doc1"
        assert suggestion.preview == "This was test document content."
        assert suggestion.id is not None
        
        # Verify API was called
        mock_api.getDocText.assert_called_with("doc1")
        
        # Clean up
        ai_core.rollback_transaction(txn_id)

    def test_preview_suggestion_invalid_range(self, ai_core, mock_api):
        """Test preview suggestion with invalid range."""
        txn_id = ai_core.begin_transaction()
        
        # Invalid range (end before start)
        rng = TextRange(start=10, end=5)
        with pytest.raises(NWAiApiError) as exc:
            ai_core.previewSuggestion("doc1", rng, "replacement")
        assert "Invalid text range" in str(exc.value)
        
        ai_core.rollback_transaction(txn_id)

    def test_preview_suggestion_no_transaction(self, ai_core):
        """Test preview suggestion without active transaction."""
        rng = TextRange(start=5, end=7)
        with pytest.raises(NWAiApiError) as exc:
            ai_core.previewSuggestion("doc1", rng, "replacement")
        assert "transaction must be active" in str(exc.value)

    def test_transaction_management(self, ai_core):
        """Test transaction begin, commit, and rollback."""
        # Begin transaction
        txn_id = ai_core.begin_transaction()
        assert txn_id.startswith("txn_")
        assert len(ai_core._transaction_stack) == 1
        
        # Commit transaction
        result = ai_core.commit_transaction(txn_id)
        assert result is True
        assert len(ai_core._transaction_stack) == 0
        
        # Begin another transaction
        txn_id2 = ai_core.begin_transaction()
        assert txn_id2 != txn_id
        
        # Rollback transaction
        result = ai_core.rollback_transaction(txn_id2)
        assert result is True
        assert len(ai_core._transaction_stack) == 0

    def test_transaction_mismatch(self, ai_core):
        """Test transaction operations with mismatched IDs."""
        txn_id = ai_core.begin_transaction()
        
        with pytest.raises(NWAiApiError) as exc:
            ai_core.commit_transaction("wrong_id")
        assert "Transaction mismatch" in str(exc.value)
        
        # Clean up
        ai_core.rollback_transaction(txn_id)

    def test_conversation_memory(self, ai_core):
        """Test conversation memory management."""
        memory = ai_core.getConversationMemory()
        assert memory is not None
        assert memory == ai_core._conversation_memory

    @patch('novelwriter.ai.ai_core.CONFIG')
    def test_provider_management(self, mock_config, ai_core):
        """Test AI provider management."""
        # Setup mock config
        mock_ai_config = Mock()
        mock_ai_config.enabled = True
        mock_ai_config.provider = "test_provider"
        mock_ai_config.create_provider.return_value = Mock()
        mock_config.ai = mock_ai_config
        
        # Ensure provider
        provider = ai_core._ensure_provider()
        assert provider is not None
        assert ai_core._provider is provider
        mock_ai_config.create_provider.assert_called_once()
        
        # Reset provider
        ai_core.resetProvider()
        assert ai_core._provider is None

    @patch('novelwriter.ai.ai_core.CONFIG')
    def test_provider_disabled(self, mock_config, ai_core):
        """Test provider when AI is disabled."""
        mock_ai_config = Mock()
        mock_ai_config.enabled = False
        mock_config.ai = mock_ai_config
        
        with pytest.raises(NWAiApiError) as exc:
            ai_core._ensure_provider()
        assert "AI provider support is disabled" in str(exc.value)

    def test_build_model_info(self, ai_core):
        """Test building model info from provider payload."""
        payload = {
            "id": "model-123",
            "name": "Test Model",
            "description": "A test model",
            "input_token_limit": 4096,
            "output_token_limit": 2048,
            "owned_by": "test_org",
            "capabilities": ["chat", "completion"],
        }
        
        info = ai_core._build_model_info(payload)
        assert isinstance(info, ModelInfo)
        assert info.id == "model-123"
        assert info.display_name == "Test Model"
        assert info.description == "A test model"
        assert info.input_token_limit == 4096
        assert info.output_token_limit == 2048

    def test_build_model_info_invalid(self, ai_core):
        """Test building model info with invalid payload."""
        # Empty payload
        assert ai_core._build_model_info({}) is None
        
        # Non-mapping payload
        assert ai_core._build_model_info("not a dict") is None
        
        # Missing ID
        assert ai_core._build_model_info({"name": "Test"}) is None

    @patch('novelwriter.ai.ai_core.CONFIG')
    def test_stream_chat_completion(self, mock_config, ai_core):
        """Test streaming chat completion."""
        # Setup mock provider
        mock_provider = Mock()
        mock_provider.chat_completion.return_value = iter(["Hello", " ", "world"])
        
        mock_ai_config = Mock()
        mock_ai_config.enabled = True
        mock_ai_config.provider = "test_provider"
        mock_ai_config.enable_cache = False
        mock_ai_config.create_provider.return_value = mock_provider
        mock_config.ai = mock_ai_config
        
        # Stream completion
        messages = [{"role": "user", "content": "Test message"}]
        result = ai_core.streamChatCompletion(messages)
        
        # Collect streamed chunks
        chunks = list(result)
        assert chunks == ["Hello", " ", "world"]
        
        mock_provider.chat_completion.assert_called_once()

    def test_proofread_document(self, ai_core, mock_api):
        """Test document proofreading."""
        # Mock the provider and stream completion
        with patch.object(ai_core, '_ensure_provider') as mock_ensure:
            with patch.object(ai_core, 'streamChatCompletion') as mock_stream:
                mock_stream.return_value = iter(["Grammar ", "looks ", "good."])
                
                result = ai_core.proofreadDocument("doc1")
                
                assert isinstance(result, ProofreadResult)
                assert result.transaction_id.startswith("txn_")
                assert isinstance(result.suggestion, Suggestion)
                assert result.original_text == "This is test document content."
                assert result.diff_stats == {"lines_changed": 0}
                
                mock_api.getDocText.assert_called_with("doc1")
                mock_stream.assert_called_once()

    def test_format_memory_context(self, ai_core):
        """Test formatting memory context."""
        # Add some turns to memory
        ai_core._conversation_memory.add_turn(
            user_input="Hello",
            ai_response="Hi there!",
            context_scope="current_document",
            context_summary="Greeting"
        )
        
        context = ai_core._format_memory_context(
            "current_document",
            max_turns=5,
            include_cross_scope=True
        )
        
        assert "Conversation Memory" in context
        assert "Hello" in context
        assert "Hi there!" in context

    def test_format_memory_context_empty(self, ai_core):
        """Test formatting empty memory context."""
        context = ai_core._format_memory_context(
            "current_document",
            max_turns=5,
            include_cross_scope=True
        )
        assert context == ""

    def test_audit_logging(self, ai_core):
        """Test audit logging functionality."""
        # Record an audit entry
        ai_core._record_audit(
            transaction_id="test_txn",
            operation="test.operation",
            target="test_target",
            summary="Test summary",
            level="info",
            metadata={"key": "value"}
        )
        
        # Check audit log
        assert len(ai_core._audit_log) == 1
        entry = ai_core._audit_log[0]
        assert entry.transaction_id == "test_txn"
        assert entry.operation == "test.operation"
        assert entry.target == "test_target"
        assert entry.summary == "Test summary"
        assert entry.level == "info"
        assert entry.metadata["key"] == "value"

    def test_pending_operations_recording(self, ai_core):
        """Test recording pending operations."""
        from novelwriter.ai.ai_core import _PendingOperation
        
        operations = [
            _PendingOperation(
                operation="test.op1",
                target="target1",
                summary="Operation 1",
                metadata={"test": True}
            ),
            _PendingOperation(
                operation="test.op2",
                target="target2",
                summary="Operation 2"
            )
        ]
        
        ai_core._record_pending_operations(
            "test_txn",
            operations,
            success=True
        )
        
        # Check audit log has entries for each operation
        audit_entries = [e for e in ai_core._audit_log 
                         if e.operation == "transaction.operation.committed"]
        assert len(audit_entries) == 2

    def test_rollback_pending_operations(self, ai_core):
        """Test rolling back pending operations."""
        from novelwriter.ai.ai_core import _PendingOperation
        
        undo_called = []
        
        def undo1():
            undo_called.append("undo1")
        
        def undo2():
            undo_called.append("undo2")
        
        operations = [
            _PendingOperation(
                operation="test.op1",
                target="target1",
                summary="Operation 1",
                undo=undo1
            ),
            _PendingOperation(
                operation="test.op2",
                target="target2",
                summary="Operation 2",
                undo=undo2
            )
        ]
        
        ai_core._rollback_pending_operations("test_txn", operations)
        
        # Verify undo callbacks were called in reverse order
        assert undo_called == ["undo2", "undo1"]
        
        # Check audit log
        audit_entries = [e for e in ai_core._audit_log 
                         if e.operation == "transaction.operation.rolled_back"]
        assert len(audit_entries) == 2
