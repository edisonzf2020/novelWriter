"""novelWriter – NovelWriter API Tests.

====================================

Tests for the unified data access API.

File History:
Created: 2025-09-24 [MCP-v1.0] API Tests

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

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from novelwriter.api import (
    APIError,
    APINotFoundError,
    APIOperationError,
    APIPermissionError,
    APIValidationError,
    NovelWriterAPI,
)
from novelwriter.enum import nwItemClass, nwItemLayout


class TestNovelWriterAPI:
    """Test the NovelWriter unified API."""

    @pytest.fixture
    def mockProject(self):
        """Create a mock project for testing."""
        project = MagicMock()

        # Mock project data
        project.data.name = "Test Project"
        project.data.title = "Test Novel"
        project.data.author = "Test Author"
        project.data.language = "en_US"
        project.data.spellCheck = True
        project.data.spellLang = "en_US"
        project.data.createdTime = 1234567890
        project.data.saveCount = 10
        project.data.autoCount = 5
        project.data.editTime = 3600
        project.data.lastHandle = {"editor": "test_doc_001"}
        project.data.doBackup = True
        project.data.uuid = "test-uuid-1234"

        # Mock storage
        project.storage.getPath.return_value = Path("/test/project")
        # Mock tree structure
        mock_item1 = MagicMock()
        mock_item1.itemHandle = "doc_001"
        mock_item1.itemName = "Chapter 1"
        mock_item1.itemClass = nwItemClass.NOVEL
        mock_item1.itemLayout = nwItemLayout.DOCUMENT
        mock_item1.itemStatus = "draft"
        mock_item1.itemParent = None
        mock_item1.itemOrder = 0
        mock_item1.isExpanded = True
        mock_item1.wordCount = 1500
        mock_item1.charCount = 7500
        mock_item1.paraCount = 10
        mock_item1.cursorPos = 0
        mock_item1.isFileType.return_value = True
        mock_item1.isFolderType.return_value = False

        mock_item2 = MagicMock()
        mock_item2.itemHandle = "doc_002"
        mock_item2.itemName = "Character Notes"
        mock_item2.itemClass = nwItemClass.CHARACTER
        mock_item2.itemLayout = nwItemLayout.NOTE
        mock_item2.itemStatus = None
        mock_item2.itemParent = None
        mock_item2.itemOrder = 1
        mock_item2.isExpanded = False
        mock_item2.wordCount = 500
        mock_item2.charCount = 2500
        mock_item2.paraCount = 5
        mock_item2.cursorPos = 0
        mock_item2.isFileType.return_value = True
        mock_item2.isFolderType.return_value = False

        # Create a mock tree that behaves like a list but also supports item lookup
        mock_tree = MagicMock()
        mock_tree.__iter__ = lambda self: iter([mock_item1, mock_item2])
        mock_tree.__getitem__ = MagicMock(side_effect=lambda handle: {
            "doc_001": mock_item1,
            "doc_002": mock_item2,
        }.get(handle))
        project.tree = mock_tree

        # Mock index
        project.index.iterTags.return_value = [
            ("@char:john", nwItemClass.CHARACTER),
            ("@plot:main", nwItemClass.PLOT),
        ]
        project.index.getTagHandles.side_effect = lambda tag: {
            "@char:john": ["doc_001", "doc_002"],
            "@plot:main": ["doc_001"],
        }.get(tag, [])

        return project

    @pytest.fixture
    def api(self, mockProject):
        """Create an API instance with mock project."""
        return NovelWriterAPI(project=mockProject, enable_performance=False)

    @pytest.fixture
    def readOnlyApi(self, mockProject):
        """Create a read-only API instance."""
        return NovelWriterAPI(project=mockProject, readOnly=True, enable_performance=False)

    # ======================================================================
    # Test Initialization
    # ======================================================================

    def test_api_initialization(self):
        """Test API initialization without project."""
        api = NovelWriterAPI(enable_performance=False)
        assert api.isProjectLoaded is False
        assert api.isReadOnly is False

    def test_api_initialization_with_project(self, mockProject):
        """Test API initialization with project."""
        api = NovelWriterAPI(project=mockProject, enable_performance=False)
        assert api.isProjectLoaded is True
        assert api.isReadOnly is False

    def test_api_set_project(self, mockProject):
        """Test setting project after initialization."""
        api = NovelWriterAPI(enable_performance=False)
        assert api.isProjectLoaded is False

        api.setProject(mockProject)
        assert api.isProjectLoaded is True

    def test_api_set_project_none(self):
        """Test setting None as project raises error."""
        api = NovelWriterAPI(enable_performance=False)
        with pytest.raises(APIValidationError) as exc:
            api.setProject(None)
        assert "Project cannot be None" in str(exc.value)

    # ======================================================================
    # Test Core Methods
    # ======================================================================

    def test_get_project_meta(self, api):
        """Test getting project metadata."""
        meta = api.getProjectMeta()

        assert meta["name"] == "Test Project"
        assert meta["title"] == "Test Novel"
        assert meta["author"] == "Test Author"
        assert meta["language"] == "en_US"
        assert meta["uuid"] == "test-uuid-1234"
        assert meta["stats"]["numChapters"] == 1
        assert meta["stats"]["numScenes"] == 0
        assert meta["stats"]["totalWords"] == 2000  # 1500 + 500

        # Test caching
        meta2 = api.getProjectMeta()
        assert meta is meta2  # Same object due to cache

    def test_get_project_meta_no_project(self):
        """Test getting metadata without project raises error."""
        api = NovelWriterAPI(enable_performance=False)
        with pytest.raises(APIPermissionError) as exc:
            api.getProjectMeta()
        assert "No project is currently loaded" in str(exc.value)

    def test_list_documents(self, api):
        """Test listing documents."""
        # List all documents
        docs = api.listDocuments("all")
        assert len(docs) == 2
        assert docs[0]["handle"] == "doc_001"
        assert docs[0]["name"] == "Chapter 1"
        assert docs[0]["class"] == "NOVEL"

        # List novel documents only
        docs = api.listDocuments("novel")
        assert len(docs) == 1
        assert docs[0]["handle"] == "doc_001"

        # List notes only
        docs = api.listDocuments("notes")
        assert len(docs) == 1
        assert docs[0]["handle"] == "doc_002"

    def test_list_documents_invalid_scope(self, api):
        """Test listing documents with invalid scope."""
        with pytest.raises(APIValidationError) as exc:
            api.listDocuments("invalid")
        assert "Invalid scope" in str(exc.value)

    @patch("novelwriter.core.document.NWDocument")
    def test_get_doc_text(self, mock_doc_class, api):
        """Test getting document text."""
        # Setup mock document
        mock_doc = MagicMock()
        mock_doc.readDocument.return_value = True
        mock_doc.getText.return_value = "Test document content"
        mock_doc_class.return_value = mock_doc

        text = api.getDocText("doc_001")
        assert text == "Test document content"
        mock_doc_class.assert_called_once_with(api._project, "doc_001")
        mock_doc.readDocument.assert_called_once()

    def test_get_doc_text_invalid_handle(self, api):
        """Test getting text with invalid handle."""
        with pytest.raises(APIValidationError) as exc:
            api.getDocText("")
        assert "Document handle must be a non-empty string" in str(exc.value)

        with pytest.raises(APIValidationError) as exc:
            api.getDocText(None)
        assert "Document handle must be a non-empty string" in str(exc.value)

    def test_get_doc_text_not_found(self, api):
        """Test getting text for non-existent document."""
        with pytest.raises(APINotFoundError) as exc:
            api.getDocText("invalid_handle")
        assert "Document not found" in str(exc.value)

    @patch("novelwriter.core.document.NWDocument")
    def test_set_doc_text(self, mock_doc_class, api):
        """Test setting document text."""
        # Setup mock document
        mock_doc = MagicMock()
        mock_doc.writeDocument.return_value = True
        mock_doc.charCount = 100
        mock_doc.wordCount = 20
        mock_doc.paraCount = 2
        mock_doc_class.return_value = mock_doc

        result = api.setDocText("doc_001", "New content")
        assert result is True
        mock_doc.setText.assert_called_once_with("New content")
        mock_doc.writeDocument.assert_called_once()

    def test_set_doc_text_read_only(self, readOnlyApi):
        """Test setting text in read-only mode raises error."""
        with pytest.raises(APIPermissionError) as exc:
            readOnlyApi.setDocText("doc_001", "New content")
        assert "not allowed in read-only mode" in str(exc.value)

    def test_set_doc_text_invalid_params(self, api):
        """Test setting text with invalid parameters."""
        with pytest.raises(APIValidationError) as exc:
            api.setDocText("", "content")
        assert "Document handle must be a non-empty string" in str(exc.value)

        with pytest.raises(APIValidationError) as exc:
            api.setDocText("doc_001", 123)
        assert "Document text must be a string" in str(exc.value)

    def test_get_project_tree(self, api):
        """Test getting project tree structure."""
        tree = api.getProjectTree()

        assert len(tree) == 2
        assert tree[0]["handle"] == "doc_001"
        assert tree[0]["type"] == "file"
        assert tree[0]["wordCount"] == 1500
        assert tree[1]["handle"] == "doc_002"

    @patch("novelwriter.core.document.NWDocument")
    def test_search_project(self, mock_doc_class, api):
        """Test searching project."""
        # Setup mock document
        mock_doc = MagicMock()
        mock_doc.readDocument.return_value = True

        # Setup getText to return different content for each call
        mock_doc.getText.side_effect = [
            "This is chapter one.\nIt contains test content.\nMore lines here.",
            "Character notes.\nTest character description."
        ]
        mock_doc_class.return_value = mock_doc

        results = api.searchProject("test")
        assert len(results) == 2
        assert results[0]["document"] == "Chapter 1"
        assert results[0]["match"] == "test"
        assert "test content" in results[0]["context"]

    def test_search_project_invalid_query(self, api):
        """Test searching with invalid query."""
        with pytest.raises(APIValidationError) as exc:
            api.searchProject("")
        assert "Search query must be a non-empty string" in str(exc.value)

    @patch("novelwriter.core.document.NWDocument")
    def test_search_project_regex(self, mock_doc_class, api):
        """Test searching with regex."""
        # Setup mock document to return different content for different handles
        def mock_doc_init(project, handle):
            mock_doc = MagicMock()
            mock_doc.readDocument.return_value = True
            if handle == "doc_001":
                mock_doc.getText.return_value = "Test line 1\nAnother line"
            else:
                mock_doc.getText.return_value = "Character notes\nNo matches here"
            return mock_doc

        mock_doc_class.side_effect = mock_doc_init

        results = api.searchProject(r"test.*\d", {"regex": True})
        assert len(results) == 1
        assert results[0]["handle"] == "doc_001"
        assert results[0]["match"] == "Test line 1"

    def test_search_project_invalid_regex(self, api):
        """Test searching with invalid regex pattern."""
        with pytest.raises(APIValidationError) as exc:
            api.searchProject(r"[invalid(", {"regex": True})
        assert "Invalid regex pattern" in str(exc.value)

    def test_get_tag_list(self, api):
        """Test getting tag list."""
        tags = api.getTagList()

        assert len(tags) == 2
        assert tags[0]["name"] == "@char:john"
        assert tags[0]["class"] == "CHARACTER"
        assert tags[0]["count"] == 2
        assert tags[1]["name"] == "@plot:main"
        assert tags[1]["count"] == 1

    def test_get_statistics(self, api):
        """Test getting statistics."""
        # Project statistics
        stats = api.getStatistics("project")
        assert stats["totalWords"] == 2000
        assert stats["totalChars"] == 10000
        assert stats["documentCount"] == 2
        assert stats["chapterCount"] == 1  # Based on mock data

        # Novel statistics
        stats = api.getStatistics("novel")
        assert stats["totalWords"] == 1500
        assert stats["documentCount"] == 1

        # Notes statistics
        stats = api.getStatistics("notes")
        assert stats["totalWords"] == 500
        assert stats["documentCount"] == 1

    def test_get_statistics_specific_document(self, api):
        """Test getting statistics for specific document."""
        stats = api.getStatistics("doc_001")
        assert stats["handle"] == "doc_001"
        assert stats["name"] == "Chapter 1"
        assert stats["words"] == 1500
        assert stats["chars"] == 7500

    def test_get_statistics_invalid_document(self, api):
        """Test getting statistics for invalid document."""
        with pytest.raises(APINotFoundError) as exc:
            api.getStatistics("invalid_doc")
        assert "Document not found" in str(exc.value)

    # ======================================================================
    # Test Performance
    # ======================================================================

    def test_api_performance(self, api):
        """Test that API calls complete within 5ms threshold."""
        # This test uses the mock, so should be very fast
        start = time.perf_counter()
        api.getProjectMeta()
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < 5.0, f"API call took {duration_ms:.2f}ms, exceeding 5ms threshold"

    # ======================================================================
    # Test Utility Methods
    # ======================================================================

    def test_clear_cache(self, api):
        """Test cache clearing."""
        # Populate cache
        meta1 = api.getProjectMeta()
        assert len(api._cache) > 0

        # Clear cache
        api.clearCache()
        assert len(api._cache) == 0

        # Should get new object after cache clear
        meta2 = api.getProjectMeta()
        assert meta1 is not meta2  # Different objects

    def test_factory_method(self, mockProject):
        """Test factory method for creating API instances."""
        api = NovelWriterAPI.createInstance(mockProject, readOnly=True, enable_performance=False)
        assert api.isProjectLoaded is True
        assert api.isReadOnly is True

    # ======================================================================
    # Test Error Handling
    # ======================================================================

    def test_api_error_details(self):
        """Test API error with details."""
        error = APIError("Test error", {"key": "value"})
        assert error.message == "Test error"
        assert error.details == {"key": "value"}

    def test_validation_error_fields(self):
        """Test validation error with field information."""
        error = APIValidationError("Invalid field", field="test_field", value=123)
        assert error.field == "test_field"
        assert error.value == 123
        assert error.details["field"] == "test_field"

    def test_permission_error_fields(self):
        """Test permission error with operation information."""
        error = APIPermissionError("Access denied", operation="write", resource="doc")
        assert error.operation == "write"
        assert error.resource == "doc"

    def test_not_found_error_fields(self):
        """Test not found error with resource information."""
        error = APINotFoundError("Not found", resource_type="document", resource_id="doc_123")
        assert error.resource_type == "document"
        assert error.resource_id == "doc_123"

    def test_operation_error_with_cause(self):
        """Test operation error with cause exception."""
        cause = ValueError("Original error")
        error = APIOperationError("Operation failed", operation="test", cause=cause)
        assert error.operation == "test"
        assert error.cause is cause
        assert error.details["cause"] == "Original error"
        assert error.details["cause_type"] == "ValueError"
