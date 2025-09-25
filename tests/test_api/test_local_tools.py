"""
novelWriter – Local Tools Test Suite
=====================================

File History:
Created: 2025-09-25 [James - Dev Agent]

This file is a part of novelWriter
Copyright (C) 2025 Veronica Berglyd Olsen and novelWriter Contributors

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
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from novelwriter.api.tools.base import (
    BaseTool, ToolMetadata, ToolPermission, ToolExecutionResult
)
from novelwriter.api.tools.project_tools import ProjectInfoTool, ProjectTreeTool
from novelwriter.api.tools.document_tools import (
    DocumentListTool, DocumentReadTool, DocumentWriteTool, CreateDocumentTool
)
from novelwriter.api.tools.search_tools import (
    GlobalSearchTool, TagListTool, ProjectStatsTool
)


@pytest.fixture
def mock_api():
    """Create a mock NovelWriterAPI instance"""
    api = Mock()
    
    # Mock project meta
    api.getProjectMeta.return_value = {
        "title": "Test Project",
        "author": "Test Author",
        "created": "2025-01-01",
        "updated": "2025-01-25",
        "wordCount": 50000,
        "chapterCount": 10,
        "sceneCount": 50,
        "language": "en",
        "spellCheck": True,
        "autoSave": True,
        "backupOnClose": True
    }
    
    # Mock project stats
    api.getProjectStats.return_value = {
        "totalWords": 50000,
        "totalChars": 250000,
        "totalParagraphs": 2000,
        "avgChapterLength": 5000,
        "avgSceneLength": 1000,
        "documentCount": 60,
        "noteCount": 20,
        "chapterCount": 10,
        "sceneCount": 50
    }
    
    # Mock project tree
    api.getProjectTree.return_value = {
        "handle": "root",
        "title": "Project Root",
        "type": "root",
        "children": [
            {
                "handle": "chapter1",
                "title": "Chapter 1",
                "type": "folder",
                "wordCount": 5000,
                "children": [
                    {
                        "handle": "scene1",
                        "title": "Scene 1",
                        "type": "document",
                        "wordCount": 1000
                    }
                ]
            }
        ]
    }
    
    # Mock document list
    mock_doc = Mock()
    mock_doc.handle = "doc1"
    mock_doc.title = "Test Document"
    mock_doc.doc_type = "DOCUMENT"
    mock_doc.status = "draft"
    mock_doc.created = datetime(2025, 1, 1)
    mock_doc.updated = datetime(2025, 1, 25)
    mock_doc.word_count = 1000
    mock_doc.char_count = 5000
    mock_doc.para_count = 50
    api.listDocuments.return_value = [mock_doc]
    
    # Mock document operations
    api.getDocText.return_value = "This is test document content."
    api.setDocText.return_value = True
    api.createDocument.return_value = "new_doc_handle"
    
    # Mock search
    api.searchProject.return_value = [
        {
            "handle": "doc1",
            "title": "Test Document",
            "match_type": "content",
            "line_number": 5,
            "context": "This is a test match",
            "score": 0.95
        }
    ]
    
    # Mock tag list
    api.getTagList.return_value = [
        {
            "name": "character",
            "type": "character",
            "color": "#FF0000",
            "description": "Main character",
            "usage_count": 10
        }
    ]
    
    return api


class TestProjectInfoTool:
    """Test ProjectInfoTool functionality"""
    
    @pytest.mark.asyncio
    async def test_project_info_basic(self, mock_api):
        """Test basic project info retrieval"""
        tool = ProjectInfoTool(mock_api)
        
        result = await tool.execute()
        
        assert result.success is True
        assert result.result["title"] == "Test Project"
        assert result.result["author"] == "Test Author"
        assert result.result["word_count"] == 50000
    
    @pytest.mark.asyncio
    async def test_project_info_with_settings(self, mock_api):
        """Test project info with settings included"""
        tool = ProjectInfoTool(mock_api)
        
        result = await tool.execute(include_settings=True, include_stats=False)
        
        assert result.success is True
        assert "settings" in result.result
        assert result.result["settings"]["language"] == "en"
        assert "statistics" not in result.result
    
    @pytest.mark.asyncio
    async def test_project_info_with_stats(self, mock_api):
        """Test project info with statistics included"""
        tool = ProjectInfoTool(mock_api)
        
        result = await tool.execute(include_settings=False, include_stats=True)
        
        assert result.success is True
        assert "statistics" in result.result
        assert result.result["statistics"]["total_words"] == 50000
        assert "settings" not in result.result


class TestProjectTreeTool:
    """Test ProjectTreeTool functionality"""
    
    @pytest.mark.asyncio
    async def test_project_tree_basic(self, mock_api):
        """Test basic project tree retrieval"""
        tool = ProjectTreeTool(mock_api)
        
        result = await tool.execute()
        
        assert result.success is True
        assert result.result["tree"]["handle"] == "root"
        assert result.result["total_nodes"] == 3  # root + folder + document
        assert result.result["max_depth"] == 2
    
    @pytest.mark.asyncio
    async def test_project_tree_with_filter(self, mock_api):
        """Test project tree with type filter"""
        tool = ProjectTreeTool(mock_api)
        
        result = await tool.execute(filter_type="novel")
        
        assert result.success is True
        assert result.result["tree"] is not None
    
    @pytest.mark.asyncio
    async def test_project_tree_max_depth(self, mock_api):
        """Test project tree with max depth limit"""
        tool = ProjectTreeTool(mock_api)
        
        result = await tool.execute(max_depth=1)
        
        assert result.success is True
        # Should only include root and immediate children
        assert result.result["max_depth"] <= 1


class TestDocumentListTool:
    """Test DocumentListTool functionality"""
    
    @pytest.mark.asyncio
    async def test_document_list_basic(self, mock_api):
        """Test basic document list retrieval"""
        tool = DocumentListTool(mock_api)
        
        result = await tool.execute()
        
        assert result.success is True
        assert len(result.result) == 1
        assert result.result[0]["handle"] == "doc1"
        assert result.result[0]["title"] == "Test Document"
    
    @pytest.mark.asyncio
    async def test_document_list_with_content(self, mock_api):
        """Test document list with content preview"""
        tool = DocumentListTool(mock_api)
        
        result = await tool.execute(include_content=True)
        
        assert result.success is True
        assert "content_preview" in result.result[0]


class TestDocumentReadTool:
    """Test DocumentReadTool functionality"""
    
    @pytest.mark.asyncio
    async def test_document_read_basic(self, mock_api):
        """Test basic document reading"""
        tool = DocumentReadTool(mock_api)
        
        result = await tool.execute(handle="doc1")
        
        assert result.success is True
        assert result.result["handle"] == "doc1"
        assert result.result["content"] == "This is test document content."
        assert result.result["length"] == 30
    
    @pytest.mark.asyncio
    async def test_document_read_with_metadata(self, mock_api):
        """Test document reading with metadata"""
        tool = DocumentReadTool(mock_api)
        
        result = await tool.execute(handle="doc1", include_metadata=True)
        
        assert result.success is True
        assert "metadata" in result.result
        assert result.result["metadata"]["title"] == "Test Document"


class TestDocumentWriteTool:
    """Test DocumentWriteTool functionality"""
    
    @pytest.mark.asyncio
    async def test_document_write_basic(self, mock_api):
        """Test basic document writing"""
        tool = DocumentWriteTool(mock_api)
        
        content = "New document content"
        result = await tool.execute(handle="doc1", content=content)
        
        assert result.success is True
        assert result.result["success"] is True
        assert result.result["content_length"] == len(content)
        mock_api.setDocText.assert_called_once_with("doc1", content)
    
    @pytest.mark.asyncio
    async def test_document_write_with_backup(self, mock_api):
        """Test document writing with backup"""
        tool = DocumentWriteTool(mock_api)
        
        result = await tool.execute(
            handle="doc1",
            content="New content",
            create_backup=True
        )
        
        assert result.success is True
        # Should have tried to read original content for backup
        mock_api.getDocText.assert_called_once_with("doc1")


class TestCreateDocumentTool:
    """Test CreateDocumentTool functionality"""
    
    @pytest.mark.asyncio
    async def test_create_document_basic(self, mock_api):
        """Test basic document creation"""
        tool = CreateDocumentTool(mock_api)
        
        result = await tool.execute(
            title="New Document",
            doc_type="DOCUMENT"
        )
        
        assert result.success is True
        assert result.result["handle"] == "new_doc_handle"
        assert result.result["title"] == "New Document"
        mock_api.createDocument.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_document_with_content(self, mock_api):
        """Test document creation with initial content"""
        tool = CreateDocumentTool(mock_api)
        
        result = await tool.execute(
            title="New Document",
            content="Initial content"
        )
        
        assert result.success is True
        assert result.result["content_set"] is True
        mock_api.setDocText.assert_called_once_with("new_doc_handle", "Initial content")


class TestGlobalSearchTool:
    """Test GlobalSearchTool functionality"""
    
    @pytest.mark.asyncio
    async def test_global_search_basic(self, mock_api):
        """Test basic global search"""
        tool = GlobalSearchTool(mock_api)
        
        result = await tool.execute(query="test")
        
        assert result.success is True
        assert result.result["query"] == "test"
        assert len(result.result["results"]) == 1
        assert result.result["results"][0]["handle"] == "doc1"
    
    @pytest.mark.asyncio
    async def test_global_search_with_options(self, mock_api):
        """Test global search with options"""
        tool = GlobalSearchTool(mock_api)
        
        result = await tool.execute(
            query="test",
            search_type="title",
            case_sensitive=True,
            whole_word=True,
            max_results=10
        )
        
        assert result.success is True
        assert result.result["statistics"]["search_type"] == "title"
        mock_api.searchProject.assert_called_once()


class TestTagListTool:
    """Test TagListTool functionality"""
    
    @pytest.mark.asyncio
    async def test_tag_list_basic(self, mock_api):
        """Test basic tag list retrieval"""
        tool = TagListTool(mock_api)
        
        result = await tool.execute()
        
        assert result.success is True
        assert result.result["total_tags"] == 1
        assert result.result["tags"][0]["name"] == "character"
    
    @pytest.mark.asyncio
    async def test_tag_list_with_counts(self, mock_api):
        """Test tag list with usage counts"""
        tool = TagListTool(mock_api)
        
        result = await tool.execute(include_counts=True, sort_by="count")
        
        assert result.success is True
        assert result.result["tags"][0]["usage_count"] == 10
        assert result.result["total_usage"] == 10


class TestProjectStatsTool:
    """Test ProjectStatsTool functionality"""
    
    @pytest.mark.asyncio
    async def test_project_stats_basic(self, mock_api):
        """Test basic project statistics"""
        tool = ProjectStatsTool(mock_api)
        
        result = await tool.execute()
        
        assert result.success is True
        assert result.result["content_statistics"]["total_words"] == 50000
        assert result.result["document_statistics"]["total_documents"] == 60
        assert result.result["structure_statistics"]["chapter_count"] == 10


class TestToolMetadata:
    """Test tool metadata and registration"""
    
    def test_tool_metadata_complete(self, mock_api):
        """Test that all tools have complete metadata"""
        tools = [
            ProjectInfoTool(mock_api),
            ProjectTreeTool(mock_api),
            DocumentListTool(mock_api),
            DocumentReadTool(mock_api),
            DocumentWriteTool(mock_api),
            CreateDocumentTool(mock_api),
            GlobalSearchTool(mock_api),
            TagListTool(mock_api),
            ProjectStatsTool(mock_api)
        ]
        
        for tool in tools:
            metadata = tool.metadata
            assert metadata.name
            assert metadata.description
            assert metadata.parameters_schema is not None
            assert isinstance(metadata.required_permissions, list)
            assert metadata.version == "1.0.0"


class TestErrorHandling:
    """Test error handling in tools"""
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_api):
        """Test that tools handle errors gracefully"""
        mock_api.getProjectMeta.side_effect = Exception("API Error")
        
        tool = ProjectInfoTool(mock_api)
        result = await tool.execute()
        
        assert result.success is False
        assert "API Error" in result.error_message
        assert result.execution_time_ms >= 0
