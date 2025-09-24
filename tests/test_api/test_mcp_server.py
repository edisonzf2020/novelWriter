"""
novelWriter – MCP Server Tests
===============================

File History:
Created: 2025-09-24 [James]

This file is a part of novelWriter
Copyright (C) 2025 Bruno Martins

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

import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from novelwriter.api.exceptions import (
    MCPServerError, MCPToolNotFoundError, MCPConnectionError,
    MCPValidationError, MCPExecutionError
)


# Skip tests if MCP not available
try:
    from novelwriter.api.mcp_server import (
        MCPServer, MCPServerConfig, ServerStatus, ToolExecutionResult
    )
    MCP_AVAILABLE = True
except (ImportError, RuntimeError):
    MCP_AVAILABLE = False
    # Create dummy classes for test collection
    MCPServer = None
    MCPServerConfig = None
    ServerStatus = None
    ToolExecutionResult = None


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
class TestMCPServer:
    """Test MCP server core functionality"""
    
    @pytest.fixture
    def mock_nw_api(self):
        """Create a mock NovelWriterAPI"""
        api = Mock()
        api.getProjectMeta = Mock(return_value={"title": "Test Project"})
        api.listDocuments = Mock(return_value=[])
        api.getDocText = Mock(return_value="Test content")
        api.setDocText = Mock(return_value=True)
        return api
    
    @pytest.fixture
    def server_config(self):
        """Create test server configuration"""
        return MCPServerConfig(
            enabled=True,
            host="127.0.0.1",
            port=3001,
            max_concurrent_calls=5,
            timeout_ms=5000
        )
    
    @pytest.fixture
    def mcp_server(self, mock_nw_api, server_config):
        """Create MCP server instance"""
        with patch('novelwriter.api.mcp_server.MCP_AVAILABLE', True):
            with patch('novelwriter.api.mcp_server.FastMCP'):
                server = MCPServer(mock_nw_api, server_config)
                yield server
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_nw_api, server_config):
        """Test server initialization"""
        with patch('novelwriter.api.mcp_server.MCP_AVAILABLE', True):
            with patch('novelwriter.api.mcp_server.FastMCP'):
                server = MCPServer(mock_nw_api, server_config)
                
                assert server._config == server_config
                assert server._status == ServerStatus.STOPPED
                assert server._nw_api == mock_nw_api
                assert not server.isRunning()
    
    @pytest.mark.asyncio
    async def test_server_start_stop(self, mcp_server):
        """Test server start and stop lifecycle"""
        # Start server
        await mcp_server.start()
        assert mcp_server.isRunning()
        assert mcp_server._status == ServerStatus.RUNNING
        
        # Get status
        status = mcp_server.getStatus()
        assert status["status"] == "running"
        assert status["local_tools_count"] == 0
        
        # Stop server
        await mcp_server.stop()
        assert not mcp_server.isRunning()
        assert mcp_server._status == ServerStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_server_restart(self, mcp_server):
        """Test server restart functionality"""
        # Start server
        await mcp_server.start()
        assert mcp_server.isRunning()
        
        # Restart
        await mcp_server.restart()
        assert mcp_server.isRunning()
        
        # Stop
        await mcp_server.stop()
        assert not mcp_server.isRunning()
    
    @pytest.mark.asyncio
    async def test_tool_call_not_running(self, mcp_server):
        """Test tool call when server not running"""
        with pytest.raises(MCPServerError, match="not running"):
            await mcp_server.callTool("test_tool", {})
    
    @pytest.mark.asyncio
    async def test_tool_call_not_found(self, mcp_server):
        """Test calling non-existent tool"""
        await mcp_server.start()
        
        with pytest.raises(MCPToolNotFoundError, match="not found"):
            await mcp_server.callTool("non_existent_tool", {})
    
    @pytest.mark.asyncio
    async def test_local_tool_call_success(self, mcp_server):
        """Test successful local tool call"""
        await mcp_server.start()
        
        # Add mock local tool
        mock_tool = Mock(return_value={"result": "success"})
        mcp_server._local_tools["test_tool"] = mock_tool
        
        # Call tool
        result = await mcp_server.callTool("test_tool", {"param": "value"})
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.tool_type == "local"
        assert result.execution_time_ms >= 0
        assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_tool_call_history(self, mcp_server):
        """Test tool call history recording"""
        await mcp_server.start()
        
        # Add mock tool
        mock_tool = Mock(return_value={"result": f"call"})
        mcp_server._local_tools["test_tool"] = mock_tool
        
        # Make multiple calls
        for i in range(3):
            await mcp_server.callTool("test_tool", {"call": i})
        
        # Check history
        assert len(mcp_server._call_history) == 3
        assert all(r.success for r in mcp_server._call_history)
    
    @pytest.mark.asyncio
    async def test_tool_type_determination(self, mcp_server):
        """Test tool type determination logic"""
        # Add local tool
        mcp_server._local_tools["local_tool"] = Mock()
        
        # Add external tool
        mcp_server._external_connections["conn1"] = ["external_tool"]
        
        assert mcp_server._determineToolType("local_tool") == "local"
        assert mcp_server._determineToolType("external_tool") == "external"
        assert mcp_server._determineToolType("unknown_tool") == "unknown"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mcp_server):
        """Test performance monitoring"""
        await mcp_server.start()
        
        # Add mock tool with delay
        async def slow_tool(**kwargs):
            await asyncio.sleep(0.01)  # 10ms delay
            return {"result": "done"}
        
        mcp_server._local_tools["slow_tool"] = slow_tool
        
        # Call tool
        result = await mcp_server.callTool("slow_tool", {})
        
        # Check execution time
        assert result.execution_time_ms >= 10
        assert result.execution_time_ms < 100  # Should be less than 100ms


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
class TestToolRegistry:
    """Test tool registry functionality"""
    
    @pytest.fixture
    def registry(self):
        """Create tool registry"""
        from novelwriter.api.tools.registry import ToolRegistry
        return ToolRegistry()
    
    def test_register_tool(self, registry):
        """Test tool registration"""
        handler = Mock()
        
        registry.registerTool(
            name="test_tool",
            handler=handler,
            description="Test tool",
            category="test"
        )
        
        assert registry.hasTool("test_tool")
        assert "test_tool" in registry.listTools()
        
        # Get tool
        registration = registry.getTool("test_tool")
        assert registration.handler == handler
        assert registration.metadata.name == "test_tool"
        assert registration.metadata.category == "test"
    
    def test_unregister_tool(self, registry):
        """Test tool unregistration"""
        registry.registerTool(
            name="test_tool",
            handler=Mock(),
            description="Test tool"
        )
        
        assert registry.hasTool("test_tool")
        
        registry.unregisterTool("test_tool")
        assert not registry.hasTool("test_tool")
    
    def test_tool_statistics(self, registry):
        """Test tool statistics tracking"""
        registry.registerTool(
            name="test_tool",
            handler=Mock(),
            description="Test tool"
        )
        
        # Record calls
        registry.recordToolCall("test_tool", success=True)
        registry.recordToolCall("test_tool", success=True)
        registry.recordToolCall("test_tool", success=False, error="Test error")
        
        stats = registry.getToolStatistics("test_tool")
        assert stats["call_count"] == 3
        assert stats["error_count"] == 1
        assert stats["success_rate"] == pytest.approx(2/3)
        assert stats["last_error"] == "Test error"
    
    def test_registry_locking(self, registry):
        """Test registry locking mechanism"""
        registry.registerTool(
            name="tool1",
            handler=Mock(),
            description="Tool 1"
        )
        
        # Lock registry
        registry.lock()
        assert registry.isLocked()
        
        # Try to register new tool
        with pytest.raises(MCPValidationError, match="locked"):
            registry.registerTool(
                name="tool2",
                handler=Mock(),
                description="Tool 2"
            )
        
        # Unlock and register
        registry.unlock()
        registry.registerTool(
            name="tool2",
            handler=Mock(),
            description="Tool 2"
        )
        assert registry.hasTool("tool2")


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
class TestLocalToolWrapper:
    """Test local tool wrapper"""
    
    @pytest.fixture
    def mock_nw_api(self):
        """Create mock NovelWriterAPI"""
        api = Mock()
        api.getProjectMeta = Mock(return_value={
            "title": "Test Project",
            "author": "Test Author"
        })
        api.listDocuments = Mock(return_value=[
            {"handle": "doc1", "title": "Document 1"},
            {"handle": "doc2", "title": "Document 2"}
        ])
        api.getDocText = Mock(return_value="Test document content")
        api.setDocText = Mock(return_value=True)
        api.getDocumentMetadata = Mock(return_value={
            "created": "2025-01-01",
            "modified": "2025-01-02"
        })
        api.searchContent = Mock(return_value=[
            {"handle": "doc1", "title": "Document 1", "matches": ["match1"]}
        ])
        api.getProjectStatistics = Mock(return_value={
            "word_count": 1000,
            "chapter_count": 5
        })
        return api
    
    @pytest.fixture
    def wrapper(self, mock_nw_api):
        """Create local tool wrapper"""
        from novelwriter.api.tools.registry import ToolRegistry
        from novelwriter.api.tools.local_tools import LocalToolWrapper
        
        registry = ToolRegistry()
        return LocalToolWrapper(mock_nw_api, registry)
    
    @pytest.mark.asyncio
    async def test_builtin_tools_registered(self, wrapper):
        """Test that built-in tools are registered"""
        from novelwriter.api.tools.registry import ToolRegistry
        
        registry = wrapper._registry
        
        # Check built-in tools
        assert registry.hasTool("get_project_info")
        assert registry.hasTool("list_documents")
        assert registry.hasTool("read_document")
        assert registry.hasTool("write_document")
        assert registry.hasTool("search_content")
    
    @pytest.mark.asyncio
    async def test_execute_get_project_info(self, wrapper, mock_nw_api):
        """Test get_project_info tool execution"""
        result = await wrapper.executeTool("get_project_info", {"include_stats": True})
        
        assert result["result"]["title"] == "Test Project"
        assert result["result"]["author"] == "Test Author"
        assert "statistics" in result["result"]
        assert result["execution_time_ms"] >= 0
        
        mock_nw_api.getProjectMeta.assert_called_once()
        mock_nw_api.getProjectStatistics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_list_documents(self, wrapper, mock_nw_api):
        """Test list_documents tool execution"""
        result = await wrapper.executeTool("list_documents", {"scope": "all"})
        
        assert len(result["result"]) == 2
        assert result["result"][0]["handle"] == "doc1"
        
        mock_nw_api.listDocuments.assert_called_once_with("all")
    
    @pytest.mark.asyncio
    async def test_execute_read_document(self, wrapper, mock_nw_api):
        """Test read_document tool execution"""
        result = await wrapper.executeTool(
            "read_document",
            {"item_handle": "doc1", "include_metadata": True}
        )
        
        assert result["result"]["content"] == "Test document content"
        assert "metadata" in result["result"]
        
        mock_nw_api.getDocText.assert_called_once_with("doc1")
        mock_nw_api.getDocumentMetadata.assert_called_once_with("doc1")
    
    @pytest.mark.asyncio
    async def test_execute_write_document(self, wrapper, mock_nw_api):
        """Test write_document tool execution"""
        result = await wrapper.executeTool(
            "write_document",
            {"item_handle": "doc1", "content": "New content", "append": False}
        )
        
        assert result["result"]["success"] is True
        assert result["result"]["content_length"] == len("New content")
        
        mock_nw_api.setDocText.assert_called_once_with("doc1", "New content")
    
    @pytest.mark.asyncio
    async def test_execute_search_content(self, wrapper, mock_nw_api):
        """Test search_content tool execution"""
        result = await wrapper.executeTool(
            "search_content",
            {"query": "test", "scope": "all", "case_sensitive": False}
        )
        
        assert len(result["result"]) == 1
        assert result["result"][0]["item_handle"] == "doc1"
        
        mock_nw_api.searchContent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tool_not_found(self, wrapper):
        """Test executing non-existent tool"""
        with pytest.raises(MCPToolNotFoundError):
            await wrapper.executeTool("non_existent_tool", {})
    
    @pytest.mark.asyncio
    async def test_disabled_tool(self, wrapper):
        """Test executing disabled tool"""
        wrapper._registry.disableTool("get_project_info")
        
        with pytest.raises(MCPExecutionError, match="disabled"):
            await wrapper.executeTool("get_project_info", {})
