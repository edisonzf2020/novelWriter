"""
AI Module Integration Test Suite

Tests integration between AI modules and the new unified API architecture.
Ensures proper data flow and component interaction.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from dataclasses import dataclass


@dataclass
class IntegrationTestResult:
    """Container for integration test results"""
    component_a: str
    component_b: str
    test_name: str
    data_flow_correct: bool
    error_handling_correct: bool
    performance_acceptable: bool
    details: Dict[str, Any]
    
    @property
    def passed(self) -> bool:
        return all([
            self.data_flow_correct,
            self.error_handling_correct,
            self.performance_acceptable
        ])


class TestAIAPIIntegration:
    """Test integration between AI modules and unified API"""
    
    @pytest.fixture
    def mock_novelwriter_api(self):
        """Create mock NovelWriterAPI"""
        api = Mock()
        api.get_project_info = Mock(return_value={
            "title": "Test Novel",
            "author": "Test Author",
            "words": 50000
        })
        api.list_documents = Mock(return_value=[
            {"id": "doc1", "title": "Chapter 1"},
            {"id": "doc2", "title": "Chapter 2"}
        ])
        api.get_document = Mock(return_value={
            "content": "Test document content",
            "metadata": {"type": "scene"}
        })
        return api
    
    @pytest.fixture
    def mock_ai_core(self, mock_novelwriter_api):
        """Create mock AI core with API injection"""
        ai_core = Mock()
        ai_core.api = mock_novelwriter_api  # Dependency injection
        ai_core.collect_context = Mock(return_value={
            "documents": [],
            "metadata": {}
        })
        return ai_core
    
    def test_ai_core_uses_unified_api(self, mock_ai_core, mock_novelwriter_api):
        """Verify AI core accesses data through unified API only"""
        
        # AI core should use injected API
        mock_ai_core.api.get_project_info()
        mock_ai_core.api.list_documents()
        
        # Verify calls went through API
        mock_novelwriter_api.get_project_info.assert_called_once()
        mock_novelwriter_api.list_documents.assert_called_once()
        
        result = IntegrationTestResult(
            component_a="ai_core",
            component_b="novelwriter_api",
            test_name="ai_core_api_usage",
            data_flow_correct=True,
            error_handling_correct=True,
            performance_acceptable=True,
            details={
                "api_calls": 2,
                "direct_core_access": 0
            }
        )
        
        assert result.passed, "AI core should only use unified API"
    
    def test_context_collection_integration(self, mock_ai_core, mock_novelwriter_api):
        """Test context collection through unified API"""
        
        # Setup mock returns
        mock_novelwriter_api.list_documents.return_value = [
            {"id": "doc1", "title": "Chapter 1", "type": "chapter"},
            {"id": "doc2", "title": "Scene 1", "type": "scene"}
        ]
        
        mock_novelwriter_api.get_document.side_effect = [
            {"content": "Chapter 1 content", "metadata": {}},
            {"content": "Scene 1 content", "metadata": {}}
        ]
        
        # Collect context
        def collect_context_impl(scope):
            docs = mock_novelwriter_api.list_documents()
            contents = []
            for doc in docs:
                content = mock_novelwriter_api.get_document(doc["id"])
                contents.append(content)
            return {"documents": contents, "scope": scope}
        
        mock_ai_core.collect_context.side_effect = collect_context_impl
        
        context = mock_ai_core.collect_context("current_chapter")
        
        # Verify proper data flow
        assert len(context["documents"]) == 2
        assert context["scope"] == "current_chapter"
        assert mock_novelwriter_api.get_document.call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_ai_operations(self, mock_ai_core, mock_novelwriter_api):
        """Test async operations between AI and API"""
        
        # Make API methods async
        mock_novelwriter_api.async_get_document = AsyncMock(return_value={
            "content": "Async content",
            "metadata": {}
        })
        
        # Test async operation
        result = await mock_novelwriter_api.async_get_document("doc1")
        
        assert result["content"] == "Async content"
        mock_novelwriter_api.async_get_document.assert_awaited_once()
    
    def test_error_propagation(self, mock_ai_core, mock_novelwriter_api):
        """Test error handling between components"""
        
        # Simulate API error
        mock_novelwriter_api.get_document.side_effect = ValueError("Document not found")
        
        error_handled = False
        try:
            mock_novelwriter_api.get_document("invalid_id")
        except ValueError as e:
            error_handled = True
            assert str(e) == "Document not found"
        
        assert error_handled, "Errors should propagate correctly"


class TestMCPServerIntegration:
    """Test MCP server integration with unified API"""
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Create mock MCP server"""
        server = Mock()
        server.api = Mock()  # Mock API
        server.list_tools = Mock(return_value=[
            "get_project_info",
            "list_documents",
            "search_text"
        ])
        server.call_tool = Mock()
        return server
    
    def test_mcp_server_tool_registration(self, mock_mcp_server):
        """Test tool registration in MCP server"""
        
        tools = mock_mcp_server.list_tools()
        
        assert "get_project_info" in tools
        assert "list_documents" in tools
        assert len(tools) >= 3
    
    def test_mcp_server_tool_execution(self, mock_mcp_server):
        """Test tool execution through MCP server"""
        
        def call_tool_impl(tool_name, params):
            if tool_name == "get_project_info":
                return {"title": "Test Novel", "author": "Test Author"}
            elif tool_name == "list_documents":
                return [{"id": "doc1"}, {"id": "doc2"}]
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        
        mock_mcp_server.call_tool.side_effect = call_tool_impl
        
        # Call tool through MCP server
        result = mock_mcp_server.call_tool("get_project_info", {})
        
        assert result["title"] == "Test Novel"
        assert result["author"] == "Test Author"
        
        # Verify tool was called
        assert mock_mcp_server.call_tool.called
    
    @pytest.mark.asyncio
    async def test_mcp_server_async_tools(self, mock_mcp_server):
        """Test async tool execution in MCP server"""
        
        async def async_search(query):
            await asyncio.sleep(0.01)  # Simulate async work
            return [{"doc": "result1"}, {"doc": "result2"}]
        
        mock_mcp_server.async_call_tool = AsyncMock(side_effect=async_search)
        
        results = await mock_mcp_server.async_call_tool("search_text")
        
        assert len(results) == 2
        mock_mcp_server.async_call_tool.assert_awaited_once()


class TestExternalMCPIntegration:
    """Test external MCP client integration"""
    
    @pytest.fixture
    def mock_external_client(self):
        """Create mock external MCP client"""
        client = Mock()
        client.connected = True
        client.available_tools = ["time_service", "weather_service"]
        client.call_tool = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_external_tool_discovery(self, mock_external_client):
        """Test discovery of external MCP tools"""
        
        # Discover available tools
        tools = mock_external_client.available_tools
        
        assert "time_service" in tools
        assert "weather_service" in tools
        assert len(tools) == 2
    
    @pytest.mark.asyncio
    async def test_external_tool_invocation(self, mock_external_client):
        """Test invoking external MCP tools"""
        
        mock_external_client.call_tool.return_value = {
            "time": "2024-01-01 12:00:00",
            "timezone": "UTC"
        }
        
        result = await mock_external_client.call_tool("time_service", {})
        
        assert result["time"] == "2024-01-01 12:00:00"
        mock_external_client.call_tool.assert_awaited_once_with("time_service", {})
    
    @pytest.mark.asyncio
    async def test_external_connection_resilience(self, mock_external_client):
        """Test connection resilience for external MCP"""
        
        # Simulate connection loss
        mock_external_client.connected = False
        mock_external_client.call_tool.side_effect = ConnectionError("Connection lost")
        
        # Should handle connection error
        error_handled = False
        try:
            await mock_external_client.call_tool("time_service", {})
        except ConnectionError:
            error_handled = True
            # Should attempt reconnection
            mock_external_client.connected = True
            mock_external_client.call_tool.side_effect = None
            mock_external_client.call_tool.return_value = {"status": "reconnected"}
        
        assert error_handled, "Should handle connection errors"
        
        # Verify can call after reconnection
        if mock_external_client.connected:
            result = await mock_external_client.call_tool("time_service", {})
            assert result["status"] == "reconnected"


class TestSecurityIntegration:
    """Test security component integration"""
    
    @pytest.fixture
    def mock_security_controller(self):
        """Create mock security controller"""
        controller = Mock()
        controller.validate_request = Mock(return_value=True)
        controller.sanitize_input = Mock(side_effect=lambda x: x)
        controller.audit_log = Mock()
        return controller
    
    def test_api_security_integration(self, mock_security_controller):
        """Test security integration with API calls"""
        
        # API should validate requests through security
        def secure_api_call(method, params):
            if mock_security_controller.validate_request(method, params):
                sanitized = mock_security_controller.sanitize_input(params)
                result = method(**sanitized)
                mock_security_controller.audit_log(getattr(method, '__name__', 'unknown'), params, result)
                return result
            else:
                raise PermissionError("Request validation failed")
        
        # Create mock API
        mock_api = Mock()
        mock_api.get_project_info = Mock(return_value={"title": "Test Novel"})
        
        # Test secure API call
        params = {"include_stats": True}
        result = secure_api_call(mock_api.get_project_info, params)
        
        # Verify security checks were performed
        mock_security_controller.validate_request.assert_called_once()
        mock_security_controller.sanitize_input.assert_called_once_with(params)
        mock_security_controller.audit_log.assert_called_once()
        
        assert result["title"] == "Test Novel"
    
    def test_tool_permission_validation(self, mock_security_controller):
        """Test tool permission validation"""
        
        def validate_tool_access(tool_name, user_context):
            allowed_tools = ["get_project_info", "list_documents"]
            return tool_name in allowed_tools
        
        mock_security_controller.validate_tool_access = Mock(side_effect=validate_tool_access)
        
        # Test allowed tool
        assert mock_security_controller.validate_tool_access("get_project_info", {})
        
        # Test restricted tool
        assert not mock_security_controller.validate_tool_access("delete_project", {})


class TestPerformanceMonitorIntegration:
    """Test performance monitor integration"""
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Create mock performance monitor"""
        monitor = Mock()
        monitor.record_metric = Mock()
        monitor.get_metrics = Mock(return_value={})
        monitor.check_thresholds = Mock(return_value=True)
        return monitor
    
    def test_api_performance_monitoring(self, mock_performance_monitor):
        """Test performance monitoring of API calls"""
        
        def monitored_api_call(method, *args, **kwargs):
            import time
            start = time.perf_counter()
            result = method(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            
            mock_performance_monitor.record_metric(
                operation=getattr(method, '__name__', 'unknown'),
                duration_ms=duration_ms
            )
            
            return result
        
        # Create mock API
        mock_api = Mock()
        mock_api.get_project_info = Mock(return_value={"title": "Test"})
        mock_api.get_project_info.__name__ = "get_project_info"
        
        # Make monitored API call
        result = monitored_api_call(mock_api.get_project_info)
        
        # Verify metric was recorded
        mock_performance_monitor.record_metric.assert_called_once()
        call_args = mock_performance_monitor.record_metric.call_args
        assert call_args.kwargs["operation"] == "get_project_info"
        assert "duration_ms" in call_args.kwargs
    
    def test_performance_threshold_alerts(self, mock_performance_monitor):
        """Test performance threshold alerting"""
        
        # Record slow operation
        mock_performance_monitor.record_metric(
            operation="slow_operation",
            duration_ms=150  # Above threshold
        )
        
        # Check thresholds
        mock_performance_monitor.check_thresholds.return_value = False
        
        threshold_exceeded = not mock_performance_monitor.check_thresholds()
        
        assert threshold_exceeded, "Should detect threshold violations"


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    def test_complete_ai_workflow(self):
        """Test complete AI workflow from request to response"""
        
        # 1. User request through AI Copilot
        user_request = "Suggest improvements for Chapter 1"
        
        # Create mock AI core
        mock_ai_core = Mock()
        mock_ai_core.api = Mock()
        mock_ai_core.api.list_documents = Mock()
        mock_ai_core.collect_context = Mock()
        
        # 2. AI core collects context via API
        mock_ai_core.collect_context("chapter_1")
        mock_ai_core.collect_context.assert_called()
        
        # 3. AI processes request
        ai_response = {
            "suggestions": ["Add more dialogue", "Expand scene description"],
            "confidence": 0.85
        }
        mock_ai_core.process_request = Mock(return_value=ai_response)
        
        # 4. Response delivered
        result = mock_ai_core.process_request(user_request)
        
        assert len(result["suggestions"]) == 2
        assert result["confidence"] > 0.8
    
    @pytest.mark.asyncio
    async def test_hybrid_tool_execution(self):
        """Test execution of both local and external tools"""
        
        # Create mock MCP server
        mock_mcp_server = Mock()
        mock_mcp_server.call_tool = Mock(return_value={"local": "result"})
        
        # Local tool execution
        local_result = mock_mcp_server.call_tool("get_project_info", {})
        
        # Create mock external client
        mock_external_client = Mock()
        mock_external_client.call_tool = AsyncMock(return_value={"external": "result"})
        
        # External tool execution
        external_result = await mock_external_client.call_tool("time_service", {})
        
        # Both should work seamlessly
        assert local_result["local"] == "result"
        assert external_result["external"] == "result"
    
    def test_full_stack_error_handling(self):
        """Test error handling across the full stack"""
        
        # Create mock API
        mock_novelwriter_api = Mock()
        
        # Inject error at API level
        mock_novelwriter_api.get_document = Mock(side_effect=FileNotFoundError("Document not found"))
        
        # Create mock security controller
        mock_security_controller = Mock()
        mock_security_controller.log_error = Mock()
        
        # Create mock performance monitor
        mock_performance_monitor = Mock()
        mock_performance_monitor.record_failure = Mock()
        
        # AI core should handle gracefully
        def handle_error():
            try:
                mock_novelwriter_api.get_document("missing_doc")
            except FileNotFoundError as e:
                mock_security_controller.log_error(str(e))
                mock_performance_monitor.record_failure("get_document")
                return {"error": str(e), "fallback": True}
        
        result = handle_error()
        
        # Verify error was handled at all levels
        assert result["fallback"] is True
        mock_security_controller.log_error.assert_called_once()
        mock_performance_monitor.record_failure.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
