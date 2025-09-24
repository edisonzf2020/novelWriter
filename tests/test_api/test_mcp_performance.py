"""
novelWriter – MCP Performance Benchmark Tests
==============================================

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
import statistics
from unittest.mock import Mock, AsyncMock, patch

import pytest

from novelwriter.api.exceptions import MCPServerError


# Skip tests if MCP not available
try:
    from novelwriter.api.mcp_server import (
        MCPServer, MCPServerConfig, ServerStatus
    )
    from novelwriter.api.tools.registry import ToolRegistry
    from novelwriter.api.tools.local_tools import LocalToolWrapper
    MCP_AVAILABLE = True
except (ImportError, RuntimeError):
    MCP_AVAILABLE = False
    MCPServer = None
    MCPServerConfig = None
    ServerStatus = None
    ToolRegistry = None
    LocalToolWrapper = None


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
class TestMCPPerformance:
    """Performance benchmark tests for MCP server"""
    
    @pytest.fixture
    def mock_nw_api(self):
        """Create a mock NovelWriterAPI with minimal latency"""
        api = Mock()
        api.getProjectMeta = Mock(return_value={"title": "Test"})
        api.listDocuments = Mock(return_value=[])
        api.getDocText = Mock(return_value="Content")
        api.setDocText = Mock(return_value=True)
        api.searchContent = Mock(return_value=[])
        api.getProjectStatistics = Mock(return_value={"words": 1000})
        api.getDocumentMetadata = Mock(return_value={"created": "2025-01-01"})
        return api
    
    @pytest.fixture
    def server_config(self):
        """Create performance test configuration"""
        return MCPServerConfig(
            enabled=True,
            host="127.0.0.1",
            port=3002,
            max_concurrent_calls=10,
            timeout_ms=5000
        )
    
    @pytest.fixture
    def mcp_server(self, mock_nw_api, server_config):
        """Create MCP server for performance testing"""
        with patch('novelwriter.api.mcp_server.MCP_AVAILABLE', True):
            with patch('novelwriter.api.mcp_server.FastMCP'):
                server = MCPServer(mock_nw_api, server_config)
                yield server
    
    @pytest.mark.asyncio
    async def test_local_tool_latency_p95(self, mock_nw_api):
        """Test that local tool calls meet P95 < 10ms requirement"""
        # Create wrapper with real registry
        registry = ToolRegistry()
        wrapper = LocalToolWrapper(mock_nw_api, registry)
        
        # Warm up
        for _ in range(10):
            await wrapper.executeTool("get_project_info", {})
        
        # Measure latencies
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start = time.perf_counter()
            await wrapper.executeTool("get_project_info", {"include_stats": False})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate P95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]
        
        # Also calculate other metrics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Log results
        print(f"\nLocal Tool Performance:")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  Mean: {mean_latency:.2f}ms")
        print(f"  Median: {median_latency:.2f}ms")
        print(f"  Min: {min_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        
        # Assert P95 < 10ms requirement
        assert p95_latency < 10, f"P95 latency {p95_latency:.2f}ms exceeds 10ms requirement"
    
    @pytest.mark.asyncio
    async def test_multiple_local_tools_latency(self, mock_nw_api):
        """Test latency for different local tools"""
        registry = ToolRegistry()
        wrapper = LocalToolWrapper(mock_nw_api, registry)
        
        tools_to_test = [
            ("list_documents", {"scope": "all"}),
            ("read_document", {"item_handle": "test"}),
            ("search_content", {"query": "test", "scope": "all"}),
        ]
        
        results = {}
        
        for tool_name, params in tools_to_test:
            latencies = []
            
            # Warm up
            for _ in range(5):
                try:
                    await wrapper.executeTool(tool_name, params)
                except:
                    pass  # Ignore errors from mock data
            
            # Measure
            for _ in range(50):
                start = time.perf_counter()
                try:
                    await wrapper.executeTool(tool_name, params)
                except:
                    pass  # Ignore errors from mock data
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            latencies.sort()
            p95_index = int(len(latencies) * 0.95)
            results[tool_name] = {
                "p95": latencies[p95_index],
                "mean": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies)
            }
        
        # Log results
        print("\nLocal Tools Performance Summary:")
        for tool_name, metrics in results.items():
            print(f"  {tool_name}:")
            print(f"    P95: {metrics['p95']:.2f}ms")
            print(f"    Mean: {metrics['mean']:.2f}ms")
            
            # Assert each tool meets requirement
            assert metrics['p95'] < 10, f"{tool_name} P95 {metrics['p95']:.2f}ms exceeds 10ms"
    
    @pytest.mark.asyncio
    async def test_external_tool_latency_simulation(self, mcp_server):
        """Simulate external tool latency (should be < 200ms)"""
        await mcp_server.start()
        
        # Create mock external tool with simulated network delay
        async def mock_external_tool(self, name, parameters, timeout_ms=None):
            # Simulate network latency (50-150ms)
            await asyncio.sleep(0.05)  # 50ms base latency
            return {"result": "external", "data": parameters}
        
        # Register as external
        mcp_server._external_connections["test_conn"] = ["external_tool"]
        
        # Mock the external call
        with patch.object(mcp_server, '_callExternalTool', mock_external_tool):
            latencies = []
            
            # Warm up
            for _ in range(5):
                await mcp_server.callTool("external_tool", {})
            
            # Measure
            for _ in range(50):
                start = time.perf_counter()
                result = await mcp_server.callTool("external_tool", {"test": "data"})
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
                assert result.success
            
            # Calculate metrics
            latencies.sort()
            p95_index = int(len(latencies) * 0.95)
            p95_latency = latencies[p95_index]
            
            print(f"\nExternal Tool Performance (Simulated):")
            print(f"  P95: {p95_latency:.2f}ms")
            print(f"  Mean: {statistics.mean(latencies):.2f}ms")
            print(f"  Median: {statistics.median(latencies):.2f}ms")
            
            # Assert P95 < 200ms requirement
            assert p95_latency < 200, f"P95 latency {p95_latency:.2f}ms exceeds 200ms requirement"
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self, mock_nw_api):
        """Test performance under concurrent load"""
        registry = ToolRegistry()
        wrapper = LocalToolWrapper(mock_nw_api, registry)
        
        async def single_call():
            start = time.perf_counter()
            await wrapper.executeTool("get_project_info", {})
            return (time.perf_counter() - start) * 1000
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        print("\nConcurrent Calls Performance:")
        for level in concurrency_levels:
            # Run concurrent calls
            tasks = [single_call() for _ in range(level * 10)]
            start = time.perf_counter()
            latencies = await asyncio.gather(*tasks)
            total_time = (time.perf_counter() - start) * 1000
            
            latencies.sort()
            p95_index = int(len(latencies) * 0.95)
            p95_latency = latencies[p95_index]
            
            print(f"  Concurrency {level}:")
            print(f"    P95: {p95_latency:.2f}ms")
            print(f"    Mean: {statistics.mean(latencies):.2f}ms")
            print(f"    Total time: {total_time:.2f}ms")
            print(f"    Throughput: {len(tasks) / (total_time / 1000):.0f} calls/sec")
            
            # Even under load, P95 should stay under reasonable threshold
            assert p95_latency < 50, f"P95 {p95_latency:.2f}ms too high under load"
    
    @pytest.mark.asyncio
    async def test_tool_registry_performance(self):
        """Test tool registry operations performance"""
        registry = ToolRegistry()
        
        # Register many tools
        for i in range(100):
            registry.registerTool(
                name=f"tool_{i}",
                handler=Mock(),
                description=f"Test tool {i}",
                category="test"
            )
        
        # Measure lookup performance
        lookup_times = []
        for _ in range(1000):
            tool_name = f"tool_{_ % 100}"
            start = time.perf_counter()
            registry.hasTool(tool_name)
            registry.getTool(tool_name)
            end = time.perf_counter()
            lookup_times.append((end - start) * 1000)
        
        mean_lookup = statistics.mean(lookup_times)
        max_lookup = max(lookup_times)
        
        print(f"\nRegistry Performance (100 tools):")
        print(f"  Mean lookup: {mean_lookup:.3f}ms")
        print(f"  Max lookup: {max_lookup:.3f}ms")
        
        # Registry operations should be very fast
        assert mean_lookup < 0.1, f"Mean lookup {mean_lookup:.3f}ms too slow"
        assert max_lookup < 1.0, f"Max lookup {max_lookup:.3f}ms too slow"
