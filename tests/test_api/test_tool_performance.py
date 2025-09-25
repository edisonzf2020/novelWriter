"""
novelWriter – Tool Performance Test Suite
==========================================

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
import time
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime
from typing import List

from novelwriter.api.tools.project_tools import ProjectInfoTool, ProjectTreeTool
from novelwriter.api.tools.document_tools import (
    DocumentListTool, DocumentReadTool, DocumentWriteTool, CreateDocumentTool
)
from novelwriter.api.tools.search_tools import (
    GlobalSearchTool, TagListTool, ProjectStatsTool
)


@pytest.fixture
def fast_mock_api():
    """Create a mock API with minimal latency for performance testing"""
    api = Mock()
    
    # Configure all methods to return quickly
    api.getProjectMeta.return_value = {"title": "Test", "wordCount": 1000}
    api.getProjectStats.return_value = {"totalWords": 1000}
    api.getProjectTree.return_value = {"handle": "root", "title": "Root"}
    api.listDocuments.return_value = []
    api.getDocText.return_value = "content"
    api.setDocText.return_value = True
    api.createDocument.return_value = "handle"
    api.searchProject.return_value = []
    api.getTagList.return_value = []
    
    return api


class PerformanceTestBase:
    """Base class for performance testing"""
    
    async def measure_latency(self, tool, params=None, iterations=100):
        """
        Measure tool execution latency
        
        Args:
            tool: Tool instance to test
            params: Parameters to pass to tool
            iterations: Number of iterations for testing
            
        Returns:
            Dictionary with performance metrics
        """
        if params is None:
            params = {}
        
        latencies = []
        
        # Warm-up runs
        for _ in range(10):
            await tool.execute(**params)
        
        # Actual measurement
        for _ in range(iterations):
            start = time.perf_counter()
            result = await tool.execute(**params)
            end = time.perf_counter()
            
            if result.success:
                latencies.append((end - start) * 1000)  # Convert to ms
        
        if not latencies:
            pytest.fail("No successful executions")
        
        return {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "std": np.std(latencies)
        }
    
    def assert_performance(self, metrics, p95_threshold=10.0):
        """
        Assert performance meets requirements
        
        Args:
            metrics: Performance metrics dictionary
            p95_threshold: P95 latency threshold in ms
        """
        assert metrics["p95"] < p95_threshold, (
            f"P95 latency {metrics['p95']:.2f}ms exceeds {p95_threshold}ms threshold"
        )


class TestProjectToolsPerformance(PerformanceTestBase):
    """Performance tests for project tools"""
    
    @pytest.mark.asyncio
    async def test_project_info_tool_performance(self, fast_mock_api):
        """Test ProjectInfoTool meets <10ms P95 latency requirement"""
        tool = ProjectInfoTool(fast_mock_api)
        
        metrics = await self.measure_latency(tool)
        
        print(f"\nProjectInfoTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)
    
    @pytest.mark.asyncio
    async def test_project_tree_tool_performance(self, fast_mock_api):
        """Test ProjectTreeTool meets <10ms P95 latency requirement"""
        tool = ProjectTreeTool(fast_mock_api)
        
        metrics = await self.measure_latency(tool)
        
        print(f"\nProjectTreeTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)


class TestDocumentToolsPerformance(PerformanceTestBase):
    """Performance tests for document tools"""
    
    @pytest.mark.asyncio
    async def test_document_list_tool_performance(self, fast_mock_api):
        """Test DocumentListTool meets <10ms P95 latency requirement"""
        tool = DocumentListTool(fast_mock_api)
        
        metrics = await self.measure_latency(tool)
        
        print(f"\nDocumentListTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)
    
    @pytest.mark.asyncio
    async def test_document_read_tool_performance(self, fast_mock_api):
        """Test DocumentReadTool meets <10ms P95 latency requirement"""
        tool = DocumentReadTool(fast_mock_api)
        
        params = {"handle": "test_doc"}
        metrics = await self.measure_latency(tool, params)
        
        print(f"\nDocumentReadTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)
    
    @pytest.mark.asyncio
    async def test_document_write_tool_performance(self, fast_mock_api):
        """Test DocumentWriteTool meets <10ms P95 latency requirement"""
        tool = DocumentWriteTool(fast_mock_api)
        
        params = {"handle": "test_doc", "content": "test content"}
        metrics = await self.measure_latency(tool, params)
        
        print(f"\nDocumentWriteTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)
    
    @pytest.mark.asyncio
    async def test_create_document_tool_performance(self, fast_mock_api):
        """Test CreateDocumentTool meets <10ms P95 latency requirement"""
        tool = CreateDocumentTool(fast_mock_api)
        
        params = {"title": "New Doc"}
        metrics = await self.measure_latency(tool, params)
        
        print(f"\nCreateDocumentTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)


class TestSearchToolsPerformance(PerformanceTestBase):
    """Performance tests for search tools"""
    
    @pytest.mark.asyncio
    async def test_global_search_tool_performance(self, fast_mock_api):
        """Test GlobalSearchTool meets <10ms P95 latency requirement"""
        tool = GlobalSearchTool(fast_mock_api)
        
        params = {"query": "test"}
        metrics = await self.measure_latency(tool, params)
        
        print(f"\nGlobalSearchTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)
    
    @pytest.mark.asyncio
    async def test_tag_list_tool_performance(self, fast_mock_api):
        """Test TagListTool meets <10ms P95 latency requirement"""
        tool = TagListTool(fast_mock_api)
        
        metrics = await self.measure_latency(tool)
        
        print(f"\nTagListTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)
    
    @pytest.mark.asyncio
    async def test_project_stats_tool_performance(self, fast_mock_api):
        """Test ProjectStatsTool meets <10ms P95 latency requirement"""
        tool = ProjectStatsTool(fast_mock_api)
        
        metrics = await self.measure_latency(tool)
        
        print(f"\nProjectStatsTool Performance:")
        print(f"  Mean: {metrics['mean']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        
        self.assert_performance(metrics)


class TestConcurrentPerformance(PerformanceTestBase):
    """Test concurrent tool execution performance"""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, fast_mock_api):
        """Test multiple tools executing concurrently"""
        tools = [
            ProjectInfoTool(fast_mock_api),
            DocumentListTool(fast_mock_api),
            TagListTool(fast_mock_api),
            ProjectStatsTool(fast_mock_api)
        ]
        
        async def execute_tool(tool):
            start = time.perf_counter()
            result = await tool.execute()
            end = time.perf_counter()
            return (end - start) * 1000, result.success
        
        # Test concurrent execution
        latencies = []
        for _ in range(20):
            tasks = [execute_tool(tool) for tool in tools]
            results = await asyncio.gather(*tasks)
            
            for latency, success in results:
                if success:
                    latencies.append(latency)
        
        p95 = np.percentile(latencies, 95)
        print(f"\nConcurrent Execution Performance:")
        print(f"  P95: {p95:.2f}ms")
        
        assert p95 < 10.0, f"Concurrent P95 latency {p95:.2f}ms exceeds 10ms threshold"
    
    @pytest.mark.asyncio
    async def test_high_load_performance(self, fast_mock_api):
        """Test performance under high load"""
        tool = ProjectInfoTool(fast_mock_api)
        
        async def stress_test():
            tasks = []
            for _ in range(100):
                tasks.append(tool.execute())
            
            start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            end = time.perf_counter()
            
            successful = sum(1 for r in results if r.success)
            total_time = (end - start) * 1000
            
            return successful, total_time
        
        successful, total_time = await stress_test()
        
        print(f"\nHigh Load Performance (100 concurrent requests):")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Average per request: {total_time/100:.2f}ms")
        print(f"  Successful: {successful}/100")
        
        assert successful == 100, f"Only {successful}/100 requests succeeded"
        assert total_time/100 < 20.0, "Average latency under load exceeds 20ms"


class TestMemoryEfficiency:
    """Test memory efficiency of tools"""
    
    @pytest.mark.asyncio
    async def test_tool_memory_usage(self, fast_mock_api):
        """Test that tools don't leak memory"""
        import gc
        import sys
        
        tool = ProjectInfoTool(fast_mock_api)
        
        # Get initial memory baseline
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Execute tool many times
        for _ in range(1000):
            await tool.execute()
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Check for memory leaks
        object_growth = final_objects - initial_objects
        
        print(f"\nMemory Efficiency:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Object growth: {object_growth}")
        
        # Allow some growth but not excessive
        # Note: Pydantic models and async operations can create many objects
        # Allow up to 10000 objects growth for 1000 iterations
        assert object_growth < 10000, f"Excessive object growth: {object_growth}"


class TestPerformanceRegression:
    """Test for performance regression"""
    
    @pytest.mark.asyncio
    async def test_performance_baseline(self, fast_mock_api):
        """Establish and verify performance baseline"""
        baseline = {
            "ProjectInfoTool": 5.0,
            "ProjectTreeTool": 5.0,
            "DocumentListTool": 5.0,
            "DocumentReadTool": 5.0,
            "DocumentWriteTool": 5.0,
            "CreateDocumentTool": 5.0,
            "GlobalSearchTool": 5.0,
            "TagListTool": 5.0,
            "ProjectStatsTool": 5.0
        }
        
        tools = {
            "ProjectInfoTool": ProjectInfoTool(fast_mock_api),
            "ProjectTreeTool": ProjectTreeTool(fast_mock_api),
            "DocumentListTool": DocumentListTool(fast_mock_api),
            "DocumentReadTool": DocumentReadTool(fast_mock_api),
            "DocumentWriteTool": DocumentWriteTool(fast_mock_api),
            "CreateDocumentTool": CreateDocumentTool(fast_mock_api),
            "GlobalSearchTool": GlobalSearchTool(fast_mock_api),
            "TagListTool": TagListTool(fast_mock_api),
            "ProjectStatsTool": ProjectStatsTool(fast_mock_api)
        }
        
        print("\nPerformance Baseline Verification:")
        
        for tool_name, tool in tools.items():
            # Measure current performance
            latencies = []
            for _ in range(50):
                start = time.perf_counter()
                
                # Add required params for specific tools
                if tool_name == "DocumentReadTool":
                    result = await tool.execute(handle="test")
                elif tool_name == "DocumentWriteTool":
                    result = await tool.execute(handle="test", content="content")
                elif tool_name == "CreateDocumentTool":
                    result = await tool.execute(title="Test")
                elif tool_name == "GlobalSearchTool":
                    result = await tool.execute(query="test")
                else:
                    result = await tool.execute()
                
                if result.success:
                    latencies.append((time.perf_counter() - start) * 1000)
            
            if latencies:
                p95 = np.percentile(latencies, 95)
                baseline_p95 = baseline[tool_name]
                
                print(f"  {tool_name}: {p95:.2f}ms (baseline: {baseline_p95:.2f}ms)")
                
                # Allow 50% regression from baseline
                assert p95 < baseline_p95 * 1.5, (
                    f"{tool_name} P95 {p95:.2f}ms exceeds 150% of baseline {baseline_p95:.2f}ms"
                )
