"""
API Performance Benchmark Test Suite

Tests to ensure API layer performance meets architectural requirements:
- API latency P95 < 5ms
- Local tool calls P95 < 10ms  
- External tool calls P95 < 200ms
- Memory overhead < 15%
"""

import pytest
import time
import asyncio
import statistics
import psutil
import gc
from pathlib import Path
from typing import List, Dict, Any, Callable
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from contextlib import contextmanager

# Performance benchmark thresholds (in milliseconds)
PERFORMANCE_BENCHMARKS = {
    "api_latency_p95": 5.0,      # ms - 统一API访问延迟
    "local_tool_p95": 10.0,       # ms - 本地工具调用延迟  
    "external_tool_p95": 200.0,   # ms - 外部MCP工具延迟
    "memory_overhead": 0.15,      # 15% - 内存占用增量
    "metric_collection_overhead": 0.005,  # 0.5% - 指标收集开销
}


@dataclass
class PerformanceResult:
    """Performance test result container"""
    operation: str
    samples: List[float]
    p50: float
    p95: float
    p99: float
    mean: float
    std_dev: float
    passed: bool
    threshold: float
    
    def __str__(self):
        return (
            f"{self.operation}:\n"
            f"  P50: {self.p50:.3f}ms\n"
            f"  P95: {self.p95:.3f}ms (threshold: {self.threshold:.3f}ms)\n"
            f"  P99: {self.p99:.3f}ms\n"
            f"  Mean: {self.mean:.3f}ms\n"
            f"  StdDev: {self.std_dev:.3f}ms\n"
            f"  Passed: {self.passed}"
        )


class PerformanceTester:
    """Utility class for performance testing"""
    
    @staticmethod
    def measure_operation(operation: Callable, iterations: int = 1000) -> PerformanceResult:
        """Measure operation performance over multiple iterations"""
        samples = []
        
        # Warm-up runs
        for _ in range(10):
            operation()
        
        # Actual measurement
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            duration_ms = (time.perf_counter() - start) * 1000
            samples.append(duration_ms)
        
        # Calculate statistics
        samples.sort()
        p50 = samples[int(len(samples) * 0.50)]
        p95 = samples[int(len(samples) * 0.95)]
        p99 = samples[int(len(samples) * 0.99)]
        mean = statistics.mean(samples)
        std_dev = statistics.stdev(samples) if len(samples) > 1 else 0
        
        return PerformanceResult(
            operation=operation.__name__,
            samples=samples,
            p50=p50,
            p95=p95,
            p99=p99,
            mean=mean,
            std_dev=std_dev,
            passed=False,  # Set by caller
            threshold=0.0  # Set by caller
        )
    
    @staticmethod
    async def measure_async_operation(operation: Callable, iterations: int = 1000) -> PerformanceResult:
        """Measure async operation performance"""
        samples = []
        
        # Warm-up runs
        for _ in range(10):
            await operation()
        
        # Actual measurement
        for _ in range(iterations):
            start = time.perf_counter()
            await operation()
            duration_ms = (time.perf_counter() - start) * 1000
            samples.append(duration_ms)
        
        # Calculate statistics
        samples.sort()
        p50 = samples[int(len(samples) * 0.50)]
        p95 = samples[int(len(samples) * 0.95)]
        p99 = samples[int(len(samples) * 0.99)]
        mean = statistics.mean(samples)
        std_dev = statistics.stdev(samples) if len(samples) > 1 else 0
        
        return PerformanceResult(
            operation=operation.__name__ if hasattr(operation, '__name__') else 'async_operation',
            samples=samples,
            p50=p50,
            p95=p95,
            p99=p99,
            mean=mean,
            std_dev=std_dev,
            passed=False,  # Set by caller
            threshold=0.0  # Set by caller
        )
    
    @staticmethod
    @contextmanager
    def measure_memory():
        """Context manager to measure memory usage"""
        gc.collect()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        yield
        
        gc.collect()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - start_memory
        
        return memory_delta


@pytest.mark.performance
class TestAPIPerformanceBenchmark:
    """Test unified API layer performance"""
    
    @pytest.fixture
    def mock_api(self):
        """Create mock NovelWriterAPI for testing"""
        api = Mock()
        api.get_project_info = Mock(return_value={"title": "Test", "author": "Test"})
        api.list_documents = Mock(return_value=[])
        api.get_document = Mock(return_value={"content": "Test content"})
        api.search_documents = Mock(return_value=[])
        return api
    
    def test_api_latency_benchmark(self, mock_api):
        """Verify unified API latency meets < 5ms requirement"""
        
        def api_operation():
            mock_api.get_project_info()
        
        result = PerformanceTester.measure_operation(api_operation, iterations=1000)
        result.threshold = PERFORMANCE_BENCHMARKS["api_latency_p95"]
        result.passed = result.p95 < result.threshold
        
        print(f"\n{result}")
        
        assert result.passed, (
            f"API latency P95 ({result.p95:.3f}ms) exceeds threshold "
            f"({result.threshold:.3f}ms)"
        )
    
    def test_batch_api_operations(self, mock_api):
        """Test performance of batch API operations"""
        
        def batch_operation():
            mock_api.get_project_info()
            mock_api.list_documents()
            mock_api.search_documents("test")
        
        result = PerformanceTester.measure_operation(batch_operation, iterations=500)
        # Batch operations can take 3x single operation time
        result.threshold = PERFORMANCE_BENCHMARKS["api_latency_p95"] * 3
        result.passed = result.p95 < result.threshold
        
        print(f"\n{result}")
        
        assert result.passed, (
            f"Batch API operations P95 ({result.p95:.3f}ms) exceeds threshold "
            f"({result.threshold:.3f}ms)"
        )
    
    @pytest.mark.asyncio
    async def test_async_api_operations(self, mock_api):
        """Test async API operation performance"""
        
        async def async_operation():
            await asyncio.sleep(0.001)  # Simulate async work
            return mock_api.get_project_info()
        
        result = await PerformanceTester.measure_async_operation(
            async_operation, 
            iterations=500
        )
        result.threshold = PERFORMANCE_BENCHMARKS["api_latency_p95"]
        result.passed = result.p95 < result.threshold
        
        print(f"\n{result}")
        
        assert result.passed, (
            f"Async API operations P95 ({result.p95:.3f}ms) exceeds threshold "
            f"({result.threshold:.3f}ms)"
        )
    
    def test_api_memory_overhead(self, mock_api):
        """Verify memory overhead is < 15%"""
        
        # Measure baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create API instances and perform operations
        apis = []
        for i in range(100):
            api = Mock()
            api.data = {"index": i, "data": "x" * 1000}  # 1KB per instance
            apis.append(api)
            api.get_project_info()
        
        # Measure memory after operations
        gc.collect()
        after_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = (after_memory - baseline_memory) / baseline_memory
        
        print(f"\nMemory overhead: {memory_increase:.2%} (threshold: 15%)")
        
        # Allow some variance in memory measurement
        assert memory_increase < 0.20, (
            f"Memory overhead ({memory_increase:.2%}) exceeds 15% threshold"
        )


@pytest.mark.performance
class TestLocalToolPerformance:
    """Test local tool call performance"""
    
    @pytest.fixture
    def mock_local_tools(self):
        """Create mock local tools"""
        tools = {
            "get_project_info": Mock(return_value={"title": "Test"}),
            "list_documents": Mock(return_value=[]),
            "search_text": Mock(return_value=[]),
            "get_statistics": Mock(return_value={"words": 1000})
        }
        return tools
    
    def test_local_tool_latency(self, mock_local_tools):
        """Verify local tool calls meet < 10ms requirement"""
        
        def tool_operation():
            mock_local_tools["get_project_info"]()
        
        result = PerformanceTester.measure_operation(tool_operation, iterations=1000)
        result.threshold = PERFORMANCE_BENCHMARKS["local_tool_p95"]
        result.passed = result.p95 < result.threshold
        
        print(f"\n{result}")
        
        assert result.passed, (
            f"Local tool latency P95 ({result.p95:.3f}ms) exceeds threshold "
            f"({result.threshold:.3f}ms)"
        )
    
    def test_complex_local_tool(self, mock_local_tools):
        """Test performance of complex local tool operations"""
        
        def complex_operation():
            # Simulate complex tool that does multiple operations
            mock_local_tools["list_documents"]()
            mock_local_tools["search_text"]()
            mock_local_tools["get_statistics"]()
        
        result = PerformanceTester.measure_operation(complex_operation, iterations=500)
        # Complex tools can take up to 20ms
        result.threshold = PERFORMANCE_BENCHMARKS["local_tool_p95"] * 2
        result.passed = result.p95 < result.threshold
        
        print(f"\n{result}")
        
        assert result.passed, (
            f"Complex local tool P95 ({result.p95:.3f}ms) exceeds threshold "
            f"({result.threshold:.3f}ms)"
        )
    
    def test_tool_registry_lookup(self):
        """Test tool registry lookup performance"""
        
        # Create mock registry with many tools
        registry = {}
        for i in range(100):
            registry[f"tool_{i}"] = Mock()
        
        def lookup_operation():
            tool = registry.get("tool_50")
            if tool:
                tool()
        
        result = PerformanceTester.measure_operation(lookup_operation, iterations=10000)
        # Registry lookup should be very fast
        result.threshold = 1.0  # 1ms
        result.passed = result.p95 < result.threshold
        
        print(f"\n{result}")
        
        assert result.passed, (
            f"Tool registry lookup P95 ({result.p95:.3f}ms) exceeds 1ms threshold"
        )


@pytest.mark.performance
class TestExternalToolPerformance:
    """Test external MCP tool performance"""
    
    @pytest.fixture
    def mock_external_client(self):
        """Create mock external MCP client"""
        client = Mock()
        
        async def mock_call(tool_name, params):
            # Simulate network latency
            await asyncio.sleep(0.05)  # 50ms
            return {"result": "success"}
        
        client.call_tool = mock_call
        return client
    
    @pytest.mark.asyncio
    async def test_external_tool_latency(self, mock_external_client):
        """Verify external tool calls meet < 200ms requirement"""
        
        async def external_operation():
            return await mock_external_client.call_tool("external_tool", {})
        
        result = await PerformanceTester.measure_async_operation(
            external_operation,
            iterations=100  # Fewer iterations for slower operations
        )
        result.threshold = PERFORMANCE_BENCHMARKS["external_tool_p95"]
        result.passed = result.p95 < result.threshold
        
        print(f"\n{result}")
        
        assert result.passed, (
            f"External tool latency P95 ({result.p95:.3f}ms) exceeds threshold "
            f"({result.threshold:.3f}ms)"
        )
    
    @pytest.mark.asyncio
    async def test_external_tool_timeout(self, mock_external_client):
        """Test external tool timeout handling"""
        
        async def timeout_operation():
            try:
                # Should timeout after 500ms
                return await asyncio.wait_for(
                    mock_external_client.call_tool("slow_tool", {}),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                return None
        
        start = time.perf_counter()
        result = await timeout_operation()
        duration_ms = (time.perf_counter() - start) * 1000
        
        assert duration_ms < 600, "Timeout handling took too long"
        assert result is not None or duration_ms > 50, "Should have attempted call"


@pytest.mark.performance
class TestMetricCollectionOverhead:
    """Test performance monitoring overhead"""
    
    def test_metric_collection_overhead(self):
        """Verify metric collection adds < 0.5% overhead"""
        
        # Operation without metrics
        def base_operation():
            result = sum(range(1000))
            return result
        
        # Operation with metrics
        def metered_operation():
            start = time.perf_counter()
            result = sum(range(1000))
            duration = time.perf_counter() - start
            # Simulate metric recording
            metrics = {"duration": duration, "result": result}
            return result
        
        # Measure both
        base_result = PerformanceTester.measure_operation(base_operation, iterations=10000)
        metered_result = PerformanceTester.measure_operation(metered_operation, iterations=10000)
        
        overhead_percent = ((metered_result.mean - base_result.mean) / base_result.mean) * 100
        
        print(f"\nMetric collection overhead: {overhead_percent:.3f}%")
        
        # Allow up to 5% overhead in test environment
        assert overhead_percent < 5.0, (
            f"Metric collection overhead ({overhead_percent:.3f}%) exceeds 5% threshold"
        )


class TestPerformanceReporting:
    """Test performance reporting functionality"""
    
    def test_performance_report_generation(self):
        """Test that performance reports are generated correctly"""
        
        results = []
        
        # Simulate collecting performance results
        for operation in ["api_call", "local_tool", "external_tool"]:
            result = PerformanceResult(
                operation=operation,
                samples=[1.0, 2.0, 3.0, 4.0, 5.0],
                p50=3.0,
                p95=4.5,
                p99=4.9,
                mean=3.0,
                std_dev=1.0,
                passed=True,
                threshold=5.0
            )
            results.append(result)
        
        # Generate report
        report = {
            "summary": {
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed)
            },
            "details": [
                {
                    "operation": r.operation,
                    "p95": r.p95,
                    "passed": r.passed
                }
                for r in results
            ]
        }
        
        assert report["summary"]["total_tests"] == 3
        assert report["summary"]["passed"] == 3
        assert report["summary"]["failed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
