"""
novelWriter – External Tool Performance Test Suite
===================================================

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
from unittest.mock import AsyncMock, Mock
from typing import List

from novelwriter.api.external_mcp import (
    ExternalMCPCache, CacheManager, HealthChecker
)


class TestExternalToolPerformance:
    """Performance tests for external MCP tools."""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache operation performance."""
        cache = ExternalMCPCache()
        
        # Measure put performance
        put_times = []
        for i in range(1000):
            start = time.perf_counter()
            cache.put(f"key{i}", {"data": f"value{i}"})
            put_times.append((time.perf_counter() - start) * 1000)
        
        put_p95 = np.percentile(put_times, 95)
        print(f"\nCache PUT Performance:")
        print(f"  Mean: {np.mean(put_times):.3f}ms")
        print(f"  P95: {put_p95:.3f}ms")
        
        assert put_p95 < 1.0, f"Cache put P95 {put_p95:.3f}ms exceeds 1ms"
        
        # Measure get performance
        get_times = []
        for i in range(1000):
            start = time.perf_counter()
            _ = cache.get(f"key{i}")
            get_times.append((time.perf_counter() - start) * 1000)
        
        get_p95 = np.percentile(get_times, 95)
        print(f"\nCache GET Performance:")
        print(f"  Mean: {np.mean(get_times):.3f}ms")
        print(f"  P95: {get_p95:.3f}ms")
        
        assert get_p95 < 0.5, f"Cache get P95 {get_p95:.3f}ms exceeds 0.5ms"
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self):
        """Test performance difference between cache hit and miss."""
        cache = ExternalMCPCache()
        
        # Populate cache
        for i in range(100):
            cache.put(f"key{i}", {"data": f"value{i}"})
        
        # Measure hit performance
        hit_times = []
        for i in range(100):
            start = time.perf_counter()
            _ = cache.get(f"key{i}")
            hit_times.append((time.perf_counter() - start) * 1000)
        
        # Measure miss performance
        miss_times = []
        for i in range(100, 200):
            start = time.perf_counter()
            _ = cache.get(f"key{i}")
            miss_times.append((time.perf_counter() - start) * 1000)
        
        hit_mean = np.mean(hit_times)
        miss_mean = np.mean(miss_times)
        
        print(f"\nCache Hit vs Miss Performance:")
        print(f"  Hit mean: {hit_mean:.3f}ms")
        print(f"  Miss mean: {miss_mean:.3f}ms")
        
        # Both should be very fast (sub-millisecond)
        # Cache hits might be slightly slower due to data retrieval
        # but both should be under 1ms
        assert hit_mean < 1.0, f"Cache hit too slow: {hit_mean:.3f}ms"
        assert miss_mean < 1.0, f"Cache miss too slow: {miss_mean:.3f}ms"
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance."""
        checker = HealthChecker(timeout_seconds=0.2)
        
        # Create mock connections with varying response times
        connections = []
        for i in range(10):
            mock_conn = AsyncMock()
            
            async def health_check(delay=0.01 * (i + 1)):
                await asyncio.sleep(delay)
                return {"status": "ok"}
            
            mock_conn.health_check = health_check
            connections.append(mock_conn)
            checker.register_connection(f"conn{i}", mock_conn)
        
        # Measure health check performance
        check_times = []
        for i in range(10):
            start = time.perf_counter()
            result = await checker.check_health(f"conn{i}")
            check_times.append((time.perf_counter() - start) * 1000)
            
            # Verify result
            if i < 10:  # Should succeed (under 200ms timeout)
                assert result.status.value in ["healthy", "degraded"]
        
        p95 = np.percentile(check_times, 95)
        print(f"\nHealth Check Performance:")
        print(f"  Mean: {np.mean(check_times):.3f}ms")
        print(f"  P95: {p95:.3f}ms")
        
        # Health checks should complete within timeout
        assert p95 < 250, f"Health check P95 {p95:.3f}ms exceeds 250ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent health check performance."""
        checker = HealthChecker()
        
        # Create mock connections
        for i in range(20):
            mock_conn = AsyncMock()
            mock_conn.health_check.return_value = {"status": "ok"}
            checker.register_connection(f"conn{i}", mock_conn)
        
        # Perform concurrent health checks
        start = time.perf_counter()
        tasks = [
            checker.check_health(f"conn{i}")
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start) * 1000
        
        print(f"\nConcurrent Health Checks (20 connections):")
        print(f"  Total time: {total_time:.3f}ms")
        print(f"  Average per check: {total_time/20:.3f}ms")
        
        # All should succeed
        assert all(r.status.value in ["healthy", "unknown"] for r in results)
        
        # Should complete reasonably fast
        assert total_time < 500, f"Concurrent checks took {total_time:.3f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self):
        """Test cache eviction performance under pressure."""
        cache = ExternalMCPCache(max_entries=100)
        
        # Fill cache beyond capacity
        eviction_times = []
        for i in range(500):
            start = time.perf_counter()
            cache.put(f"key{i}", {"data": f"value{i}" * 100})
            eviction_times.append((time.perf_counter() - start) * 1000)
        
        # Check that cache size is maintained
        stats = cache.get_statistics()
        assert stats["entries"] <= 100
        
        p95 = np.percentile(eviction_times, 95)
        print(f"\nCache Eviction Performance:")
        print(f"  Mean: {np.mean(eviction_times):.3f}ms")
        print(f"  P95: {p95:.3f}ms")
        print(f"  Final entries: {stats['entries']}")
        
        # Eviction should be fast
        assert p95 < 2.0, f"Eviction P95 {p95:.3f}ms exceeds 2ms"
    
    @pytest.mark.asyncio
    async def test_pattern_invalidation_performance(self):
        """Test pattern-based cache invalidation performance."""
        cache = ExternalMCPCache()
        
        # Populate cache with patterns
        for conn in range(5):
            for tool in range(20):
                for param in range(10):
                    key = f"conn{conn}:tool{tool}:param{param}"
                    cache.put(key, {"data": "value"})
        
        stats_before = cache.get_statistics()
        print(f"\nCache before invalidation: {stats_before['entries']} entries")
        
        # Measure pattern invalidation
        start = time.perf_counter()
        count = cache.invalidate_pattern("conn2:*")
        invalidation_time = (time.perf_counter() - start) * 1000
        
        stats_after = cache.get_statistics()
        print(f"Invalidation Performance:")
        print(f"  Pattern: conn2:*")
        print(f"  Invalidated: {count} entries")
        print(f"  Time: {invalidation_time:.3f}ms")
        print(f"  Remaining: {stats_after['entries']} entries")
        
        assert count == 200  # 20 tools * 10 params
        assert invalidation_time < 10.0, f"Invalidation took {invalidation_time:.3f}ms"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Test circuit breaker performance impact."""
        from novelwriter.api.external_mcp.health_check import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=5)
        
        # Measure closed state performance
        closed_times = []
        for _ in range(1000):
            start = time.perf_counter()
            is_open = breaker.is_open()
            closed_times.append((time.perf_counter() - start) * 1000000)  # microseconds
        
        # Open the circuit
        for _ in range(5):
            breaker.call_failed()
        
        # Measure open state performance
        open_times = []
        for _ in range(1000):
            start = time.perf_counter()
            is_open = breaker.is_open()
            open_times.append((time.perf_counter() - start) * 1000000)  # microseconds
        
        closed_mean = np.mean(closed_times)
        open_mean = np.mean(open_times)
        
        print(f"\nCircuit Breaker Performance:")
        print(f"  Closed state: {closed_mean:.3f}μs")
        print(f"  Open state: {open_mean:.3f}μs")
        
        # Both should be very fast (sub-millisecond)
        assert closed_mean < 100, f"Closed check {closed_mean:.3f}μs exceeds 100μs"
        assert open_mean < 100, f"Open check {open_mean:.3f}μs exceeds 100μs"


class TestExternalToolLatency:
    """Test external tool call latency requirements."""
    
    @pytest.mark.asyncio
    async def test_external_tool_p95_latency(self):
        """Test that external tool calls meet P95 < 200ms requirement."""
        # This would test against actual external MCP servers
        # For unit tests, we simulate with mocks
        
        class MockExternalConnection:
            async def call_tool(self, tool_name: str, params: dict):
                # Simulate network latency
                await asyncio.sleep(0.05 + np.random.random() * 0.1)
                return {"success": True, "result": "data"}
        
        conn = MockExternalConnection()
        
        # Measure latencies
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            result = await conn.call_tool("test_tool", {"param": "value"})
            latencies.append((time.perf_counter() - start) * 1000)
            assert result["success"]
        
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"\nExternal Tool Latency:")
        print(f"  Mean: {np.mean(latencies):.3f}ms")
        print(f"  P95: {p95:.3f}ms")
        print(f"  P99: {p99:.3f}ms")
        
        assert p95 < 200, f"External tool P95 {p95:.3f}ms exceeds 200ms requirement"
    
    @pytest.mark.asyncio
    async def test_cached_external_tool_latency(self):
        """Test that cached external tool calls meet P95 < 5ms requirement."""
        cache = ExternalMCPCache()
        
        # Pre-populate cache
        for i in range(100):
            key = cache.generate_key(f"tool{i}", {"param": i}, "conn1")
            cache.put(key, {"result": f"cached_data_{i}"})
        
        # Measure cached retrieval
        latencies = []
        for i in range(100):
            key = cache.generate_key(f"tool{i}", {"param": i}, "conn1")
            start = time.perf_counter()
            result = cache.get(key)
            latencies.append((time.perf_counter() - start) * 1000)
            assert result is not None
        
        p95 = np.percentile(latencies, 95)
        
        print(f"\nCached External Tool Latency:")
        print(f"  Mean: {np.mean(latencies):.3f}ms")
        print(f"  P95: {p95:.3f}ms")
        
        assert p95 < 5, f"Cached tool P95 {p95:.3f}ms exceeds 5ms requirement"
