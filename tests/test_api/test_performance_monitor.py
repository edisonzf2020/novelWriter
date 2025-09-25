"""
novelWriter – Performance Monitor Tests
========================================

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

import asyncio
import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import statistics as stats

from novelwriter.api.base.performance import (
    MetricType,
    PerformanceAlert,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceStatistics,
    PerformanceThreshold,
    SlidingWindowStats,
    TDigest,
    get_performance_monitor,
    monitor_performance,
    set_performance_monitor,
)
from novelwriter.api.base.performance_stats import (
    AnomalyDetector,
    PerformanceBaseline,
    PerformanceComparator,
    PerformanceHotspotDetector,
    StatisticsAggregator,
    TrendAnalyzer,
)


class TestTDigest:
    """Test T-Digest algorithm."""
    
    def test_basic_quantiles(self):
        """Test basic quantile calculations."""
        digest = TDigest(compression=100)
        
        # Add values
        for i in range(100):
            digest.add(float(i))
        
        # Test quantiles (with wider tolerance for approximation)
        assert abs(digest.quantile(0.5) - 50) < 10  # Median
        assert abs(digest.quantile(0.95) - 95) < 10  # P95
        assert abs(digest.quantile(0.99) - 99) < 10  # P99
    
    def test_compression(self):
        """Test that compression maintains accuracy."""
        digest = TDigest(compression=50)
        
        # Add many values
        for i in range(10000):
            digest.add(i)
        
        # Should still be accurate despite compression
        assert abs(digest.quantile(0.5) - 5000) < 100
        assert abs(digest.quantile(0.95) - 9500) < 100
    
    def test_empty_digest(self):
        """Test empty digest behavior."""
        digest = TDigest()
        assert digest.quantile(0.5) == 0.0


class TestSlidingWindowStats:
    """Test sliding window statistics."""
    
    def test_window_expiration(self):
        """Test that old data expires from window."""
        window = SlidingWindowStats(window_size=1)  # 1 second window
        
        # Add old value
        old_time = datetime.now() - timedelta(seconds=2)
        window.add(100, old_time)
        
        # Add new value
        window.add(200)
        
        # Stats should only include new value
        stats = window.get_stats()
        assert stats["count"] == 1
        assert stats["mean"] == 200
    
    def test_statistics_calculation(self):
        """Test statistics calculations."""
        window = SlidingWindowStats()
        
        # Add values
        values = [10, 20, 30, 40, 50]
        for v in values:
            window.add(v)
        
        stats = window.get_stats()
        assert stats["count"] == 5
        assert stats["mean"] == 30
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["std_dev"] > 0


class TestPerformanceMonitor:
    """Test performance monitor."""
    
    def test_metric_recording(self):
        """Test recording performance metrics."""
        monitor = PerformanceMonitor()
        
        # Record metric
        monitor.record_metric(
            MetricType.LATENCY,
            "test_component",
            "test_operation",
            100.5
        )
        
        # Check metric was recorded
        assert len(monitor.metrics) == 1
        metric = monitor.metrics[0]
        assert metric.metric_type == MetricType.LATENCY
        assert metric.component == "test_component"
        assert metric.operation == "test_operation"
        assert metric.value == 100.5
    
    def test_operation_tracking(self):
        """Test operation start/end tracking."""
        monitor = PerformanceMonitor()
        
        # Start operation
        op_id = monitor.start_operation("component", "operation")
        assert monitor.active_operations["component:operation"] == 1
        
        # Simulate some work
        time.sleep(0.01)
        
        # End operation
        monitor.end_operation(op_id, success=True)
        assert monitor.active_operations["component:operation"] == 0
        
        # Check metrics were recorded
        latency_metrics = [
            m for m in monitor.metrics
            if m.metric_type == MetricType.LATENCY
        ]
        assert len(latency_metrics) == 1
        assert latency_metrics[0].value > 0  # Should have some latency
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        monitor = PerformanceMonitor()
        
        # Record multiple metrics
        for i in range(10):
            monitor.record_metric(
                MetricType.LATENCY,
                "component",
                "operation",
                float(i * 10)
            )
        
        # Get statistics
        stats = monitor.get_statistics("component", "operation")
        assert stats.count == 10
        assert stats.mean == 45.0  # Average of 0,10,20...90
        assert stats.min == 0.0
        assert stats.max == 90.0
    
    def test_threshold_checking(self):
        """Test threshold violation detection."""
        monitor = PerformanceMonitor()
        alerts_received = []
        
        # Add alert callback
        monitor.add_alert_callback(lambda a: alerts_received.append(a))
        
        # Add threshold
        monitor.add_threshold(PerformanceThreshold(
            metric_type=MetricType.LATENCY,
            component="api",
            operation="*",
            warning_threshold=100,
            critical_threshold=500
        ))
        
        # Record metric below threshold
        monitor.record_metric(MetricType.LATENCY, "api", "test", 50)
        assert len(alerts_received) == 0
        
        # Record metric above warning threshold
        monitor.record_metric(MetricType.LATENCY, "api", "test", 150)
        assert len(alerts_received) == 1
        assert alerts_received[0].level == "WARNING"
        
        # Record metric above critical threshold
        monitor.record_metric(MetricType.LATENCY, "api", "test", 600)
        assert len(alerts_received) == 2
        assert alerts_received[1].level == "CRITICAL"
    
    def test_baseline_comparison(self):
        """Test baseline saving and comparison."""
        monitor = PerformanceMonitor()
        
        # Record baseline metrics
        for i in range(10):
            monitor.record_metric(MetricType.LATENCY, "comp", "op", 100.0)
        
        # Save baseline
        monitor.save_baseline("comp", "op")
        assert "comp:op" in monitor.baselines
        
        # Record new metrics (20% worse)
        for i in range(10):
            monitor.record_metric(MetricType.LATENCY, "comp", "op", 120.0)
        
        # Compare to baseline
        comparison = monitor.compare_to_baseline("comp", "op")
        assert comparison is not None
        assert comparison["latency_change"] > 0  # Performance degraded
    
    def test_export_import(self):
        """Test metrics export and import."""
        monitor = PerformanceMonitor()
        
        # Record some metrics
        monitor.record_metric(MetricType.LATENCY, "comp", "op", 100)
        monitor.save_baseline("comp", "op")
        
        # Export to file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)
        
        monitor.export_metrics(filepath)
        assert filepath.exists()
        
        # Import baselines
        new_monitor = PerformanceMonitor()
        new_monitor.import_baselines(filepath)
        assert "comp:op" in new_monitor.baselines
        
        # Cleanup
        filepath.unlink()
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        monitor = PerformanceMonitor()
        errors = []
        
        def record_metrics():
            try:
                for i in range(100):
                    monitor.record_metric(
                        MetricType.LATENCY,
                        f"comp_{threading.current_thread().name}",
                        "op",
                        float(i)
                    )
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_metrics, name=f"thread_{i}")
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Should have metrics from all threads
        assert len(monitor.metrics) == 500


class TestPerformanceDecorator:
    """Test performance monitoring decorator."""
    
    def test_sync_function_monitoring(self):
        """Test monitoring synchronous functions."""
        monitor = PerformanceMonitor()
        set_performance_monitor(monitor)
        
        @monitor_performance(component="test", operation="sync_func")
        def test_func(x):
            time.sleep(0.01)
            return x * 2
        
        result = test_func(5)
        assert result == 10
        
        # Check metrics were recorded
        assert len(monitor.metrics) > 0
        latency_metrics = [
            m for m in monitor.metrics
            if m.metric_type == MetricType.LATENCY
        ]
        assert len(latency_metrics) == 1
        assert latency_metrics[0].value > 0
    
    @pytest.mark.asyncio
    async def test_async_function_monitoring(self):
        """Test monitoring asynchronous functions."""
        monitor = PerformanceMonitor()
        set_performance_monitor(monitor)
        
        @monitor_performance(component="test", operation="async_func")
        async def test_func(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        result = await test_func(5)
        assert result == 10
        
        # Check metrics were recorded
        assert len(monitor.metrics) > 0
    
    def test_exception_handling(self):
        """Test decorator handles exceptions properly."""
        monitor = PerformanceMonitor()
        set_performance_monitor(monitor)
        
        @monitor_performance(component="test", operation="failing_func")
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
        
        # Should still record metric with failure
        error_metrics = [
            m for m in monitor.metrics
            if m.metric_type == MetricType.ERROR_RATE
        ]
        assert len(error_metrics) == 1


class TestAnomalyDetector:
    """Test anomaly detection."""
    
    def test_anomaly_detection(self):
        """Test detecting anomalous values."""
        detector = AnomalyDetector(sensitivity=2.0)
        
        # Add normal values
        for i in range(20):
            is_anomaly = detector.add_value(100 + i)
            assert not is_anomaly  # Should not be anomalous
        
        # Add anomalous value
        is_anomaly = detector.add_value(500)
        assert is_anomaly  # Should be detected as anomaly
    
    def test_get_anomalies(self):
        """Test retrieving recent anomalies."""
        detector = AnomalyDetector()
        
        # Add values
        for i in range(20):
            detector.add_value(100)
        
        # Add anomalies
        detector.add_value(500)
        detector.add_value(600)
        
        anomalies = detector.get_anomalies(window_minutes=60)
        assert len(anomalies) >= 2


class TestTrendAnalyzer:
    """Test trend analysis."""
    
    def test_trend_detection(self):
        """Test detecting performance trends."""
        analyzer = TrendAnalyzer()
        
        # Add improving trend (decreasing latency)
        for i in range(20):
            analyzer.add_point(100 - i * 2)
        
        trend = analyzer.analyze_trend()
        assert trend.trend == "improving"
        assert trend.slope < 0
        assert trend.r_squared > 0.8  # Strong correlation
    
    def test_stable_trend(self):
        """Test detecting stable performance."""
        analyzer = TrendAnalyzer()
        
        # Add stable values with small variation
        for i in range(20):
            analyzer.add_point(100 + (i % 2))
        
        trend = analyzer.analyze_trend()
        assert trend.trend == "stable"
        assert abs(trend.slope) < 1
    
    def test_degrading_trend(self):
        """Test detecting degrading performance."""
        analyzer = TrendAnalyzer()
        
        # Add degrading trend (increasing latency)
        for i in range(20):
            analyzer.add_point(100 + i * 3)
        
        trend = analyzer.analyze_trend()
        assert trend.trend == "degrading"
        assert trend.slope > 0


class TestPerformanceComparator:
    """Test performance comparison."""
    
    def test_baseline_comparison(self):
        """Test comparing to baseline."""
        comparator = PerformanceComparator()
        
        # Save baseline
        baseline_stats = {
            "mean": 100,
            "p95": 150,
            "p99": 200,
            "success_rate": 0.99,
            "throughput": 100
        }
        comparator.save_baseline("api", "call", baseline_stats)
        
        # Compare with worse performance
        current_stats = {
            "mean": 120,  # 20% worse
            "p95": 180,   # 20% worse
            "p99": 250,   # 25% worse
            "success_rate": 0.95,  # 4% worse
            "throughput": 90  # 10% worse
        }
        
        comparison = comparator.compare_to_baseline("api", "call", current_stats)
        assert comparison["has_baseline"]
        assert comparison["mean_latency_change"] == 20.0
        assert comparison["success_rate_change"] == -4.0
        assert comparison["performance_degraded"]
    
    def test_regression_detection(self):
        """Test detecting performance regression."""
        comparator = PerformanceComparator()
        
        # Save baseline
        baseline_stats = {"mean": 100, "p95": 150, "success_rate": 0.99}
        comparator.save_baseline("api", "call", baseline_stats)
        
        # Test with small degradation (no regression)
        current_stats = {"mean": 105, "p95": 155, "success_rate": 0.98}
        assert not comparator.detect_regression("api", "call", current_stats)
        
        # Test with significant degradation (regression)
        current_stats = {"mean": 115, "p95": 180, "success_rate": 0.85}
        assert comparator.detect_regression("api", "call", current_stats)


class TestPerformanceHotspotDetector:
    """Test hotspot detection."""
    
    def test_hotspot_detection(self):
        """Test detecting performance hotspots."""
        detector = PerformanceHotspotDetector()
        
        # Record calls
        detector.record_call("db", "query", 100)
        detector.record_call("db", "query", 150)
        detector.record_call("api", "process", 50)
        detector.record_call("api", "process", 60)
        detector.record_call("cache", "get", 5)
        
        # Get hotspots
        hotspots = detector.get_hotspots(top_n=2)
        
        assert len(hotspots) <= 2
        assert hotspots[0]["component"] == "db"  # DB should be top hotspot
        assert hotspots[0]["total_time_ms"] == 250
        assert hotspots[0]["call_count"] == 2
        assert hotspots[0]["avg_time_ms"] == 125
    
    def test_reset(self):
        """Test resetting hotspot data."""
        detector = PerformanceHotspotDetector()
        
        detector.record_call("api", "call", 100)
        assert len(detector.get_hotspots()) == 1
        
        detector.reset()
        assert len(detector.get_hotspots()) == 0


class TestStatisticsAggregator:
    """Test statistics aggregation."""
    
    def test_multi_window_aggregation(self):
        """Test aggregating across multiple time windows."""
        aggregator = StatisticsAggregator()
        
        # Add samples
        for i in range(10):
            aggregator.add_sample(float(i * 10))
        
        # Get aggregated stats
        stats = aggregator.get_aggregated_stats()
        
        # All windows should have data
        assert "1min" in stats
        assert "5min" in stats
        assert "15min" in stats
        assert "1hour" in stats
        
        # Check 1 minute window
        assert stats["1min"]["count"] == 10
        assert stats["1min"]["mean"] == 45.0
        assert stats["1min"]["min"] == 0.0
        assert stats["1min"]["max"] == 90.0


class TestPerformanceRequirements:
    """Test performance requirements."""
    
    def test_metric_collection_overhead(self):
        """Test that metric collection overhead is < 0.5%."""
        monitor = PerformanceMonitor()
        
        # Measure baseline execution time
        def test_function():
            total = 0
            for i in range(1000):
                total += i
            return total
        
        # Baseline timing
        start = time.perf_counter()
        for _ in range(100):
            test_function()
        baseline_time = time.perf_counter() - start
        
        # Timing with monitoring
        @monitor_performance(component="test", operation="overhead")
        def monitored_function():
            total = 0
            for i in range(1000):
                total += i
            return total
        
        set_performance_monitor(monitor)
        
        start = time.perf_counter()
        for _ in range(100):
            monitored_function()
        monitored_time = time.perf_counter() - start
        
        # Calculate overhead
        overhead = ((monitored_time - baseline_time) / baseline_time) * 100
        
        # Should be less than 0.5%
        assert overhead < 0.5, f"Overhead {overhead:.2f}% exceeds 0.5% limit"
    
    def test_memory_usage(self):
        """Test that memory usage is bounded."""
        monitor = PerformanceMonitor()
        
        # Record many metrics
        for i in range(20000):
            monitor.record_metric(
                MetricType.LATENCY,
                f"comp_{i % 10}",
                f"op_{i % 100}",
                float(i)
            )
        
        # Metrics deque should be bounded
        assert len(monitor.metrics) <= 10000  # Max size
        
        # Alerts deque should be bounded
        assert len(monitor.alerts) <= 100  # Max size
    
    def test_statistics_accuracy(self):
        """Test P95/P99 accuracy."""
        monitor = PerformanceMonitor()
        
        # Generate known distribution (simple range for predictability)
        values = list(range(1, 1001))  # 1 to 1000
        
        # Record values
        for v in values:
            monitor.record_metric(MetricType.LATENCY, "test", "op", float(v))
        
        # Get statistics
        statistics = monitor.get_statistics("test", "op")
        
        # Calculate actual percentiles
        sorted_values = sorted(values)
        n = len(values)
        actual_p95 = sorted_values[int(n * 0.95)]
        actual_p99 = sorted_values[int(n * 0.99)]
        
        # Check accuracy (within 5% for T-Digest approximation)
        p95_error = abs(statistics.p95 - actual_p95) / actual_p95
        p99_error = abs(statistics.p99 - actual_p99) / actual_p99
        
        assert p95_error < 0.05, f"P95 error {p95_error:.2%} exceeds 5%"
        assert p99_error < 0.05, f"P99 error {p99_error:.2%} exceeds 5%"


@pytest.mark.integration
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_monitoring(self):
        """Test complete monitoring workflow."""
        monitor = PerformanceMonitor()
        set_performance_monitor(monitor)
        
        # Simulate API calls
        @monitor_performance(component="api", operation="get_document")
        def get_document(doc_id):
            time.sleep(0.01)  # Simulate work
            return f"document_{doc_id}"
        
        @monitor_performance(component="api", operation="save_document")
        def save_document(doc_id, content):
            time.sleep(0.02)  # Simulate work
            if doc_id == "error":
                raise ValueError("Save failed")
            return True
        
        # Make successful calls
        for i in range(10):
            get_document(f"doc_{i}")
            save_document(f"doc_{i}", "content")
        
        # Make error call
        try:
            save_document("error", "content")
        except ValueError:
            pass
        
        # Check statistics
        get_stats = monitor.get_statistics("api", "get_document")
        assert get_stats.count == 10
        assert get_stats.success_rate == 1.0
        
        save_stats = monitor.get_statistics("api", "save_document")
        assert save_stats.count == 11
        assert save_stats.success_rate < 1.0  # Had one error
        assert save_stats.error_count == 1
        
        # Check hotspots
        detector = PerformanceHotspotDetector()
        for metric in monitor.metrics:
            if metric.metric_type == MetricType.LATENCY:
                detector.record_call(
                    metric.component,
                    metric.operation,
                    metric.value
                )
        
        hotspots = detector.get_hotspots()
        assert len(hotspots) > 0
        # save_document should be the top hotspot (slower operation)
        assert hotspots[0]["operation"] == "save_document"
    
    def test_alert_workflow(self):
        """Test complete alert workflow."""
        monitor = PerformanceMonitor()
        alerts_received = []
        
        # Setup alert handling
        monitor.add_alert_callback(lambda a: alerts_received.append(a))
        
        # Add thresholds
        monitor.add_threshold(PerformanceThreshold(
            metric_type=MetricType.LATENCY,
            component="api",
            warning_threshold=50,
            critical_threshold=100
        ))
        
        monitor.add_threshold(PerformanceThreshold(
            metric_type=MetricType.SUCCESS_RATE,
            component="api",
            warning_threshold=0.95,
            critical_threshold=0.90,
            comparison="<"
        ))
        
        # Simulate normal operation
        for i in range(10):
            op_id = monitor.start_operation("api", "call")
            time.sleep(0.01)
            monitor.end_operation(op_id, success=True)
        
        assert len(alerts_received) == 0  # No alerts yet
        
        # Simulate slow operation
        op_id = monitor.start_operation("api", "call")
        time.sleep(0.06)  # Exceeds warning threshold
        monitor.end_operation(op_id, success=True)
        
        assert len(alerts_received) == 1
        assert alerts_received[0].level == "WARNING"
        
        # Simulate very slow operation
        op_id = monitor.start_operation("api", "call")
        time.sleep(0.11)  # Exceeds critical threshold
        monitor.end_operation(op_id, success=True)
        
        assert len(alerts_received) == 2
        assert alerts_received[1].level == "CRITICAL"
        
        # Acknowledge alerts
        for alert in alerts_received:
            monitor.acknowledge_alert(alert.id)
        
        assert len(monitor.get_active_alerts()) == 0
