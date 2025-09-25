"""
novelWriter – Performance Monitor
==================================

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

import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from novelwriter import CONFIG

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class MetricType(Enum):
    """Types of performance metrics."""
    
    LATENCY = "latency"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    CONCURRENCY = "concurrency"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"


class PerformanceMetric(BaseModel):
    """Single performance metric data point."""
    
    metric_type: MetricType
    component: str
    operation: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceStatistics(BaseModel):
    """Aggregated performance statistics."""
    
    component: str
    operation: str
    window_minutes: int
    count: int = 0
    mean: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    std_dev: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)


class PerformanceThreshold(BaseModel):
    """Performance threshold configuration."""
    
    metric_type: MetricType
    component: str = "*"
    operation: str = "*"
    warning_threshold: float
    critical_threshold: float
    comparison: str = ">"  # >, <, >=, <=
    enabled: bool = True


class PerformanceAlert(BaseModel):
    """Performance alert."""
    
    id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    metric_type: MetricType
    component: str
    operation: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    acknowledged: bool = False


class TDigest:
    """Simple percentile estimation using reservoir sampling."""
    
    def __init__(self, compression: float = 100.0):
        """Initialize T-Digest.
        
        Args:
            compression: Maximum number of values to keep (multiplied by 10 for accuracy)
        """
        self.max_size = int(compression * 10)  # Keep more values for accuracy
        self.values: List[float] = []
        self._lock = threading.Lock()
        self._sorted = False
        self._count = 0  # Total values added
    
    def add(self, value: float, weight: int = 1) -> None:
        """Add a value to the digest using reservoir sampling."""
        import random
        with self._lock:
            # Add the value 'weight' times
            for _ in range(weight):
                self._count += 1
                if len(self.values) < self.max_size:
                    # Reservoir not full, just add
                    self.values.append(value)
                else:
                    # Reservoir sampling: randomly replace with decreasing probability
                    j = random.randint(0, self._count - 1)
                    if j < self.max_size:
                        self.values[j] = value
                self._sorted = False
    
    def quantile(self, q: float) -> float:
        """Get quantile value.
        
        Args:
            q: Quantile (0.0 to 1.0)
            
        Returns:
            Estimated value at quantile
        """
        if not self.values or q < 0 or q > 1:
            return 0.0
        
        with self._lock:
            if not self._sorted:
                self.values.sort()
                self._sorted = True
            
            if len(self.values) == 1:
                return self.values[0]
            
            # Calculate position
            pos = q * (len(self.values) - 1)
            lower = int(pos)
            upper = min(lower + 1, len(self.values) - 1)
            fraction = pos - lower
            
            # Interpolate
            return self.values[lower] + fraction * (self.values[upper] - self.values[lower])


class SlidingWindowStats:
    """Sliding window statistics calculator."""
    
    def __init__(self, window_size: int = 60):
        """Initialize sliding window.
        
        Args:
            window_size: Window size in seconds
        """
        self.window_size = window_size
        self.data: deque = deque()
        self.digest = TDigest()
        self._lock = threading.Lock()
    
    def add(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the window."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            # Remove old data
            cutoff = datetime.now() - timedelta(seconds=self.window_size)
            while self.data and self.data[0][1] < cutoff:
                self.data.popleft()
            
            # Add new data
            self.data.append((value, timestamp))
            self.digest.add(value)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        with self._lock:
            if not self.data:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "std_dev": 0.0
                }
            
            values = [v for v, _ in self.data]
            count = len(values)
            mean = sum(values) / count
            
            # Calculate standard deviation
            variance = sum((v - mean) ** 2 for v in values) / count
            std_dev = math.sqrt(variance)
            
            return {
                "count": count,
                "mean": mean,
                "min": min(values),
                "max": max(values),
                "p50": self.digest.quantile(0.5),
                "p95": self.digest.quantile(0.95),
                "p99": self.digest.quantile(0.99),
                "std_dev": std_dev
            }


class PerformanceMonitor:
    """Performance monitoring and metrics collection system."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize performance monitor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled = True
        
        # Metrics storage
        self.metrics: deque = deque(maxlen=10000)
        self.windows: Dict[str, Dict[str, SlidingWindowStats]] = defaultdict(
            lambda: defaultdict(lambda: SlidingWindowStats())
        )
        
        # Thresholds and alerts
        self.thresholds: List[PerformanceThreshold] = []
        self.alerts: deque = deque(maxlen=100)
        self.alert_callbacks: List[Callable] = []
        
        # Concurrency tracking
        self.active_operations: Dict[str, int] = defaultdict(int)
        
        # Performance baselines
        self.baselines: Dict[str, PerformanceStatistics] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background thread for cleanup (only start if explicitly enabled)
        self._stop_event = threading.Event()
        self._cleanup_thread = None
        self._cleanup_enabled = False
        
        # Load default thresholds
        self._load_default_thresholds()
        
        logger.debug("PerformanceMonitor initialized")
    
    def _load_default_thresholds(self) -> None:
        """Load default performance thresholds."""
        # API latency thresholds
        self.add_threshold(PerformanceThreshold(
            metric_type=MetricType.LATENCY,
            component="api",
            warning_threshold=1000,  # 1 second
            critical_threshold=5000,  # 5 seconds
        ))
        
        # Success rate thresholds
        self.add_threshold(PerformanceThreshold(
            metric_type=MetricType.SUCCESS_RATE,
            component="*",
            warning_threshold=0.95,
            critical_threshold=0.90,
            comparison="<"
        ))
        
        # Error rate thresholds
        self.add_threshold(PerformanceThreshold(
            metric_type=MetricType.ERROR_RATE,
            component="*",
            warning_threshold=0.05,
            critical_threshold=0.10,
            comparison=">"
        ))
    
    def record_metric(
        self,
        metric_type: MetricType,
        component: str,
        operation: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric.
        
        Args:
            metric_type: Type of metric
            component: Component name
            operation: Operation name
            value: Metric value
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            component=component,
            operation=operation,
            value=value,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Store metric
            self.metrics.append(metric)
            
            # Update sliding windows
            key = f"{component}:{operation}"
            self.windows[metric_type.value][key].add(value)
            
            # Check thresholds
            self._check_thresholds(metric)
    
    def start_operation(self, component: str, operation: str) -> str:
        """Start tracking an operation.
        
        Args:
            component: Component name
            operation: Operation name
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{component}:{operation}:{time.time_ns()}"
        
        key = f"{component}:{operation}"
        with self._lock:
            self.active_operations[key] += 1
            concurrency = self.active_operations[key]
        
        # Record concurrency metric (outside the lock to avoid deadlock)
        self.record_metric(
            MetricType.CONCURRENCY,
            component,
            operation,
            concurrency
        )
        
        return operation_id
    
    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """End tracking an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            metadata: Additional metadata
        """
        parts = operation_id.rsplit(":", 1)
        if len(parts) != 2:
            return
        
        key, start_time_ns = parts[0], int(parts[1])
        component, operation = key.split(":", 1)
        
        # Calculate latency
        latency_ms = (time.time_ns() - start_time_ns) / 1_000_000
        
        with self._lock:
            # Update concurrency
            self.active_operations[key] = max(0, self.active_operations[key] - 1)
        
        # Record metrics (outside the lock to avoid deadlock)
        self.record_metric(
            MetricType.LATENCY,
            component,
            operation,
            latency_ms,
            metadata
        )
        
        # Record success/error
        if success:
            self.record_metric(
                MetricType.SUCCESS_RATE,
                component,
                operation,
                1.0,
                metadata
            )
        else:
            self.record_metric(
                MetricType.ERROR_RATE,
                component,
                operation,
                1.0,
                metadata
            )
    
    def get_statistics(
        self,
        component: str,
        operation: str,
        window_minutes: int = 5
    ) -> PerformanceStatistics:
        """Get performance statistics.
        
        Args:
            component: Component name
            operation: Operation name
            window_minutes: Time window in minutes
            
        Returns:
            Performance statistics
        """
        key = f"{component}:{operation}"
        stats = PerformanceStatistics(
            component=component,
            operation=operation,
            window_minutes=window_minutes
        )
        
        # Get latency stats
        if key in self.windows[MetricType.LATENCY.value]:
            latency_stats = self.windows[MetricType.LATENCY.value][key].get_stats()
            stats.count = latency_stats["count"]
            stats.mean = latency_stats["mean"]
            stats.min = latency_stats["min"]
            stats.max = latency_stats["max"]
            stats.p50 = latency_stats["p50"]
            stats.p95 = latency_stats["p95"]
            stats.p99 = latency_stats["p99"]
            stats.std_dev = latency_stats["std_dev"]
        
        # Calculate success rate
        if key in self.windows[MetricType.SUCCESS_RATE.value]:
            success_stats = self.windows[MetricType.SUCCESS_RATE.value][key].get_stats()
            if key in self.windows[MetricType.ERROR_RATE.value]:
                error_stats = self.windows[MetricType.ERROR_RATE.value][key].get_stats()
                total = success_stats["count"] + error_stats["count"]
                if total > 0:
                    stats.success_rate = success_stats["count"] / total
                    stats.error_count = error_stats["count"]
        
        return stats
    
    def get_all_statistics(self, window_minutes: int = 5) -> List[PerformanceStatistics]:
        """Get statistics for all tracked operations.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            List of performance statistics
        """
        stats_list = []
        
        # Get all unique component:operation pairs
        all_keys = set()
        for metric_type_windows in self.windows.values():
            all_keys.update(metric_type_windows.keys())
        
        for key in all_keys:
            component, operation = key.split(":", 1)
            stats = self.get_statistics(component, operation, window_minutes)
            if stats.count > 0:
                stats_list.append(stats)
        
        return stats_list
    
    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add a performance threshold.
        
        Args:
            threshold: Threshold configuration
        """
        with self._lock:
            self.thresholds.append(threshold)
    
    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric violates any thresholds.
        
        Args:
            metric: Performance metric to check
        """
        for threshold in self.thresholds:
            if not threshold.enabled:
                continue
            
            # Check if threshold applies
            if threshold.metric_type != metric.metric_type:
                continue
            
            if threshold.component != "*" and threshold.component != metric.component:
                continue
            
            if threshold.operation != "*" and threshold.operation != metric.operation:
                continue
            
            # Check threshold violation
            violated = False
            level = "INFO"
            threshold_value = 0.0
            
            if threshold.comparison == ">":
                if metric.value > threshold.critical_threshold:
                    violated = True
                    level = "CRITICAL"
                    threshold_value = threshold.critical_threshold
                elif metric.value > threshold.warning_threshold:
                    violated = True
                    level = "WARNING"
                    threshold_value = threshold.warning_threshold
            elif threshold.comparison == "<":
                if metric.value < threshold.critical_threshold:
                    violated = True
                    level = "CRITICAL"
                    threshold_value = threshold.critical_threshold
                elif metric.value < threshold.warning_threshold:
                    violated = True
                    level = "WARNING"
                    threshold_value = threshold.warning_threshold
            
            if violated:
                self._create_alert(metric, level, threshold_value)
    
    def _create_alert(
        self,
        metric: PerformanceMetric,
        level: str,
        threshold_value: float
    ) -> None:
        """Create a performance alert.
        
        Args:
            metric: Metric that triggered alert
            level: Alert level
            threshold_value: Threshold that was violated
        """
        alert = PerformanceAlert(
            id=f"{metric.component}:{metric.operation}:{time.time_ns()}",
            level=level,
            metric_type=metric.metric_type,
            component=metric.component,
            operation=metric.operation,
            message=f"{metric.metric_type.value} for {metric.component}:{metric.operation} "
                   f"is {metric.value:.2f} (threshold: {threshold_value:.2f})",
            current_value=metric.value,
            threshold_value=threshold_value
        )
        
        self.alerts.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add an alert callback.
        
        Args:
            callback: Function to call when alert is created
        """
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get active (unacknowledged) alerts.
        
        Returns:
            List of active alerts
        """
        with self._lock:
            return [a for a in self.alerts if not a.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if alert was found and acknowledged
        """
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
        return False
    
    def save_baseline(self, component: str, operation: str) -> None:
        """Save current performance as baseline.
        
        Args:
            component: Component name
            operation: Operation name
        """
        stats = self.get_statistics(component, operation)
        if stats.count > 0:
            key = f"{component}:{operation}"
            self.baselines[key] = stats
            logger.info(f"Saved baseline for {key}")
    
    def compare_to_baseline(
        self,
        component: str,
        operation: str
    ) -> Optional[Dict[str, float]]:
        """Compare current performance to baseline.
        
        Args:
            component: Component name
            operation: Operation name
            
        Returns:
            Comparison results or None if no baseline
        """
        key = f"{component}:{operation}"
        if key not in self.baselines:
            return None
        
        baseline = self.baselines[key]
        current = self.get_statistics(component, operation)
        
        if current.count == 0:
            return None
        
        return {
            "latency_change": (current.mean - baseline.mean) / baseline.mean * 100,
            "p95_change": (current.p95 - baseline.p95) / baseline.p95 * 100,
            "p99_change": (current.p99 - baseline.p99) / baseline.p99 * 100,
            "success_rate_change": current.success_rate - baseline.success_rate,
            "error_count_change": current.error_count - baseline.error_count
        }
    
    def export_metrics(self, filepath: Path) -> None:
        """Export metrics to file.
        
        Args:
            filepath: Path to export file
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": [
                stats.model_dump() for stats in self.get_all_statistics()
            ],
            "baselines": {
                key: baseline.model_dump()
                for key, baseline in self.baselines.items()
            },
            "alerts": [
                alert.model_dump() for alert in self.alerts
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported metrics to {filepath}")
    
    def import_baselines(self, filepath: Path) -> None:
        """Import baselines from file.
        
        Args:
            filepath: Path to import file
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        if "baselines" in data:
            for key, baseline_data in data["baselines"].items():
                self.baselines[key] = PerformanceStatistics(**baseline_data)
        
        logger.info(f"Imported {len(self.baselines)} baselines from {filepath}")
    
    def _cleanup_loop(self) -> None:
        """Background thread for cleaning up old data."""
        while not self._stop_event.is_set():
            # Wait for 60 seconds or until stop event is set
            if self._stop_event.wait(60):
                break
            try:
                cutoff = datetime.now() - timedelta(hours=1)
                
                with self._lock:
                    # Clean old metrics
                    while self.metrics and self.metrics[0].timestamp < cutoff:
                        self.metrics.popleft()
                    
                    # Clean old alerts
                    acknowledged_cutoff = datetime.now() - timedelta(hours=24)
                    temp_alerts = deque()
                    for alert in self.alerts:
                        if not alert.acknowledged or alert.timestamp > acknowledged_cutoff:
                            temp_alerts.append(alert)
                    self.alerts = temp_alerts
                    
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def enable_cleanup(self) -> None:
        """Enable background cleanup thread."""
        if not self._cleanup_enabled:
            self._cleanup_enabled = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
            logger.debug("Cleanup thread started")
    
    def shutdown(self) -> None:
        """Shutdown the performance monitor."""
        self._stop_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)


def monitor_performance(
    component: str = "unknown",
    operation: str = "unknown"
) -> Callable[[F], F]:
    """Decorator to monitor function performance.
    
    Args:
        component: Component name
        operation: Operation name
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get monitor instance
            monitor = get_performance_monitor()
            if not monitor or not monitor.enabled:
                return func(*args, **kwargs)
            
            # Start tracking
            op_id = monitor.start_operation(component, operation)
            
            try:
                result = func(*args, **kwargs)
                monitor.end_operation(op_id, success=True)
                return result
            except Exception as e:
                monitor.end_operation(op_id, success=False, metadata={"error": str(e)})
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get monitor instance
            monitor = get_performance_monitor()
            if not monitor or not monitor.enabled:
                return await func(*args, **kwargs)
            
            # Start tracking
            op_id = monitor.start_operation(component, operation)
            
            try:
                result = await func(*args, **kwargs)
                monitor.end_operation(op_id, success=True)
                return result
            except Exception as e:
                monitor.end_operation(op_id, success=False, metadata={"error": str(e)})
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global instance management
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get global performance monitor instance.
    
    Returns:
        Global PerformanceMonitor instance
    """
    return _performance_monitor


def set_performance_monitor(monitor: PerformanceMonitor) -> None:
    """Set global performance monitor instance.
    
    Args:
        monitor: PerformanceMonitor instance
    """
    global _performance_monitor
    _performance_monitor = monitor
