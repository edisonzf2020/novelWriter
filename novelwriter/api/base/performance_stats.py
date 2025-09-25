"""
novelWriter – Performance Statistics Calculator
================================================

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

import logging
import math
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Use standard library instead of numpy for basic statistics
import statistics
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    
    trend: str  # "improving", "stable", "degrading"
    slope: float
    r_squared: float
    confidence: float
    prediction_next: float
    anomalies: List[Tuple[datetime, float]] = Field(default_factory=list)


class PerformanceBaseline(BaseModel):
    """Performance baseline for comparison."""
    
    component: str
    operation: str
    timestamp: datetime
    mean_latency: float
    p95_latency: float
    p99_latency: float
    success_rate: float
    throughput: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnomalyDetector:
    """Detect anomalies in performance metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        """Initialize anomaly detector.
        
        Args:
            sensitivity: Number of standard deviations for anomaly threshold
        """
        self.sensitivity = sensitivity
        self.history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> bool:
        """Add a value and check if it's anomalous.
        
        Args:
            value: Metric value
            timestamp: Timestamp of value
            
        Returns:
            True if value is anomalous
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            is_anomaly = False
            
            if len(self.history) >= 10:
                values = [v for v, _ in self.history]
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0
                
                # Check if value is outside threshold
                if std > 0 and abs(value - mean) > self.sensitivity * std:
                    is_anomaly = True
            
            self.history.append((value, timestamp))
            return is_anomaly
    
    def get_anomalies(self, window_minutes: int = 60) -> List[Tuple[datetime, float]]:
        """Get recent anomalies.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            List of anomalous values with timestamps
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        anomalies = []
        
        with self._lock:
            if len(self.history) < 10:
                return anomalies
            
            values = [v for v, _ in self.history]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            
            for value, timestamp in self.history:
                if timestamp > cutoff and std > 0:
                    if abs(value - mean) > self.sensitivity * std:
                        anomalies.append((timestamp, value))
        
        return anomalies


class TrendAnalyzer:
    """Analyze performance trends over time."""
    
    def __init__(self, window_size: int = 100):
        """Initialize trend analyzer.
        
        Args:
            window_size: Number of points for trend analysis
        """
        self.window_size = window_size
        self.data_points: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def add_point(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a data point.
        
        Args:
            value: Metric value
            timestamp: Timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.data_points.append((timestamp, value))
    
    def analyze_trend(self) -> TrendAnalysis:
        """Analyze the trend in the data.
        
        Returns:
            Trend analysis results
        """
        with self._lock:
            if len(self.data_points) < 3:
                return TrendAnalysis(
                    trend="stable",
                    slope=0.0,
                    r_squared=0.0,
                    confidence=0.0,
                    prediction_next=0.0
                )
            
            # Convert to lists for analysis
            timestamps = []
            values = []
            start_time = self.data_points[0][0]
            
            for ts, val in self.data_points:
                # Convert to seconds since start
                time_diff = (ts - start_time).total_seconds()
                timestamps.append(time_diff)
                values.append(val)
            
            # Linear regression using standard library
            n = len(timestamps)
            x_mean = statistics.mean(timestamps)
            y_mean = statistics.mean(values)
            
            # Calculate slope and intercept
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(timestamps, values))
            denominator = sum((x - x_mean) ** 2 for x in timestamps)
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            intercept = y_mean - slope * x_mean
            
            # Calculate R-squared
            y_pred = [slope * x + intercept for x in timestamps]
            ss_res = sum((y - yp) ** 2 for y, yp in zip(values, y_pred))
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            
            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Determine trend
            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0:
                trend = "degrading"  # Higher values = worse performance
            else:
                trend = "improving"
            
            # Predict next value
            if len(timestamps) > 0:
                next_time = timestamps[-1] + (timestamps[-1] - timestamps[0]) / len(timestamps)
                prediction_next = slope * next_time + intercept
            else:
                prediction_next = y_mean
            
            # Detect anomalies using residuals
            anomalies = []
            if len(y_pred) > 0:
                residuals = [y - yp for y, yp in zip(values, y_pred)]
                std_residual = statistics.stdev(residuals) if len(residuals) > 1 else 0
                
                if std_residual > 0:
                    for i, residual in enumerate(residuals):
                        if abs(residual) > 2 * std_residual:
                            anomalies.append((self.data_points[i][0], values[i]))
            
            # Calculate confidence based on R-squared and data points
            confidence = r_squared * min(1.0, len(self.data_points) / 20)
            
            return TrendAnalysis(
                trend=trend,
                slope=slope,
                r_squared=r_squared,
                confidence=confidence,
                prediction_next=prediction_next,
                anomalies=anomalies
            )


class PerformanceComparator:
    """Compare performance across different time periods or versions."""
    
    def __init__(self):
        """Initialize performance comparator."""
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self._lock = threading.Lock()
    
    def save_baseline(
        self,
        component: str,
        operation: str,
        stats: Dict[str, float]
    ) -> None:
        """Save performance baseline.
        
        Args:
            component: Component name
            operation: Operation name
            stats: Performance statistics
        """
        key = f"{component}:{operation}"
        
        baseline = PerformanceBaseline(
            component=component,
            operation=operation,
            timestamp=datetime.now(),
            mean_latency=stats.get("mean", 0.0),
            p95_latency=stats.get("p95", 0.0),
            p99_latency=stats.get("p99", 0.0),
            success_rate=stats.get("success_rate", 1.0),
            throughput=stats.get("throughput", 0.0)
        )
        
        with self._lock:
            self.baselines[key] = baseline
    
    def compare_to_baseline(
        self,
        component: str,
        operation: str,
        current_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare current performance to baseline.
        
        Args:
            component: Component name
            operation: Operation name
            current_stats: Current performance statistics
            
        Returns:
            Comparison results
        """
        key = f"{component}:{operation}"
        
        with self._lock:
            if key not in self.baselines:
                return {"has_baseline": False}
            
            baseline = self.baselines[key]
        
        # Calculate percentage changes
        def calc_change(current: float, baseline_val: float) -> float:
            if baseline_val == 0:
                return 0.0
            return ((current - baseline_val) / baseline_val) * 100
        
        comparison = {
            "has_baseline": True,
            "baseline_timestamp": baseline.timestamp,
            "mean_latency_change": calc_change(
                current_stats.get("mean", 0.0),
                baseline.mean_latency
            ),
            "p95_latency_change": calc_change(
                current_stats.get("p95", 0.0),
                baseline.p95_latency
            ),
            "p99_latency_change": calc_change(
                current_stats.get("p99", 0.0),
                baseline.p99_latency
            ),
            "success_rate_change": (
                current_stats.get("success_rate", 1.0) - baseline.success_rate
            ) * 100,
            "throughput_change": calc_change(
                current_stats.get("throughput", 0.0),
                baseline.throughput
            )
        }
        
        # Determine if performance has degraded
        degraded = False
        if comparison["mean_latency_change"] > 20:  # 20% worse
            degraded = True
        if comparison["p95_latency_change"] > 30:  # 30% worse
            degraded = True
        if comparison["success_rate_change"] < -5:  # 5% worse
            degraded = True
        
        comparison["performance_degraded"] = degraded
        
        # Calculate overall score
        score = 100.0
        score -= max(0, comparison["mean_latency_change"]) * 0.3
        score -= max(0, comparison["p95_latency_change"]) * 0.2
        score -= max(0, -comparison["success_rate_change"]) * 2.0
        score = max(0, min(100, score))
        
        comparison["performance_score"] = score
        
        return comparison
    
    def detect_regression(
        self,
        component: str,
        operation: str,
        current_stats: Dict[str, float],
        threshold: float = 0.1
    ) -> bool:
        """Detect performance regression.
        
        Args:
            component: Component name
            operation: Operation name
            current_stats: Current performance statistics
            threshold: Regression threshold (0.1 = 10% degradation)
            
        Returns:
            True if regression detected
        """
        comparison = self.compare_to_baseline(component, operation, current_stats)
        
        if not comparison["has_baseline"]:
            return False
        
        # Check for regression
        if comparison["mean_latency_change"] > threshold * 100:
            return True
        if comparison["p95_latency_change"] > threshold * 150:  # More sensitive for p95
            return True
        if comparison["success_rate_change"] < -threshold * 100:
            return True
        
        return False


class PerformanceHotspotDetector:
    """Detect performance hotspots in the system."""
    
    def __init__(self):
        """Initialize hotspot detector."""
        self.call_counts: Dict[str, int] = {}
        self.total_time: Dict[str, float] = {}
        self.max_time: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def record_call(
        self,
        component: str,
        operation: str,
        duration_ms: float
    ) -> None:
        """Record a function call.
        
        Args:
            component: Component name
            operation: Operation name
            duration_ms: Call duration in milliseconds
        """
        key = f"{component}:{operation}"
        
        with self._lock:
            self.call_counts[key] = self.call_counts.get(key, 0) + 1
            self.total_time[key] = self.total_time.get(key, 0.0) + duration_ms
            self.max_time[key] = max(self.max_time.get(key, 0.0), duration_ms)
    
    def get_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top performance hotspots.
        
        Args:
            top_n: Number of top hotspots to return
            
        Returns:
            List of hotspot information
        """
        with self._lock:
            hotspots = []
            
            for key in self.total_time:
                component, operation = key.split(":", 1)
                
                hotspots.append({
                    "component": component,
                    "operation": operation,
                    "total_time_ms": self.total_time[key],
                    "call_count": self.call_counts.get(key, 0),
                    "avg_time_ms": (
                        self.total_time[key] / self.call_counts[key]
                        if self.call_counts.get(key, 0) > 0 else 0
                    ),
                    "max_time_ms": self.max_time.get(key, 0.0)
                })
            
            # Sort by total time (most time consuming first)
            hotspots.sort(key=lambda x: x["total_time_ms"], reverse=True)
            
            return hotspots[:top_n]
    
    def reset(self) -> None:
        """Reset hotspot data."""
        with self._lock:
            self.call_counts.clear()
            self.total_time.clear()
            self.max_time.clear()


class StatisticsAggregator:
    """Aggregate statistics across multiple time windows."""
    
    def __init__(self):
        """Initialize statistics aggregator."""
        self.windows = {
            "1min": deque(maxlen=60),
            "5min": deque(maxlen=300),
            "15min": deque(maxlen=900),
            "1hour": deque(maxlen=3600)
        }
        self._lock = threading.Lock()
    
    def add_sample(self, value: float) -> None:
        """Add a sample to all windows.
        
        Args:
            value: Sample value
        """
        timestamp = datetime.now()
        sample = (timestamp, value)
        
        with self._lock:
            for window in self.windows.values():
                window.append(sample)
    
    def get_aggregated_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated statistics for all windows.
        
        Returns:
            Statistics for each time window
        """
        results = {}
        
        with self._lock:
            for window_name, window_data in self.windows.items():
                # Parse window duration
                if "min" in window_name:
                    duration = int(window_name.replace("min", ""))
                    cutoff = datetime.now() - timedelta(minutes=duration)
                elif "hour" in window_name:
                    duration = int(window_name.replace("hour", ""))
                    cutoff = datetime.now() - timedelta(hours=duration)
                else:
                    continue
                
                # Filter samples within window
                samples = [
                    value for timestamp, value in window_data
                    if timestamp > cutoff
                ]
                
                if samples:
                    sorted_samples = sorted(samples)
                    n = len(samples)
                    
                    # Calculate percentiles manually
                    def percentile(data, p):
                        k = (n - 1) * p / 100
                        f = math.floor(k)
                        c = math.ceil(k)
                        if f == c:
                            return data[int(k)]
                        d0 = data[int(f)] * (c - k)
                        d1 = data[int(c)] * (k - f)
                        return d0 + d1
                    
                    results[window_name] = {
                        "count": len(samples),
                        "mean": statistics.mean(samples),
                        "min": min(samples),
                        "max": max(samples),
                        "std": statistics.stdev(samples) if len(samples) > 1 else 0,
                        "p50": percentile(sorted_samples, 50),
                        "p95": percentile(sorted_samples, 95),
                        "p99": percentile(sorted_samples, 99)
                    }
                else:
                    results[window_name] = {
                        "count": 0,
                        "mean": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "std": 0.0,
                        "p50": 0.0,
                        "p95": 0.0,
                        "p99": 0.0
                    }
        
        return results
