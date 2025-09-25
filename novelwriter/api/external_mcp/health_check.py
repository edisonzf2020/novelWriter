"""
novelWriter – External MCP Health Check System
===============================================

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
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from novelwriter.api.external_mcp.exceptions import ExternalMCPHealthCheckError

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    connection_id: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if result indicates healthy status."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def is_available(self) -> bool:
        """Check if service is available (healthy or degraded)."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


@dataclass
class HealthMetrics:
    """Health metrics for a connection."""
    
    connection_id: str
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    consecutive_failures: int = 0
    average_response_time_ms: float = 0.0
    last_check_time: Optional[datetime] = None
    last_healthy_time: Optional[datetime] = None
    uptime_percentage: float = 100.0
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, result: HealthCheckResult) -> None:
        """Update metrics with new health check result.
        
        Args:
            result: Health check result
        """
        self.total_checks += 1
        self.last_check_time = result.timestamp
        
        if result.is_healthy:
            self.successful_checks += 1
            self.consecutive_failures = 0
            self.last_healthy_time = result.timestamp
        else:
            self.failed_checks += 1
            self.consecutive_failures += 1
        
        if result.response_time_ms is not None:
            self.recent_response_times.append(result.response_time_ms)
            self.average_response_time_ms = sum(self.recent_response_times) / len(self.recent_response_times)
        
        if self.total_checks > 0:
            self.uptime_percentage = (self.successful_checks / self.total_checks) * 100


class HealthChecker:
    """Health checker for external MCP connections."""
    
    def __init__(
        self,
        check_interval_seconds: int = 30,
        timeout_seconds: int = 5,
        failure_threshold: int = 3,
        recovery_threshold: int = 2
    ):
        """Initialize health checker.
        
        Args:
            check_interval_seconds: Interval between health checks
            timeout_seconds: Timeout for health check requests
            failure_threshold: Consecutive failures before marking unhealthy
            recovery_threshold: Consecutive successes before marking healthy
        """
        self._check_interval = check_interval_seconds
        self._timeout = timeout_seconds
        self._failure_threshold = failure_threshold
        self._recovery_threshold = recovery_threshold
        
        self._connections: Dict[str, Any] = {}  # connection_id -> connection object
        self._metrics: Dict[str, HealthMetrics] = {}
        self._status_cache: Dict[str, HealthStatus] = {}
        self._check_tasks: Dict[str, asyncio.Task] = {}
        self._callbacks: List[Callable[[HealthCheckResult], None]] = []
        self._stop_checking = False
        
        logger.debug(
            f"HealthChecker initialized: interval={check_interval_seconds}s, "
            f"timeout={timeout_seconds}s, failure_threshold={failure_threshold}"
        )
    
    def register_connection(self, connection_id: str, connection: Any) -> None:
        """Register connection for health checking.
        
        Args:
            connection_id: Connection identifier
            connection: Connection object with health_check method
        """
        self._connections[connection_id] = connection
        self._metrics[connection_id] = HealthMetrics(connection_id)
        self._status_cache[connection_id] = HealthStatus.UNKNOWN
        
        logger.info(f"Registered connection for health checking: {connection_id}")
    
    def unregister_connection(self, connection_id: str) -> None:
        """Unregister connection from health checking.
        
        Args:
            connection_id: Connection identifier
        """
        if connection_id in self._connections:
            del self._connections[connection_id]
            
            if connection_id in self._metrics:
                del self._metrics[connection_id]
            
            if connection_id in self._status_cache:
                del self._status_cache[connection_id]
            
            # Cancel check task if running
            if connection_id in self._check_tasks:
                self._check_tasks[connection_id].cancel()
                del self._check_tasks[connection_id]
            
            logger.info(f"Unregistered connection from health checking: {connection_id}")
    
    async def check_health(self, connection_id: str) -> HealthCheckResult:
        """Perform single health check.
        
        Args:
            connection_id: Connection to check
            
        Returns:
            Health check result
            
        Raises:
            ExternalMCPHealthCheckError: If connection not registered
        """
        if connection_id not in self._connections:
            raise ExternalMCPHealthCheckError(
                f"Connection not registered: {connection_id}",
                connection_id=connection_id
            )
        
        connection = self._connections[connection_id]
        start_time = time.time()
        
        try:
            # Call connection's health check method with timeout
            result = await asyncio.wait_for(
                connection.health_check(),
                timeout=self._timeout
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine status based on response
            if result.get("status") == "ok":
                status = HealthStatus.HEALTHY
                error_message = None
            else:
                status = HealthStatus.DEGRADED
                error_message = result.get("message", "Health check returned non-ok status")
            
        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            status = HealthStatus.UNHEALTHY
            error_message = f"Health check timed out after {self._timeout}s"
            logger.warning(f"Health check timeout for {connection_id}")
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            status = HealthStatus.OFFLINE
            error_message = str(e)
            logger.error(f"Health check failed for {connection_id}: {e}")
        
        # Create result
        result = HealthCheckResult(
            connection_id=connection_id,
            status=status,
            timestamp=datetime.now(),
            response_time_ms=response_time_ms,
            error_message=error_message
        )
        
        # Update metrics
        self._update_metrics(result)
        
        # Update status cache with thresholds
        self._update_status_cache(connection_id, result)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in health check callback: {e}")
        
        return result
    
    def _update_metrics(self, result: HealthCheckResult) -> None:
        """Update metrics for connection.
        
        Args:
            result: Health check result
        """
        if result.connection_id in self._metrics:
            self._metrics[result.connection_id].update(result)
    
    def _update_status_cache(self, connection_id: str, result: HealthCheckResult) -> None:
        """Update status cache with threshold logic.
        
        Args:
            connection_id: Connection identifier
            result: Health check result
        """
        metrics = self._metrics.get(connection_id)
        if not metrics:
            return
        
        current_status = self._status_cache.get(connection_id, HealthStatus.UNKNOWN)
        
        # Apply threshold logic
        if result.is_healthy:
            # Need consecutive successes to recover
            if current_status != HealthStatus.HEALTHY:
                if metrics.consecutive_failures == 0:
                    # Check if we have enough consecutive successes
                    recent_results = list(self._get_recent_results(connection_id, self._recovery_threshold))
                    if len(recent_results) >= self._recovery_threshold:
                        if all(r.is_healthy for r in recent_results):
                            self._status_cache[connection_id] = HealthStatus.HEALTHY
                            logger.info(f"Connection {connection_id} recovered to HEALTHY")
            else:
                self._status_cache[connection_id] = HealthStatus.HEALTHY
        else:
            # Need consecutive failures to mark unhealthy
            if metrics.consecutive_failures >= self._failure_threshold:
                if current_status != HealthStatus.OFFLINE:
                    self._status_cache[connection_id] = HealthStatus.OFFLINE
                    logger.warning(f"Connection {connection_id} marked as OFFLINE")
            elif metrics.consecutive_failures > 0:
                if current_status == HealthStatus.HEALTHY:
                    self._status_cache[connection_id] = HealthStatus.DEGRADED
                    logger.warning(f"Connection {connection_id} degraded")
    
    def _get_recent_results(self, connection_id: str, count: int) -> List[HealthCheckResult]:
        """Get recent health check results (stub for actual implementation).
        
        Args:
            connection_id: Connection identifier
            count: Number of recent results
            
        Returns:
            List of recent results
        """
        # This would typically query a history store
        # For now, return empty list
        return []
    
    def get_status(self, connection_id: str) -> HealthStatus:
        """Get current health status.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Current health status
        """
        return self._status_cache.get(connection_id, HealthStatus.UNKNOWN)
    
    def get_metrics(self, connection_id: str) -> Optional[HealthMetrics]:
        """Get health metrics for connection.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Health metrics or None if not found
        """
        return self._metrics.get(connection_id)
    
    def get_all_statuses(self) -> Dict[str, HealthStatus]:
        """Get all connection statuses.
        
        Returns:
            Dictionary of connection_id -> status
        """
        return self._status_cache.copy()
    
    def add_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """Add callback for health check results.
        
        Args:
            callback: Callback function
        """
        self._callbacks.append(callback)
        logger.debug("Added health check callback")
    
    def remove_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """Remove callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug("Removed health check callback")
    
    async def start_monitoring(self) -> None:
        """Start health monitoring for all connections."""
        self._stop_checking = False
        
        for connection_id in self._connections:
            if connection_id not in self._check_tasks:
                task = asyncio.create_task(
                    self._monitor_connection(connection_id)
                )
                self._check_tasks[connection_id] = task
        
        logger.info(f"Started health monitoring for {len(self._connections)} connections")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._stop_checking = True
        
        # Cancel all check tasks
        for task in self._check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._check_tasks:
            await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)
        
        self._check_tasks.clear()
        logger.info("Stopped health monitoring")
    
    async def _monitor_connection(self, connection_id: str) -> None:
        """Monitor single connection health.
        
        Args:
            connection_id: Connection to monitor
        """
        logger.debug(f"Started monitoring connection: {connection_id}")
        
        while not self._stop_checking:
            try:
                # Perform health check
                await self.check_health(connection_id)
                
                # Wait for next check
                await asyncio.sleep(self._check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring connection {connection_id}: {e}")
                await asyncio.sleep(self._check_interval)
        
        logger.debug(f"Stopped monitoring connection: {connection_id}")


class CircuitBreaker:
    """Circuit breaker for external MCP connections."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout_seconds: Time before trying half-open
            half_open_max_calls: Max calls in half-open state
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._half_open_max_calls = half_open_max_calls
        
        self._state = "closed"  # closed, open, half_open
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._half_open_successes = 0
        
        logger.debug(
            f"CircuitBreaker initialized: threshold={failure_threshold}, "
            f"recovery={recovery_timeout_seconds}s"
        )
    
    def call_succeeded(self) -> None:
        """Record successful call."""
        if self._state == "half_open":
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_max_calls:
                self._state = "closed"
                self._failure_count = 0
                logger.info("Circuit breaker closed after recovery")
        elif self._state == "closed":
            self._failure_count = 0
    
    def call_failed(self) -> None:
        """Record failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == "closed":
            if self._failure_count >= self._failure_threshold:
                self._state = "open"
                logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
        elif self._state == "half_open":
            self._state = "open"
            logger.warning("Circuit breaker reopened from half-open state")
    
    def is_open(self) -> bool:
        """Check if circuit is open.
        
        Returns:
            True if circuit is open
        """
        if self._state == "open":
            # Check if we should try half-open
            if self._last_failure_time:
                if time.time() - self._last_failure_time > self._recovery_timeout:
                    self._state = "half_open"
                    self._half_open_calls = 0
                    self._half_open_successes = 0
                    logger.info("Circuit breaker entering half-open state")
                    return False
            return True
        
        if self._state == "half_open":
            self._half_open_calls += 1
            if self._half_open_calls > self._half_open_max_calls:
                return True
        
        return False
    
    def get_state(self) -> str:
        """Get current circuit breaker state.
        
        Returns:
            Current state (closed, open, half_open)
        """
        return self._state
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = "closed"
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._half_open_successes = 0
        logger.debug("Circuit breaker reset")
