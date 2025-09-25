"""
novelWriter – Fault Handling and Recovery System
=================================================

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
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    get_circuit_breaker_manager,
)
from .degradation_service import (
    DegradationService,
    ServiceDegradationLevel,
    get_degradation_service,
)
from .error_classifier import (
    ErrorClassification,
    ErrorClassifier,
    ErrorContext,
    ErrorSeverity,
)
from .retry_manager import (
    RetryManager,
    RetryPolicy,
    get_retry_manager,
    with_retry,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert(BaseModel):
    """System alert."""
    id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    level: AlertLevel
    component: str
    title: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class ErrorTracker(BaseModel):
    """Error tracking information."""
    trace_id: str
    parent_trace_id: Optional[str] = None
    component: str
    operation: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    errors: List[ErrorContext] = Field(default_factory=list)
    status: str = "in_progress"  # in_progress, success, failed
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FaultRecoveryStrategy(BaseModel):
    """Fault recovery strategy."""
    name: str
    description: str
    applicable_errors: List[ErrorClassification]
    recovery_actions: List[str]
    auto_execute: bool = False
    success_rate: float = 0.0


class FaultHandlingMetrics(BaseModel):
    """Fault handling system metrics."""
    total_errors: int = 0
    recovered_errors: int = 0
    unrecovered_errors: int = 0
    active_alerts: int = 0
    circuit_breakers_open: int = 0
    current_degradation_level: ServiceDegradationLevel = ServiceDegradationLevel.FULL
    mean_time_to_recovery: float = 0.0  # seconds
    error_rate_per_minute: float = 0.0
    recovery_success_rate: float = 0.0


class FaultHandlingSystem:
    """Central fault handling and recovery system."""
    
    def __init__(self):
        """Initialize fault handling system."""
        # Core components
        self.error_classifier = ErrorClassifier()
        self.retry_manager = get_retry_manager()
        self.circuit_breaker_manager = get_circuit_breaker_manager()
        self.degradation_service = get_degradation_service()
        
        # Error tracking
        self._error_traces: Dict[str, ErrorTracker] = {}
        self._error_correlations: Dict[str, List[str]] = defaultdict(list)
        
        # Alerting
        self._alerts: deque = deque(maxlen=1000)
        self._alert_callbacks: List[Callable] = []
        
        # Recovery strategies
        self._recovery_strategies: Dict[str, FaultRecoveryStrategy] = {}
        self._register_default_strategies()
        
        # Metrics
        self.metrics = FaultHandlingMetrics()
        self._recovery_times: deque = deque(maxlen=100)
        
        # State management
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Start monitoring
        self._start_monitoring()
    
    def handle_error(self,
                    error: Exception,
                    component: str,
                    operation: str,
                    trace_id: Optional[str] = None,
                    parent_trace_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle an error with full fault tolerance.
        
        Args:
            error: The exception
            component: Component where error occurred
            operation: Operation that failed
            trace_id: Trace ID for correlation
            parent_trace_id: Parent trace ID for correlation
            metadata: Additional metadata
            
        Returns:
            Error context with handling information
        """
        with self._lock:
            # Classify error
            error_context = self.error_classifier.classify(
                error, component, operation, metadata=metadata
            )
            
            # Create or update trace
            if not trace_id:
                import uuid
                trace_id = f"trace_{uuid.uuid4().hex[:8]}"
            
            if trace_id not in self._error_traces:
                self._error_traces[trace_id] = ErrorTracker(
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    component=component,
                    operation=operation
                )
            
            tracker = self._error_traces[trace_id]
            tracker.errors.append(error_context)
            
            # Correlate errors
            if parent_trace_id:
                self._error_correlations[parent_trace_id].append(trace_id)
            
            # Update metrics
            self.metrics.total_errors += 1
            
            # Check severity and create alert if needed
            if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self._create_alert(
                    level=AlertLevel.ERROR if error_context.severity == ErrorSeverity.HIGH else AlertLevel.CRITICAL,
                    component=component,
                    title=f"{operation} failed",
                    message=str(error),
                    metadata={"trace_id": trace_id, "error_context": error_context.model_dump()}
                )
            
            # Check for cascade risk
            cascade_risk = self.circuit_breaker_manager.check_cascade_risk()
            if cascade_risk["risk_level"] in ["high", "critical"]:
                self._create_alert(
                    level=AlertLevel.CRITICAL,
                    component="system",
                    title="Cascade failure risk detected",
                    message=f"{cascade_risk['open_breakers']} circuit breakers open",
                    metadata=cascade_risk
                )
            
            # Attempt recovery if applicable
            if error_context.classification != ErrorClassification.PERMANENT:
                self._attempt_recovery(error_context, trace_id)
            
            return error_context
    
    def create_trace(self,
                    component: str,
                    operation: str,
                    parent_trace_id: Optional[str] = None) -> str:
        """Create a new error trace.
        
        Args:
            component: Component name
            operation: Operation name
            parent_trace_id: Parent trace ID
            
        Returns:
            Trace ID
        """
        import uuid
        trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            self._error_traces[trace_id] = ErrorTracker(
                trace_id=trace_id,
                parent_trace_id=parent_trace_id,
                component=component,
                operation=operation
            )
            
            if parent_trace_id:
                self._error_correlations[parent_trace_id].append(trace_id)
        
        return trace_id
    
    def complete_trace(self,
                      trace_id: str,
                      success: bool = True) -> None:
        """Complete an error trace.
        
        Args:
            trace_id: Trace ID
            success: Whether operation succeeded
        """
        with self._lock:
            if trace_id in self._error_traces:
                tracker = self._error_traces[trace_id]
                tracker.end_time = datetime.now()
                tracker.status = "success" if success else "failed"
                
                # Calculate recovery time if recovered
                if success and tracker.errors:
                    recovery_time = (tracker.end_time - tracker.start_time).total_seconds()
                    self._recovery_times.append(recovery_time)
                    self.metrics.recovered_errors += 1
                elif not success:
                    self.metrics.unrecovered_errors += 1
    
    def get_trace(self, trace_id: str) -> Optional[ErrorTracker]:
        """Get error trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Error tracker or None
        """
        return self._error_traces.get(trace_id)
    
    def get_correlated_traces(self, trace_id: str) -> List[ErrorTracker]:
        """Get correlated error traces.
        
        Args:
            trace_id: Parent trace ID
            
        Returns:
            List of correlated traces
        """
        with self._lock:
            child_ids = self._error_correlations.get(trace_id, [])
            return [
                self._error_traces[child_id]
                for child_id in child_ids
                if child_id in self._error_traces
            ]
    
    def create_alert(self,
                    level: AlertLevel,
                    component: str,
                    title: str,
                    message: str,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a system alert.
        
        Args:
            level: Alert level
            component: Component name
            title: Alert title
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Alert ID
        """
        return self._create_alert(level, component, title, message, metadata)
    
    def acknowledge_alert(self,
                         alert_id: str,
                         acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged
            
        Returns:
            True if acknowledged
        """
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id and not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_at = datetime.now()
                    alert.acknowledged_by = acknowledged_by
                    self.metrics.active_alerts -= 1
                    return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unacknowledged) alerts.
        
        Returns:
            List of active alerts
        """
        with self._lock:
            return [
                alert for alert in self._alerts
                if not alert.acknowledged
            ]
    
    def get_metrics(self) -> FaultHandlingMetrics:
        """Get fault handling metrics.
        
        Returns:
            Current metrics
        """
        with self._lock:
            # Update calculated metrics
            if self._recovery_times:
                self.metrics.mean_time_to_recovery = sum(self._recovery_times) / len(self._recovery_times)
            
            if self.metrics.total_errors > 0:
                self.metrics.recovery_success_rate = self.metrics.recovered_errors / self.metrics.total_errors
            
            # Get current states
            self.metrics.circuit_breakers_open = len(self.circuit_breaker_manager.get_open_breakers())
            self.metrics.current_degradation_level = self.degradation_service.get_state().level
            
            # Calculate error rate
            recent_errors = self.error_classifier.analyze_pattern(timedelta(minutes=1))
            self.metrics.error_rate_per_minute = recent_errors.get("error_rate", 0.0)
            
            return self.metrics.model_copy()
    
    def register_recovery_strategy(self, strategy: FaultRecoveryStrategy) -> None:
        """Register a recovery strategy.
        
        Args:
            strategy: Recovery strategy
        """
        self._recovery_strategies[strategy.name] = strategy
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback.
        
        Args:
            callback: Callback function(alert)
        """
        self._alert_callbacks.append(callback)
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform system health check.
        
        Returns:
            Health check results
        """
        with self._lock:
            # Analyze recent errors
            error_analysis = self.error_classifier.analyze_pattern()
            
            # Check circuit breakers
            cascade_risk = self.circuit_breaker_manager.check_cascade_risk()
            
            # Check degradation
            degradation_state = self.degradation_service.get_state()
            
            # Calculate health score
            health_score = 100.0
            
            if error_analysis["failure_prediction"] == "high":
                health_score -= 30
            elif error_analysis["failure_prediction"] == "medium":
                health_score -= 20
            elif error_analysis["failure_prediction"] == "low":
                health_score -= 10
            
            if cascade_risk["risk_level"] == "critical":
                health_score -= 40
            elif cascade_risk["risk_level"] == "high":
                health_score -= 25
            elif cascade_risk["risk_level"] == "medium":
                health_score -= 15
            
            if degradation_state.level == ServiceDegradationLevel.EMERGENCY:
                health_score -= 50
            elif degradation_state.level == ServiceDegradationLevel.OFFLINE:
                health_score -= 30
            elif degradation_state.level == ServiceDegradationLevel.LIMITED:
                health_score -= 15
            
            health_score = max(0, health_score)
            
            # Determine health status
            if health_score >= 80:
                health_status = "healthy"
            elif health_score >= 60:
                health_status = "degraded"
            elif health_score >= 40:
                health_status = "unhealthy"
            else:
                health_status = "critical"
            
            return {
                "status": health_status,
                "score": health_score,
                "error_analysis": error_analysis,
                "cascade_risk": cascade_risk,
                "degradation": {
                    "level": degradation_state.level.value,
                    "reason": degradation_state.reason,
                    "features_affected": len(degradation_state.features_disabled + degradation_state.features_limited)
                },
                "active_alerts": self.metrics.active_alerts,
                "metrics": self.get_metrics().model_dump()
            }
    
    def _create_alert(self,
                     level: AlertLevel,
                     component: str,
                     title: str,
                     message: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create an alert (internal).
        
        Returns:
            Alert ID
        """
        import uuid
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        alert = Alert(
            id=alert_id,
            level=level,
            component=component,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._alerts.append(alert)
            self.metrics.active_alerts += 1
            
            # Notify callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert created: [{level.value}] {title}")
        return alert_id
    
    def _attempt_recovery(self,
                         error_context: ErrorContext,
                         trace_id: str) -> bool:
        """Attempt automatic recovery.
        
        Args:
            error_context: Error context
            trace_id: Trace ID
            
        Returns:
            True if recovery attempted
        """
        # Find applicable strategies
        applicable_strategies = [
            strategy for strategy in self._recovery_strategies.values()
            if error_context.classification in strategy.applicable_errors
            and strategy.auto_execute
        ]
        
        if not applicable_strategies:
            return False
        
        # Sort by success rate
        applicable_strategies.sort(key=lambda s: s.success_rate, reverse=True)
        
        # Try best strategy
        strategy = applicable_strategies[0]
        logger.info(f"Attempting recovery with strategy: {strategy.name}")
        
        # Execute recovery actions
        for action in strategy.recovery_actions:
            try:
                # Execute action (simplified for now)
                if action == "reset_circuit_breaker":
                    breaker = self.circuit_breaker_manager.get(error_context.component)
                    if breaker:
                        breaker.reset()
                elif action == "clear_cache":
                    # Would clear relevant caches
                    pass
                elif action == "restart_component":
                    # Would restart component
                    pass
            except Exception as e:
                logger.error(f"Recovery action failed: {action}, error: {e}")
        
        return True
    
    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        strategies = [
            FaultRecoveryStrategy(
                name="network_recovery",
                description="Recovery for network errors",
                applicable_errors=[ErrorClassification.NETWORK],
                recovery_actions=["reset_circuit_breaker", "clear_connection_pool"],
                auto_execute=True,
                success_rate=0.7
            ),
            FaultRecoveryStrategy(
                name="timeout_recovery",
                description="Recovery for timeout errors",
                applicable_errors=[ErrorClassification.TIMEOUT],
                recovery_actions=["increase_timeout", "reduce_load"],
                auto_execute=False,
                success_rate=0.6
            ),
            FaultRecoveryStrategy(
                name="transient_recovery",
                description="Recovery for transient errors",
                applicable_errors=[ErrorClassification.TRANSIENT],
                recovery_actions=["clear_cache", "retry_operation"],
                auto_execute=True,
                success_rate=0.8
            ),
            FaultRecoveryStrategy(
                name="degradation_recovery",
                description="Recovery through degradation",
                applicable_errors=[ErrorClassification.DEGRADATION],
                recovery_actions=["enable_degraded_mode", "disable_non_critical_features"],
                auto_execute=True,
                success_rate=0.9
            )
        ]
        
        for strategy in strategies:
            self.register_recovery_strategy(strategy)
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Perform health check
                health = self.perform_health_check()
                
                # Check for degradation
                if health["status"] in ["unhealthy", "critical"]:
                    # Evaluate degradation
                    decision = self.degradation_service.evaluate_degradation(
                        error_rate=self.metrics.error_rate_per_minute / 60,  # Convert to rate
                        latency=0.0,  # Would get from performance monitor
                        cpu_usage=0.0,  # Would get from system
                        memory_usage=0.0  # Would get from system
                    )
                    
                    if decision.should_degrade:
                        self.degradation_service.apply_degradation(decision)
                
                # Attempt recovery if degraded
                elif health["status"] == "healthy" and self.degradation_service.get_state().level != ServiceDegradationLevel.FULL:
                    self.degradation_service.attempt_recovery()
                
                # Clean old traces
                self._clean_old_traces()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait before next check
            self._stop_monitoring.wait(30)  # Check every 30 seconds
    
    def _clean_old_traces(self) -> None:
        """Clean old error traces."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            old_traces = [
                trace_id for trace_id, tracker in self._error_traces.items()
                if tracker.start_time < cutoff_time and tracker.status != "in_progress"
            ]
            
            for trace_id in old_traces:
                del self._error_traces[trace_id]
                if trace_id in self._error_correlations:
                    del self._error_correlations[trace_id]
    
    def _start_monitoring(self) -> None:
        """Start background monitoring."""
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)


def with_fault_handling(component: str,
                       operation: str,
                       retry_policy: Optional[Union[RetryPolicy, str]] = None,
                       circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
    """Decorator for complete fault handling.
    
    Args:
        component: Component name
        operation: Operation name
        retry_policy: Retry policy or tool type
        circuit_breaker_config: Circuit breaker configuration
    """
    def decorator(func: F) -> F:
        # Apply circuit breaker
        if circuit_breaker_config:
            func = with_circuit_breaker(
                f"{component}:{operation}",
                circuit_breaker_config
            )(func)
        
        # Apply retry
        if retry_policy:
            func = with_retry(
                retry_policy,
                component,
                operation
            )(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            fault_system = get_fault_handling_system()
            trace_id = fault_system.create_trace(component, operation)
            
            try:
                result = func(*args, **kwargs)
                fault_system.complete_trace(trace_id, success=True)
                return result
            except Exception as e:
                fault_system.handle_error(
                    e, component, operation, trace_id
                )
                fault_system.complete_trace(trace_id, success=False)
                raise
        
        return wrapper  # type: ignore
    
    return decorator


# Global fault handling system
_fault_handling_system: Optional[FaultHandlingSystem] = None


def get_fault_handling_system() -> FaultHandlingSystem:
    """Get global fault handling system.
    
    Returns:
        Fault handling system
    """
    global _fault_handling_system
    if _fault_handling_system is None:
        _fault_handling_system = FaultHandlingSystem()
    return _fault_handling_system
