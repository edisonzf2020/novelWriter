"""
novelWriter – Circuit Breaker Pattern Implementation
=====================================================

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
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Blocking requests
    HALF_OPEN = "half_open"    # Testing recovery


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures to trigger open
    recovery_timeout: int = 60           # Seconds before half-open
    success_threshold: int = 3           # Successes to close from half-open
    monitoring_window: int = 300         # Seconds for failure rate calculation
    failure_rate_threshold: float = 0.5  # Failure rate to trigger open
    min_calls_in_window: int = 10        # Minimum calls to calculate failure rate
    excluded_exceptions: List[str] = Field(default_factory=list)


class CircuitBreakerMetrics(BaseModel):
    """Circuit breaker metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_failure_rate: float = 0.0
    time_in_current_state: float = 0.0


class CallRecord(BaseModel):
    """Record of a call through circuit breaker."""
    timestamp: datetime
    success: bool
    duration: float
    error_type: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        
        # State management
        self._lock = threading.RLock()
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = datetime.now()
        self._last_failure_time: Optional[datetime] = None
        
        # Call history for monitoring window
        self._call_history: deque = deque(maxlen=1000)
        
        # Callbacks
        self._state_change_callbacks: List[Callable] = []
        
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitBreakerState.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == CircuitBreakerState.CLOSED
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        with self._lock:
            # Check if call is permitted
            if not self._is_call_permitted():
                self.metrics.rejected_calls += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )
            
            # Record call start
            start_time = time.time()
            self.metrics.total_calls += 1
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                duration = time.time() - start_time
                self._on_success(duration)
                
                return result
                
            except Exception as e:
                # Check if exception should be ignored
                if self._should_ignore_exception(e):
                    raise
                
                # Record failure
                duration = time.time() - start_time
                self._on_failure(e, duration)
                raise
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitBreakerState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._call_history.clear()
            logger.info(f"Circuit breaker '{self.name}' reset")
    
    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            self._transition_to(CircuitBreakerState.OPEN)
            logger.warning(f"Circuit breaker '{self.name}' forced open")
    
    def get_state(self) -> CircuitBreakerState:
        """Get current state.
        
        Returns:
            Current state
        """
        with self._lock:
            self._check_recovery_timeout()
            return self.state
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics.
        
        Returns:
            Metrics
        """
        with self._lock:
            # Update time in current state
            self.metrics.time_in_current_state = (
                datetime.now() - self._last_state_change
            ).total_seconds()
            
            # Calculate current failure rate
            self.metrics.current_failure_rate = self._calculate_failure_rate()
            
            return self.metrics.model_copy()
    
    def add_state_change_callback(self, callback: Callable) -> None:
        """Add state change callback.
        
        Args:
            callback: Callback function(old_state, new_state, breaker_name)
        """
        self._state_change_callbacks.append(callback)
    
    def _is_call_permitted(self) -> bool:
        """Check if call is permitted.
        
        Returns:
            True if call is permitted
        """
        # Check for recovery timeout
        self._check_recovery_timeout()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited calls in half-open state
            return True
        
        return False
    
    def _on_success(self, duration: float) -> None:
        """Handle successful call.
        
        Args:
            duration: Call duration
        """
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = datetime.now()
        
        # Record call
        self._record_call(True, duration)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            
            # Check if enough successes to close
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)
                logger.info(
                    f"Circuit breaker '{self.name}' closed after "
                    f"{self._success_count} successful calls"
                )
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0
    
    def _on_failure(self, error: Exception, duration: float) -> None:
        """Handle failed call.
        
        Args:
            error: The exception
            duration: Call duration
        """
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = datetime.now()
        self._last_failure_time = datetime.now()
        
        # Record call
        self._record_call(False, duration, type(error).__name__)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open state reopens the circuit
            self._transition_to(CircuitBreakerState.OPEN)
            logger.warning(
                f"Circuit breaker '{self.name}' reopened after failure in half-open state"
            )
        
        elif self.state == CircuitBreakerState.CLOSED:
            self._failure_count += 1
            
            # Check failure threshold
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after "
                    f"{self._failure_count} failures"
                )
            
            # Also check failure rate
            elif self._should_open_by_failure_rate():
                self._transition_to(CircuitBreakerState.OPEN)
                logger.warning(
                    f"Circuit breaker '{self.name}' opened due to high failure rate"
                )
    
    def _check_recovery_timeout(self) -> None:
        """Check if recovery timeout has elapsed."""
        if self.state == CircuitBreakerState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            
            if elapsed >= self.config.recovery_timeout:
                self._transition_to(CircuitBreakerState.HALF_OPEN)
                logger.info(
                    f"Circuit breaker '{self.name}' half-open after "
                    f"{elapsed:.1f}s recovery timeout"
                )
    
    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to new state.
        
        Args:
            new_state: New state
        """
        if self.state == new_state:
            return
        
        old_state = self.state
        self.state = new_state
        self._last_state_change = datetime.now()
        self.metrics.state_transitions += 1
        
        # Reset counters
        if new_state == CircuitBreakerState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self._success_count = 0
        
        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state, self.name)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def _should_open_by_failure_rate(self) -> bool:
        """Check if should open based on failure rate.
        
        Returns:
            True if should open
        """
        failure_rate = self._calculate_failure_rate()
        
        # Need minimum calls in window
        recent_calls = self._get_recent_calls()
        if len(recent_calls) < self.config.min_calls_in_window:
            return False
        
        return failure_rate >= self.config.failure_rate_threshold
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate.
        
        Returns:
            Failure rate (0.0 to 1.0)
        """
        recent_calls = self._get_recent_calls()
        
        if not recent_calls:
            return 0.0
        
        failures = sum(1 for call in recent_calls if not call.success)
        return failures / len(recent_calls)
    
    def _get_recent_calls(self) -> List[CallRecord]:
        """Get calls within monitoring window.
        
        Returns:
            Recent calls
        """
        cutoff_time = datetime.now() - timedelta(
            seconds=self.config.monitoring_window
        )
        
        return [
            call for call in self._call_history
            if call.timestamp > cutoff_time
        ]
    
    def _record_call(self, 
                    success: bool,
                    duration: float,
                    error_type: Optional[str] = None) -> None:
        """Record call in history.
        
        Args:
            success: Whether call succeeded
            duration: Call duration
            error_type: Error type if failed
        """
        record = CallRecord(
            timestamp=datetime.now(),
            success=success,
            duration=duration,
            error_type=error_type
        )
        self._call_history.append(record)
    
    def _should_ignore_exception(self, error: Exception) -> bool:
        """Check if exception should be ignored.
        
        Args:
            error: The exception
            
        Returns:
            True if should ignore
        """
        if not self.config.excluded_exceptions:
            return False
        
        error_type = type(error).__name__
        return error_type in self.config.excluded_exceptions


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
    def get_or_create(self,
                     name: str,
                     config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration
            
        Returns:
            Circuit breaker
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker or None
        """
        return self._breakers.get(name)
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers.
        
        Returns:
            Metrics by breaker name
        """
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }
    
    def get_open_breakers(self) -> List[str]:
        """Get names of open circuit breakers.
        
        Returns:
            List of breaker names
        """
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]
    
    def check_cascade_risk(self) -> Dict[str, Any]:
        """Check for cascade failure risk.
        
        Returns:
            Risk assessment
        """
        open_count = len(self.get_open_breakers())
        total_count = len(self._breakers)
        
        if total_count == 0:
            risk_level = "none"
        elif open_count == 0:
            risk_level = "low"
        elif open_count / total_count < 0.3:
            risk_level = "medium"
        elif open_count / total_count < 0.6:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        return {
            "risk_level": risk_level,
            "open_breakers": open_count,
            "total_breakers": total_count,
            "open_percentage": (open_count / total_count * 100) if total_count > 0 else 0
        }


def with_circuit_breaker(name: str,
                        config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection.
    
    Args:
        name: Circuit breaker name
        config: Configuration
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get circuit breaker
            manager = get_circuit_breaker_manager()
            breaker = manager.get_or_create(name, config)
            
            # Execute through circuit breaker
            return breaker.call(func, *args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# Global circuit breaker manager
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager.
    
    Returns:
        Circuit breaker manager
    """
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager
