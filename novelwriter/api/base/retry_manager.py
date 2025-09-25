"""
novelWriter – Retry Manager with Exponential Backoff
======================================================

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
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from .error_classifier import ErrorClassification, ErrorContext

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1  # ±10% jitter
    applicable_errors: List[ErrorClassification] = Field(
        default_factory=lambda: [
            ErrorClassification.TRANSIENT,
            ErrorClassification.TIMEOUT,
            ErrorClassification.NETWORK
        ]
    )
    retry_on_exception_types: List[str] = Field(default_factory=list)


class RetryContext(BaseModel):
    """Context for retry operations."""
    attempt_number: int = 0
    total_delay: float = 0.0
    start_time: datetime = Field(default_factory=datetime.now)
    errors: List[ErrorContext] = Field(default_factory=list)
    success: bool = False
    final_result: Optional[Any] = None
    final_error: Optional[str] = None


class RetryStatistics(BaseModel):
    """Statistics for retry operations."""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_time: float = 0.0
    success_rate: float = 0.0
    average_attempts: float = 0.0
    operations_by_component: Dict[str, int] = Field(default_factory=dict)


class RetryBudget(BaseModel):
    """Retry budget management."""
    max_retries_per_minute: int = 100
    max_retries_per_hour: int = 1000
    current_minute_count: int = 0
    current_hour_count: int = 0
    last_minute_reset: datetime = Field(default_factory=datetime.now)
    last_hour_reset: datetime = Field(default_factory=datetime.now)


class RetryManager:
    """Manages retry operations with exponential backoff."""
    
    # Default policies for different tool types
    DEFAULT_POLICIES = {
        "local_tool": RetryPolicy(
            max_attempts=2,
            base_delay=0.1,
            max_delay=1.0,
            jitter=False
        ),
        "external_mcp": RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        ),
        "ai_service": RetryPolicy(
            max_attempts=5,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=1.5,
            jitter=True
        ),
        "network": RetryPolicy(
            max_attempts=4,
            base_delay=1.0,
            max_delay=20.0,
            exponential_base=2.0,
            jitter=True,
            applicable_errors=[ErrorClassification.NETWORK, ErrorClassification.TIMEOUT]
        )
    }
    
    def __init__(self):
        """Initialize retry manager."""
        self.policies: Dict[str, RetryPolicy] = self.DEFAULT_POLICIES.copy()
        self.retry_contexts: Dict[str, RetryContext] = {}
        self.statistics = RetryStatistics()
        self.retry_budget = RetryBudget()
        self.retry_history: List[RetryContext] = []
        
    def get_policy(self, tool_type: str) -> RetryPolicy:
        """Get retry policy for tool type.
        
        Args:
            tool_type: Type of tool
            
        Returns:
            Retry policy
        """
        return self.policies.get(tool_type, RetryPolicy())
    
    def set_policy(self, tool_type: str, policy: RetryPolicy) -> None:
        """Set retry policy for tool type.
        
        Args:
            tool_type: Type of tool
            policy: Retry policy
        """
        self.policies[tool_type] = policy
    
    def calculate_delay(self, 
                       attempt: int,
                       policy: RetryPolicy) -> float:
        """Calculate delay for retry attempt.
        
        Args:
            attempt: Attempt number (1-based)
            policy: Retry policy
            
        Returns:
            Delay in seconds
        """
        if attempt <= 0:
            return 0.0
        
        # Exponential backoff
        delay = policy.base_delay * (policy.exponential_base ** (attempt - 1))
        
        # Apply max delay cap
        delay = min(delay, policy.max_delay)
        
        # Add jitter if enabled
        if policy.jitter:
            jitter_amount = delay * policy.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure minimum delay
        
        return delay
    
    def should_retry(self,
                    error: Exception,
                    error_context: ErrorContext,
                    policy: RetryPolicy,
                    attempt: int) -> bool:
        """Determine if operation should be retried.
        
        Args:
            error: The exception
            error_context: Error classification context
            policy: Retry policy
            attempt: Current attempt number
            
        Returns:
            True if should retry
        """
        # Check attempt limit
        if attempt >= policy.max_attempts:
            return False
        
        # Check error classification
        if error_context.classification not in policy.applicable_errors:
            return False
        
        # Check specific exception types if configured
        if policy.retry_on_exception_types:
            error_type = type(error).__name__
            if error_type not in policy.retry_on_exception_types:
                return False
        
        # Check retry budget
        if not self._check_budget():
            logger.warning("Retry budget exceeded")
            return False
        
        return True
    
    def create_context(self, operation_id: str) -> RetryContext:
        """Create retry context for operation.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Retry context
        """
        context = RetryContext()
        self.retry_contexts[operation_id] = context
        return context
    
    def record_attempt(self,
                      context: RetryContext,
                      error: Optional[ErrorContext] = None,
                      delay: float = 0.0) -> None:
        """Record retry attempt.
        
        Args:
            context: Retry context
            error: Error context if failed
            delay: Delay before retry
        """
        context.attempt_number += 1
        context.total_delay += delay
        
        if error:
            context.errors.append(error)
        
        # Update statistics
        self.statistics.total_attempts += 1
        self.statistics.total_delay_time += delay
        
        # Update budget
        self._update_budget()
    
    def record_success(self, context: RetryContext, result: Any) -> None:
        """Record successful retry.
        
        Args:
            context: Retry context
            result: Operation result
        """
        context.success = True
        context.final_result = result
        
        # Update statistics
        if context.attempt_number > 1:
            self.statistics.successful_retries += 1
        
        self._update_statistics()
        
        # Add to history
        self.retry_history.append(context)
    
    def record_failure(self, context: RetryContext, error: str) -> None:
        """Record failed retry.
        
        Args:
            context: Retry context
            error: Final error message
        """
        context.success = False
        context.final_error = error
        
        # Update statistics
        self.statistics.failed_retries += 1
        self._update_statistics()
        
        # Add to history
        self.retry_history.append(context)
    
    def get_statistics(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics.
        
        Args:
            component: Optional component filter
            
        Returns:
            Statistics dictionary
        """
        stats = self.statistics.model_dump()
        
        if component:
            stats["component_attempts"] = self.statistics.operations_by_component.get(
                component, 0
            )
        
        # Add recent history summary
        recent_contexts = self.retry_history[-100:]  # Last 100 operations
        if recent_contexts:
            stats["recent_success_rate"] = sum(
                1 for c in recent_contexts if c.success
            ) / len(recent_contexts)
            stats["recent_average_attempts"] = sum(
                c.attempt_number for c in recent_contexts
            ) / len(recent_contexts)
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset retry statistics."""
        self.statistics = RetryStatistics()
        self.retry_history.clear()
    
    def _check_budget(self) -> bool:
        """Check if retry budget allows retry.
        
        Returns:
            True if budget allows retry
        """
        now = datetime.now()
        
        # Reset minute counter if needed
        if (now - self.retry_budget.last_minute_reset) > timedelta(minutes=1):
            self.retry_budget.current_minute_count = 0
            self.retry_budget.last_minute_reset = now
        
        # Reset hour counter if needed
        if (now - self.retry_budget.last_hour_reset) > timedelta(hours=1):
            self.retry_budget.current_hour_count = 0
            self.retry_budget.last_hour_reset = now
        
        # Check limits
        if self.retry_budget.current_minute_count >= self.retry_budget.max_retries_per_minute:
            return False
        if self.retry_budget.current_hour_count >= self.retry_budget.max_retries_per_hour:
            return False
        
        return True
    
    def _update_budget(self) -> None:
        """Update retry budget counters."""
        self.retry_budget.current_minute_count += 1
        self.retry_budget.current_hour_count += 1
    
    def _update_statistics(self) -> None:
        """Update retry statistics."""
        total = self.statistics.successful_retries + self.statistics.failed_retries
        if total > 0:
            self.statistics.success_rate = self.statistics.successful_retries / total
            self.statistics.average_attempts = self.statistics.total_attempts / total


def with_retry(policy: Optional[Union[RetryPolicy, str]] = None,
               component: str = "unknown",
               operation: str = "unknown"):
    """Decorator for retry with exponential backoff.
    
    Args:
        policy: Retry policy or tool type name
        component: Component name
        operation: Operation name
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from .error_classifier import ErrorClassifier
            
            # Get retry manager and classifier
            retry_manager = get_retry_manager()
            error_classifier = ErrorClassifier()
            
            # Get policy
            if isinstance(policy, str):
                retry_policy = retry_manager.get_policy(policy)
            elif isinstance(policy, RetryPolicy):
                retry_policy = policy
            else:
                retry_policy = RetryPolicy()
            
            # Create context
            operation_id = f"{component}:{operation}:{time.time()}"
            context = retry_manager.create_context(operation_id)
            
            last_error = None
            for attempt in range(1, retry_policy.max_attempts + 1):
                try:
                    # Record attempt
                    retry_manager.record_attempt(context)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record success
                    retry_manager.record_success(context, result)
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    # Classify error
                    error_context = error_classifier.classify(
                        e, component, operation, attempt - 1
                    )
                    
                    # Check if should retry
                    if not retry_manager.should_retry(
                        e, error_context, retry_policy, attempt
                    ):
                        break
                    
                    # Calculate delay
                    delay = retry_manager.calculate_delay(attempt, retry_policy)
                    
                    # Log retry
                    logger.info(
                        f"Retrying {operation} after {delay:.2f}s "
                        f"(attempt {attempt}/{retry_policy.max_attempts})"
                    )
                    
                    # Record attempt with error
                    retry_manager.record_attempt(context, error_context, delay)
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # All retries failed
            retry_manager.record_failure(context, str(last_error))
            raise last_error
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from .error_classifier import ErrorClassifier
            
            # Get retry manager and classifier
            retry_manager = get_retry_manager()
            error_classifier = ErrorClassifier()
            
            # Get policy
            if isinstance(policy, str):
                retry_policy = retry_manager.get_policy(policy)
            elif isinstance(policy, RetryPolicy):
                retry_policy = policy
            else:
                retry_policy = RetryPolicy()
            
            # Create context
            operation_id = f"{component}:{operation}:{time.time()}"
            context = retry_manager.create_context(operation_id)
            
            last_error = None
            for attempt in range(1, retry_policy.max_attempts + 1):
                try:
                    # Record attempt
                    retry_manager.record_attempt(context)
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    retry_manager.record_success(context, result)
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    # Classify error
                    error_context = error_classifier.classify(
                        e, component, operation, attempt - 1
                    )
                    
                    # Check if should retry
                    if not retry_manager.should_retry(
                        e, error_context, retry_policy, attempt
                    ):
                        break
                    
                    # Calculate delay
                    delay = retry_manager.calculate_delay(attempt, retry_policy)
                    
                    # Log retry
                    logger.info(
                        f"Retrying {operation} after {delay:.2f}s "
                        f"(attempt {attempt}/{retry_policy.max_attempts})"
                    )
                    
                    # Record attempt with error
                    retry_manager.record_attempt(context, error_context, delay)
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
            
            # All retries failed
            retry_manager.record_failure(context, str(last_error))
            raise last_error
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


# Global retry manager instance
_retry_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """Get global retry manager instance.
    
    Returns:
        Retry manager
    """
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = RetryManager()
    return _retry_manager
