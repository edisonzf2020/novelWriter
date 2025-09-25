"""
novelWriter – Error Classification System
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

import logging
import re
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ErrorClassification(Enum):
    """Error classification types."""
    TRANSIENT = "transient"           # Temporary error, can retry
    PERMANENT = "permanent"           # Permanent error, cannot retry
    DEGRADATION = "degradation"       # Degradation error, needs fallback
    TIMEOUT = "timeout"               # Timeout error, special retry strategy
    NETWORK = "network"               # Network error, connection handling


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorContext(BaseModel):
    """Error context information."""
    error_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    classification: ErrorClassification
    severity: ErrorSeverity
    component: str
    operation: str
    retry_count: int = 0
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    impact_scope: str = "local"  # local, component, system
    recovery_suggestion: Optional[str] = None


class ErrorPattern(BaseModel):
    """Pattern for error classification."""
    pattern: str  # Regex pattern
    classification: ErrorClassification
    severity: ErrorSeverity
    recovery_suggestion: Optional[str] = None


class ErrorMetrics(BaseModel):
    """Error metrics for analysis."""
    total_errors: int = 0
    errors_by_classification: Dict[str, int] = Field(default_factory=dict)
    errors_by_severity: Dict[str, int] = Field(default_factory=dict)
    errors_by_component: Dict[str, int] = Field(default_factory=dict)
    error_rate: float = 0.0
    last_error_time: Optional[datetime] = None


class ErrorClassifier:
    """Classifies and analyzes errors."""
    
    # Error patterns for classification
    ERROR_PATTERNS = [
        # Network errors
        ErrorPattern(
            pattern=r"(connection|network|socket|timeout|refused|unreachable)",
            classification=ErrorClassification.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestion="Check network connectivity and retry"
        ),
        # Timeout errors
        ErrorPattern(
            pattern=r"(timeout|timed out|deadline exceeded)",
            classification=ErrorClassification.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestion="Increase timeout or retry with backoff"
        ),
        # Permission errors
        ErrorPattern(
            pattern=r"(permission|denied|unauthorized|forbidden)",
            classification=ErrorClassification.PERMANENT,
            severity=ErrorSeverity.HIGH,
            recovery_suggestion="Check permissions and authentication"
        ),
        # Resource errors
        ErrorPattern(
            pattern=r"(memory|disk|space|resource|quota)",
            classification=ErrorClassification.DEGRADATION,
            severity=ErrorSeverity.HIGH,
            recovery_suggestion="Free up resources or enable degraded mode"
        ),
        # Temporary errors
        ErrorPattern(
            pattern=r"(temporary|transient|retry|busy|locked)",
            classification=ErrorClassification.TRANSIENT,
            severity=ErrorSeverity.LOW,
            recovery_suggestion="Retry with exponential backoff"
        ),
        # File not found
        ErrorPattern(
            pattern=r"(not found|does not exist|missing)",
            classification=ErrorClassification.PERMANENT,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestion="Verify resource exists"
        ),
    ]
    
    def __init__(self, max_history: int = 1000):
        """Initialize error classifier.
        
        Args:
            max_history: Maximum error history to maintain
        """
        self.error_history: deque = deque(maxlen=max_history)
        self.error_patterns: List[ErrorPattern] = self.ERROR_PATTERNS.copy()
        self.metrics = ErrorMetrics()
        self.error_aggregates: Dict[str, List[ErrorContext]] = defaultdict(list)
        
    def classify(self, 
                 error: Exception,
                 component: str,
                 operation: str,
                 retry_count: int = 0,
                 metadata: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Classify an error and create context.
        
        Args:
            error: The exception to classify
            component: Component where error occurred
            operation: Operation that failed
            retry_count: Number of retries attempted
            metadata: Additional metadata
            
        Returns:
            ErrorContext with classification
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Default classification
        classification = ErrorClassification.TRANSIENT
        severity = ErrorSeverity.MEDIUM
        recovery_suggestion = "Retry operation"
        
        # Match against patterns
        for pattern in self.error_patterns:
            if re.search(pattern.pattern, error_str, re.IGNORECASE):
                classification = pattern.classification
                severity = pattern.severity
                recovery_suggestion = pattern.recovery_suggestion
                break
        
        # Adjust severity based on retry count
        if retry_count > 3:
            severity = ErrorSeverity.HIGH
        if retry_count > 5:
            severity = ErrorSeverity.CRITICAL
            
        # Special handling for specific exception types
        if isinstance(error, TimeoutError):
            classification = ErrorClassification.TIMEOUT
        elif isinstance(error, ConnectionError):
            classification = ErrorClassification.NETWORK
        elif isinstance(error, PermissionError):
            classification = ErrorClassification.PERMANENT
            severity = ErrorSeverity.HIGH
        elif isinstance(error, MemoryError):
            classification = ErrorClassification.DEGRADATION
            severity = ErrorSeverity.CRITICAL
            
        # Create error context
        context = ErrorContext(
            error_id=self._generate_error_id(),
            classification=classification,
            severity=severity,
            component=component,
            operation=operation,
            retry_count=retry_count,
            error_type=error_type,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            metadata=metadata or {},
            recovery_suggestion=recovery_suggestion
        )
        
        # Determine impact scope
        context.impact_scope = self._determine_impact_scope(context)
        
        # Record in history
        self._record_error(context)
        
        return context
    
    def analyze_pattern(self, time_window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Analyze error patterns within time window.
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            Analysis results
        """
        cutoff_time = datetime.now() - time_window
        recent_errors = [
            e for e in self.error_history 
            if e.timestamp > cutoff_time
        ]
        
        if not recent_errors:
            return {
                "pattern_detected": False,
                "error_count": 0,
                "dominant_classification": None,
                "failure_prediction": None
            }
        
        # Count by classification
        classification_counts = defaultdict(int)
        component_counts = defaultdict(int)
        
        for error in recent_errors:
            classification_counts[error.classification.value] += 1
            component_counts[error.component] += 1
        
        # Find dominant classification
        dominant_classification = max(
            classification_counts.items(),
            key=lambda x: x[1]
        )[0] if classification_counts else None
        
        # Predict failure if error rate is high
        error_rate = len(recent_errors) / (time_window.total_seconds() / 60)
        failure_prediction = None
        
        if error_rate > 10:  # More than 10 errors per minute
            failure_prediction = "high"
        elif error_rate > 5:
            failure_prediction = "medium"
        elif error_rate > 2:
            failure_prediction = "low"
            
        return {
            "pattern_detected": failure_prediction is not None,
            "error_count": len(recent_errors),
            "error_rate": error_rate,
            "dominant_classification": dominant_classification,
            "failure_prediction": failure_prediction,
            "affected_components": list(component_counts.keys()),
            "classification_distribution": dict(classification_counts)
        }
    
    def aggregate_errors(self, dedup_window: timedelta = timedelta(seconds=60)) -> List[ErrorContext]:
        """Aggregate and deduplicate errors.
        
        Args:
            dedup_window: Time window for deduplication
            
        Returns:
            Aggregated unique errors
        """
        cutoff_time = datetime.now() - dedup_window
        recent_errors = [
            e for e in self.error_history 
            if e.timestamp > cutoff_time
        ]
        
        # Group by error signature
        error_groups = defaultdict(list)
        for error in recent_errors:
            signature = self._get_error_signature(error)
            error_groups[signature].append(error)
        
        # Create aggregated errors
        aggregated = []
        for signature, errors in error_groups.items():
            if errors:
                # Use the most recent error as representative
                representative = max(errors, key=lambda e: e.timestamp)
                representative.metadata["occurrence_count"] = len(errors)
                representative.metadata["first_occurrence"] = min(
                    e.timestamp for e in errors
                ).isoformat()
                aggregated.append(representative)
        
        return aggregated
    
    def predict_failure(self) -> Optional[Dict[str, Any]]:
        """Predict potential failures based on error patterns.
        
        Returns:
            Failure prediction or None
        """
        analysis = self.analyze_pattern(timedelta(minutes=10))
        
        if not analysis["pattern_detected"]:
            return None
        
        prediction = {
            "likelihood": analysis["failure_prediction"],
            "affected_components": analysis["affected_components"],
            "recommended_action": None,
            "estimated_impact": "unknown"
        }
        
        # Determine recommended action
        dominant = analysis.get("dominant_classification")
        if dominant == ErrorClassification.NETWORK.value:
            prediction["recommended_action"] = "Check network connectivity"
            prediction["estimated_impact"] = "service_disruption"
        elif dominant == ErrorClassification.TIMEOUT.value:
            prediction["recommended_action"] = "Increase timeouts or reduce load"
            prediction["estimated_impact"] = "performance_degradation"
        elif dominant == ErrorClassification.DEGRADATION.value:
            prediction["recommended_action"] = "Enable degraded mode"
            prediction["estimated_impact"] = "feature_limitation"
        elif analysis["error_rate"] > 10:
            prediction["recommended_action"] = "Immediate investigation required"
            prediction["estimated_impact"] = "system_failure"
            
        return prediction
    
    def get_metrics(self) -> ErrorMetrics:
        """Get current error metrics.
        
        Returns:
            Error metrics
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset error metrics."""
        self.metrics = ErrorMetrics()
        self.error_history.clear()
        self.error_aggregates.clear()
    
    def add_pattern(self, pattern: ErrorPattern) -> None:
        """Add custom error pattern.
        
        Args:
            pattern: Error pattern to add
        """
        self.error_patterns.append(pattern)
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID.
        
        Returns:
            Error ID
        """
        import uuid
        return f"err_{uuid.uuid4().hex[:8]}"
    
    def _determine_impact_scope(self, context: ErrorContext) -> str:
        """Determine error impact scope.
        
        Args:
            context: Error context
            
        Returns:
            Impact scope
        """
        if context.severity == ErrorSeverity.CRITICAL:
            return "system"
        elif context.severity == ErrorSeverity.HIGH:
            return "component"
        else:
            return "local"
    
    def _get_error_signature(self, error: ErrorContext) -> str:
        """Get error signature for deduplication.
        
        Args:
            error: Error context
            
        Returns:
            Error signature
        """
        return f"{error.component}:{error.operation}:{error.error_type}:{error.classification.value}"
    
    def _record_error(self, context: ErrorContext) -> None:
        """Record error in history and update metrics.
        
        Args:
            context: Error context
        """
        # Add to history
        self.error_history.append(context)
        
        # Update metrics
        self.metrics.total_errors += 1
        self.metrics.errors_by_classification[context.classification.value] = \
            self.metrics.errors_by_classification.get(context.classification.value, 0) + 1
        self.metrics.errors_by_severity[context.severity.value] = \
            self.metrics.errors_by_severity.get(context.severity.value, 0) + 1
        self.metrics.errors_by_component[context.component] = \
            self.metrics.errors_by_component.get(context.component, 0) + 1
        self.metrics.last_error_time = context.timestamp
        
        # Calculate error rate (errors per minute over last 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_count = sum(
            1 for e in self.error_history 
            if e.timestamp > cutoff_time
        )
        self.metrics.error_rate = recent_count / 5.0
        
        # Add to aggregates
        signature = self._get_error_signature(context)
        self.error_aggregates[signature].append(context)
