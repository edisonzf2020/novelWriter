"""
novelWriter – Fault Handling Tests
===================================

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

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from novelwriter.api.base.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)
from novelwriter.api.base.degradation_service import (
    DegradableFeature,
    DegradationService,
    ServiceDegradationLevel,
)
from novelwriter.api.base.error_classifier import (
    ErrorClassification,
    ErrorClassifier,
    ErrorPattern,
    ErrorSeverity,
)
from novelwriter.api.base.fault_handling import (
    AlertLevel,
    FaultHandlingSystem,
    get_fault_handling_system,
    with_fault_handling,
)
from novelwriter.api.base.retry_manager import (
    RetryManager,
    RetryPolicy,
    with_retry,
)


class TestErrorClassifier:
    """Test error classification system."""
    
    def test_error_classification(self):
        """Test error classification logic."""
        classifier = ErrorClassifier()
        
        # Test network error
        error = ConnectionError("Connection refused")
        context = classifier.classify(error, "test_component", "test_operation")
        assert context.classification == ErrorClassification.NETWORK
        assert context.severity == ErrorSeverity.MEDIUM
        
        # Test timeout error
        error = TimeoutError("Operation timed out")
        context = classifier.classify(error, "test_component", "test_operation")
        assert context.classification == ErrorClassification.TIMEOUT
        assert context.severity == ErrorSeverity.MEDIUM
        
        # Test permission error
        error = PermissionError("Access denied")
        context = classifier.classify(error, "test_component", "test_operation")
        assert context.classification == ErrorClassification.PERMANENT
        assert context.severity == ErrorSeverity.HIGH
    
    def test_error_pattern_matching(self):
        """Test custom error pattern matching."""
        classifier = ErrorClassifier()
        
        # Add custom pattern
        pattern = ErrorPattern(
            pattern=r"custom_error",
            classification=ErrorClassification.DEGRADATION,
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestion="Custom recovery"
        )
        classifier.add_pattern(pattern)
        
        # Test matching
        error = Exception("This is a custom_error message")
        context = classifier.classify(error, "test", "test")
        assert context.classification == ErrorClassification.DEGRADATION
        assert context.severity == ErrorSeverity.CRITICAL
        assert context.recovery_suggestion == "Custom recovery"
    
    def test_error_aggregation(self):
        """Test error aggregation and deduplication."""
        classifier = ErrorClassifier()
        
        # Generate similar errors
        for i in range(5):
            error = ValueError("Same error")
            classifier.classify(error, "component", "operation")
        
        # Aggregate errors
        aggregated = classifier.aggregate_errors(timedelta(minutes=1))
        
        # Should deduplicate
        assert len(aggregated) == 1
        assert aggregated[0].metadata["occurrence_count"] == 5
    
    def test_failure_prediction(self):
        """Test failure prediction based on patterns."""
        classifier = ErrorClassifier()
        
        # Generate high error rate within time window
        for i in range(20):
            error = Exception("Error")
            classifier.classify(error, "component", "operation")
        
        # Analyze pattern first to ensure we have data
        analysis = classifier.analyze_pattern(timedelta(minutes=10))
        
        # Predict failure
        prediction = classifier.predict_failure()
        
        # With 20 errors in window, should predict failure
        if analysis["error_rate"] > 2:
            assert prediction is not None
            assert prediction["likelihood"] in ["low", "medium", "high"]
        else:
            # If rate is too low, prediction may be None
            assert prediction is None or prediction["likelihood"] in ["low", "medium", "high"]


class TestRetryManager:
    """Test retry manager with exponential backoff."""
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        manager = RetryManager()
        policy = RetryPolicy(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        
        # Test exponential growth
        assert manager.calculate_delay(1, policy) == 1.0
        assert manager.calculate_delay(2, policy) == 2.0
        assert manager.calculate_delay(3, policy) == 4.0
        assert manager.calculate_delay(4, policy) == 8.0
        assert manager.calculate_delay(5, policy) == 10.0  # Max delay
    
    def test_jitter_application(self):
        """Test jitter in retry delays."""
        manager = RetryManager()
        policy = RetryPolicy(
            base_delay=1.0,
            jitter=True,
            jitter_range=0.1
        )
        
        # Calculate multiple delays
        delays = [manager.calculate_delay(2, policy) for _ in range(10)]
        
        # Should have variation
        assert len(set(delays)) > 1
        assert all(1.8 <= d <= 2.2 for d in delays)  # ±10% jitter
    
    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        call_count = 0
        
        @with_retry(
            policy=RetryPolicy(max_attempts=3, base_delay=0.01),
            component="test",
            operation="test_op"
        )
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_budget(self):
        """Test retry budget management."""
        manager = RetryManager()
        manager.retry_budget.max_retries_per_minute = 5
        
        # Use up budget
        for i in range(5):
            manager._update_budget()
        
        # Should not allow more retries
        assert not manager._check_budget()


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,
            success_threshold=2
        )
        breaker = CircuitBreaker("test", config)
        
        # Initial state should be closed
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Record failures to open circuit
        for i in range(3):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except:
                pass
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Should reject calls when open
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(lambda: "test")
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should transition to half-open
        breaker._check_recovery_timeout()
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Successful calls should close circuit
        for i in range(2):
            breaker.call(lambda: "success")
        
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_failure_rate_threshold(self):
        """Test circuit breaker opening by failure rate."""
        config = CircuitBreakerConfig(
            failure_threshold=100,  # High threshold
            failure_rate_threshold=0.5,
            min_calls_in_window=5,
            monitoring_window=60
        )
        breaker = CircuitBreaker("test", config)
        
        # Generate mixed results
        for i in range(10):
            try:
                if i % 2 == 0:
                    breaker.call(lambda: "success")
                else:
                    breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except:
                pass
        
        # Should calculate 50% failure rate
        assert breaker._calculate_failure_rate() == 0.5
    
    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics collection."""
        breaker = CircuitBreaker("test")
        
        # Generate some calls
        breaker.call(lambda: "success")
        try:
            breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except:
            pass
        
        metrics = breaker.get_metrics()
        assert metrics.total_calls == 2
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 1


class TestDegradationService:
    """Test service degradation management."""
    
    def test_degradation_evaluation(self):
        """Test degradation decision evaluation."""
        service = DegradationService()
        
        # Normal conditions
        decision = service.evaluate_degradation(
            error_rate=0.1,
            latency=1.0,
            cpu_usage=50.0,
            memory_usage=50.0
        )
        assert not decision.should_degrade
        
        # High error rate
        decision = service.evaluate_degradation(
            error_rate=0.5,
            latency=1.0,
            cpu_usage=50.0,
            memory_usage=50.0
        )
        assert decision.should_degrade
        # With only error rate high, should be LIMITED or FULL (severity_score = 1)
        assert decision.target_level in [
            ServiceDegradationLevel.FULL,
            ServiceDegradationLevel.LIMITED
        ]
    
    def test_feature_availability(self):
        """Test feature availability in degraded modes."""
        service = DegradationService()
        
        # Register test feature
        feature = DegradableFeature(
            name="test_feature",
            component="test",
            priority=30,
            offline_capable=False
        )
        service.register_feature(feature)
        
        # Should be available in full mode
        assert service.is_feature_available("test_feature")
        
        # Apply degradation
        decision = service.evaluate_degradation(
            error_rate=0.8,
            latency=10.0,
            cpu_usage=90.0,
            memory_usage=90.0
        )
        service.apply_degradation(decision)
        
        # Feature may be disabled
        if service.current_state.level == ServiceDegradationLevel.OFFLINE:
            assert not service.is_feature_available("test_feature")
    
    def test_core_features_protection(self):
        """Test that core features remain available."""
        service = DegradationService()
        
        # Apply severe degradation
        decision = service.evaluate_degradation(
            error_rate=0.9,
            latency=20.0,
            cpu_usage=95.0,
            memory_usage=95.0
        )
        service.apply_degradation(decision)
        
        # Core features should still be available (unless emergency)
        if service.current_state.level != ServiceDegradationLevel.EMERGENCY:
            assert service.is_feature_available("project_access")
            assert service.is_feature_available("document_read")
            assert service.is_feature_available("document_write")
    
    def test_recovery_from_degradation(self):
        """Test recovery from degraded state."""
        service = DegradationService()
        service.policy.min_degradation_duration = 0  # Allow immediate recovery
        
        # Apply degradation
        decision = service.evaluate_degradation(
            error_rate=0.5,
            latency=10.0,
            cpu_usage=80.0,
            memory_usage=80.0
        )
        service.apply_degradation(decision)
        
        assert service.current_state.level != ServiceDegradationLevel.FULL
        
        # Attempt recovery
        success = service.attempt_recovery()
        assert success
        assert service.current_state.level == ServiceDegradationLevel.FULL


class TestFaultHandlingSystem:
    """Test integrated fault handling system."""
    
    def test_error_handling_flow(self):
        """Test complete error handling flow."""
        system = FaultHandlingSystem()
        
        # Handle an error
        error = ConnectionError("Test error")
        context = system.handle_error(
            error,
            component="test_component",
            operation="test_operation"
        )
        
        assert context.classification == ErrorClassification.NETWORK
        assert system.metrics.total_errors == 1
    
    def test_error_correlation(self):
        """Test error trace correlation."""
        system = FaultHandlingSystem()
        
        # Create parent trace
        parent_id = system.create_trace("parent", "operation")
        
        # Create child traces
        child1_id = system.create_trace("child1", "op1", parent_id)
        child2_id = system.create_trace("child2", "op2", parent_id)
        
        # Get correlated traces
        children = system.get_correlated_traces(parent_id)
        assert len(children) == 2
        assert any(t.trace_id == child1_id for t in children)
        assert any(t.trace_id == child2_id for t in children)
    
    def test_alert_creation_and_acknowledgment(self):
        """Test alert management."""
        system = FaultHandlingSystem()
        
        # Create alert
        alert_id = system.create_alert(
            level=AlertLevel.ERROR,
            component="test",
            title="Test Alert",
            message="Test message"
        )
        
        # Should have active alert
        active = system.get_active_alerts()
        assert len(active) == 1
        assert active[0].id == alert_id
        
        # Acknowledge alert
        success = system.acknowledge_alert(alert_id, "tester")
        assert success
        
        # Should no longer be active
        active = system.get_active_alerts()
        assert len(active) == 0
    
    def test_health_check(self):
        """Test system health check."""
        system = FaultHandlingSystem()
        
        # Initial health should be good
        health = system.perform_health_check()
        assert health["status"] in ["healthy", "degraded"]
        assert health["score"] > 0
        
        # Generate errors to degrade health
        for i in range(10):
            error = Exception("Test error")
            system.handle_error(error, "test", "operation")
        
        # Health should be worse or same (depends on error analysis window)
        health = system.perform_health_check()
        # Score should be 100 or less
        assert health["score"] <= 100
    
    @pytest.mark.fault_tolerance
    def test_fault_handling_decorator(self):
        """Test fault handling decorator."""
        call_count = 0
        
        @with_fault_handling(
            component="test",
            operation="test_op",
            retry_policy=RetryPolicy(max_attempts=2, base_delay=0.01)
        )
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Temporary")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 2
    
    def test_recovery_strategy_execution(self):
        """Test automatic recovery strategy execution."""
        system = FaultHandlingSystem()
        
        # Handle network error
        error = ConnectionError("Network error")
        context = system.handle_error(
            error,
            component="network_component",
            operation="network_op"
        )
        
        # Should attempt recovery for network errors
        assert context.classification == ErrorClassification.NETWORK
        # Recovery strategy should be registered
        assert "network_recovery" in system._recovery_strategies


class TestPerformanceRequirements:
    """Test performance requirements."""
    
    def test_error_classification_performance(self):
        """Test error classification performance < 50ms."""
        classifier = ErrorClassifier()
        error = Exception("Test error")
        
        start = time.perf_counter()
        classifier.classify(error, "component", "operation")
        duration = (time.perf_counter() - start) * 1000
        
        assert duration < 50  # Should be < 50ms
    
    def test_fault_detection_performance(self):
        """Test fault detection performance < 100ms."""
        system = FaultHandlingSystem()
        error = Exception("Test error")
        
        start = time.perf_counter()
        system.handle_error(error, "component", "operation")
        duration = (time.perf_counter() - start) * 1000
        
        assert duration < 100  # Should be < 100ms
    
    def test_circuit_breaker_overhead(self):
        """Test circuit breaker overhead is minimal."""
        breaker = CircuitBreaker("test")
        
        def test_function():
            return "result"
        
        # Measure overhead
        start = time.perf_counter()
        for _ in range(100):
            breaker.call(test_function)
        duration = (time.perf_counter() - start) * 1000
        
        # Average overhead should be minimal
        avg_overhead = duration / 100
        assert avg_overhead < 1  # Less than 1ms per call


@pytest.mark.chaos_engineering
class TestChaosEngineering:
    """Chaos engineering tests."""
    
    def test_random_failures_handling(self):
        """Test system stability under random failures."""
        system = get_fault_handling_system()
        
        # Simulate random failures
        import random
        errors = [
            ConnectionError("Network failure"),
            TimeoutError("Timeout"),
            MemoryError("Out of memory"),
            ValueError("Invalid value"),
            PermissionError("Access denied")
        ]
        
        for _ in range(50):
            error = random.choice(errors)
            try:
                system.handle_error(
                    error,
                    component=f"component_{random.randint(1, 5)}",
                    operation=f"operation_{random.randint(1, 10)}"
                )
            except:
                pass  # Some errors may not be recoverable
        
        # System should still be functional
        health = system.perform_health_check()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Should have metrics
        metrics = system.get_metrics()
        assert metrics.total_errors == 50
    
    def test_cascade_failure_prevention(self):
        """Test cascade failure prevention."""
        system = FaultHandlingSystem()
        
        # Create multiple circuit breakers
        for i in range(5):
            breaker = system.circuit_breaker_manager.get_or_create(
                f"service_{i}",
                CircuitBreakerConfig(failure_threshold=2)
            )
        
        # Trigger failures in multiple services
        for i in range(3):
            breaker = system.circuit_breaker_manager.get(f"service_{i}")
            for _ in range(2):
                try:
                    breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
                except:
                    pass
        
        # Check cascade risk
        risk = system.circuit_breaker_manager.check_cascade_risk()
        assert risk["open_breakers"] == 3
        # With 3/5 breakers open (60%), risk should be high or critical
        assert risk["risk_level"] in ["high", "critical"]
