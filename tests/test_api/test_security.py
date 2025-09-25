"""
novelWriter – Security Controller Tests
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

import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from novelwriter.api.base.security import (
    SecurityController, SecurityContext, SecurityPermission,
    PermissionValidator, ParameterSanitizer, DataClassifier,
    ResourceLimiter, AuditLogger, AuditLogEntry,
    DataSensitivity, RiskLevel,
    requires_permission, audit_operation
)
from novelwriter.api.base.audit import AuditLogManager, AuditLogViewer


class TestSecurityContext:
    """Test security context functionality."""
    
    def test_context_creation(self):
        """Test security context creation."""
        context = SecurityContext(
            session_id="test_session",
            permissions=[SecurityPermission.READ, SecurityPermission.WRITE]
        )
        
        assert context.session_id == "test_session"
        assert SecurityPermission.READ in context.permissions
        assert SecurityPermission.WRITE in context.permissions
    
    def test_has_permission(self):
        """Test permission checking."""
        context = SecurityContext(
            session_id="test",
            permissions=[SecurityPermission.READ]
        )
        
        assert context.has_permission(SecurityPermission.READ) is True
        assert context.has_permission(SecurityPermission.WRITE) is False
        
        # Admin permission grants all
        admin_context = SecurityContext(
            session_id="admin",
            permissions=[SecurityPermission.ADMIN]
        )
        assert admin_context.has_permission(SecurityPermission.WRITE) is True
    
    def test_resource_usage_tracking(self):
        """Test resource usage tracking."""
        context = SecurityContext(
            session_id="test",
            permissions=[],
            resource_quotas={"api_calls": 10}
        )
        
        # Within quota
        for i in range(10):
            assert context.update_usage("api_calls", 1) is True
        
        # Exceeds quota
        assert context.update_usage("api_calls", 1) is False
        
        # No quota set
        assert context.update_usage("unlimited", 100) is True


class TestPermissionValidator:
    """Test permission validation."""
    
    def test_default_permissions(self):
        """Test default permission mappings."""
        validator = PermissionValidator()
        
        read_context = SecurityContext(
            session_id="test",
            permissions=[SecurityPermission.READ]
        )
        
        write_context = SecurityContext(
            session_id="test",
            permissions=[SecurityPermission.WRITE]
        )
        
        # Read operations
        assert validator.validate("get_document", read_context) is True
        assert validator.validate("save_document", read_context) is False
        
        # Write operations
        assert validator.validate("save_document", write_context) is True
        assert validator.validate("get_document", write_context) is False
    
    def test_register_operation(self):
        """Test registering new operations."""
        validator = PermissionValidator()
        
        validator.register_operation(
            "custom_operation",
            {SecurityPermission.TOOL_CALL}
        )
        
        context = SecurityContext(
            session_id="test",
            permissions=[SecurityPermission.TOOL_CALL]
        )
        
        assert validator.validate("custom_operation", context) is True


class TestParameterSanitizer:
    """Test parameter sanitization."""
    
    def test_sql_injection_protection(self):
        """Test SQL injection protection."""
        sanitizer = ParameterSanitizer()
        
        malicious = "'; DROP TABLE users; --"
        sanitized = sanitizer.sanitize(malicious)
        
        assert "DROP TABLE" not in sanitized
        assert malicious != sanitized
    
    def test_xss_protection(self):
        """Test XSS attack protection."""
        sanitizer = ParameterSanitizer()
        
        xss = "<script>alert('XSS')</script>"
        sanitized = sanitizer.sanitize(xss)
        
        assert "<script>" not in sanitized
        assert "javascript:" not in sanitized
    
    def test_path_traversal_protection(self):
        """Test path traversal protection."""
        sanitizer = ParameterSanitizer()
        
        traversal = "../../etc/passwd"
        sanitized = sanitizer.sanitize(traversal, "path")
        
        assert "../" not in sanitized
        assert traversal != sanitized
    
    def test_nested_sanitization(self):
        """Test sanitization of nested structures."""
        sanitizer = ParameterSanitizer()
        
        nested = {
            "query": "SELECT * FROM users",
            "params": ["<script>", "../../file"],
            "nested": {
                "xss": "javascript:alert(1)"
            }
        }
        
        sanitized = sanitizer.sanitize(nested)
        
        assert "SELECT" not in sanitized["query"]
        assert "<script>" not in sanitized["params"][0]
        assert "javascript:" not in sanitized["nested"]["xss"]
    
    def test_whitelist_validation(self):
        """Test whitelist validation."""
        sanitizer = ParameterSanitizer()
        
        whitelist = {"option1", "option2", "option3"}
        
        assert sanitizer.validate_whitelist("option1", whitelist) is True
        assert sanitizer.validate_whitelist("invalid", whitelist) is False


class TestDataClassifier:
    """Test data sensitivity classification."""
    
    def test_secret_detection(self):
        """Test secret data detection."""
        classifier = DataClassifier()
        
        # API key pattern
        api_key = "sk_test_4eC39HqLyjWDarjtT1zdp7dcsktest123456"
        assert classifier.classify(api_key) == DataSensitivity.SECRET
        
        # Private key
        private_key = "-----BEGIN PRIVATE KEY-----\nMIIEvQ..."
        assert classifier.classify(private_key) == DataSensitivity.SECRET
        
        # Password
        password = "password: mysecretpass123"
        assert classifier.classify(password) == DataSensitivity.SECRET
    
    def test_confidential_detection(self):
        """Test confidential data detection."""
        classifier = DataClassifier()
        
        # SSN
        ssn = "123-45-6789"
        assert classifier.classify(ssn) == DataSensitivity.CONFIDENTIAL
        
        # Credit card
        cc = "4532-1234-5678-9012"
        assert classifier.classify(cc) == DataSensitivity.CONFIDENTIAL
        
        # Email
        email = "user@example.com"
        assert classifier.classify(email) == DataSensitivity.CONFIDENTIAL
    
    def test_data_masking(self):
        """Test sensitive data masking."""
        classifier = DataClassifier()
        
        # Secret masking
        secret = "api_key_12345"
        masked = classifier.mask_sensitive_data(secret, DataSensitivity.SECRET)
        assert masked == "***SECRET***"
        
        # Confidential partial masking
        confidential = "sensitive123"
        masked = classifier.mask_sensitive_data(confidential, DataSensitivity.CONFIDENTIAL)
        assert masked.startswith("se")
        assert masked.endswith("23")
        assert "*" in masked
        
        # Internal masking
        internal = "internal_data"
        masked = classifier.mask_sensitive_data(internal, DataSensitivity.INTERNAL)
        assert "INTERNAL" in masked
        assert "13 chars" in masked


class TestResourceLimiter:
    """Test resource limiting."""
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        limiter = ResourceLimiter()
        limiter.set_limit("api_calls_per_minute", 5)
        
        context = SecurityContext(session_id="test", permissions=[])
        
        # Within limit
        for i in range(5):
            assert limiter.check_rate_limit("api_calls", context, 60) is True
        
        # Exceeds limit
        assert limiter.check_rate_limit("api_calls", context, 60) is False
    
    def test_rate_limit_window(self):
        """Test rate limit time window."""
        limiter = ResourceLimiter()
        limiter.set_limit("api_calls_per_minute", 2)
        
        context = SecurityContext(session_id="test", permissions=[])
        
        # Use up limit
        assert limiter.check_rate_limit("api_calls", context, 60) is True
        assert limiter.check_rate_limit("api_calls", context, 60) is True
        assert limiter.check_rate_limit("api_calls", context, 60) is False
        
        # Wait for window to expire (using very short window for testing)
        time.sleep(0.1)
        assert limiter.check_rate_limit("api_calls", context, 0.05) is True


class TestAuditLogger:
    """Test audit logging."""
    
    def test_log_creation(self):
        """Test audit log entry creation."""
        logger = AuditLogger()
        context = SecurityContext(session_id="test", permissions=[])
        
        entry = logger.log(
            operation="test_operation",
            context=context,
            result="success",
            resource="test_resource",
            parameters={"param": "value"},
            risk_level=RiskLevel.LOW,
            execution_time_ms=10
        )
        
        assert entry.operation == "test_operation"
        assert entry.result == "success"
        assert entry.resource == "test_resource"
        assert entry.risk_level == RiskLevel.LOW
        assert entry.execution_time_ms == 10
        assert entry.integrity_hash is not None
    
    def test_log_query(self):
        """Test audit log querying."""
        logger = AuditLogger()
        context = SecurityContext(session_id="test", permissions=[])
        
        # Create multiple entries
        for i in range(5):
            logger.log(
                operation=f"op_{i}",
                context=context,
                result="success" if i % 2 == 0 else "failed",
                risk_level=RiskLevel.LOW if i < 3 else RiskLevel.HIGH
            )
        
        # Query all
        all_entries = logger.query()
        assert len(all_entries) == 5
        
        # Query by operation
        op_entries = logger.query(operation="op_1")
        assert len(op_entries) == 1
        
        # Query by risk level
        high_risk = logger.query(risk_level=RiskLevel.HIGH)
        assert len(high_risk) == 2


class TestSecurityController:
    """Test main security controller."""
    
    def test_controller_initialization(self):
        """Test security controller initialization."""
        controller = SecurityController()
        
        assert controller.permission_validator is not None
        assert controller.parameter_sanitizer is not None
        assert controller.data_classifier is not None
        assert controller.resource_limiter is not None
        assert controller.audit_logger is not None
    
    def test_context_management(self):
        """Test security context management."""
        controller = SecurityController()
        
        context = controller.create_context(
            "session_123",
            [SecurityPermission.READ, SecurityPermission.WRITE]
        )
        
        assert context.session_id == "session_123"
        assert SecurityPermission.READ in context.permissions
        
        # Retrieve context
        retrieved = controller.get_context("session_123")
        assert retrieved == context
    
    def test_validate_and_log(self):
        """Test validation and logging."""
        controller = SecurityController()
        
        context = controller.create_context(
            "test",
            [SecurityPermission.READ]
        )
        
        # Allowed operation
        result = controller.validate_and_log(
            "get_document",
            context,
            resource="doc_123",
            parameters={"format": "json"}
        )
        assert result is True
        
        # Denied operation
        result = controller.validate_and_log(
            "save_document",
            context,
            resource="doc_123"
        )
        assert result is False
        
        # Check audit logs
        logs = controller.audit_logger.query()
        assert len(logs) == 2
        assert logs[0].result == "allowed"
        assert logs[1].result == "denied"
    
    def test_performance_requirement(self):
        """Test that permission validation meets <1ms requirement."""
        controller = SecurityController()
        context = controller.create_context("test", [SecurityPermission.READ])
        
        # Warm up
        controller.validate_and_log("get_document", context)
        
        # Measure performance
        times = []
        for _ in range(100):
            start = time.perf_counter()
            controller.permission_validator.validate("get_document", context)
            duration = (time.perf_counter() - start) * 1000
            times.append(duration)
        
        # Check P95 < 1ms
        times.sort()
        p95 = times[94]
        assert p95 < 1.0, f"Permission validation P95 {p95:.3f}ms exceeds 1ms"


class TestSecurityDecorators:
    """Test security decorators."""
    
    def test_requires_permission_decorator(self):
        """Test permission requirement decorator."""
        
        class TestClass:
            def __init__(self):
                self._security_context = SecurityContext(
                    session_id="test",
                    permissions=[SecurityPermission.READ]
                )
            
            @requires_permission(SecurityPermission.READ)
            def read_operation(self):
                return "success"
            
            @requires_permission(SecurityPermission.WRITE)
            def write_operation(self):
                return "success"
        
        obj = TestClass()
        
        # Allowed operation
        assert obj.read_operation() == "success"
        
        # Denied operation
        with pytest.raises(PermissionError):
            obj.write_operation()
    
    def test_audit_operation_decorator(self):
        """Test audit operation decorator."""
        
        class TestClass:
            def __init__(self):
                self._security_controller = SecurityController()
                self._security_context = self._security_controller.create_context(
                    "test", []
                )
            
            @audit_operation("test_op", RiskLevel.MEDIUM)
            def operation(self, value: str):
                return f"processed: {value}"
        
        obj = TestClass()
        result = obj.operation("test")
        
        assert result == "processed: test"
        
        # Check audit log
        logs = obj._security_controller.audit_logger.query()
        assert len(logs) == 1
        assert logs[0].operation == "test_op"
        assert logs[0].risk_level == RiskLevel.MEDIUM


class TestAuditLogManager:
    """Test audit log management."""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Create temporary log directory."""
        return tmp_path / "audit_logs"
    
    def test_log_manager_initialization(self, temp_log_dir):
        """Test log manager initialization."""
        manager = AuditLogManager(
            log_dir=temp_log_dir,
            max_file_size_mb=1,
            retention_days=7
        )
        
        assert manager.log_dir.exists()
        assert manager.max_file_size == 1024 * 1024
        assert manager.retention_days == 7
    
    def test_log_writing(self, temp_log_dir):
        """Test writing logs to file."""
        manager = AuditLogManager(temp_log_dir)
        
        entry = AuditLogEntry(
            session_id="test",
            user_context="test_user",
            operation="test_op",
            resource="test_resource",
            result="success"
        )
        
        manager.write(entry)
        
        # Check file was created
        log_files = list(temp_log_dir.glob("audit_*.jsonl"))
        assert len(log_files) == 1
    
    def test_log_rotation(self, temp_log_dir):
        """Test log file rotation."""
        manager = AuditLogManager(
            log_dir=temp_log_dir,
            max_file_size_mb=0.001  # Very small for testing
        )
        
        # Write enough entries to trigger rotation
        for i in range(100):
            entry = AuditLogEntry(
                session_id=f"session_{i}",
                user_context="user",
                operation=f"op_{i}",
                resource="resource",
                result="success"
            )
            manager.write(entry)
        
        # Check multiple files created
        log_files = list(temp_log_dir.glob("audit_*.jsonl*"))
        assert len(log_files) > 1
    
    def test_log_cleanup(self, temp_log_dir):
        """Test old log cleanup."""
        manager = AuditLogManager(
            log_dir=temp_log_dir,
            retention_days=0  # Delete immediately for testing
        )
        
        # Create old log file
        old_log = temp_log_dir / "audit_20240101_000000.jsonl"
        old_log.touch()
        
        # Run cleanup
        manager.cleanup_old_logs()
        
        # Check old file was deleted
        assert not old_log.exists()


class TestAuditLogViewer:
    """Test audit log viewer."""
    
    @pytest.fixture
    def viewer_setup(self, tmp_path):
        """Setup viewer with test data."""
        log_dir = tmp_path / "audit_logs"
        manager = AuditLogManager(log_dir)
        viewer = AuditLogViewer(manager)
        
        # Add test entries
        for i in range(10):
            entry = AuditLogEntry(
                session_id=f"session_{i}",
                user_context="user",
                operation=f"op_{i % 3}",
                resource=f"resource_{i}",
                result="success" if i % 2 == 0 else "failed",
                risk_level=RiskLevel.LOW if i < 5 else RiskLevel.HIGH
            )
            manager.write(entry)
        
        return viewer, manager
    
    def test_search_logs(self, viewer_setup):
        """Test log searching."""
        viewer, manager = viewer_setup
        
        # Search by operation
        results = viewer.search(query="", operation="op_1")
        assert all(e.operation == "op_1" for e in results)
        
        # Search by risk level
        results = viewer.search(query="", risk_level=RiskLevel.HIGH)
        assert all(e.risk_level == RiskLevel.HIGH for e in results)
        
        # Text search
        results = viewer.search(query="session_5")
        assert any("session_5" in e.session_id for e in results)
