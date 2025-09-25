"""
novelWriter – Security Controller
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

import time
import re
import hashlib
import logging
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from threading import Lock
import json

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SecurityPermission(str, Enum):
    """Security permission levels."""
    
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    ADMIN = "admin"
    TOOL_CALL = "tool_call"
    EXTERNAL_TOOL = "external_tool"


class DataSensitivity(str, Enum):
    """Data sensitivity levels."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class RiskLevel(str, Enum):
    """Risk level classification."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityContext(BaseModel):
    """Security context for a session."""
    
    session_id: str
    permissions: List[SecurityPermission] = Field(default_factory=list)
    resource_quotas: Dict[str, int] = Field(default_factory=dict)
    current_usage: Dict[str, int] = Field(default_factory=dict)
    last_activity: datetime = Field(default_factory=datetime.now)
    trusted_sources: List[str] = Field(default_factory=list)
    
    def has_permission(self, permission: SecurityPermission) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions or SecurityPermission.ADMIN in self.permissions
    
    def update_usage(self, resource: str, amount: int = 1) -> bool:
        """Update resource usage and check quota."""
        if resource not in self.resource_quotas:
            return True  # No quota set
        
        current = self.current_usage.get(resource, 0)
        quota = self.resource_quotas[resource]
        
        if current + amount > quota:
            return False  # Quota exceeded
        
        self.current_usage[resource] = current + amount
        self.last_activity = datetime.now()
        return True


class AuditLogEntry(BaseModel):
    """Audit log entry."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str
    user_context: str
    operation: str
    resource: str
    permission_required: Optional[SecurityPermission] = None
    result: str  # "allowed", "denied", "error"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    integrity_hash: Optional[str] = None
    
    def calculate_hash(self) -> str:
        """Calculate integrity hash for the log entry."""
        data = f"{self.timestamp.isoformat()}:{self.session_id}:{self.operation}:{self.result}"
        return hashlib.sha256(data.encode()).hexdigest()


class PermissionValidator:
    """Permission validation system."""
    
    def __init__(self):
        """Initialize permission validator."""
        self._permission_map: Dict[str, Set[SecurityPermission]] = {}
        self._lock = Lock()
        self._setup_default_permissions()
    
    def _setup_default_permissions(self) -> None:
        """Setup default permission mappings."""
        # Read operations
        self._permission_map["get_project_info"] = {SecurityPermission.READ}
        self._permission_map["get_document"] = {SecurityPermission.READ}
        self._permission_map["search"] = {SecurityPermission.READ}
        
        # Write operations
        self._permission_map["save_document"] = {SecurityPermission.WRITE}
        self._permission_map["update_document"] = {SecurityPermission.WRITE}
        
        # Create operations
        self._permission_map["create_document"] = {SecurityPermission.CREATE}
        self._permission_map["create_folder"] = {SecurityPermission.CREATE}
        
        # Delete operations
        self._permission_map["delete_document"] = {SecurityPermission.DELETE}
        self._permission_map["delete_folder"] = {SecurityPermission.DELETE}
        
        # Tool operations
        self._permission_map["call_tool"] = {SecurityPermission.TOOL_CALL}
        self._permission_map["call_external_tool"] = {SecurityPermission.EXTERNAL_TOOL}
        
        # Admin operations
        self._permission_map["manage_permissions"] = {SecurityPermission.ADMIN}
        self._permission_map["view_audit_logs"] = {SecurityPermission.ADMIN}
    
    def validate(
        self,
        operation: str,
        context: SecurityContext,
        resource: Optional[str] = None
    ) -> bool:
        """Validate permission for operation.
        
        Args:
            operation: Operation to validate
            context: Security context
            resource: Optional resource identifier
            
        Returns:
            True if permission granted
        """
        with self._lock:
            required_permissions = self._permission_map.get(operation, set())
            
            if not required_permissions:
                # No permissions required (public operation)
                return True
            
            # Check if context has any of the required permissions
            for perm in required_permissions:
                if context.has_permission(perm):
                    logger.debug(f"Permission granted: {operation} with {perm}")
                    return True
            
            logger.warning(f"Permission denied: {operation} for session {context.session_id}")
            return False
    
    def register_operation(
        self,
        operation: str,
        permissions: Set[SecurityPermission]
    ) -> None:
        """Register operation with required permissions.
        
        Args:
            operation: Operation name
            permissions: Required permissions
        """
        with self._lock:
            self._permission_map[operation] = permissions


class ParameterSanitizer:
    """Parameter sanitization and validation."""
    
    # Patterns for common attacks
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
        re.IGNORECASE
    )
    XSS_PATTERN = re.compile(
        r"(<script|javascript:|onerror=|onclick=|<iframe|<object|<embed)",
        re.IGNORECASE
    )
    PATH_TRAVERSAL_PATTERN = re.compile(r"\.\./|\.\\/|%2e%2e|%252e")
    
    def sanitize(self, value: Any, param_type: str = "general") -> Any:
        """Sanitize parameter value.
        
        Args:
            value: Value to sanitize
            param_type: Type of parameter for specific sanitization
            
        Returns:
            Sanitized value
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            return self._sanitize_string(value, param_type)
        elif isinstance(value, dict):
            return {k: self.sanitize(v, param_type) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.sanitize(item, param_type) for item in value]
        else:
            return value
    
    def _sanitize_string(self, value: str, param_type: str) -> str:
        """Sanitize string value.
        
        Args:
            value: String to sanitize
            param_type: Parameter type
            
        Returns:
            Sanitized string
        """
        # Check for SQL injection
        if self.SQL_INJECTION_PATTERN.search(value):
            logger.warning(f"SQL injection attempt detected: {value[:50]}...")
            value = self.SQL_INJECTION_PATTERN.sub("", value)
        
        # Check for XSS
        if self.XSS_PATTERN.search(value):
            logger.warning(f"XSS attempt detected: {value[:50]}...")
            value = self.XSS_PATTERN.sub("", value)
        
        # Check for path traversal
        if param_type == "path" and self.PATH_TRAVERSAL_PATTERN.search(value):
            logger.warning(f"Path traversal attempt detected: {value}")
            value = self.PATH_TRAVERSAL_PATTERN.sub("", value)
        
        # HTML entity encoding for display values
        if param_type == "display":
            value = value.replace("<", "&lt;").replace(">", "&gt;")
            value = value.replace('"', "&quot;").replace("'", "&#x27;")
        
        return value.strip()
    
    def validate_whitelist(self, value: str, whitelist: Set[str]) -> bool:
        """Validate value against whitelist.
        
        Args:
            value: Value to validate
            whitelist: Allowed values
            
        Returns:
            True if value is in whitelist
        """
        return value in whitelist


class DataClassifier:
    """Data sensitivity classifier."""
    
    # Patterns for sensitive data detection
    PATTERNS = {
        DataSensitivity.SECRET: [
            re.compile(r"\b[A-Za-z0-9_-]{32,}\b"),  # API keys (more flexible)
            re.compile(r"-----BEGIN.*KEY-----"),  # Private keys
            re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
        ],
        DataSensitivity.CONFIDENTIAL: [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit card
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
        ],
        DataSensitivity.INTERNAL: [
            re.compile(r"/internal/|/private/", re.IGNORECASE),
            re.compile(r"\bconfidential\b", re.IGNORECASE),
        ]
    }
    
    def classify(self, data: Any) -> DataSensitivity:
        """Classify data sensitivity.
        
        Args:
            data: Data to classify
            
        Returns:
            Data sensitivity level
        """
        if not isinstance(data, str):
            data = str(data)
        
        # Check patterns from most to least sensitive
        for sensitivity in [DataSensitivity.SECRET, DataSensitivity.CONFIDENTIAL, DataSensitivity.INTERNAL]:
            patterns = self.PATTERNS.get(sensitivity, [])
            for pattern in patterns:
                if pattern.search(data):
                    return sensitivity
        
        return DataSensitivity.PUBLIC
    
    def mask_sensitive_data(self, data: str, sensitivity: DataSensitivity) -> str:
        """Mask sensitive data based on classification.
        
        Args:
            data: Data to mask
            sensitivity: Sensitivity level
            
        Returns:
            Masked data
        """
        if sensitivity == DataSensitivity.SECRET:
            return "***SECRET***"
        elif sensitivity == DataSensitivity.CONFIDENTIAL:
            # Partial masking
            if len(data) > 4:
                return data[:2] + "*" * (len(data) - 4) + data[-2:]
            else:
                return "*" * len(data)
        elif sensitivity == DataSensitivity.INTERNAL:
            return f"[INTERNAL: {len(data)} chars]"
        else:
            return data


class ResourceLimiter:
    """Resource usage limiter."""
    
    def __init__(self):
        """Initialize resource limiter."""
        self._limits = {
            "api_calls_per_minute": 60,
            "api_calls_per_hour": 1000,
            "concurrent_calls": 10,
            "memory_mb": 500,
            "cpu_percent": 50,
        }
        self._usage_tracker: Dict[str, List[float]] = {}
        self._lock = Lock()
    
    def check_rate_limit(
        self,
        resource: str,
        context: SecurityContext,
        window_seconds: int = 60
    ) -> bool:
        """Check if rate limit is exceeded.
        
        Args:
            resource: Resource identifier
            context: Security context
            window_seconds: Time window for rate limiting
            
        Returns:
            True if within limits
        """
        with self._lock:
            key = f"{context.session_id}:{resource}"
            now = time.time()
            
            # Initialize or clean old entries
            if key not in self._usage_tracker:
                self._usage_tracker[key] = []
            
            # Remove entries outside window
            self._usage_tracker[key] = [
                t for t in self._usage_tracker[key]
                if now - t < window_seconds
            ]
            
            # Check limit
            limit_key = f"{resource}_per_minute" if window_seconds == 60 else f"{resource}_per_hour"
            limit = self._limits.get(limit_key, float('inf'))
            
            if len(self._usage_tracker[key]) >= limit:
                logger.warning(f"Rate limit exceeded for {key}")
                return False
            
            # Record usage
            self._usage_tracker[key].append(now)
            return True
    
    def set_limit(self, resource: str, limit: int) -> None:
        """Set resource limit.
        
        Args:
            resource: Resource name
            limit: Limit value
        """
        self._limits[resource] = limit


class AuditLogger:
    """Audit logging system."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self._log_file = log_file
        self._entries: List[AuditLogEntry] = []
        self._lock = Lock()
    
    def log(
        self,
        operation: str,
        context: SecurityContext,
        result: str,
        resource: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        risk_level: RiskLevel = RiskLevel.LOW,
        execution_time_ms: int = 0
    ) -> AuditLogEntry:
        """Log an audit entry.
        
        Args:
            operation: Operation performed
            context: Security context
            result: Operation result
            resource: Resource accessed
            parameters: Operation parameters
            risk_level: Risk level
            execution_time_ms: Execution time
            
        Returns:
            Created audit log entry
        """
        entry = AuditLogEntry(
            session_id=context.session_id,
            user_context=str(context.permissions),
            operation=operation,
            resource=resource or "",
            result=result,
            parameters=parameters or {},
            risk_level=risk_level,
            execution_time_ms=execution_time_ms
        )
        
        # Calculate integrity hash
        entry.integrity_hash = entry.calculate_hash()
        
        with self._lock:
            self._entries.append(entry)
            
            # Write to file if configured
            if self._log_file:
                self._write_to_file(entry)
        
        logger.debug(f"Audit log: {operation} - {result}")
        return entry
    
    def _write_to_file(self, entry: AuditLogEntry) -> None:
        """Write entry to log file.
        
        Args:
            entry: Log entry to write
        """
        try:
            with open(self._log_file, 'a') as f:
                f.write(entry.model_dump_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operation: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None
    ) -> List[AuditLogEntry]:
        """Query audit logs.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            operation: Operation filter
            risk_level: Risk level filter
            
        Returns:
            Filtered audit log entries
        """
        with self._lock:
            results = self._entries.copy()
        
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        if operation:
            results = [e for e in results if e.operation == operation]
        if risk_level:
            results = [e for e in results if e.risk_level == risk_level]
        
        return results


class SecurityController:
    """Main security controller."""
    
    def __init__(self):
        """Initialize security controller."""
        self.permission_validator = PermissionValidator()
        self.parameter_sanitizer = ParameterSanitizer()
        self.data_classifier = DataClassifier()
        self.resource_limiter = ResourceLimiter()
        self.audit_logger = AuditLogger()
        self._contexts: Dict[str, SecurityContext] = {}
        self._lock = Lock()
    
    def create_context(
        self,
        session_id: str,
        permissions: List[SecurityPermission]
    ) -> SecurityContext:
        """Create security context.
        
        Args:
            session_id: Session identifier
            permissions: Granted permissions
            
        Returns:
            Created security context
        """
        context = SecurityContext(
            session_id=session_id,
            permissions=permissions,
            resource_quotas={
                "api_calls": 1000,
                "external_calls": 100,
                "memory_mb": 500
            }
        )
        
        with self._lock:
            self._contexts[session_id] = context
        
        return context
    
    def get_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Security context or None
        """
        with self._lock:
            return self._contexts.get(session_id)
    
    def validate_and_log(
        self,
        operation: str,
        context: SecurityContext,
        resource: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate permission and log operation.
        
        Args:
            operation: Operation to perform
            context: Security context
            resource: Resource to access
            parameters: Operation parameters
            
        Returns:
            True if operation allowed
        """
        start_time = time.perf_counter()
        
        # Check rate limit
        if not self.resource_limiter.check_rate_limit("api_calls", context):
            self.audit_logger.log(
                operation, context, "denied",
                resource, parameters, RiskLevel.MEDIUM,
                int((time.perf_counter() - start_time) * 1000)
            )
            return False
        
        # Validate permission
        if not self.permission_validator.validate(operation, context, resource):
            self.audit_logger.log(
                operation, context, "denied",
                resource, parameters, RiskLevel.HIGH,
                int((time.perf_counter() - start_time) * 1000)
            )
            return False
        
        # Sanitize parameters
        if parameters:
            sanitized_params = self.parameter_sanitizer.sanitize(parameters)
        else:
            sanitized_params = {}
        
        # Log successful validation
        self.audit_logger.log(
            operation, context, "allowed",
            resource, sanitized_params, RiskLevel.LOW,
            int((time.perf_counter() - start_time) * 1000)
        )
        
        return True


# Security decorators
def requires_permission(*permissions: SecurityPermission):
    """Decorator to require specific permissions.
    
    Args:
        permissions: Required permissions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get security context from first argument (assuming it's self)
            if hasattr(args[0], '_security_context'):
                context = args[0]._security_context
            else:
                # Try to get from kwargs
                context = kwargs.get('security_context')
            
            if not context:
                raise PermissionError("No security context provided")
            
            # Check permissions
            has_permission = any(
                context.has_permission(perm) for perm in permissions
            )
            
            if not has_permission:
                raise PermissionError(
                    f"Missing required permissions: {permissions}"
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def audit_operation(operation: str, risk_level: RiskLevel = RiskLevel.LOW):
    """Decorator to audit operations.
    
    Args:
        operation: Operation name
        risk_level: Risk level
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            # Get security controller
            if hasattr(args[0], '_security_controller'):
                controller = args[0]._security_controller
                context = args[0]._security_context
            else:
                # No controller available, execute without audit
                return func(*args, **kwargs)
            
            try:
                result = func(*args, **kwargs)
                
                # Log success
                controller.audit_logger.log(
                    operation, context, "success",
                    parameters=kwargs,
                    risk_level=risk_level,
                    execution_time_ms=int((time.perf_counter() - start_time) * 1000)
                )
                
                return result
                
            except Exception as e:
                # Log error
                controller.audit_logger.log(
                    operation, context, "error",
                    parameters=kwargs,
                    risk_level=RiskLevel.HIGH,
                    execution_time_ms=int((time.perf_counter() - start_time) * 1000)
                )
                raise
        
        return wrapper
    return decorator


# Global security controller instance
_security_controller: Optional[SecurityController] = None


def get_security_controller() -> SecurityController:
    """Get global security controller instance.
    
    Returns:
        Security controller instance
    """
    global _security_controller
    if _security_controller is None:
        _security_controller = SecurityController()
    return _security_controller
