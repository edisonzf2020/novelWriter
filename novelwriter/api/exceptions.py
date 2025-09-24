"""
novelWriter – API Exceptions
=============================

Unified exception handling for the API module.

File History:
Created: 2025-09-24 [MCP-v1.0] API Exceptions
Updated: 2025-09-24 [MCP-v1.0] API Exceptions
This file is a part of novelWriter
Copyright (C) 2025 Veronica Berglyd Olsen and novelWriter contributors

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

from typing import Any


class APIError(Exception):
    """Base exception for all API-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize an API error with message and optional details.

        Args:
            message: Human-readable error message
            details: Additional error details for debugging

        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class APIValidationError(APIError):
    """Raised when API input validation fails."""

    def __init__(self, message: str, field: str | None = None,
                 value: Any = None, **details: Any) -> None:
        """Initialize a validation error.

        Args:
            message: Validation error message
            field: Field that failed validation
            value: Invalid value that was provided
            **details: Additional error details

        """
        error_details = {"field": field, "value": value, **details}
        super().__init__(message, error_details)
        self.field = field
        self.value = value


class APIPermissionError(APIError):
    """Raised when an operation is not permitted."""

    def __init__(self, message: str, operation: str | None = None,
                 resource: str | None = None, **details: Any) -> None:
        """Initialize a permission error.

        Args:
            message: Permission error message
            operation: Operation that was denied
            resource: Resource that was being accessed
            **details: Additional error details

        """
        error_details = {"operation": operation, "resource": resource, **details}
        super().__init__(message, error_details)
        self.operation = operation
        self.resource = resource


class APINotFoundError(APIError):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str, resource_type: str | None = None,
                 resource_id: str | None = None, **details: Any) -> None:
        """Initialize a not found error.

        Args:
            message: Not found error message
            resource_type: Type of resource not found
            resource_id: ID of resource not found
            **details: Additional error details

        """
        error_details = {"resource_type": resource_type,
                        "resource_id": resource_id, **details}
        super().__init__(message, error_details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class APIOperationError(APIError):
    """Raised when an API operation fails."""

    def __init__(self, message: str, operation: str | None = None,
                 cause: Exception | None = None, **details: Any) -> None:
        """Initialize an operation error.

        Args:
            message: Operation error message
            operation: Operation that failed
            cause: Original exception that caused the failure
            **details: Additional error details

        """
        error_details = {"operation": operation, **details}
        if cause:
            error_details["cause"] = str(cause)
            error_details["cause_type"] = type(cause).__name__
        super().__init__(message, error_details)
        self.operation = operation
        self.cause = cause


# MCP-specific exceptions

class MCPServerError(APIError):
    """Raised when MCP server encounters an error."""
    pass


class MCPToolNotFoundError(APINotFoundError):
    """Raised when a requested MCP tool is not found."""
    
    def __init__(self, message: str, tool_name: str | None = None, **details: Any) -> None:
        """Initialize a tool not found error.
        
        Args:
            message: Error message
            tool_name: Name of the tool not found
            **details: Additional error details
        """
        super().__init__(message, resource_type="tool", resource_id=tool_name, **details)
        self.tool_name = tool_name


class MCPConnectionError(APIError):
    """Raised when MCP connection fails."""
    
    def __init__(self, message: str, connection_id: str | None = None,
                 server_url: str | None = None, **details: Any) -> None:
        """Initialize a connection error.
        
        Args:
            message: Error message
            connection_id: Connection identifier
            server_url: Server URL
            **details: Additional error details
        """
        error_details = {"connection_id": connection_id, "server_url": server_url, **details}
        super().__init__(message, error_details)
        self.connection_id = connection_id
        self.server_url = server_url


class MCPAuthenticationError(APIPermissionError):
    """Raised when MCP authentication fails."""
    pass


class MCPValidationError(APIValidationError):
    """Raised when MCP parameter validation fails."""
    pass


class MCPExecutionError(APIOperationError):
    """Raised when MCP tool execution fails."""
    
    def __init__(self, message: str, tool_name: str | None = None,
                 execution_time_ms: int | None = None, **details: Any) -> None:
        """Initialize an execution error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that failed
            execution_time_ms: Execution time before failure
            **details: Additional error details
        """
        error_details = {"tool_name": tool_name, "execution_time_ms": execution_time_ms, **details}
        super().__init__(message, operation=f"execute_tool:{tool_name}", **error_details)
        self.tool_name = tool_name
        self.execution_time_ms = execution_time_ms
