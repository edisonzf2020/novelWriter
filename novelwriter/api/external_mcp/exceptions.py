"""
novelWriter – External MCP Exceptions
======================================

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

from typing import Any, Optional

from novelwriter.api.exceptions import APIError, APIOperationError


class ExternalMCPError(APIError):
    """Base exception for external MCP operations."""
    pass


class ExternalMCPConnectionError(ExternalMCPError):
    """Raised when external MCP connection fails."""
    
    def __init__(
        self,
        message: str,
        connection_id: Optional[str] = None,
        server_url: Optional[str] = None,
        **details: Any
    ) -> None:
        """Initialize connection error.
        
        Args:
            message: Error message
            connection_id: Connection identifier
            server_url: Server URL that failed
            **details: Additional error details
        """
        super().__init__(message, details)
        self.connection_id = connection_id
        self.server_url = server_url


class ExternalMCPTimeoutError(ExternalMCPError):
    """Raised when external MCP call times out."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **details: Any
    ) -> None:
        """Initialize timeout error.
        
        Args:
            message: Error message
            tool_name: Tool that timed out
            timeout_ms: Timeout duration in milliseconds
            **details: Additional error details
        """
        super().__init__(message, details)
        self.tool_name = tool_name
        self.timeout_ms = timeout_ms


class ExternalMCPProtocolError(ExternalMCPError):
    """Raised when MCP protocol violation occurs."""
    
    def __init__(
        self,
        message: str,
        protocol_version: Optional[str] = None,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
        **details: Any
    ) -> None:
        """Initialize protocol error.
        
        Args:
            message: Error message
            protocol_version: MCP protocol version
            expected_format: Expected message format
            actual_format: Actual message format received
            **details: Additional error details
        """
        super().__init__(message, details)
        self.protocol_version = protocol_version
        self.expected_format = expected_format
        self.actual_format = actual_format


class ExternalMCPAuthenticationError(ExternalMCPError):
    """Raised when external MCP authentication fails."""
    
    def __init__(
        self,
        message: str,
        connection_id: Optional[str] = None,
        auth_method: Optional[str] = None,
        **details: Any
    ) -> None:
        """Initialize authentication error.
        
        Args:
            message: Error message
            connection_id: Connection identifier
            auth_method: Authentication method that failed
            **details: Additional error details
        """
        super().__init__(message, details)
        self.connection_id = connection_id
        self.auth_method = auth_method


class ExternalMCPDiscoveryError(ExternalMCPError):
    """Raised when tool discovery fails."""
    
    def __init__(
        self,
        message: str,
        server_url: Optional[str] = None,
        discovery_method: Optional[str] = None,
        **details: Any
    ) -> None:
        """Initialize discovery error.
        
        Args:
            message: Error message
            server_url: Server URL being discovered
            discovery_method: Discovery method that failed
            **details: Additional error details
        """
        super().__init__(message, details)
        self.server_url = server_url
        self.discovery_method = discovery_method


class ExternalMCPCacheError(ExternalMCPError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        **details: Any
    ) -> None:
        """Initialize cache error.
        
        Args:
            message: Error message
            cache_key: Cache key involved
            operation: Cache operation that failed
            **details: Additional error details
        """
        super().__init__(message, details)
        self.cache_key = cache_key
        self.operation = operation


class ExternalMCPHealthCheckError(ExternalMCPError):
    """Raised when health check fails."""
    
    def __init__(
        self,
        message: str,
        connection_id: Optional[str] = None,
        health_status: Optional[str] = None,
        **details: Any
    ) -> None:
        """Initialize health check error.
        
        Args:
            message: Error message
            connection_id: Connection identifier
            health_status: Current health status
            **details: Additional error details
        """
        super().__init__(message, details)
        self.connection_id = connection_id
        self.health_status = health_status


class ExternalToolTimeoutError(ExternalMCPTimeoutError):
    """Specific timeout error for external tool calls."""
    pass


class ExternalToolNotFoundError(ExternalMCPError):
    """Raised when external tool is not found."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        connection_id: Optional[str] = None,
        **details: Any
    ) -> None:
        """Initialize tool not found error.
        
        Args:
            message: Error message
            tool_name: Tool name that was not found
            connection_id: Connection where tool was sought
            **details: Additional error details
        """
        super().__init__(message, details)
        self.tool_name = tool_name
        self.connection_id = connection_id
