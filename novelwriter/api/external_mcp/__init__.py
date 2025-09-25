"""
novelWriter – External MCP Module
==================================

File History:
Created: 2025-09-24 [James]

This file is a part of novelWriter
Copyright (C) 2025 Bruno Martins

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

from novelwriter.api.external_mcp.connection import MCPConnection, ConnectionPool
from novelwriter.api.external_mcp.discovery import ToolDiscovery
from novelwriter.api.external_mcp.client import MCPClient
from novelwriter.api.external_mcp.cache import ExternalMCPCache, CacheManager
from novelwriter.api.external_mcp.health_check import (
    HealthChecker, HealthStatus, HealthCheckResult, CircuitBreaker
)
from novelwriter.api.external_mcp.exceptions import (
    ExternalMCPError, ExternalMCPConnectionError, ExternalMCPTimeoutError,
    ExternalToolTimeoutError, ExternalToolNotFoundError
)

__all__ = [
    # Connection management
    "MCPConnection", "ConnectionPool",
    # Discovery
    "ToolDiscovery", 
    # Client
    "MCPClient",
    # Cache
    "ExternalMCPCache", "CacheManager",
    # Health check
    "HealthChecker", "HealthStatus", "HealthCheckResult", "CircuitBreaker",
    # Exceptions
    "ExternalMCPError", "ExternalMCPConnectionError", "ExternalMCPTimeoutError",
    "ExternalToolTimeoutError", "ExternalToolNotFoundError"
]
