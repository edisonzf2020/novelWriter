"""
novelWriter – MCP Client
=========================

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

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from novelwriter.api.exceptions import (
    MCPConnectionError, MCPToolNotFoundError, MCPExecutionError
)
from novelwriter.api.external_mcp.connection import ConnectionPool
from novelwriter.api.external_mcp.discovery import ToolDiscovery

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP client for managing external tool connections and calls.
    
    Coordinates between connection pool and tool discovery to provide
    unified external tool access.
    """
    
    def __init__(
        self,
        max_connections: int = 10,
        default_timeout_ms: int = 30000
    ) -> None:
        """Initialize MCP client.
        
        Args:
            max_connections: Maximum number of connections
            default_timeout_ms: Default timeout for tool calls
        """
        self._pool = ConnectionPool(max_connections)
        self._discovery = ToolDiscovery()
        self._default_timeout_ms = default_timeout_ms
        self._call_history: List[Dict[str, Any]] = []
        
        logger.info(f"MCPClient initialized with max_connections={max_connections}")
    
    async def addConnection(
        self,
        server_url: str,
        connection_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Add a new external MCP connection.
        
        Args:
            server_url: MCP server URL
            connection_id: Optional connection ID (generated if not provided)
            **kwargs: Additional connection parameters
        
        Returns:
            Connection ID
        
        Raises:
            MCPConnectionError: If connection fails
        """
        # Generate connection ID if not provided
        if not connection_id:
            connection_id = f"mcp_{uuid.uuid4().hex[:8]}"
        
        # Add to pool and connect
        connection = await self._pool.addConnection(
            connection_id,
            server_url,
            **kwargs
        )
        
        # Discover tools
        await self._discovery.discoverTools(connection_id, connection)
        
        logger.info(f"Added external connection: {connection_id}")
        return connection_id
    
    async def removeConnection(self, connection_id: str) -> None:
        """Remove an external connection.
        
        Args:
            connection_id: Connection identifier
        """
        # Clear discovered tools
        self._discovery.clearConnectionTools(connection_id)
        
        # Remove from pool
        await self._pool.removeConnection(connection_id)
        
        logger.info(f"Removed external connection: {connection_id}")
    
    async def callExternalTool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call an external tool.
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            timeout_ms: Optional timeout override
        
        Returns:
            Tool execution result
        
        Raises:
            MCPToolNotFoundError: If tool not found
            MCPConnectionError: If connection fails
            MCPExecutionError: If execution fails
        """
        # Find tool and connection
        tool_info = self._discovery.getTool(tool_name)
        if not tool_info:
            raise MCPToolNotFoundError(f"External tool '{tool_name}' not found")
        
        connection = self._pool.getConnection(tool_info.connection_id)
        if not connection:
            raise MCPConnectionError(f"Connection '{tool_info.connection_id}' not found")
        
        if not connection.isConnected():
            raise MCPConnectionError(f"Connection '{tool_info.connection_id}' not connected")
        
        # Prepare call
        call_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        try:
            logger.debug(f"Calling external tool '{tool_name}' via {tool_info.connection_id}")
            
            # Execute tool call with timeout
            timeout = (timeout_ms or self._default_timeout_ms) / 1000
            result = await asyncio.wait_for(
                connection.callTool(tool_name, parameters),
                timeout=timeout
            )
            
            # Record success
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            self._recordCall(
                call_id=call_id,
                tool_name=tool_name,
                connection_id=tool_info.connection_id,
                parameters=parameters,
                result=result,
                success=True,
                execution_time_ms=execution_time_ms
            )
            
            return {
                "call_id": call_id,
                "result": result,
                "execution_time_ms": execution_time_ms,
                "connection_id": tool_info.connection_id
            }
            
        except asyncio.TimeoutError:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Tool call timed out after {timeout}s"
            
            self._recordCall(
                call_id=call_id,
                tool_name=tool_name,
                connection_id=tool_info.connection_id,
                parameters=parameters,
                result=None,
                success=False,
                execution_time_ms=execution_time_ms,
                error=error_msg
            )
            
            raise MCPExecutionError(error_msg)
            
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = str(e)
            
            self._recordCall(
                call_id=call_id,
                tool_name=tool_name,
                connection_id=tool_info.connection_id,
                parameters=parameters,
                result=None,
                success=False,
                execution_time_ms=execution_time_ms,
                error=error_msg
            )
            
            logger.error(f"External tool call failed: {e}")
            raise MCPExecutionError(f"Tool execution failed: {e}")
    
    async def refreshDiscovery(self) -> Dict[str, List[str]]:
        """Refresh tool discovery for all connections.
        
        Returns:
            Dictionary of connection_id to tool names
        """
        connections = {
            conn_id: self._pool.getConnection(conn_id)
            for conn_id in self._pool.listConnections()
        }
        
        results = await self._discovery.refreshDiscovery(connections)
        
        return {
            conn_id: [tool.name for tool in tools]
            for conn_id, tools in results.items()
        }
    
    def listExternalTools(self, connection_id: Optional[str] = None) -> List[str]:
        """List available external tools.
        
        Args:
            connection_id: Optional filter by connection
        
        Returns:
            List of tool names
        """
        return self._discovery.listTools(connection_id)
    
    def getToolInfo(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an external tool.
        
        Args:
            tool_name: Tool name
        
        Returns:
            Tool information if found
        """
        tool_info = self._discovery.getTool(tool_name)
        return tool_info.toDict() if tool_info else None
    
    def listConnections(self) -> List[str]:
        """List all connection IDs"""
        return self._pool.listConnections()
    
    def getConnectionStatus(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific connection.
        
        Args:
            connection_id: Connection identifier
        
        Returns:
            Connection status if found
        """
        connection = self._pool.getConnection(connection_id)
        return connection.getStatus() if connection else None
    
    async def healthCheck(self) -> Dict[str, Any]:
        """Perform health check on all connections.
        
        Returns:
            Health check results
        """
        results = {}
        
        for conn_id in self._pool.listConnections():
            connection = self._pool.getConnection(conn_id)
            if connection:
                results[conn_id] = {
                    "connected": connection.isConnected(),
                    "healthy": connection.isHealthy(),
                    "tools_count": len(self._discovery.getToolsForConnection(conn_id))
                }
        
        return {
            "connections": results,
            "total_connections": len(results),
            "healthy_connections": sum(1 for r in results.values() if r["healthy"]),
            "total_tools": len(self._discovery.listTools())
        }
    
    async def shutdown(self) -> None:
        """Shutdown the MCP client and close all connections"""
        logger.info("Shutting down MCP client")
        
        # Clear discovery
        self._discovery.clearAll()
        
        # Close all connections
        await self._pool.closeAll()
        
        logger.info("MCP client shutdown complete")
    
    def _recordCall(
        self,
        call_id: str,
        tool_name: str,
        connection_id: str,
        parameters: Dict[str, Any],
        result: Any,
        success: bool,
        execution_time_ms: int,
        error: Optional[str] = None
    ) -> None:
        """Record a tool call for history and statistics.
        
        Args:
            call_id: Call identifier
            tool_name: Tool name
            connection_id: Connection identifier
            parameters: Call parameters
            result: Call result
            success: Whether call succeeded
            execution_time_ms: Execution time
            error: Error message if failed
        """
        record = {
            "call_id": call_id,
            "timestamp": time.time(),
            "tool_name": tool_name,
            "connection_id": connection_id,
            "parameters": parameters,
            "result": result,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "error": error
        }
        
        self._call_history.append(record)
        
        # Keep only last 100 calls
        if len(self._call_history) > 100:
            self._call_history = self._call_history[-100:]
    
    def getCallHistory(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent call history.
        
        Args:
            limit: Maximum number of records
        
        Returns:
            List of call records
        """
        return self._call_history[-limit:]
    
    def getStatistics(self) -> Dict[str, Any]:
        """Get client statistics.
        
        Returns:
            Client statistics
        """
        total_calls = len(self._call_history)
        successful_calls = sum(1 for c in self._call_history if c["success"])
        failed_calls = total_calls - successful_calls
        
        avg_latency = 0
        if total_calls > 0:
            avg_latency = sum(c["execution_time_ms"] for c in self._call_history) / total_calls
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "average_latency_ms": avg_latency,
            "connections_count": len(self._pool.listConnections()),
            "tools_count": len(self._discovery.listTools()),
            "discovery_stats": self._discovery.getStatistics()
        }
