"""
novelWriter – MCP Connection Management
========================================

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
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime, timedelta
from enum import Enum

from novelwriter.api.exceptions import MCPConnectionError

logger = logging.getLogger(__name__)

# Try to import httpx for HTTP connections
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.debug("httpx not available for external MCP connections")


class ConnectionStatus(str, Enum):
    """Connection status states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class MCPConnection:
    """
    Represents a connection to an external MCP server.
    
    Manages connection lifecycle, health checks, and automatic reconnection.
    """
    
    def __init__(
        self,
        connection_id: str,
        server_url: str,
        transport_type: Literal["streamable-http", "sse"] = "streamable-http",
        auth_config: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
        retry_count: int = 3,
        health_check_interval: int = 60
    ) -> None:
        """Initialize MCP connection.
        
        Args:
            connection_id: Unique connection identifier
            server_url: MCP server URL
            transport_type: Transport protocol
            auth_config: Optional authentication configuration
            timeout_ms: Connection timeout in milliseconds
            retry_count: Number of retry attempts
            health_check_interval: Health check interval in seconds
        """
        self.connection_id = connection_id
        self.server_url = server_url
        self.transport_type = transport_type
        self.auth_config = auth_config or {}
        self.timeout_ms = timeout_ms
        self.retry_count = retry_count
        self.health_check_interval = health_check_interval
        
        self._status = ConnectionStatus.DISCONNECTED
        self._client: Optional[httpx.AsyncClient] = None
        self._available_tools: List[str] = []
        self._last_health_check: Optional[datetime] = None
        self._health_status: Literal["healthy", "degraded", "offline"] = "offline"
        self._connection_established: Optional[datetime] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_latency_ms": 0
        }
        
        logger.info(f"MCPConnection initialized: {connection_id} -> {server_url}")
    
    async def connect(self) -> None:
        """Establish connection to MCP server.
        
        Raises:
            MCPConnectionError: If connection fails
        """
        if not HTTPX_AVAILABLE:
            raise MCPConnectionError("httpx not installed, cannot connect to external MCP servers")
        
        if self._status == ConnectionStatus.CONNECTED:
            logger.debug(f"Already connected to {self.connection_id}")
            return
        
        self._status = ConnectionStatus.CONNECTING
        retry_count = 0
        
        while retry_count < self.retry_count:
            try:
                logger.info(f"Connecting to MCP server: {self.server_url}")
                
                # Create HTTP client
                self._client = httpx.AsyncClient(
                    base_url=self.server_url,
                    timeout=httpx.Timeout(self.timeout_ms / 1000),
                    headers=self._buildHeaders()
                )
                
                # Test connection and discover tools
                await self._performHandshake()
                
                self._status = ConnectionStatus.CONNECTED
                self._connection_established = datetime.now()
                self._health_status = "healthy"
                
                logger.info(f"Connected to {self.connection_id} with {len(self._available_tools)} tools")
                
                # Start health check task
                asyncio.create_task(self._healthCheckLoop())
                
                return
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Connection attempt {retry_count} failed: {e}")
                
                if retry_count < self.retry_count:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    self._status = ConnectionStatus.ERROR
                    raise MCPConnectionError(f"Failed to connect after {self.retry_count} attempts: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server"""
        if self._status == ConnectionStatus.DISCONNECTED:
            return
        
        logger.info(f"Disconnecting from {self.connection_id}")
        
        # Cancel reconnect task if running
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        
        # Close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None
        
        self._status = ConnectionStatus.DISCONNECTED
        self._health_status = "offline"
        self._available_tools = []
        
        logger.info(f"Disconnected from {self.connection_id}")
    
    async def reconnect(self) -> None:
        """Reconnect to MCP server"""
        logger.info(f"Reconnecting to {self.connection_id}")
        await self.disconnect()
        await asyncio.sleep(1)
        await self.connect()
    
    async def callTool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on the remote MCP server.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
        
        Returns:
            Tool execution result
        
        Raises:
            MCPConnectionError: If not connected or call fails
        """
        if self._status != ConnectionStatus.CONNECTED:
            raise MCPConnectionError(f"Not connected to {self.connection_id}")
        
        if not self._client:
            raise MCPConnectionError("HTTP client not initialized")
        
        start_time = time.perf_counter()
        
        try:
            # Prepare request
            request_data = {
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Make request
            response = await self._client.post(
                "/mcp/tools/call",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Update metrics
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._metrics["total_calls"] += 1
            self._metrics["successful_calls"] += 1
            self._metrics["total_latency_ms"] += latency_ms
            
            return result
            
        except Exception as e:
            self._metrics["total_calls"] += 1
            self._metrics["failed_calls"] += 1
            
            logger.error(f"Tool call failed on {self.connection_id}: {e}")
            raise MCPConnectionError(f"Tool call failed: {e}")
    
    async def _performHandshake(self) -> None:
        """Perform initial handshake with MCP server"""
        if not self._client:
            raise MCPConnectionError("HTTP client not initialized")
        
        # Get server info and available tools
        response = await self._client.get("/mcp/tools/list")
        response.raise_for_status()
        
        data = response.json()
        self._available_tools = data.get("tools", [])
        
        logger.debug(f"Handshake complete, discovered tools: {self._available_tools}")
    
    async def _healthCheckLoop(self) -> None:
        """Background task for periodic health checks"""
        while self._status == ConnectionStatus.CONNECTED:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._performHealthCheck()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed for {self.connection_id}: {e}")
                self._health_status = "degraded"
    
    async def _performHealthCheck(self) -> None:
        """Perform a health check on the connection"""
        if not self._client:
            self._health_status = "offline"
            return
        
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
            
            self._health_status = "healthy"
            self._last_health_check = datetime.now()
            
        except Exception as e:
            logger.warning(f"Health check failed for {self.connection_id}: {e}")
            self._health_status = "degraded"
            
            # Trigger reconnection if needed
            if self._status == ConnectionStatus.CONNECTED:
                self._reconnect_task = asyncio.create_task(self._autoReconnect())
    
    async def _autoReconnect(self) -> None:
        """Automatically reconnect on connection failure"""
        self._status = ConnectionStatus.RECONNECTING
        
        for attempt in range(self.retry_count):
            try:
                logger.info(f"Auto-reconnect attempt {attempt + 1} for {self.connection_id}")
                await self.connect()
                return
            except Exception as e:
                logger.warning(f"Auto-reconnect failed: {e}")
                await asyncio.sleep(2 ** attempt)
        
        self._status = ConnectionStatus.ERROR
        self._health_status = "offline"
    
    def _buildHeaders(self) -> Dict[str, str]:
        """Build HTTP headers including authentication"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Add authentication headers if configured
        if "api_key" in self.auth_config:
            headers["Authorization"] = f"Bearer {self.auth_config['api_key']}"
        elif "token" in self.auth_config:
            headers["X-Auth-Token"] = self.auth_config["token"]
        
        return headers
    
    def getStatus(self) -> Dict[str, Any]:
        """Get connection status and metrics"""
        avg_latency = 0
        if self._metrics["successful_calls"] > 0:
            avg_latency = self._metrics["total_latency_ms"] / self._metrics["successful_calls"]
        
        uptime = None
        if self._connection_established and self._status == ConnectionStatus.CONNECTED:
            uptime = (datetime.now() - self._connection_established).total_seconds()
        
        return {
            "connection_id": self.connection_id,
            "server_url": self.server_url,
            "status": self._status.value,
            "health_status": self._health_status,
            "available_tools": self._available_tools,
            "uptime_seconds": uptime,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "metrics": {
                **self._metrics,
                "average_latency_ms": avg_latency,
                "success_rate": (
                    self._metrics["successful_calls"] / self._metrics["total_calls"]
                    if self._metrics["total_calls"] > 0 else 0
                )
            }
        }
    
    def isConnected(self) -> bool:
        """Check if connection is established"""
        return self._status == ConnectionStatus.CONNECTED
    
    def isHealthy(self) -> bool:
        """Check if connection is healthy"""
        return self._health_status == "healthy"
    
    def getAvailableTools(self) -> List[str]:
        """Get list of available tools"""
        return self._available_tools.copy()


class ConnectionPool:
    """
    Manages a pool of MCP connections.
    
    Provides connection pooling, load balancing, and failover.
    """
    
    def __init__(self, max_connections: int = 10) -> None:
        """Initialize connection pool.
        
        Args:
            max_connections: Maximum number of connections
        """
        self.max_connections = max_connections
        self._connections: Dict[str, MCPConnection] = {}
        self._lock = asyncio.Lock()
        
        logger.debug(f"ConnectionPool initialized with max_connections={max_connections}")
    
    async def addConnection(
        self,
        connection_id: str,
        server_url: str,
        **kwargs
    ) -> MCPConnection:
        """Add a new connection to the pool.
        
        Args:
            connection_id: Unique connection identifier
            server_url: MCP server URL
            **kwargs: Additional connection parameters
        
        Returns:
            Created connection
        
        Raises:
            MCPConnectionError: If pool is full or connection exists
        """
        async with self._lock:
            if len(self._connections) >= self.max_connections:
                raise MCPConnectionError(f"Connection pool full (max: {self.max_connections})")
            
            if connection_id in self._connections:
                raise MCPConnectionError(f"Connection '{connection_id}' already exists")
            
            # Create and connect
            connection = MCPConnection(connection_id, server_url, **kwargs)
            await connection.connect()
            
            self._connections[connection_id] = connection
            
            logger.info(f"Added connection to pool: {connection_id}")
            return connection
    
    async def removeConnection(self, connection_id: str) -> None:
        """Remove a connection from the pool.
        
        Args:
            connection_id: Connection identifier
        """
        async with self._lock:
            if connection_id in self._connections:
                connection = self._connections[connection_id]
                await connection.disconnect()
                del self._connections[connection_id]
                logger.info(f"Removed connection from pool: {connection_id}")
    
    def getConnection(self, connection_id: str) -> Optional[MCPConnection]:
        """Get a specific connection.
        
        Args:
            connection_id: Connection identifier
        
        Returns:
            Connection if exists, None otherwise
        """
        return self._connections.get(connection_id)
    
    def getHealthyConnection(self) -> Optional[MCPConnection]:
        """Get a healthy connection from the pool.
        
        Returns:
            Healthy connection if available, None otherwise
        """
        for connection in self._connections.values():
            if connection.isConnected() and connection.isHealthy():
                return connection
        return None
    
    def listConnections(self) -> List[str]:
        """List all connection IDs"""
        return list(self._connections.keys())
    
    async def closeAll(self) -> None:
        """Close all connections in the pool"""
        async with self._lock:
            for connection in self._connections.values():
                await connection.disconnect()
            self._connections.clear()
            logger.info("Closed all connections in pool")
