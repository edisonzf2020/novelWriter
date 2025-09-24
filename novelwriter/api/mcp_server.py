"""
novelWriter – MCP Server Infrastructure
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
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Literal
from datetime import datetime
from enum import Enum

from novelwriter.api.exceptions import (
    MCPServerError, MCPToolNotFoundError, MCPConnectionError,
    MCPAuthenticationError, MCPValidationError
)

if TYPE_CHECKING:
    from novelwriter.api.novelwriter_api import NovelWriterAPI

logger = logging.getLogger(__name__)

# Try to import MCP SDK
MCP_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import Tool, TextContent
    from pydantic import BaseModel, Field
    MCP_AVAILABLE = True
except ImportError as e:
    logger.debug(f"MCP SDK not available: {e}")
    
    # Provide stub implementations for graceful degradation
    class BaseModel:
        pass
    
    class FastMCP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "MCP functionality not installed. "
                "Install with: pip install 'novelwriter[ai-mcp]'"
            )
    
    class Tool:
        pass
    
    class TextContent:
        pass
    
    def Field(*args, **kwargs):
        return None


class TransportType(str, Enum):
    """Supported MCP transport types"""
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"


class ServerStatus(str, Enum):
    """MCP server status states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ToolExecutionResult(BaseModel):
    """Standard result format for tool execution"""
    call_id: str = Field(..., description="Unique identifier for the tool call")
    success: bool = Field(..., description="Whether the tool executed successfully")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    tool_type: Literal["local", "external"] = Field(..., description="Type of tool executed")


class MCPServerConfig(BaseModel):
    """Configuration for MCP server"""
    enabled: bool = Field(True, description="Whether MCP server is enabled")
    transport_type: TransportType = Field(
        TransportType.STREAMABLE_HTTP,
        description="Transport protocol to use"
    )
    host: str = Field("127.0.0.1", description="Server host address")
    port: int = Field(3001, description="Server port")
    max_concurrent_calls: int = Field(10, description="Maximum concurrent tool calls")
    timeout_ms: int = Field(30000, description="Default timeout in milliseconds")
    retry_count: int = Field(3, description="Number of retries for failed calls")
    auth_enabled: bool = Field(False, description="Whether authentication is required")


class MCPServer:
    """
    Mixed MCP Server Infrastructure
    
    Provides unified tool calling interface for both local and external tools,
    based on FastMCP framework with streamable-http transport support.
    """
    
    def __init__(
        self,
        nw_api: NovelWriterAPI,
        config: Optional[MCPServerConfig] = None
    ) -> None:
        """Initialize MCP server with NovelWriterAPI integration.
        
        Args:
            nw_api: NovelWriterAPI instance for data access
            config: Optional server configuration
        """
        if not MCP_AVAILABLE:
            raise MCPServerError(
                "MCP SDK not available. Install with: pip install 'novelwriter[ai-mcp]'"
            )
        
        self._nw_api = nw_api
        self._config = config or MCPServerConfig()
        self._status = ServerStatus.STOPPED
        self._server: Optional[FastMCP] = None
        self._local_tools: Dict[str, Any] = {}
        self._external_connections: Dict[str, Any] = {}
        self._call_history: List[ToolExecutionResult] = []
        self._start_time: Optional[datetime] = None
        
        # Initialize components
        self._initializeServer()
        self._registerLocalTools()
        
        logger.info("MCPServer initialized with config: %s", self._config)
    
    def _initializeServer(self) -> None:
        """Initialize FastMCP server instance"""
        try:
            self._server = FastMCP(
                name="novelWriter MCP Server",
                version="1.0.0"
            )
            logger.debug("FastMCP server instance created")
        except Exception as e:
            logger.error(f"Failed to initialize FastMCP server: {e}")
            raise MCPServerError(f"Server initialization failed: {e}")
    
    def _registerLocalTools(self) -> None:
        """Register local tools with the server"""
        # This will be expanded in Task 3
        logger.debug("Registering local tools...")
        # Placeholder for local tool registration
    
    async def start(self) -> None:
        """Start the MCP server"""
        if self._status == ServerStatus.RUNNING:
            logger.warning("Server already running")
            return
        
        try:
            self._status = ServerStatus.STARTING
            logger.info(f"Starting MCP server on {self._config.host}:{self._config.port}")
            
            # Configure transport based on config
            if self._config.transport_type == TransportType.STREAMABLE_HTTP:
                await self._startStreamableHttp()
            elif self._config.transport_type == TransportType.STDIO:
                await self._startStdio()
            else:
                raise MCPServerError(f"Unsupported transport: {self._config.transport_type}")
            
            self._status = ServerStatus.RUNNING
            self._start_time = datetime.now()
            logger.info("MCP server started successfully")
            
        except Exception as e:
            self._status = ServerStatus.ERROR
            logger.error(f"Failed to start server: {e}")
            raise MCPServerError(f"Server start failed: {e}")
    
    async def _startStreamableHttp(self) -> None:
        """Start server with streamable-http transport"""
        # Implementation will be completed in Task 5
        logger.debug("Starting streamable-http transport")
    
    async def _startStdio(self) -> None:
        """Start server with stdio transport (for development)"""
        logger.debug("Starting stdio transport")
    
    async def stop(self) -> None:
        """Stop the MCP server"""
        if self._status != ServerStatus.RUNNING:
            logger.warning("Server not running")
            return
        
        try:
            self._status = ServerStatus.STOPPING
            logger.info("Stopping MCP server...")
            
            # Close external connections
            for conn_id in list(self._external_connections.keys()):
                await self._closeExternalConnection(conn_id)
            
            # Stop server
            if self._server:
                # Server shutdown logic here
                pass
            
            self._status = ServerStatus.STOPPED
            logger.info("MCP server stopped")
            
        except Exception as e:
            self._status = ServerStatus.ERROR
            logger.error(f"Error stopping server: {e}")
            raise MCPServerError(f"Server stop failed: {e}")
    
    async def restart(self) -> None:
        """Restart the MCP server"""
        logger.info("Restarting MCP server...")
        await self.stop()
        await asyncio.sleep(1)  # Brief pause before restart
        await self.start()
    
    async def _closeExternalConnection(self, connection_id: str) -> None:
        """Close an external MCP connection"""
        if connection_id in self._external_connections:
            # Connection closing logic here
            del self._external_connections[connection_id]
            logger.debug(f"Closed external connection: {connection_id}")
    
    def getStatus(self) -> Dict[str, Any]:
        """Get current server status"""
        uptime = None
        if self._start_time and self._status == ServerStatus.RUNNING:
            uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "status": self._status.value,
            "config": self._config.model_dump() if hasattr(self._config, 'model_dump') else {},
            "uptime_seconds": uptime,
            "local_tools_count": len(self._local_tools),
            "external_connections_count": len(self._external_connections),
            "total_calls": len(self._call_history)
        }
    
    def isRunning(self) -> bool:
        """Check if server is running"""
        return self._status == ServerStatus.RUNNING
    
    async def callTool(
        self,
        name: str,
        parameters: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> ToolExecutionResult:
        """Unified interface for calling tools.
        
        Routes tool calls to either local or external handlers based on
        tool registration. Provides consistent error handling and monitoring.
        
        Args:
            name: Tool name
            parameters: Tool parameters
            timeout_ms: Optional timeout override
        
        Returns:
            Standardized tool execution result
        
        Raises:
            MCPToolNotFoundError: If tool not found
            MCPValidationError: If parameters invalid
            MCPExecutionError: If execution fails
        """
        if not self.isRunning():
            raise MCPServerError("MCP server not running")
        
        # Generate call ID
        call_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        # Determine tool type and route accordingly
        tool_type = self._determineToolType(name)
        
        try:
            if tool_type == "local":
                result = await self._callLocalTool(name, parameters, timeout_ms)
            elif tool_type == "external":
                result = await self._callExternalTool(name, parameters, timeout_ms)
            else:
                raise MCPToolNotFoundError(f"Tool '{name}' not found")
            
            # Create success result
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            execution_result = ToolExecutionResult(
                call_id=call_id,
                success=True,
                result=result,
                error_message=None,
                execution_time_ms=execution_time_ms,
                tool_type=tool_type
            )
            
            # Record in history
            self._call_history.append(execution_result)
            
            # Trim history if needed
            if len(self._call_history) > 100:
                self._call_history = self._call_history[-100:]
            
            logger.info(f"Tool '{name}' executed successfully in {execution_time_ms}ms")
            return execution_result
            
        except Exception as e:
            # Create error result
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            # Ensure tool_type is valid for error cases
            if tool_type == "unknown":
                tool_type = "local"  # Default to local for unknown tools
            execution_result = ToolExecutionResult(
                call_id=call_id,
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                tool_type=tool_type
            )
            
            # Record in history
            self._call_history.append(execution_result)
            
            logger.error(f"Tool '{name}' failed: {e}")
            raise
    
    def _determineToolType(self, tool_name: str) -> str:
        """Determine if a tool is local or external.
        
        Args:
            tool_name: Tool name
        
        Returns:
            'local', 'external', or 'unknown'
        """
        # Check local tools first
        if tool_name in self._local_tools:
            return "local"
        
        # Check external tools
        for conn_id, tools in self._external_connections.items():
            if isinstance(tools, list) and tool_name in tools:
                return "external"
        
        return "unknown"
    
    async def _callLocalTool(
        self,
        name: str,
        parameters: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call a local tool.
        
        Args:
            name: Tool name
            parameters: Tool parameters
            timeout_ms: Optional timeout
        
        Returns:
            Tool result
        """
        if name not in self._local_tools:
            raise MCPToolNotFoundError(f"Local tool '{name}' not found")
        
        tool_handler = self._local_tools[name]
        logger.debug(f"Calling local tool: {name}")
        
        # Check if handler is callable
        if callable(tool_handler):
            # Check if it's an async function
            if asyncio.iscoroutinefunction(tool_handler):
                # Execute async function
                if timeout_ms:
                    result = await asyncio.wait_for(
                        tool_handler(**parameters),
                        timeout=timeout_ms / 1000
                    )
                else:
                    result = await tool_handler(**parameters)
            else:
                # Execute sync function in executor
                loop = asyncio.get_event_loop()
                from functools import partial
                result = await loop.run_in_executor(
                    None, partial(tool_handler, **parameters)
                )
            return result
        else:
            # Fallback for non-callable entries
            return {"message": f"Local tool '{name}' called", "parameters": parameters}
    
    async def _callExternalTool(
        self,
        name: str,
        parameters: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call an external tool.
        
        Args:
            name: Tool name
            parameters: Tool parameters
            timeout_ms: Optional timeout
        
        Returns:
            Tool result
        """
        # This will be implemented with MCPClient integration
        logger.debug(f"Calling external tool: {name}")
        return {"message": f"External tool '{name}' called", "parameters": parameters}
