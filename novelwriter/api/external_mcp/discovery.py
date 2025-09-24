"""
novelWriter – Tool Discovery
=============================

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
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from novelwriter.api.exceptions import MCPConnectionError

logger = logging.getLogger(__name__)


class ToolInfo:
    """Information about a discovered tool"""
    
    def __init__(
        self,
        name: str,
        description: str,
        connection_id: str,
        parameters_schema: Optional[Dict[str, Any]] = None,
        is_external: bool = True
    ) -> None:
        """Initialize tool info.
        
        Args:
            name: Tool name
            description: Tool description
            connection_id: Connection that provides this tool
            parameters_schema: JSON schema for parameters
            is_external: Whether this is an external tool
        """
        self.name = name
        self.description = description
        self.connection_id = connection_id
        self.parameters_schema = parameters_schema or {}
        self.is_external = is_external
        self.discovered_at = datetime.now()
    
    def toDict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "connection_id": self.connection_id,
            "parameters_schema": self.parameters_schema,
            "is_external": self.is_external,
            "discovered_at": self.discovered_at.isoformat()
        }


class ToolDiscovery:
    """
    Manages tool discovery across external MCP connections.
    
    Provides automatic discovery, caching, and tool routing.
    """
    
    def __init__(self) -> None:
        """Initialize tool discovery service"""
        self._discovered_tools: Dict[str, ToolInfo] = {}
        self._connection_tools: Dict[str, Set[str]] = {}
        self._discovery_cache_ttl = 300  # 5 minutes
        self._last_discovery: Optional[datetime] = None
        self._lock = asyncio.Lock()
        
        logger.debug("ToolDiscovery service initialized")
    
    async def discoverTools(
        self,
        connection_id: str,
        connection
    ) -> List[ToolInfo]:
        """Discover tools from a specific connection.
        
        Args:
            connection_id: Connection identifier
            connection: MCPConnection instance
        
        Returns:
            List of discovered tools
        
        Raises:
            MCPConnectionError: If discovery fails
        """
        if not connection.isConnected():
            raise MCPConnectionError(f"Connection '{connection_id}' not connected")
        
        async with self._lock:
            try:
                logger.info(f"Discovering tools from {connection_id}")
                
                # Get tools from connection
                available_tools = connection.getAvailableTools()
                
                # Clear old tools for this connection
                if connection_id in self._connection_tools:
                    for tool_name in self._connection_tools[connection_id]:
                        if tool_name in self._discovered_tools:
                            del self._discovered_tools[tool_name]
                
                # Process discovered tools
                discovered = []
                tool_names = set()
                
                for tool_data in available_tools:
                    if isinstance(tool_data, dict):
                        tool_name = tool_data.get("name")
                        tool_desc = tool_data.get("description", "")
                        tool_schema = tool_data.get("parameters", {})
                    else:
                        # Simple string format
                        tool_name = str(tool_data)
                        tool_desc = f"External tool: {tool_name}"
                        tool_schema = {}
                    
                    if tool_name:
                        # Create tool info
                        tool_info = ToolInfo(
                            name=tool_name,
                            description=tool_desc,
                            connection_id=connection_id,
                            parameters_schema=tool_schema,
                            is_external=True
                        )
                        
                        # Register tool
                        self._discovered_tools[tool_name] = tool_info
                        tool_names.add(tool_name)
                        discovered.append(tool_info)
                
                # Update connection mapping
                self._connection_tools[connection_id] = tool_names
                self._last_discovery = datetime.now()
                
                logger.info(f"Discovered {len(discovered)} tools from {connection_id}")
                return discovered
                
            except Exception as e:
                logger.error(f"Tool discovery failed for {connection_id}: {e}")
                raise MCPConnectionError(f"Discovery failed: {e}")
    
    async def refreshDiscovery(self, connections: Dict[str, Any]) -> Dict[str, List[ToolInfo]]:
        """Refresh tool discovery for all connections.
        
        Args:
            connections: Dictionary of connection_id to connection
        
        Returns:
            Dictionary of connection_id to discovered tools
        """
        results = {}
        
        for conn_id, connection in connections.items():
            try:
                tools = await self.discoverTools(conn_id, connection)
                results[conn_id] = tools
            except Exception as e:
                logger.warning(f"Failed to refresh discovery for {conn_id}: {e}")
                results[conn_id] = []
        
        return results
    
    def getTool(self, tool_name: str) -> Optional[ToolInfo]:
        """Get tool information by name.
        
        Args:
            tool_name: Tool name
        
        Returns:
            Tool info if found, None otherwise
        """
        return self._discovered_tools.get(tool_name)
    
    def getConnectionForTool(self, tool_name: str) -> Optional[str]:
        """Get the connection ID that provides a tool.
        
        Args:
            tool_name: Tool name
        
        Returns:
            Connection ID if found, None otherwise
        """
        tool_info = self.getTool(tool_name)
        return tool_info.connection_id if tool_info else None
    
    def listTools(
        self,
        connection_id: Optional[str] = None,
        include_local: bool = False
    ) -> List[str]:
        """List discovered tool names.
        
        Args:
            connection_id: Filter by connection ID
            include_local: Include local tools
        
        Returns:
            List of tool names
        """
        if connection_id:
            return list(self._connection_tools.get(connection_id, set()))
        
        tools = []
        for tool_name, tool_info in self._discovered_tools.items():
            if tool_info.is_external or include_local:
                tools.append(tool_name)
        
        return sorted(tools)
    
    def getToolsForConnection(self, connection_id: str) -> List[ToolInfo]:
        """Get all tools provided by a connection.
        
        Args:
            connection_id: Connection identifier
        
        Returns:
            List of tool information
        """
        tools = []
        for tool_name in self._connection_tools.get(connection_id, set()):
            if tool_name in self._discovered_tools:
                tools.append(self._discovered_tools[tool_name])
        return tools
    
    def clearConnectionTools(self, connection_id: str) -> None:
        """Clear tools for a specific connection.
        
        Args:
            connection_id: Connection identifier
        """
        if connection_id in self._connection_tools:
            for tool_name in self._connection_tools[connection_id]:
                if tool_name in self._discovered_tools:
                    del self._discovered_tools[tool_name]
            del self._connection_tools[connection_id]
            
            logger.debug(f"Cleared tools for connection: {connection_id}")
    
    def clearAll(self) -> None:
        """Clear all discovered tools"""
        self._discovered_tools.clear()
        self._connection_tools.clear()
        self._last_discovery = None
        
        logger.debug("Cleared all discovered tools")
    
    def needsRefresh(self) -> bool:
        """Check if discovery cache needs refresh.
        
        Returns:
            True if refresh needed
        """
        if not self._last_discovery:
            return True
        
        age = (datetime.now() - self._last_discovery).total_seconds()
        return age > self._discovery_cache_ttl
    
    def getStatistics(self) -> Dict[str, Any]:
        """Get discovery statistics.
        
        Returns:
            Discovery statistics
        """
        return {
            "total_tools": len(self._discovered_tools),
            "total_connections": len(self._connection_tools),
            "last_discovery": self._last_discovery.isoformat() if self._last_discovery else None,
            "cache_ttl_seconds": self._discovery_cache_ttl,
            "needs_refresh": self.needsRefresh(),
            "tools_by_connection": {
                conn_id: len(tools)
                for conn_id, tools in self._connection_tools.items()
            }
        }
