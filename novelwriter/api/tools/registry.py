"""
novelWriter – Tool Registry
============================

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

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from novelwriter.api.exceptions import MCPToolNotFoundError, MCPValidationError

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolPermission(str, Enum):
    """Tool permission levels"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class ToolMetadata:
    """Metadata for a registered tool"""
    name: str
    description: str
    parameters_schema: Optional[Dict[str, Any]] = None
    required_permissions: Set[ToolPermission] = field(default_factory=set)
    is_async: bool = False
    is_local: bool = True
    category: str = "general"
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_message: Optional[str] = None


@dataclass
class ToolRegistration:
    """Complete tool registration entry"""
    metadata: ToolMetadata
    handler: Callable
    validator: Optional[Callable] = None
    health_check: Optional[Callable] = None
    enabled: bool = True
    call_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class ToolRegistry:
    """
    Central registry for all MCP tools.
    
    Manages tool discovery, registration, metadata, and access control.
    """
    
    def __init__(self) -> None:
        """Initialize the tool registry"""
        self._tools: Dict[str, ToolRegistration] = {}
        self._categories: Dict[str, List[str]] = {}
        self._locked = False  # Prevent modifications after initialization
        
        logger.debug("ToolRegistry initialized")
    
    def registerTool(
        self,
        name: str,
        handler: Callable,
        description: str,
        parameters_schema: Optional[Dict[str, Any]] = None,
        required_permissions: Optional[Set[ToolPermission]] = None,
        is_async: bool = False,
        category: str = "general",
        validator: Optional[Callable] = None,
        health_check: Optional[Callable] = None
    ) -> None:
        """Register a new tool.
        
        Args:
            name: Unique tool name
            handler: Tool implementation function
            description: Human-readable tool description
            parameters_schema: JSON schema for parameters validation
            required_permissions: Set of required permissions
            is_async: Whether the handler is async
            category: Tool category for organization
            validator: Optional custom parameter validator
            health_check: Optional health check function
        
        Raises:
            MCPValidationError: If registration fails
        """
        if self._locked:
            raise MCPValidationError("Registry is locked, cannot register new tools")
        
        if name in self._tools:
            raise MCPValidationError(f"Tool '{name}' is already registered")
        
        # Create metadata
        metadata = ToolMetadata(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            required_permissions=required_permissions or set(),
            is_async=is_async,
            is_local=True,
            category=category
        )
        
        # Create registration
        registration = ToolRegistration(
            metadata=metadata,
            handler=handler,
            validator=validator,
            health_check=health_check
        )
        
        # Register tool
        self._tools[name] = registration
        
        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        logger.info(f"Registered tool: {name} (category: {category})")
    
    def unregisterTool(self, name: str) -> None:
        """Unregister a tool.
        
        Args:
            name: Tool name to unregister
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist
        """
        if self._locked:
            raise MCPValidationError("Registry is locked, cannot unregister tools")
        
        if name not in self._tools:
            raise MCPToolNotFoundError(f"Tool '{name}' not found")
        
        registration = self._tools[name]
        category = registration.metadata.category
        
        # Remove from registry
        del self._tools[name]
        
        # Update category index
        if category in self._categories:
            self._categories[category].remove(name)
            if not self._categories[category]:
                del self._categories[category]
        
        logger.info(f"Unregistered tool: {name}")
    
    def getTool(self, name: str) -> ToolRegistration:
        """Get a tool registration.
        
        Args:
            name: Tool name
        
        Returns:
            Tool registration
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist
        """
        if name not in self._tools:
            raise MCPToolNotFoundError(f"Tool '{name}' not found")
        
        return self._tools[name]
    
    def hasTool(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Tool name
        
        Returns:
            True if tool exists
        """
        return name in self._tools
    
    def listTools(
        self,
        category: Optional[str] = None,
        include_disabled: bool = False
    ) -> List[str]:
        """List registered tools.
        
        Args:
            category: Filter by category
            include_disabled: Include disabled tools
        
        Returns:
            List of tool names
        """
        if category:
            tool_names = self._categories.get(category, [])
        else:
            tool_names = list(self._tools.keys())
        
        if not include_disabled:
            tool_names = [
                name for name in tool_names
                if self._tools[name].enabled
            ]
        
        return sorted(tool_names)
    
    def getToolMetadata(self, name: str) -> ToolMetadata:
        """Get tool metadata.
        
        Args:
            name: Tool name
        
        Returns:
            Tool metadata
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist
        """
        return self.getTool(name).metadata
    
    def enableTool(self, name: str) -> None:
        """Enable a tool.
        
        Args:
            name: Tool name
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist
        """
        self.getTool(name).enabled = True
        logger.info(f"Enabled tool: {name}")
    
    def disableTool(self, name: str) -> None:
        """Disable a tool.
        
        Args:
            name: Tool name
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist
        """
        self.getTool(name).enabled = False
        logger.info(f"Disabled tool: {name}")
    
    def isToolEnabled(self, name: str) -> bool:
        """Check if a tool is enabled.
        
        Args:
            name: Tool name
        
        Returns:
            True if tool is enabled
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist
        """
        return self.getTool(name).enabled
    
    def recordToolCall(self, name: str, success: bool, error: Optional[str] = None) -> None:
        """Record a tool call for statistics.
        
        Args:
            name: Tool name
            success: Whether the call succeeded
            error: Error message if failed
        """
        if name not in self._tools:
            return
        
        registration = self._tools[name]
        registration.call_count += 1
        
        if not success:
            registration.error_count += 1
            registration.last_error = error
    
    def getToolStatistics(self, name: str) -> Dict[str, Any]:
        """Get tool usage statistics.
        
        Args:
            name: Tool name
        
        Returns:
            Tool statistics
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist
        """
        registration = self.getTool(name)
        
        success_rate = 0.0
        if registration.call_count > 0:
            success_rate = (registration.call_count - registration.error_count) / registration.call_count
        
        return {
            "name": name,
            "call_count": registration.call_count,
            "error_count": registration.error_count,
            "success_rate": success_rate,
            "last_error": registration.last_error,
            "enabled": registration.enabled
        }
    
    def getAllStatistics(self) -> Dict[str, Any]:
        """Get statistics for all tools.
        
        Returns:
            Dictionary of all tool statistics
        """
        total_calls = sum(reg.call_count for reg in self._tools.values())
        total_errors = sum(reg.error_count for reg in self._tools.values())
        
        return {
            "total_tools": len(self._tools),
            "enabled_tools": sum(1 for reg in self._tools.values() if reg.enabled),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "categories": list(self._categories.keys()),
            "tools": {
                name: self.getToolStatistics(name)
                for name in self._tools
            }
        }
    
    def lock(self) -> None:
        """Lock the registry to prevent further modifications"""
        self._locked = True
        logger.info("Tool registry locked")
    
    def unlock(self) -> None:
        """Unlock the registry to allow modifications"""
        self._locked = False
        logger.info("Tool registry unlocked")
    
    def isLocked(self) -> bool:
        """Check if registry is locked"""
        return self._locked
    
    async def performHealthChecks(self) -> Dict[str, bool]:
        """Perform health checks on all tools with health check functions.
        
        Returns:
            Dictionary of tool names to health status
        """
        results = {}
        
        for name, registration in self._tools.items():
            if registration.health_check:
                try:
                    if registration.metadata.is_async:
                        results[name] = await registration.health_check()
                    else:
                        results[name] = registration.health_check()
                except Exception as e:
                    logger.error(f"Health check failed for tool '{name}': {e}")
                    results[name] = False
            else:
                results[name] = True  # No health check means assumed healthy
        
        return results
    
    def discoverLocalTools(self, api_instance: Any) -> None:
        """
        Automatically discover and register local tools.
        
        Args:
            api_instance: NovelWriterAPI instance to pass to tools
        """
        from novelwriter.api.tools.base import BaseTool
        from novelwriter.api.tools.project_tools import ProjectInfoTool, ProjectTreeTool
        from novelwriter.api.tools.document_tools import (
            DocumentListTool, DocumentReadTool, DocumentWriteTool, CreateDocumentTool
        )
        from novelwriter.api.tools.search_tools import (
            GlobalSearchTool, TagListTool, ProjectStatsTool
        )
        
        # List of tool classes to auto-discover
        tool_classes = [
            # Project tools
            ProjectInfoTool,
            ProjectTreeTool,
            
            # Document tools
            DocumentListTool,
            DocumentReadTool,
            DocumentWriteTool,
            CreateDocumentTool,
            
            # Search tools
            GlobalSearchTool,
            TagListTool,
            ProjectStatsTool
        ]
        
        # Register each tool
        for tool_class in tool_classes:
            try:
                # Instantiate tool
                tool_instance = tool_class(api_instance)
                
                # Create async wrapper for execute method
                async def tool_handler(**params):
                    return await tool_instance.execute(**params)
                
                # Register with metadata from tool
                metadata = tool_instance.metadata
                self.registerTool(
                    name=metadata.name,
                    handler=tool_handler,
                    description=metadata.description,
                    parameters_schema=metadata.parameters_schema,
                    required_permissions=set(metadata.required_permissions),
                    is_async=True,
                    category=metadata.tags[0] if metadata.tags else "general"
                )
                
                logger.info(f"Auto-discovered and registered tool: {metadata.name}")
                
            except Exception as e:
                logger.error(f"Failed to auto-discover tool {tool_class.__name__}: {e}")
        
        logger.info(f"Auto-discovery complete: {len(tool_classes)} tools registered")
