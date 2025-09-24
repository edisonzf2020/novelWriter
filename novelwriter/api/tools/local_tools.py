"""
novelWriter – Local Tools Wrapper
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

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from functools import wraps

from novelwriter.api.exceptions import (
    MCPToolNotFoundError, MCPValidationError, MCPExecutionError
)

if TYPE_CHECKING:
    from novelwriter.api.novelwriter_api import NovelWriterAPI
    from novelwriter.api.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class LocalToolWrapper:
    """
    Wrapper for local tools that use NovelWriterAPI.
    
    Provides a unified interface for executing local tools with
    proper parameter validation, error handling, and performance monitoring.
    """
    
    def __init__(self, nw_api: NovelWriterAPI, registry: ToolRegistry) -> None:
        """Initialize local tool wrapper.
        
        Args:
            nw_api: NovelWriterAPI instance for data access
            registry: Tool registry for tool management
        """
        self._nw_api = nw_api
        self._registry = registry
        
        # Register built-in local tools
        self._registerBuiltinTools()
        
        logger.debug("LocalToolWrapper initialized")
    
    def _registerBuiltinTools(self) -> None:
        """Register built-in local tools"""
        # Project tools
        self._registry.registerTool(
            name="get_project_info",
            handler=self._getProjectInfo,
            description="Get current project information",
            parameters_schema={
                "type": "object",
                "properties": {
                    "include_stats": {
                        "type": "boolean",
                        "description": "Include project statistics"
                    }
                },
                "required": []
            },
            is_async=False,
            category="project"
        )
        
        self._registry.registerTool(
            name="list_documents",
            handler=self._listDocuments,
            description="List documents in the project",
            parameters_schema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["all", "novel", "notes", "trash"],
                        "description": "Document scope to list"
                    }
                },
                "required": ["scope"]
            },
            is_async=False,
            category="document"
        )
        
        self._registry.registerTool(
            name="read_document",
            handler=self._readDocument,
            description="Read document content",
            parameters_schema={
                "type": "object",
                "properties": {
                    "item_handle": {
                        "type": "string",
                        "description": "Document handle identifier"
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include document metadata"
                    }
                },
                "required": ["item_handle"]
            },
            is_async=False,
            category="document"
        )
        
        self._registry.registerTool(
            name="write_document",
            handler=self._writeDocument,
            description="Write content to a document",
            parameters_schema={
                "type": "object",
                "properties": {
                    "item_handle": {
                        "type": "string",
                        "description": "Document handle identifier"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Append to existing content"
                    }
                },
                "required": ["item_handle", "content"]
            },
            is_async=False,
            category="document",
            required_permissions={"write"}
        )
        
        self._registry.registerTool(
            name="search_content",
            handler=self._searchContent,
            description="Search for content in the project",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "novel", "notes"],
                        "description": "Search scope"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case-sensitive search"
                    }
                },
                "required": ["query"]
            },
            is_async=False,
            category="search"
        )
        
        logger.info(f"Registered {len(self._registry.listTools())} built-in local tools")
    
    async def executeTool(
        self,
        name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a local tool.
        
        Args:
            name: Tool name
            parameters: Tool parameters
        
        Returns:
            Tool execution result
        
        Raises:
            MCPToolNotFoundError: If tool not found
            MCPValidationError: If parameters invalid
            MCPExecutionError: If execution fails
        """
        # Check if tool exists
        if not self._registry.hasTool(name):
            raise MCPToolNotFoundError(f"Tool '{name}' not found")
        
        # Get tool registration
        registration = self._registry.getTool(name)
        
        # Check if tool is enabled
        if not registration.enabled:
            raise MCPExecutionError(f"Tool '{name}' is disabled")
        
        # Validate parameters
        if registration.validator:
            try:
                registration.validator(parameters)
            except Exception as e:
                raise MCPValidationError(f"Parameter validation failed: {e}")
        
        # Execute tool with timing
        start_time = time.perf_counter()
        success = False
        error_msg = None
        result = None
        
        try:
            # Execute handler
            handler = registration.handler
            if registration.metadata.is_async:
                result = await handler(**parameters)
            else:
                # Run sync handler in executor to avoid blocking
                loop = asyncio.get_event_loop()
                from functools import partial
                result = await loop.run_in_executor(None, partial(handler, **parameters))
            
            success = True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool '{name}' execution failed: {e}")
            raise MCPExecutionError(f"Tool execution failed: {e}")
        
        finally:
            # Record statistics
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            self._registry.recordToolCall(name, success, error_msg)
        
        return {
            "result": result,
            "execution_time_ms": execution_time_ms
        }
    
    def createToolHandler(
        self,
        api_method: str,
        transform_params: Optional[Callable] = None,
        transform_result: Optional[Callable] = None
    ) -> Callable:
        """Create a tool handler from a NovelWriterAPI method.
        
        Args:
            api_method: Name of the NovelWriterAPI method
            transform_params: Optional function to transform parameters
            transform_result: Optional function to transform result
        
        Returns:
            Tool handler function
        """
        def handler(**kwargs):
            # Get API method
            method = getattr(self._nw_api, api_method)
            
            # Transform parameters if needed
            if transform_params:
                kwargs = transform_params(kwargs)
            
            # Call API method
            result = method(**kwargs)
            
            # Transform result if needed
            if transform_result:
                result = transform_result(result)
            
            return result
        
        return handler
    
    # Built-in tool implementations
    
    def _getProjectInfo(self, include_stats: bool = False) -> Dict[str, Any]:
        """Get project information"""
        info = self._nw_api.getProjectMeta()
        
        if include_stats:
            stats = self._nw_api.getProjectStatistics()
            info["statistics"] = stats
        
        return info
    
    def _listDocuments(self, scope: str) -> List[Dict[str, Any]]:
        """List documents in the project"""
        documents = self._nw_api.listDocuments(scope)
        return [doc.model_dump() if hasattr(doc, 'model_dump') else doc for doc in documents]
    
    def _readDocument(self, item_handle: str, include_metadata: bool = False) -> Dict[str, Any]:
        """Read document content"""
        content = self._nw_api.getDocText(item_handle)
        
        result = {"content": content}
        
        if include_metadata:
            meta = self._nw_api.getDocumentMetadata(item_handle)
            result["metadata"] = meta
        
        return result
    
    def _writeDocument(self, item_handle: str, content: str, append: bool = False) -> Dict[str, Any]:
        """Write document content"""
        if append:
            current = self._nw_api.getDocText(item_handle)
            content = current + "\n" + content
        
        success = self._nw_api.setDocText(item_handle, content)
        
        return {
            "success": success,
            "item_handle": item_handle,
            "content_length": len(content)
        }
    
    def _searchContent(
        self,
        query: str,
        scope: str = "all",
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for content in the project"""
        results = self._nw_api.searchContent(
            query=query,
            scope=scope,
            case_sensitive=case_sensitive
        )
        
        return [
            {
                "item_handle": result.get("handle"),
                "title": result.get("title"),
                "matches": result.get("matches", [])
            }
            for result in results
        ]
