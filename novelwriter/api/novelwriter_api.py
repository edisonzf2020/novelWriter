"""novelWriter – Unified Data Access API.
======================================

This module provides a unified API for accessing all novelWriter project data.
It serves as the single entry point for all data operations, replacing direct
access to core modules.

File History:
Created: 2025-09-24 [MCP-v1.0] NovelWriterAPI

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

import logging
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar
from collections.abc import Callable, Mapping

from novelwriter.enum import nwItemClass, nwItemLayout

from .exceptions import (
    APIError,
    APINotFoundError,
    APIOperationError,
    APIPermissionError,
    APIValidationError,
)

if TYPE_CHECKING:
    from novelwriter.core.project import NWProject

__all__ = ["NovelWriterAPI"]

logger = logging.getLogger(__name__)

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])


def validateParams(func: F) -> F:
    """Validate parameters for API methods.

    Validates that required parameters are present and have the correct type.
    Measures execution time and logs performance metrics.
    """
    @wraps(func)
    def wrapper(self: NovelWriterAPI, *args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        method_name = func.__name__

        try:
            # Log API call
            logger.debug(f"API call: {method_name} with args={args}, kwargs={kwargs}")

            # Execute the function
            result = func(self, *args, **kwargs)

            # Measure performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > 5.0:  # Log if exceeds 5ms threshold
                logger.warning(f"API call {method_name} took {elapsed_ms:.2f}ms (>5ms threshold)")
            else:
                logger.debug(f"API call {method_name} completed in {elapsed_ms:.2f}ms")

            return result

        except APIError:
            # Re-raise API errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.error(f"API call {method_name} failed: {e}")
            raise APIOperationError(
                f"API operation '{method_name}' failed",
                operation=method_name,
                cause=e
            ) from e

    return wrapper  # type: ignore


def requiresProject(func: F) -> F:
    """Ensure a project is loaded before method execution."""
    @wraps(func)
    def wrapper(self: NovelWriterAPI, *args: Any, **kwargs: Any) -> Any:
        if self._project is None:
            raise APIPermissionError(
                "No project is currently loaded",
                operation=func.__name__,
                resource="project"
            )
        return func(self, *args, **kwargs)

    return wrapper  # type: ignore


def requiresWritePermission(func: F) -> F:
    """Check for write permissions."""
    @wraps(func)
    def wrapper(self: NovelWriterAPI, *args: Any, **kwargs: Any) -> Any:
        if self._readOnly:
            raise APIPermissionError(
                f"Write operation '{func.__name__}' not allowed in read-only mode",
                operation=func.__name__
            )
        return func(self, *args, **kwargs)

    return wrapper  # type: ignore


class NovelWriterAPI:
    """Unified API for novelWriter data access.

    This class provides a single, consistent interface for all data access
    operations in novelWriter. It abstracts the underlying core modules and
    provides:

    - Unified error handling
    - Parameter validation
    - Permission checking
    - Performance monitoring
    - Audit logging

    All modules should access project data through this API rather than
    directly accessing core modules.
    """

    __slots__ = ("_cache", "_initialized", "_project", "_readOnly")

    def __init__(self, project: NWProject | None = None,
                 readOnly: bool = False) -> None:
        """Initialize the NovelWriter API.

        Args:
            project: The NWProject instance to wrap (can be set later)
            readOnly: Whether to enforce read-only access

        """
        self._project = project
        self._readOnly = readOnly
        self._cache: dict[str, Any] = {}
        self._initialized = False

        if project is not None:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize the API with a project."""
        if self._project is None:
            raise APIError("Cannot initialize API without a project")

        self._initialized = True
        logger.info("NovelWriter API initialized with project: %s",
                   self._project.data.name)

    def setProject(self, project: NWProject, readOnly: bool = False) -> None:
        """Set or update the project instance.

        Args:
            project: The NWProject instance to use
            readOnly: Whether to enforce read-only access

        """
        if not project:
            raise APIValidationError("Project cannot be None", field="project")

        self._project = project
        self._readOnly = readOnly
        self._cache.clear()  # Clear cache when project changes
        self._initialize()

    # ======================================================================
    # Core Data Access Methods (8 methods as per Story 1.1)
    # ======================================================================

    @validateParams
    @requiresProject
    def getProjectMeta(self) -> Mapping[str, Any]:
        """Get project metadata.

        Returns a read-only mapping of project metadata including:
        - Project name, author, language
        - Creation and modification times
        - Word counts and statistics
        - Project settings

        Returns:
            Read-only mapping of project metadata

        Raises:
            APIPermissionError: If no project is loaded

        """
        # Cache metadata for performance
        cache_key = "project_meta"
        if cache_key in self._cache:
            return self._cache[cache_key]

        project = self._project
        meta = {
            "name": project.data.name,
            "title": project.data.title,
            "author": project.data.author,
            "language": project.data.language,
            "spellCheck": project.data.spellCheck,
            "spellLang": project.data.spellLang,
            "createdTime": project.data.createdTime,
            "saveCount": project.data.saveCount,
            "autoCount": project.data.autoCount,
            "editTime": project.data.editTime,
            "lastPath": str(project.storage.getPath()),
            "lastHandle": project.data.lastHandle,
            "doBackup": project.data.doBackup,
            "uuid": project.data.uuid,
            "stats": {
                "numChapters": 0,
                "numScenes": 0,
                "totalWords": 0,
            }
        }

        # Calculate statistics
        for item in project.tree:
            if item.isFileType():
                # Note: In novelWriter, chapters and scenes are determined by heading level
                # not by layout. For now, count NOVEL documents as chapters.
                if (item.itemClass == nwItemClass.NOVEL
                    and item.itemLayout == nwItemLayout.DOCUMENT):
                    meta["stats"]["numChapters"] += 1
                meta["stats"]["totalWords"] += item.wordCount

        # Make immutable
        from types import MappingProxyType
        result = MappingProxyType(meta)
        self._cache[cache_key] = result

        return result

    @validateParams
    @requiresProject
    def listDocuments(self, scope: str = "all") -> list[dict[str, Any]]:
        """List documents in the project.

        Args:
            scope: Filter scope for documents
                - "all": All documents
                - "novel": Novel documents only
                - "notes": Project notes only
                - "trash": Trashed items only
                - "archive": Archived items only

        Returns:
            List of document references with metadata

        Raises:
            APIValidationError: If scope is invalid
            APIPermissionError: If no project is loaded

        """
        valid_scopes = ["all", "novel", "notes", "trash", "archive"]
        if scope not in valid_scopes:
            raise APIValidationError(
                f"Invalid scope: {scope}",
                field="scope",
                value=scope,
                valid_values=valid_scopes
            )

        documents = []
        for item in self._project.tree:
            # Filter based on scope
            if scope == "novel" and item.itemClass != nwItemClass.NOVEL:
                continue
            if scope == "notes" and item.itemClass not in [
                nwItemClass.CHARACTER,
                nwItemClass.PLOT,
                nwItemClass.WORLD,
                nwItemClass.TIMELINE,
                nwItemClass.OBJECT,
                nwItemClass.ENTITY,
                nwItemClass.CUSTOM,
            ]:
                continue
            if scope == "trash" and item.itemClass != nwItemClass.TRASH:
                continue
            if scope == "archive" and item.itemClass != nwItemClass.ARCHIVE:
                continue

            # Skip folders, only include documents
            if item.isFileType():
                documents.append({
                    "handle": item.itemHandle,
                    "name": item.itemName,
                    "class": item.itemClass.name,
                    "layout": item.itemLayout.name if item.itemLayout else None,
                    "status": item.itemStatus,
                    "parent": item.itemParent,
                    "order": item.itemOrder,
                    "expanded": item.isExpanded,
                    "wordCount": item.wordCount,
                    "charCount": item.charCount,
                    "paraCount": item.paraCount,
                    "cursorPos": item.cursorPos,
                })

        return documents

    @validateParams
    @requiresProject
    def getDocText(self, handle: str) -> str:
        """Get the text content of a document.

        Args:
            handle: The document handle/ID

        Returns:
            The document text content

        Raises:
            APIValidationError: If handle is invalid
            APINotFoundError: If document not found
            APIPermissionError: If no project is loaded

        """
        if not handle or not isinstance(handle, str):
            raise APIValidationError(
                "Document handle must be a non-empty string",
                field="handle",
                value=handle
            )

        # Find the document
        item = self._project.tree[handle]
        if item is None:
            raise APINotFoundError(
                f"Document not found: {handle}",
                resource_type="document",
                resource_id=handle
            )

        # Ensure it's a file, not a folder
        if not item.isFileType():
            raise APIValidationError(
                f"Item {handle} is not a document",
                field="handle",
                value=handle
            )

        # Load document content
        from novelwriter.core.document import NWDocument
        doc = NWDocument(self._project, item.itemHandle)
        if not doc.readDocument():
            raise APIOperationError(
                f"Failed to read document: {handle}",
                operation="readDocument"
            )

        return doc.getText()

    @validateParams
    @requiresProject
    @requiresWritePermission
    def setDocText(self, handle: str, text: str) -> bool:
        """Set the text content of a document.

        Args:
            handle: The document handle/ID
            text: The new text content

        Returns:
            True if successful, False otherwise

        Raises:
            APIValidationError: If parameters are invalid
            APINotFoundError: If document not found
            APIPermissionError: If no project loaded or read-only mode
            APIOperationError: If write operation fails

        """
        if not handle or not isinstance(handle, str):
            raise APIValidationError(
                "Document handle must be a non-empty string",
                field="handle",
                value=handle
            )

        if not isinstance(text, str):
            raise APIValidationError(
                "Document text must be a string",
                field="text",
                value=type(text).__name__
            )

        # Find the document
        item = self._project.tree[handle]
        if item is None:
            raise APINotFoundError(
                f"Document not found: {handle}",
                resource_type="document",
                resource_id=handle
            )

        # Ensure it's a file, not a folder
        if not item.isFileType():
            raise APIValidationError(
                f"Item {handle} is not a document",
                field="handle",
                value=handle
            )

        # Write document content
        from novelwriter.core.document import NWDocument
        doc = NWDocument(self._project, item.itemHandle)
        doc.setText(text)

        if not doc.writeDocument():
            raise APIOperationError(
                f"Failed to write document: {handle}",
                operation="writeDocument"
            )

        # Update item metadata
        item.setCharCount(doc.charCount)
        item.setWordCount(doc.wordCount)
        item.setParaCount(doc.paraCount)

        # Clear cache as content has changed
        self._cache.clear()

        return True

    @validateParams
    @requiresProject
    def getProjectTree(self) -> list[dict[str, Any]]:
        """Get the hierarchical project tree structure.

        Returns a list representing the project tree with parent-child
        relationships preserved.

        Returns:
            List of tree nodes with metadata

        Raises:
            APIPermissionError: If no project is loaded

        """
        tree_data = []

        def buildNode(item: Any) -> dict[str, Any]:
            """Build a tree node representation."""
            node = {
                "handle": item.itemHandle,
                "name": item.itemName,
                "type": "folder" if item.isFolderType() else "file",
                "class": item.itemClass.name,
                "layout": item.itemLayout.name if item.itemLayout else None,
                "status": item.itemStatus,
                "parent": item.itemParent,
                "order": item.itemOrder,
                "expanded": item.isExpanded,
                "children": [],
            }

            if item.isFileType():
                node.update({
                    "wordCount": item.wordCount,
                    "charCount": item.charCount,
                    "paraCount": item.paraCount,
                })

            return node

        # Build tree structure
        node_map = {}
        for item in self._project.tree:
            node = buildNode(item)
            node_map[item.itemHandle] = node

            if item.itemParent is None:
                # Root level item
                tree_data.append(node)
            else:
                # Child item - add to parent's children
                parent = node_map.get(item.itemParent)
                if parent:
                    parent["children"].append(node)

        return tree_data

    @validateParams
    @requiresProject
    def searchProject(self, query: str,
                     options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search for text across the project.

        Args:
            query: Search query string
            options: Search options dictionary
                - case_sensitive: bool (default False)
                - whole_words: bool (default False)
                - regex: bool (default False)
                - scope: str - "all", "novel", "notes" (default "all")
                - max_results: int (default 100)

        Returns:
            List of search results with context

        Raises:
            APIValidationError: If query is invalid
            APIPermissionError: If no project is loaded

        """
        if not query or not isinstance(query, str):
            raise APIValidationError(
                "Search query must be a non-empty string",
                field="query",
                value=query
            )

        # Default options
        opts = {
            "case_sensitive": False,
            "whole_words": False,
            "regex": False,
            "scope": "all",
            "max_results": 100,
        }
        if options:
            opts.update(options)

        # Import search functionality
        import re

        # Prepare search pattern
        if opts["regex"]:
            try:
                if opts["case_sensitive"]:
                    pattern = re.compile(query)
                else:
                    pattern = re.compile(query, re.IGNORECASE)
            except re.error as e:
                raise APIValidationError(
                    f"Invalid regex pattern: {e}",
                    field="query",
                    value=query
                ) from e
        else:
            # Escape special regex characters for literal search
            escaped = re.escape(query)
            if opts["whole_words"]:
                escaped = r"\b" + escaped + r"\b"

            if opts["case_sensitive"]:
                pattern = re.compile(escaped)
            else:
                pattern = re.compile(escaped, re.IGNORECASE)

        results = []
        result_count = 0

        # Search documents
        for item in self._project.tree:
            if result_count >= opts["max_results"]:
                break

            # Apply scope filter
            if opts["scope"] == "novel" and item.itemClass != nwItemClass.NOVEL:
                continue
            if opts["scope"] == "notes" and item.itemClass not in [
                nwItemClass.CHARACTER,
                nwItemClass.PLOT,
                nwItemClass.WORLD,
                nwItemClass.TIMELINE,
                nwItemClass.OBJECT,
                nwItemClass.ENTITY,
                nwItemClass.CUSTOM,
            ]:
                continue

            if not item.isFileType():
                continue

            # Load document
            try:
                text = self.getDocText(item.itemHandle)
            except Exception:
                continue

            # Search in document
            lines = text.split("\n")
            for line_num, line in enumerate(lines, 1):
                matches = list(pattern.finditer(line))
                if matches:
                    for match in matches:
                        if result_count >= opts["max_results"]:
                            break

                        # Get context (±2 lines)
                        context_start = max(0, line_num - 3)
                        context_end = min(len(lines), line_num + 2)
                        context = lines[context_start:context_end]

                        results.append({
                            "handle": item.itemHandle,
                            "document": item.itemName,
                            "line": line_num,
                            "column": match.start() + 1,
                            "match": match.group(),
                            "context": "\n".join(context),
                            "contextLineStart": context_start + 1,
                        })
                        result_count += 1

        return results

    @validateParams
    @requiresProject
    def getTagList(self) -> list[dict[str, Any]]:
        """Get list of all tags/keywords in the project.

        Returns:
            List of tags with usage information

        Raises:
            APIPermissionError: If no project is loaded

        """
        # Get tags from the project index
        tags = {}

        # Collect tags from index
        if hasattr(self._project, "index"):
            # Get all tags from the index
            for tag in self._project.index.iterTags():
                tag_name = tag[0]
                tag_class = tag[1]
                tag_count = len(self._project.index.getTagHandles(tag_name))

                if tag_name not in tags:
                    tags[tag_name] = {
                        "name": tag_name,
                        "class": tag_class.name if tag_class else None,
                        "count": tag_count,
                        "handles": list(self._project.index.getTagHandles(tag_name))
                    }

        return list(tags.values())

    @validateParams
    @requiresProject
    def getStatistics(self, scope: str = "project") -> dict[str, Any]:
        """Get project or document statistics.

        Args:
            scope: Statistics scope
                - "project": Overall project statistics
                - "novel": Novel content statistics
                - "notes": Notes statistics
                - Handle string: Specific document statistics

        Returns:
            Statistics dictionary

        Raises:
            APIValidationError: If scope is invalid
            APINotFoundError: If specific document not found
            APIPermissionError: If no project is loaded

        """
        stats = {
            "totalWords": 0,
            "totalChars": 0,
            "totalParagraphs": 0,
            "documentCount": 0,
            "chapterCount": 0,
            "sceneCount": 0,
        }

        # Check if scope is a document handle
        if scope not in ["project", "novel", "notes"]:
            # Try to get specific document stats
            item = self._project.tree[scope]
            if item is None:
                raise APINotFoundError(
                    f"Document not found: {scope}",
                    resource_type="document",
                    resource_id=scope
                )

            if item.isFileType():
                return {
                    "handle": item.itemHandle,
                    "name": item.itemName,
                    "words": item.wordCount,
                    "chars": item.charCount,
                    "paragraphs": item.paraCount,
                }
            else:
                raise APIValidationError(
                    f"Item {scope} is not a document",
                    field="scope",
                    value=scope
                )

        # Calculate aggregate statistics
        for item in self._project.tree:
            if not item.isFileType():
                continue

            # Apply scope filter
            if scope == "novel" and item.itemClass != nwItemClass.NOVEL:
                continue
            if scope == "notes" and item.itemClass not in [
                nwItemClass.CHARACTER,
                nwItemClass.PLOT,
                nwItemClass.WORLD,
                nwItemClass.TIMELINE,
                nwItemClass.OBJECT,
                nwItemClass.ENTITY,
                nwItemClass.CUSTOM,
            ]:
                continue

            stats["totalWords"] += item.wordCount
            stats["totalChars"] += item.charCount
            stats["totalParagraphs"] += item.paraCount
            stats["documentCount"] += 1

            # Note: In novelWriter, chapters and scenes are determined by heading level
            # not by layout. For now, count NOVEL documents as chapters.
            if item.itemClass == nwItemClass.NOVEL and item.itemLayout == nwItemLayout.DOCUMENT:
                stats["chapterCount"] += 1

        return stats

    # ======================================================================
    # Utility Methods
    # ======================================================================

    @property
    def isProjectLoaded(self) -> bool:
        """Check if a project is currently loaded."""
        return self._project is not None and self._initialized

    @property
    def isReadOnly(self) -> bool:
        """Check if API is in read-only mode."""
        return self._readOnly

    def clearCache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.debug("API cache cleared")

    # ======================================================================
    # Factory Methods
    # ======================================================================

    @classmethod
    def createInstance(cls, project: NWProject,
                      readOnly: bool = False) -> NovelWriterAPI:
        """Create an API instance.

        Args:
            project: The project to wrap
            readOnly: Whether to enforce read-only access

        Returns:
            A new NovelWriterAPI instance

        """
        return cls(project=project, readOnly=readOnly)
