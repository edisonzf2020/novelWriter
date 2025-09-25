"""novelWriter – MCP Tools Module.
===============================

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

from novelwriter.api.tools.registry import ToolRegistry
from novelwriter.api.tools.local_tools import LocalToolWrapper

# Import base classes
from novelwriter.api.tools.base import (
    BaseTool, ToolMetadata, ToolPermission, ToolExecutionResult
)

# Import project tools
from novelwriter.api.tools.project_tools import (
    ProjectInfoTool, ProjectTreeTool
)

# Import document tools
from novelwriter.api.tools.document_tools import (
    DocumentListTool, DocumentReadTool, DocumentWriteTool, CreateDocumentTool
)

# Import search tools
from novelwriter.api.tools.search_tools import (
    GlobalSearchTool, TagListTool, ProjectStatsTool
)

__all__ = [
    # Base classes
    "BaseTool",
    "CreateDocumentTool",
    # Document tools
    "DocumentListTool",
    "DocumentReadTool",
    "DocumentWriteTool",
    # Search tools
    "GlobalSearchTool",
    "LocalToolWrapper",
    # Project tools
    "ProjectInfoTool",
    "ProjectStatsTool",
    "ProjectTreeTool",
    "TagListTool",
    "ToolExecutionResult",
    "ToolMetadata",
    "ToolPermission",
    # Registry and wrapper
    "ToolRegistry"
]
