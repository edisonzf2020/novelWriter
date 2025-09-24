"""novelWriter – API Module.
========================

The API module provides a unified data access layer for all novelWriter components.
This module abstracts core project data access through a clean, consistent API.

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

from .novelwriter_api import NovelWriterAPI
from .exceptions import (
    APIError,
    APIValidationError,
    APIPermissionError,
    APINotFoundError,
    APIOperationError
)

__all__ = [
    "APIError",
    "APINotFoundError",
    "APIOperationError",
    "APIPermissionError",
    "APIValidationError",
    "NovelWriterAPI",
]

logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# API singleton instance (will be initialized by the application)
_api_instance: NovelWriterAPI | None = None


def getAPI() -> NovelWriterAPI | None:
    """Get the singleton API instance if it exists."""
    return _api_instance


def setAPI(api: NovelWriterAPI) -> None:
    """Set the singleton API instance."""
    global _api_instance
    _api_instance = api
    logger.debug("NovelWriter API instance registered")


def clearAPI() -> None:
    """Clear the singleton API instance."""
    global _api_instance
    _api_instance = None
    logger.debug("NovelWriter API instance cleared")
