"""Data transfer objects shared across the AI Copilot domain layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DocumentRef",
    "TextRange",
    "Suggestion",
    "BuildResult",
]


@dataclass
class DocumentRef:
    """Reference to a project item that the AI Copilot can safely address."""

    handle: str
    name: str
    parent: Optional[str]


@dataclass
class TextRange:
    """Inclusive text range expressed as character offsets in a document."""

    start: int
    end: int


@dataclass
class Suggestion:
    """Description of a pending AI suggestion and any preview data."""

    id: str
    handle: str
    preview: str
    diff: Optional[str]


@dataclass
class BuildResult:
    """Outcome metadata for build/export operations triggered by the AI Copilot."""

    format: str
    outputPath: str
    success: bool
    message: Optional[str] = None
