"""Data transfer objects shared across the AI Copilot domain layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "DocumentRef",
    "TextRange",
    "Suggestion",
    "BuildResult",
    "ModelInfo",
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
class ModelInfo:
    """Normalised provider model information for UI consumption."""

    id: str
    display_name: str
    description: Optional[str]
    input_token_limit: Optional[int]
    output_token_limit: Optional[int]
    owned_by: Optional[str]
    capabilities: Any
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Expose the model record as a serialisable mapping."""

        payload: dict[str, Any] = {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "input_token_limit": self.input_token_limit,
            "output_token_limit": self.output_token_limit,
            "owned_by": self.owned_by,
            "capabilities": self.capabilities,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class BuildResult:
    """Outcome metadata for build/export operations triggered by the AI Copilot."""

    format: str
    outputPath: str
    success: bool
    message: Optional[str] = None
