"""Domain API facade exposed to the AI Copilot runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .models import BuildResult, DocumentRef, Suggestion, TextRange

__all__ = ["NWAiApi"]

if TYPE_CHECKING:  # pragma: no cover - hints only
    from novelwriter.core.project import NWProject


class NWAiApi:
    """AI-facing facade encapsulating safe interactions with novelWriter data."""

    def __init__(self, project: "NWProject") -> None:
        """Create the API facade for a given project context."""

        self._project = project

    # ------------------------------------------------------------------
    # Transaction and auditing
    # ------------------------------------------------------------------
    def begin_transaction(self) -> str:
        """Open a new AI-controlled transaction scope and return its identifier."""

        raise NotImplementedError("Transaction management is pending implementation.")

    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit the given transaction and persist all pending changes."""

        raise NotImplementedError("Transaction management is pending implementation.")

    def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback the given transaction and discard pending changes."""

        raise NotImplementedError("Transaction management is pending implementation.")

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the accumulated AI audit events for inspection."""

        raise NotImplementedError("Audit logging is pending implementation.")

    # ------------------------------------------------------------------
    # Project and document access
    # ------------------------------------------------------------------
    def getProjectMeta(self) -> dict[str, Any]:
        """Fetch read-only project metadata tailored for the AI Copilot."""

        raise NotImplementedError("Project metadata access is pending implementation.")

    def listDocuments(self, scope: str = "all") -> list[DocumentRef]:
        """List project items visible to the AI Copilot for the given scope."""

        raise NotImplementedError("Document listing is pending implementation.")

    def getCurrentDocument(self) -> Optional[DocumentRef]:
        """Return the current focus document for the AI Copilot session."""

        raise NotImplementedError("Current document lookup is pending implementation.")

    # ------------------------------------------------------------------
    # Text access
    # ------------------------------------------------------------------
    def getDocText(self, handle: str) -> str:
        """Retrieve the text content of the document identified by ``handle``."""

        raise NotImplementedError("Document text retrieval is pending implementation.")

    def setDocText(self, handle: str, text: str, apply: bool = False) -> bool:
        """Apply or preview a text replacement for the given document handle."""

        raise NotImplementedError("Document text updates are pending implementation.")

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    def previewSuggestion(self, handle: str, rng: TextRange, newText: str) -> Suggestion:
        """Generate a preview suggestion for replacing ``rng`` in ``handle`` with ``newText``."""

        raise NotImplementedError("Suggestion previews are pending implementation.")

    def applySuggestion(self, suggestionId: str) -> bool:
        """Apply a previously generated suggestion if it remains valid."""

        raise NotImplementedError("Suggestion application is pending implementation.")

    # ------------------------------------------------------------------
    # Context and search utilities
    # ------------------------------------------------------------------
    def collectContext(self, mode: str) -> str:
        """Collect contextual text for the AI Copilot using the selected ``mode``."""

        raise NotImplementedError("Context collection is pending implementation.")

    def search(self, query: str, scope: str = "document", limit: int = 50) -> list[str]:
        """Search within project resources on behalf of the AI Copilot."""

        raise NotImplementedError("Search functionality is pending implementation.")

    # ------------------------------------------------------------------
    # Structured operations (future phases)
    # ------------------------------------------------------------------
    def createChapter(self, name: str, parent: Optional[str]) -> Optional[DocumentRef]:
        """Create a chapter under ``parent`` and return a document reference."""

        raise NotImplementedError("Chapter creation is scheduled for a later phase.")

    def createScene(self, name: str, parent: str, after: Optional[str] = None) -> Optional[DocumentRef]:
        """Create a scene item after ``after`` under ``parent`` and return a reference."""

        raise NotImplementedError("Scene creation is scheduled for a later phase.")

    def duplicateItem(
        self,
        handle: str,
        parent: Optional[str] = None,
        after: bool = True,
    ) -> Optional[DocumentRef]:
        """Duplicate an existing item and optionally reparent or reorder the clone."""

        raise NotImplementedError("Item duplication is scheduled for a later phase.")

    # ------------------------------------------------------------------
    # Build / export support
    # ------------------------------------------------------------------
    def build(self, fmt: str, options: Optional[dict[str, Any]] = None) -> BuildResult:
        """Trigger a project build in ``fmt`` format and return a result summary."""

        raise NotImplementedError("Build operations are pending implementation.")
