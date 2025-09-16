"""Domain API facade exposed to the AI Copilot runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional
from types import MappingProxyType

from novelwriter.enum import nwItemClass, nwItemLayout

from .errors import NWAiApiError
from .models import BuildResult, DocumentRef, Suggestion, TextRange

__all__ = ["NWAiApi"]

_SCOPE_CLASS_MAP: Mapping[str, frozenset[nwItemClass]] = MappingProxyType(
    {
        "novel": frozenset({nwItemClass.NOVEL}),
        "plot": frozenset({nwItemClass.PLOT}),
        "character": frozenset({nwItemClass.CHARACTER}),
        "world": frozenset({nwItemClass.WORLD}),
        "timeline": frozenset({nwItemClass.TIMELINE}),
        "object": frozenset({nwItemClass.OBJECT}),
        "entity": frozenset({nwItemClass.ENTITY}),
        "custom": frozenset({nwItemClass.CUSTOM}),
        "archive": frozenset({nwItemClass.ARCHIVE}),
        "template": frozenset({nwItemClass.TEMPLATE}),
        "trash": frozenset({nwItemClass.TRASH}),
        "outline": frozenset({nwItemClass.PLOT, nwItemClass.TIMELINE}),
    }
)
_SCOPE_LAYOUT_MAP: Mapping[str, nwItemLayout] = MappingProxyType(
    {
        "document": nwItemLayout.DOCUMENT,
        "documents": nwItemLayout.DOCUMENT,
        "note": nwItemLayout.NOTE,
        "notes": nwItemLayout.NOTE,
    }
)

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
    def getProjectMeta(self) -> Mapping[str, Any]:
        """Fetch read-only project metadata tailored for the AI Copilot."""

        project = self._project
        data = project.data
        novel_words, note_words, novel_chars, note_chars = data.currCounts
        meta: dict[str, Any] = {
            "uuid": data.uuid,
            "name": data.name,
            "author": data.author,
            "language": data.language,
            "spellCheck": data.spellCheck,
            "spellLanguage": data.spellLang,
            "openedAt": project.projOpened,
            "saveCount": data.saveCount,
            "autoSaveCount": data.autoCount,
            "editTime": data.editTime,
            "totalWords": novel_words + note_words,
            "totalCharacters": novel_chars + note_chars,
            "novelWords": novel_words,
            "noteWords": note_words,
            "novelCharacters": novel_chars,
            "noteCharacters": note_chars,
            "currentTotalCount": project.currentTotalCount,
            "projectState": project.state.name,
            "projectChanged": project.projChanged,
            "isValid": project.isValid,
            "lastHandles": dict(data.lastHandle),
        }
        return MappingProxyType(meta)

    def listDocuments(self, scope: str = "all") -> list[DocumentRef]:
        """List project items visible to the AI Copilot for the given scope."""

        scope_key = (scope or "all").strip().lower()
        if not scope_key:
            scope_key = "all"

        class_filter = _SCOPE_CLASS_MAP.get(scope_key)
        layout_filter = _SCOPE_LAYOUT_MAP.get(scope_key)
        if scope_key not in ("all",) and class_filter is None and layout_filter is None:
            raise NWAiApiError(f"Unknown document scope: '{scope}'.")

        documents: list[DocumentRef] = []
        tree = self._project.tree
        for item in tree:
            if not item.isFileType() or not item.isActive:
                continue
            if class_filter is not None and item.itemClass not in class_filter:
                continue
            if layout_filter is not None and item.itemLayout != layout_filter:
                continue
            documents.append(
                DocumentRef(
                    handle=item.itemHandle,
                    name=item.itemName,
                    parent=item.itemParent,
                )
            )
        return documents

    def getCurrentDocument(self) -> Optional[DocumentRef]:
        """Return the current focus document for the AI Copilot session."""

        handle = self._project.data.lastHandle.get("editor")
        if not handle:
            return None

        tree = self._project.tree
        if handle not in tree:
            return None

        item = tree[handle]
        if item is None or not item.isFileType() or not item.isActive:
            return None

        return DocumentRef(handle=item.itemHandle, name=item.itemName, parent=item.itemParent)

    # ------------------------------------------------------------------
    # Text access
    # ------------------------------------------------------------------
    def getDocText(self, handle: str) -> str:
        """Retrieve the text content of the document identified by ``handle``."""

        if not isinstance(handle, str) or not handle.strip():
            raise NWAiApiError("Document handle must be a non-empty string.")

        lookup = handle.strip()
        tree = self._project.tree
        if lookup not in tree:
            raise NWAiApiError(f"Unknown document handle: '{handle}'.")

        item = tree[lookup]
        if item is None or not item.isFileType():
            raise NWAiApiError(f"Handle '{handle}' does not reference a document.")
        if not item.isActive:
            raise NWAiApiError(f"Document '{handle}' is not active.")

        content_path = self._project.storage.contentPath
        if content_path is None:
            raise NWAiApiError("Project storage is not ready.")
        if not (content_path / f"{lookup}.nwd").is_file():
            raise NWAiApiError(f"Document file is missing for handle '{handle}'.")

        return self._project.storage.getDocumentText(lookup)

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
