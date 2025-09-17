"""Domain API facade exposed to the AI Copilot runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional
from types import MappingProxyType
from uuid import uuid4

from novelwriter.enum import nwItemClass, nwItemLayout

from .errors import NWAiApiError
from .models import BuildResult, DocumentRef, Suggestion, TextRange

__all__ = ["NWAiApi"]


@dataclass
class _PendingOperation:
    """State descriptor for a pending write prepared inside a transaction."""

    operation: str
    target: Optional[str]
    summary: Optional[str]
    undo: Optional[Callable[[], None]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TransactionContext:
    """Represents a single transaction frame, allowing nesting support."""

    transaction_id: str
    pending_operations: list[_PendingOperation] = field(default_factory=list)


@dataclass(frozen=True)
class _AuditRecord:
    """Immutable audit trail entry for AI-triggered activity."""

    timestamp: datetime
    transaction_id: Optional[str]
    operation: str
    target: Optional[str]
    summary: Optional[str]
    level: str = "info"

    def as_dict(self) -> dict[str, Any]:
        """Serialise the audit entry into a JSON-friendly mapping."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "target": self.target,
            "summary": self.summary,
            "level": self.level,
        }


_AUDIT_LOG_LIMIT = 1000

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
        self._transaction_stack: list[_TransactionContext] = []
        self._audit_log: deque[_AuditRecord] = deque(maxlen=_AUDIT_LOG_LIMIT)

    # ------------------------------------------------------------------
    # Transaction and auditing
    # ------------------------------------------------------------------
    def begin_transaction(self) -> str:
        """Open a new AI-controlled transaction scope and return its identifier."""

        transaction_id = (
            self._transaction_stack[0].transaction_id if self._transaction_stack else uuid4().hex
        )
        context = _TransactionContext(transaction_id=transaction_id)
        self._transaction_stack.append(context)
        depth = len(self._transaction_stack)
        self._record_audit(transaction_id, "transaction.begin", summary=f"depth={depth}")
        return transaction_id

    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit the given transaction and persist all pending changes."""

        context = self._pop_transaction_frame(transaction_id, action="commit")
        if self._transaction_stack:
            self._transaction_stack[-1].pending_operations.extend(context.pending_operations)
            depth = len(self._transaction_stack) + 1
            self._record_audit(
                transaction_id,
                "transaction.commit.nested",
                summary=f"depth={depth}",
            )
            return True

        self._record_pending_operations(
            transaction_id,
            context.pending_operations,
            success=True,
        )
        self._record_audit(
            transaction_id,
            "transaction.commit",
            summary=f"{len(context.pending_operations)} operation(s) committed.",
        )
        return True

    def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback the given transaction and discard pending changes."""

        context = self._pop_transaction_frame(transaction_id, action="rollback")
        self._rollback_pending_operations(transaction_id, context.pending_operations)
        if self._transaction_stack:
            depth = len(self._transaction_stack) + 1
            self._record_audit(
                transaction_id,
                "transaction.rollback.nested",
                summary=f"depth={depth}",
                level="warning",
            )
            return True

        self._record_audit(
            transaction_id,
            "transaction.rollback",
            summary="Transaction rolled back.",
            level="warning",
        )
        return True

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the accumulated AI audit events for inspection."""

        entries = list(self._audit_log)
        entries.sort(key=lambda item: item.timestamp)
        return [entry.as_dict() for entry in entries]

    def _pop_transaction_frame(self, transaction_id: str, *, action: str) -> _TransactionContext:
        """Validate ``transaction_id`` and pop the current stack frame."""

        normalised_id = self._validate_transaction_id(transaction_id, action=action)
        context = self._transaction_stack.pop()
        if context.transaction_id != normalised_id:
            # Defensive: if nesting produced an unexpected id, abort.
            self._record_audit(
                normalised_id,
                f"transaction.{action}.invalid",
                summary="Transaction stack corrupted.",
                level="error",
            )
            self._transaction_stack.clear()
            raise NWAiApiError("Transaction stack corrupted.")
        return context

    def _validate_transaction_id(self, transaction_id: str, *, action: str) -> str:
        """Ensure the provided transaction identifier is valid for the current state."""

        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise NWAiApiError("Transaction id must be a non-empty string.")

        normalised = transaction_id.strip()
        if not self._transaction_stack:
            self._record_audit(
                normalised,
                f"transaction.{action}.invalid",
                summary="No active transaction.",
                level="error",
            )
            raise NWAiApiError("No active transaction.")

        root_id = self._transaction_stack[0].transaction_id
        if normalised != root_id:
            self._record_audit(
                normalised,
                f"transaction.{action}.invalid",
                summary=f"Expected active transaction '{root_id}'.",
                level="error",
            )
            raise NWAiApiError("Transaction mismatch.")

        return normalised

    def _record_pending_operations(
        self,
        transaction_id: str,
        operations: Iterable[_PendingOperation],
        *,
        success: bool,
    ) -> None:
        """Record committed operations into the audit log."""

        status = "committed" if success else "rolled_back"
        level = "info" if success else "warning"
        for operation in operations:
            summary = operation.summary or f"{operation.operation} {status}"
            self._record_audit(
                transaction_id,
                f"transaction.operation.{status}",
                target=operation.target,
                summary=summary,
                level=level,
            )

    def _rollback_pending_operations(
        self,
        transaction_id: str,
        operations: Iterable[_PendingOperation],
    ) -> None:
        """Invoke undo callbacks (if any) and record rollback events."""

        for operation in reversed(list(operations)):
            if operation.undo is not None:
                try:
                    operation.undo()
                except Exception as exc:  # pragma: no cover - defensive logging
                    self._record_audit(
                        transaction_id,
                        "transaction.rollback.error",
                        target=operation.target,
                        summary=f"Undo failed for {operation.operation}: {exc}",
                        level="error",
                    )
            self._record_audit(
                transaction_id,
                "transaction.operation.rolled_back",
                target=operation.target,
                summary=operation.summary or f"{operation.operation} rolled back",
                level="warning",
            )

    def _record_audit(
        self,
        transaction_id: Optional[str],
        operation: str,
        *,
        target: Optional[str] = None,
        summary: Optional[str] = None,
        level: str = "info",
    ) -> None:
        """Store an audit entry with the given metadata."""

        entry = _AuditRecord(
            timestamp=datetime.now(timezone.utc),
            transaction_id=transaction_id,
            operation=operation,
            target=target,
            summary=summary,
            level=level,
        )
        self._audit_log.append(entry)

    def _assert_transaction_active(self) -> _TransactionContext:
        """Ensure a transaction is active before performing write operations."""

        if not self._transaction_stack:
            self._record_audit(
                None,
                "transaction.required",
                summary="Write operation requested without active transaction.",
                level="error",
            )
            raise NWAiApiError(
                "A transaction must be active before invoking write operations."
            )
        return self._transaction_stack[-1]

    def _queue_pending_operation(
        self,
        operation: str,
        target: Optional[str],
        summary: Optional[str],
        *,
        undo: Optional[Callable[[], None]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a pending write for later commit/rollback handling."""

        context = self._assert_transaction_active()
        payload = dict(metadata) if metadata else {}
        context.pending_operations.append(
            _PendingOperation(
                operation=operation,
                target=target,
                summary=summary,
                undo=undo,
                metadata=payload,
            )
        )

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

        self._assert_transaction_active()
        raise NotImplementedError("Document text updates are pending implementation.")

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    def previewSuggestion(self, handle: str, rng: TextRange, newText: str) -> Suggestion:
        """Generate a preview suggestion for replacing ``rng`` in ``handle`` with ``newText``."""

        raise NotImplementedError("Suggestion previews are pending implementation.")

    def applySuggestion(self, suggestionId: str) -> bool:
        """Apply a previously generated suggestion if it remains valid."""

        self._assert_transaction_active()
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

        self._assert_transaction_active()
        raise NotImplementedError("Chapter creation is scheduled for a later phase.")

    def createScene(
        self,
        name: str,
        parent: str,
        after: Optional[str] = None,
    ) -> Optional[DocumentRef]:
        """Create a scene item after ``after`` under ``parent`` and return a reference."""

        self._assert_transaction_active()
        raise NotImplementedError("Scene creation is scheduled for a later phase.")

    def duplicateItem(
        self,
        handle: str,
        parent: Optional[str] = None,
        after: bool = True,
    ) -> Optional[DocumentRef]:
        """Duplicate an existing item and optionally reparent or reorder the clone."""

        self._assert_transaction_active()
        raise NotImplementedError("Item duplication is scheduled for a later phase.")

    # ------------------------------------------------------------------
    # Build / export support
    # ------------------------------------------------------------------
    def build(self, fmt: str, options: Optional[dict[str, Any]] = None) -> BuildResult:
        """Trigger a project build in ``fmt`` format and return a result summary."""

        raise NotImplementedError("Build operations are pending implementation.")
