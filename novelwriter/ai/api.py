"""Domain API facade exposed to the AI Copilot runtime."""

from __future__ import annotations

import difflib
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import ExitStack
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Optional
from types import MappingProxyType
from uuid import uuid4

from novelwriter import CONFIG

logger = logging.getLogger(__name__)
from novelwriter.enum import nwItemClass, nwItemLayout

from .errors import NWAiApiError, NWAiConfigError
from .memory import ConversationMemory, ConversationTurn
from .models import BuildResult, DocumentRef, Suggestion, TextRange
from .providers import ProviderCapabilities

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


class _StreamingResult(Iterator[str]):
    """Wrap a streaming iterator and expose a cooperative close hook."""

    def __init__(self, iterator: Iterator[str], cancel_callbacks: list[Callable[[], None]]) -> None:
        self._iterator = iterator
        self._cancel_callbacks = cancel_callbacks
        self._lock = Lock()
        self._closed = False

    def __iter__(self) -> "_StreamingResult":
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    def close(self) -> None:
        """Close the iterator and invoke any registered cancel callbacks."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            callbacks = list(self._cancel_callbacks)
            self._cancel_callbacks.clear()

        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001 - best effort cancellation
                logger.debug("Stream cancel callback raised", exc_info=True)

        close_method = getattr(self._iterator, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:  # noqa: BLE001 - best effort cancellation
                logger.debug("Closing streaming iterator failed", exc_info=True)


_AUDIT_LOG_LIMIT = 1000

_CONTEXT_SCOPES: tuple[str, ...] = (
    "selection",
    "current_document",
    "outline",
    "project",
)

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
    from novelwriter.ai.providers.base import BaseProvider
    from novelwriter.core.project import NWProject


class NWAiApi:
    """AI-facing facade encapsulating safe interactions with novelWriter data."""

    def __init__(self, project: "NWProject") -> None:
        """Create the API facade for a given project context."""

        self._project = project
        self._transaction_stack: list[_TransactionContext] = []
        self._audit_log: deque[_AuditRecord] = deque(maxlen=_AUDIT_LOG_LIMIT)
        self._pending_suggestions: dict[str, dict[str, Any]] = {}
        self._conversation_memory = ConversationMemory()
        self._provider_lock = RLock()
        self._provider: "BaseProvider" | None = None

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------
    def getProviderCapabilities(self, *, refresh: bool = False) -> ProviderCapabilities:
        """Return cached provider capabilities, refreshing when requested."""

        provider = self._ensure_provider()
        if refresh:
            return provider.refresh_capabilities()
        return provider.ensure_capabilities()

    def getProviderCapabilitiesSummary(self, *, refresh: bool = False) -> dict[str, Any]:
        """Return a serialisable snapshot of the provider capabilities."""

        snapshot = self.getProviderCapabilities(refresh=refresh)
        return snapshot.as_dict()

    def resetProvider(self) -> None:
        """Dispose of the cached provider instance."""

        with self._provider_lock:
            if self._provider is not None:
                try:
                    logger.debug("Resetting AI provider instance")
                    self._provider.close()
                finally:
                    self._provider = None

    def _ensure_provider(self) -> "BaseProvider":
        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            raise NWAiApiError("AI configuration is not available.")
        if not getattr(ai_config, "enabled", False):
            ai_config.set_availability_reason("AI features are disabled in the preferences.")
            raise NWAiApiError("AI provider support is disabled.")

        with self._provider_lock:
            if self._provider is not None:
                return self._provider

            try:
                provider = ai_config.create_provider()
            except NWAiConfigError as exc:
                ai_config.set_availability_reason(str(exc))
                raise NWAiApiError(str(exc)) from exc

            ai_config.set_availability_reason(None)
            self._provider = provider
            logger.debug("AI provider '%s' instantiated", ai_config.provider)
            return provider

    def streamChatCompletion(
        self,
        messages: list[Mapping[str, Any]],
        *,
        stream: bool = True,
        tools: list[Mapping[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield content from a chat completion request against the provider.

        Args:
            messages: Conversation turns formatted for the provider.
            stream: When ``True`` the result is yielded incrementally.
            tools: Optional tool definitions forwarded to the provider.
            extra: Additional keyword arguments for provider payload tuning
                (for example ``{"max_output_tokens": 512}``). ``None`` values
                are ignored.

        Yields:
            Chunks of provider output in the order they are received.

        Raises:
            NWAiApiError: If the provider is unavailable or the request fails.
        """

        if not isinstance(messages, list) or not all(isinstance(item, Mapping) for item in messages):
            raise NWAiApiError("Messages must be a list of mapping payloads.")

        provider = self._ensure_provider()
        payload = {key: value for key, value in (extra or {}).items() if value is not None}

        self._record_audit(None, "provider.request.dispatched", summary=f"{len(messages)} message(s) sent")

        cancel_callbacks: list[Callable[[], None]] = []

        def _iterator() -> Iterator[str]:
            total_chars = 0
            with ExitStack() as stack:
                def register_cancel(handle: Any) -> None:
                    closer = getattr(handle, "close", None)
                    if not callable(closer):
                        return

                    executed = False

                    def _safe_close() -> None:
                        nonlocal executed
                        if executed:
                            return
                        executed = True
                        try:
                            closer()
                        except Exception:  # noqa: BLE001 - best effort cancellation
                            logger.debug("Closing provider stream failed", exc_info=True)

                    cancel_callbacks.append(_safe_close)
                    stack.callback(_safe_close)

                def consume(response: Any) -> Iterator[str]:
                    nonlocal total_chars
                    for chunk in response.iter_text(chunk_size=256):
                        if not chunk:
                            continue
                        total_chars += len(chunk)
                        yield chunk

                try:
                    if stream:
                        session = provider.generate(messages, stream=True, tools=tools, **payload)
                        if hasattr(session, "__enter__"):
                            response = stack.enter_context(session)
                            register_cancel(response)
                        else:
                            response = session
                            register_cancel(response)
                        yield from consume(response)
                    else:
                        response = provider.generate(messages, stream=False, tools=tools, **payload)
                        register_cancel(response)
                        text = response.text
                        total_chars += len(text)
                        yield text
                except Exception as exc:  # noqa: BLE001 - propagate as API error
                    message = str(exc) or exc.__class__.__name__
                    self._record_audit(None, "provider.request.failed", summary=message, level="error")
                    raise NWAiApiError(message) from exc
                else:
                    self._record_audit(
                        None,
                        "provider.request.succeeded",
                        summary=f"{total_chars} characters received",
                    )
                finally:
                    cancel_callbacks.clear()

        return _StreamingResult(_iterator(), cancel_callbacks)

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

    def _snapshot_document(self, handle: str) -> tuple[str, Callable[[], None]]:
        """Create a snapshot of a document and return original text with undo callback."""
        # Validate handle and get original text
        original_text = self.getDocText(handle)
        
        # Create undo callback that restores the original text
        def undo_callback() -> None:
            # Use NWDocument to write back the original content
            document = self._project.storage.getDocument(handle)
            if document is not None:
                document.writeDocument(original_text, forceWrite=True)
        
        return original_text, undo_callback

    def _write_document(self, handle: str, text: str) -> bool:
        """Write text to the specified document using NWDocument API."""
        document = self._project.storage.getDocument(handle)
        if document is None:
            raise NWAiApiError(f"Cannot load document '{handle}'.")
        
        # Write the document and update project counts
        success = document.writeDocument(text)
        if success:
            # Update project statistics and counts
            self._project.updateCounts()
            
        return success

    def setDocText(self, handle: str, text: str, apply: bool = False) -> bool:
        """Apply or preview a text replacement for the given document handle."""
        
        self._assert_transaction_active()
        
        # Get original text and undo callback
        original_text, undo_callback = self._snapshot_document(handle)
        
        # Check apply parameter with CONFIG.ai.dry_run_default if apply is False
        should_apply = apply or not getattr(CONFIG.ai, "dry_run_default", True)
        
        if not should_apply:
            # Generate diff preview
            old_lines = original_text.splitlines(keepends=True)
            new_lines = text.splitlines(keepends=True)
            diff_lines = list(difflib.unified_diff(
                old_lines, 
                new_lines, 
                fromfile=f"original/{handle}", 
                tofile=f"modified/{handle}",
                lineterm=""
            ))
            
            # Record audit entry for preview
            self._record_audit(
                self._transaction_stack[-1].transaction_id,
                "document.preview",
                target=handle,
                summary=f"Generated diff preview for document '{handle}'",
                level="info"
            )
            
            return False  # Preview mode, no actual write
        
        # Apply the changes
        success = self._write_document(handle, text)
        
        if success:
            # Queue the operation for audit and rollback support
            self._queue_pending_operation(
                operation="document.write",
                target=handle,
                summary=f"Updated document '{handle}' content",
                undo=undo_callback,
                metadata={
                    "original_length": len(original_text),
                    "new_length": len(text),
                    "diff_size": abs(len(text) - len(original_text))
                }
            )
        
        return success

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    def previewSuggestion(self, handle: str, rng: TextRange, newText: str) -> Suggestion:
        """Generate a preview suggestion for replacing ``rng`` in ``handle`` with ``newText``."""
        
        self._assert_transaction_active()
        
        # Get current document text
        original_text = self.getDocText(handle)
        
        # Validate range
        if rng.start < 0 or rng.end > len(original_text) or rng.start > rng.end:
            raise NWAiApiError(f"Invalid text range [{rng.start}:{rng.end}] for document '{handle}'")
        
        # Apply the replacement to generate preview
        new_full_text = original_text[:rng.start] + newText + original_text[rng.end:]
        
        # Generate diff
        old_lines = original_text.splitlines(keepends=True)
        new_lines = new_full_text.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"original/{handle}",
            tofile=f"suggested/{handle}",
            lineterm=""
        ))
        diff_text = "\n".join(diff_lines) if diff_lines else "No changes"
        
        # Generate unique suggestion ID
        suggestion_id = str(uuid4())
        
        # Store suggestion in cache
        self._pending_suggestions[suggestion_id] = {
            "handle": handle,
            "range": rng,
            "new_text": newText,
            "original_text": original_text,
            "preview_text": new_full_text,
            "diff": diff_text,
            "transaction_id": self._transaction_stack[-1].transaction_id
        }
        
        # Record audit entry
        self._record_audit(
            self._transaction_stack[-1].transaction_id,
            "suggestion.preview",
            target=handle,
            summary=f"Generated suggestion {suggestion_id} for range [{rng.start}:{rng.end}]",
            level="info"
        )
        
        # Create and return suggestion object
        return Suggestion(
            id=suggestion_id,
            handle=handle,
            preview=new_full_text,
            diff=diff_text
        )

    def applySuggestion(self, suggestionId: str) -> bool:
        """Apply a previously generated suggestion if it remains valid."""
        
        self._assert_transaction_active()
        
        # Check if suggestion exists in cache
        if suggestionId not in self._pending_suggestions:
            raise NWAiApiError(f"Unknown or expired suggestion ID: {suggestionId}")
        
        suggestion_data = self._pending_suggestions[suggestionId]
        
        # Verify suggestion is for current transaction
        current_transaction_id = self._transaction_stack[-1].transaction_id
        if suggestion_data["transaction_id"] != current_transaction_id:
            raise NWAiApiError(f"Suggestion {suggestionId} belongs to a different transaction")
        
        # Check CONFIG.ai.ask_before_apply setting
        if getattr(CONFIG.ai, "ask_before_apply", True):
            # In a real implementation, this would trigger UI confirmation
            # For now, we record the requirement and continue
            self._record_audit(
                current_transaction_id,
                "suggestion.confirmation_required",
                target=suggestion_data["handle"],
                summary=f"Suggestion {suggestionId} requires manual confirmation",
                level="warning"
            )
        
        # Apply the suggestion using setDocText with apply=True
        success = self.setDocText(
            suggestion_data["handle"], 
            suggestion_data["preview_text"], 
            apply=True
        )
        
        if success:
            # Clean up the suggestion from cache after successful application
            del self._pending_suggestions[suggestionId]
            
            # Record successful application
            self._record_audit(
                current_transaction_id,
                "suggestion.applied",
                target=suggestion_data["handle"],
                summary=f"Successfully applied suggestion {suggestionId}",
                level="info"
            )
        else:
            # Record failed application
            self._record_audit(
                current_transaction_id,
                "suggestion.apply_failed",
                target=suggestion_data["handle"],
                summary=f"Failed to apply suggestion {suggestionId}",
                level="error"
            )
        
        return success

    # ------------------------------------------------------------------
    # Context and conversation utilities
    # ------------------------------------------------------------------
    def collectContext(
        self,
        scope: str = "current_document",
        *,
        selection_text: str | None = None,
        max_length: int | None = None,
        include_memory: bool = False,
        memory_turns: int = 3,
        include_cross_scope_memory: bool = True,
        **kwargs: Any,
    ) -> str:
        """Collect contextual text for the AI Copilot using the selected scope.

        Args:
            scope: Context scope identifier (selection, current_document, outline, project).
            selection_text: Raw text that should be treated as the current selection when
                ``scope`` is ``selection``.
            max_length: Optional hard limit for the amount of text returned by the scope
                collector. When omitted, scope-specific defaults are used.
            include_memory: Whether recent conversation turns should be appended to the
                collected context payload.
            memory_turns: Maximum number of conversation turns to include when
                ``include_memory`` is ``True``.
            include_cross_scope_memory: When ``True`` the memory snippet may include turns
                captured under other scopes if additional space remains.
            **kwargs: Additional hints forwarded to scope collectors for forward
                compatibility.

        Returns:
            Textual context assembled for the requested scope, optionally augmented with
            conversation memory.

        Raises:
            NWAiApiError: If the scope is unknown or the context gathering fails.
        """

        scope_key = (scope or "current_document").strip().lower()
        if scope_key not in _CONTEXT_SCOPES:
            raise NWAiApiError(
                f"Invalid context scope: '{scope}'. Must be one of: selection, current_document, outline, project",
            )

        if selection_text is None:
            selection_text = kwargs.get("selection_text")
        if max_length is None:
            max_length = kwargs.get("max_length")

        try:
            if scope_key == "selection":
                context_text = self._collectSelectionContext(selection_text=selection_text)
            elif scope_key == "current_document":
                context_text = self._collectCurrentDocumentContext(max_length=max_length)
            elif scope_key == "outline":
                context_text = self._collectOutlineContext(max_length=max_length)
            else:
                context_text = self._collectProjectContext(max_length=max_length)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._record_audit(
                None,
                "context.collect.error",
                summary=f"Failed to collect {scope_key} context: {exc}",
                level="error",
            )
            raise NWAiApiError(
                f"Failed to collect context for scope '{scope_key}': {exc}"
            ) from exc

        if include_memory:
            memory_text = self._format_memory_context(
                scope_key,
                max_turns=memory_turns,
                include_cross_scope=include_cross_scope_memory,
            )
            if memory_text:
                context_text = (
                    f"{context_text}\n\n---\n\n{memory_text}" if context_text else memory_text
                )

        return context_text

    def _collectSelectionContext(self, *, selection_text: str | None) -> str:
        """Collect context from the current text selection."""

        if not selection_text or not selection_text.strip():
            return "No text is currently selected in the editor."

        clean_text = selection_text.strip()
        self._record_audit(
            None,
            "context.collect.selection",
            summary=f"Collected {len(clean_text)} characters from selection",
            level="info",
        )
        return clean_text

    def _collectCurrentDocumentContext(self, *, max_length: int | None) -> str:
        """Collect context from the current active document."""

        current_doc = self.getCurrentDocument()
        if current_doc is None:
            return "No document is currently active."

        limit = max_length if isinstance(max_length, int) and max_length > 0 else 50_000
        doc_text = self.getDocText(current_doc.handle)
        truncated = len(doc_text) > limit
        if truncated:
            doc_text = doc_text[:limit]

        self._record_audit(
            None,
            "context.collect.document",
            target=current_doc.handle,
            summary=f"Collected {len(doc_text)} characters from document '{current_doc.name}'",
            level="info",
        )

        if truncated:
            doc_text = f"{doc_text}\n\n[Content truncated due to length limit]"

        return f"# Document: {current_doc.name}\n\n{doc_text}"

    def _collectOutlineContext(self, *, max_length: int | None) -> str:
        """Collect context from the project outline structure."""

        limit = max_length if isinstance(max_length, int) and max_length > 0 else 20_000
        project_meta = self.getProjectMeta()
        outline_parts = [
            "# Project Outline",
            f"Project: {project_meta.get('name', 'Unnamed')}",
            f"Author: {project_meta.get('author', 'Unknown')}",
            f"Language: {project_meta.get('language', 'en')}",
            f"Total Words: {project_meta.get('totalWords', 0)}",
            "",
            "## Structure",
        ]

        try:
            outline_refs = self.listDocuments("outline")
        except Exception as exc:  # pragma: no cover - defensive logging
            outline_parts.append(f"[Error collecting outline: {exc}]")
            outline_text = "\n".join(outline_parts)
            return outline_text[:limit] + (
                "\n\n[Outline truncated due to length limit]"
                if len(outline_text) > limit
                else ""
            )

        tree = self._project.tree
        for ref in outline_refs:
            item = tree[ref.handle]
            if item is None:
                outline_parts.append(f"### {ref.name}")
                outline_parts.append("[Content unavailable]")
                outline_parts.append("")
                continue

            depth = 0
            node = tree.nodes.get(ref.handle)
            parent_node = node.parent() if node is not None else None
            while parent_node and parent_node.item.itemClass in {nwItemClass.PLOT, nwItemClass.TIMELINE}:
                depth += 1
                parent_node = parent_node.parent()

            indent = "    " * depth
            outline_parts.append(f"{indent}- {ref.name}")

            try:
                doc_text = self.getDocText(ref.handle)
            except NWAiApiError:
                outline_parts.append(f"{indent}  [Content unavailable]")
                outline_parts.append("")
                continue

            summary_lines = [line.strip() for line in doc_text.splitlines() if line.strip()]
            preview = summary_lines[0] if summary_lines else doc_text.strip()
            if preview and len(preview) > 500:
                preview = preview[:500].rstrip() + "..."
            if preview:
                outline_parts.append(f"{indent}  {preview}")
            outline_parts.append("")

        outline_text = "\n".join(outline_parts)
        if len(outline_text) > limit:
            outline_text = outline_text[:limit] + "\n\n[Outline truncated due to length limit]"

        self._record_audit(
            None,
            "context.collect.outline",
            summary=f"Collected outline context ({len(outline_text)} characters)",
            level="info",
        )
        return outline_text

    def _collectProjectContext(self, *, max_length: int | None) -> str:
        """Collect context from the entire project with sensible limits."""

        limit = max_length if isinstance(max_length, int) and max_length > 0 else 100_000
        project_meta = self.getProjectMeta()
        context_parts = [
            "# Complete Project Context",
            f"Project: {project_meta.get('name', 'Unnamed')}",
            f"Author: {project_meta.get('author', 'Unknown')}",
            f"Total Words: {project_meta.get('totalWords', 0)}",
            "",
        ]

        def append_section(header: str) -> None:
            context_parts.append(header)
            context_parts.append("")

        try:
            novel_docs = self.listDocuments("novel")
        except Exception as exc:  # pragma: no cover - defensive logging
            context_parts.append(f"[Error collecting project context: {exc}]")
            novel_docs = []

        if novel_docs:
            append_section("## Novel Content")
            for ref in novel_docs:
                try:
                    doc_text = self.getDocText(ref.handle)
                except NWAiApiError:
                    context_parts.append(f"### {ref.name}")
                    context_parts.append("[Content unavailable]")
                    context_parts.append("")
                    continue

                body = doc_text if len(doc_text) <= limit else f"{doc_text[:limit]}..."
                context_parts.append(f"### {ref.name}")
                context_parts.append("")
                context_parts.append(body)
                context_parts.append("")

        secondary_scopes = ("character", "world", "plot")
        for scope in secondary_scopes:
            try:
                scoped_docs = self.listDocuments(scope)
            except NWAiApiError:
                continue
            if not scoped_docs:
                continue

            append_section(f"## {scope.title()} Notes")
            for ref in scoped_docs[:5]:
                try:
                    doc_text = self.getDocText(ref.handle)
                except NWAiApiError:
                    continue
                preview = doc_text if len(doc_text) <= 1000 else f"{doc_text[:1000]}..."
                context_parts.append(f"### {ref.name}")
                context_parts.append(preview)
                context_parts.append("")

        project_text = "\n".join(context_parts).strip()
        if len(project_text) > limit:
            project_text = (
                f"{project_text[:limit]}\n\n[Project context truncated due to length limit. "
                "Use more specific context scopes for complete content.]"
            )

        self._record_audit(
            None,
            "context.collect.project",
            summary=f"Collected project context ({len(project_text)} characters)",
            level="info",
        )
        return project_text

    def _format_memory_context(
        self,
        scope: str,
        *,
        max_turns: int,
        include_cross_scope: bool,
    ) -> str:
        """Format conversation memory into a textual snippet."""

        if max_turns <= 0:
            return ""

        turns = self._conversation_memory.get_relevant_context(
            scope,
            max_turns=max_turns,
            include_cross_scope=include_cross_scope,
        )
        if not turns:
            return ""

        lines: list[str] = ["# Conversation Memory"]
        for index, turn in enumerate(reversed(turns), start=1):
            timestamp = turn.timestamp.isoformat()
            lines.append(f"## Turn {index} ({turn.context_scope}, {timestamp})")
            if turn.context_summary:
                lines.append(f"Context Summary: {turn.context_summary}")
            lines.append("User:")
            lines.append(turn.user_input or "[empty input]")
            if turn.ai_response:
                lines.append("AI:")
                lines.append(turn.ai_response)
            lines.append("")
        return "\n".join(lines).strip()

    def getConversationMemory(self) -> ConversationMemory:
        """Expose the conversation memory manager for the current session."""

        return self._conversation_memory

    def logConversationTurn(
        self,
        user_input: str,
        ai_response: str,
        *,
        context_scope: str = "current_document",
        context_summary: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Store a new conversation turn and record it in the audit trail."""

        scope_key = (context_scope or "current_document").strip().lower()
        if scope_key not in _CONTEXT_SCOPES:
            raise NWAiApiError(
                f"Invalid conversation scope '{context_scope}'. Valid scopes: {', '.join(_CONTEXT_SCOPES)}",
            )

        turn = self._conversation_memory.add_turn(
            user_input=user_input,
            ai_response=ai_response,
            context_scope=scope_key,
            context_summary=context_summary,
            metadata=metadata,
        )
        self._record_audit(
            None,
            "conversation.turn.recorded",
            summary=f"Recorded conversation turn {turn.turn_id} (scope={scope_key})",
            level="info",
        )
        return turn

    def getConversationHistory(
        self,
        scope: str = "current_document",
        *,
        max_turns: int = 5,
        include_cross_scope: bool = True,
    ) -> list[dict[str, Any]]:
        """Return recent conversation turns relevant to the requested scope."""

        scope_key = (scope or "current_document").strip().lower()
        if scope_key not in _CONTEXT_SCOPES:
            raise NWAiApiError(
                f"Invalid conversation scope '{scope}'. Valid scopes: {', '.join(_CONTEXT_SCOPES)}",
            )

        turns = self._conversation_memory.get_relevant_context(
            scope_key,
            max_turns=max_turns,
            include_cross_scope=include_cross_scope,
        )
        return [turn.to_dict() for turn in turns]

    def clearConversationMemory(self) -> None:
        """Reset the stored conversation memory."""

        previous_session = self._conversation_memory.session_id
        self._conversation_memory.clear_memory()
        self._record_audit(
            None,
            "conversation.memory.cleared",
            summary=f"Conversation memory reset (previous session {previous_session})",
            level="info",
        )

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
