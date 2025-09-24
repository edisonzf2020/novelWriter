"""AI Core Service - AI-specific business logic separated from data access."""

from __future__ import annotations

import difflib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import ExitStack, suppress
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, Mapping, Optional
from types import MappingProxyType
from uuid import uuid4

from novelwriter import CONFIG

logger = logging.getLogger(__name__)

from .errors import NWAiApiError, NWAiConfigError, NWAiProviderError
from .history import HistoryManager, HistoryOperation
from .memory import ConversationMemory, ConversationTurn
from .models import BuildResult, DocumentRef, ModelInfo, ProofreadResult, Suggestion, TextRange
from .providers import ProviderCapabilities
from .cache import ProviderQueryCache, build_cache_key, config_from_ai
from .performance import get_tracker, log_metric_event

if TYPE_CHECKING:
    from novelwriter.api import NovelWriterAPI
    from .providers import BaseProvider

__all__ = ["AICoreService"]


# Constants from original api.py
_AUDIT_LOG_LIMIT = 1000
_DIFF_METADATA_LIMIT = 5000
_CONTEXT_SCOPES = frozenset(("current_document", "project", "conversation"))


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

    event_id: str
    timestamp: datetime
    transaction_id: Optional[str]
    operation: str
    target: Optional[str]
    summary: Optional[str]
    level: str = "info"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the audit entry into a JSON-friendly mapping."""

        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "target": self.target,
            "summary": self.summary,
            "level": self.level,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class _StreamingResult(Iterator[str]):
    """Wrap a streaming iterator and expose a cooperative close hook."""

    def __init__(
        self,
        iterator: Iterator[str],
        cancel_callbacks: list[Callable[[], None]],
        *,
        on_close: Callable[[str], None] | None = None,
    ) -> None:
        self._iterator = iterator
        self._cancel_callbacks = cancel_callbacks
        self._on_close = on_close
        self._lock = Lock()
        self._closed = False

    def __iter__(self) -> "_StreamingResult":
        return self

    def __next__(self) -> str:
        try:
            value = next(self._iterator)
        except StopIteration:
            self._notify_close("completed")
            raise
        return value

    def close(self) -> None:
        """Close the iterator and invoke any registered cancel callbacks."""

        self._notify_close("cancelled")

    def _notify_close(self, reason: str) -> None:
        callbacks: list[Callable[[], None]]
        close_method: Callable[[str], None] | None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            callbacks = list(self._cancel_callbacks)
            close_method = self._on_close

        for callback in callbacks:
            with suppress(Exception):
                callback()

        if close_method is not None:
            with suppress(Exception):
                close_method(reason)


def _compute_diff_payload(
    original: str,
    modified: str,
    *,
    from_label: str = "original",
    to_label: str = "modified",
    include_text: bool = True,
) -> tuple[Optional[str], dict[str, int]]:
    """Generate a unified diff and statistics for the given text pair."""

    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    stats = {
        "lines_added": 0,
        "lines_removed": 0,
        "lines_changed": 0,
    }

    if not include_text:
        for line in difflib.unified_diff(original_lines, modified_lines, lineterm=""):
            if line.startswith("+") and not line.startswith("+++"):
                stats["lines_added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                stats["lines_removed"] += 1
        stats["lines_changed"] = min(stats["lines_added"], stats["lines_removed"])
        return None, stats

    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
        )
    )

    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            stats["lines_added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            stats["lines_removed"] += 1

    stats["lines_changed"] = min(stats["lines_added"], stats["lines_removed"])

    if not diff_lines:
        return None, stats

    return "".join(diff_lines), stats


class AICoreService:
    """AI Core Service - handles AI-specific business logic using dependency injection."""

    def __init__(self, api: "NovelWriterAPI") -> None:
        """Create the AI Core Service with dependency injection.
        
        Args:
            api: NovelWriterAPI instance for data access
        """
        self._api = api
        self._transaction_stack: list[_TransactionContext] = []
        self._audit_log: deque[_AuditRecord] = deque(maxlen=_AUDIT_LOG_LIMIT)
        self._pending_suggestions: dict[str, dict[str, Any]] = {}
        self._conversation_memory = ConversationMemory()
        self._provider_lock = RLock()
        self._query_cache = ProviderQueryCache()
        self._provider: "BaseProvider" | None = None
        self._history = HistoryManager()

    # ------------------------------------------------------------------
    # Provider management (AI-specific)
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

    def listAvailableModels(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Return the available model catalogue provided by the backend."""

        provider = self._ensure_provider()
        ai_config = getattr(CONFIG, "ai", None)

        try:
            raw_models = provider.list_models(force=refresh)
        except NWAiProviderError as exc:
            message = str(exc) or "Model listing failed"
            if ai_config is not None:
                ai_config.set_availability_reason(message)
            self._record_audit(None, "provider.models.failed", summary=message, level="error")
            raise NWAiApiError(message) from exc
        except Exception as exc:  # noqa: BLE001 - propagate as API error
            message = str(exc) or exc.__class__.__name__
            if ai_config is not None:
                ai_config.set_availability_reason(message)
            self._record_audit(None, "provider.models.failed", summary=message, level="error")
            raise NWAiApiError(message) from exc

        if ai_config is not None:
            ai_config.set_availability_reason(None)

        models: list[ModelInfo] = []
        for entry in raw_models:
            info = self._build_model_info(entry)
            if info is not None:
                models.append(info)

        self._record_audit(
            None,
            "provider.models.listed",
            summary=f"{len(models)} model(s) discovered",
        )
        return models

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

    def _build_model_info(self, payload: Mapping[str, Any]) -> ModelInfo | None:
        """Translate provider payload into a :class:`ModelInfo` record."""

        if not isinstance(payload, Mapping):
            return None

        model_id = str(payload.get("id") or "").strip()
        if not model_id:
            return None

        display_name = payload.get("display_name") or payload.get("name") or model_id
        description = payload.get("description")
        if isinstance(description, str):
            description = description.strip() or None
        else:
            description = None

        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = dict(payload)
        else:
            metadata = dict(metadata)

        def coerce_int(value: Any) -> int | None:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit():
                    return int(stripped)
            return None

        input_limit = coerce_int(payload.get("input_token_limit"))
        output_limit = coerce_int(payload.get("output_token_limit"))

        info = ModelInfo(
            id=model_id,
            display_name=str(display_name),
            description=description,
            input_token_limit=input_limit,
            output_token_limit=output_limit,
            owned_by=str(payload.get("owned_by")) if payload.get("owned_by") is not None else None,
            capabilities=payload.get("capabilities"),
            metadata=metadata,
        )
        return info

    # ------------------------------------------------------------------
    # AI Chat Completion (Core AI functionality)
    # ------------------------------------------------------------------
    def streamChatCompletion(
        self,
        messages: list[Mapping[str, Any]],
        *,
        stream: bool = True,
        tools: list[Mapping[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream a chat completion response from the configured AI provider.
        
        This is the core AI conversation functionality.
        """
        provider = self._ensure_provider()
        ai_config = getattr(CONFIG, "ai", None)
        
        # Build cache key for potential caching
        cache_key = None
        cache = None
        if ai_config is not None and getattr(ai_config, "enable_cache", False):
            cache = self._query_cache
            cache_key = build_cache_key(messages, tools, extra)
            
            # Check cache first
            cached_entry = cache.get(cache_key) if cache_key else None
            if cached_entry is not None:
                def _cached_iter() -> Iterator[str]:
                    for chunk in cached_entry.chunks:
                        if chunk:
                            yield chunk

                def _on_cached_close(reason: str) -> None:
                    # Log cache hit completion
                    logger.debug("Cached response completed: %s", reason)

                return _StreamingResult(_cached_iter(), [], on_close=_on_cached_close)

        # Not cached, make actual provider call
        cancel_callbacks: list[Callable[[], None]] = []
        collected_chunks: list[str] | None = [] if cache is not None and cache_key is not None else None

        def _on_stream_close(reason: str) -> None:
            if reason == "completed":
                if collected_chunks is not None and cache is not None and cache_key is not None:
                    cache.store(cache_key, collected_chunks)
                logger.debug("Stream completed and cached")
            else:
                logger.debug("Stream closed: %s", reason)

        def _iterator() -> Iterator[str]:
            collected = collected_chunks

            def _record_chunk(text: str) -> None:
                if not text:
                    return
                if collected is not None:
                    collected.append(text)

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
                        with suppress(Exception):
                            closer()

                    cancel_callbacks.append(_safe_close)
                    stack.callback(_safe_close)

                try:
                    response = provider.chat_completion(
                        messages=messages,
                        stream=stream,
                        tools=tools,
                        extra=extra,
                    )
                    register_cancel(response)
                    
                    # Simple text streaming for now
                    for chunk in response:
                        if chunk:
                            _record_chunk(chunk)
                            yield chunk
                        
                except NWAiProviderError as exc:
                    logger.error("Provider error: %s", exc)
                    raise NWAiApiError(str(exc)) from exc
                except Exception as exc:
                    logger.error("Unexpected error: %s", exc)
                    raise NWAiApiError(str(exc)) from exc

        return _StreamingResult(_iterator(), cancel_callbacks, on_close=_on_stream_close)

    # ------------------------------------------------------------------
    # AI-specific operations (suggestions, context, memory)
    # ------------------------------------------------------------------
    def previewSuggestion(self, handle: str, rng: TextRange, newText: str) -> Suggestion:
        """Generate a preview suggestion for replacing text range with new text.
        
        This is AI-specific business logic for suggestion system.
        """
        self._assert_transaction_active()
        
        # Get current document text through API
        original_text = self._api.getDocText(handle)
        
        # Validate range
        if rng.start < 0 or rng.end > len(original_text) or rng.start > rng.end:
            raise NWAiApiError(f"Invalid text range [{rng.start}:{rng.end}] for document '{handle}'")
        
        # Apply the replacement to generate preview
        new_full_text = original_text[:rng.start] + newText + original_text[rng.end:]
        
        diff_text, diff_stats = _compute_diff_payload(
            original_text,
            new_full_text,
            from_label=f"original/{handle}",
            to_label=f"suggested/{handle}",
            include_text=True,
        )
        if not diff_text:
            diff_text = "No changes"
        
        # Generate unique suggestion ID
        suggestion_id = str(uuid4())

        suggestion_metadata = {
            "handle": handle,
            "range": (rng.start, rng.end),
            "new_text": newText,
            "original_text": original_text,
            "preview_text": new_full_text,
            "diff": diff_text,
            "diff_stats": diff_stats,
            "transaction_id": self._transaction_stack[-1].transaction_id,
        }

        # Store suggestion in cache
        self._pending_suggestions[suggestion_id] = suggestion_metadata

        # Record audit entry
        self._record_audit(
            self._transaction_stack[-1].transaction_id,
            "suggestion.preview",
            target=handle,
            summary=f"Generated suggestion {suggestion_id} for range [{rng.start}:{rng.end}]",
            level="info",
            metadata={
                "suggestion_id": suggestion_id,
                "range": {"start": rng.start, "end": rng.end},
                "diff": diff_stats,
                "preview_length": len(new_full_text),
            },
        )
        
        # Create and return suggestion object
        return Suggestion(
            id=suggestion_id,
            handle=handle,
            preview=new_full_text,
            diff=diff_text
        )

    def collectContext(
        self,
        scope: str = "current_document",
        *,
        limit: int = 10000,
        include_memory: bool = True,
    ) -> str:
        """Collect contextual information for AI operations.
        
        This is AI-specific context collection logic.
        """
        scope_key = (scope or "current_document").strip().lower()
        if scope_key not in _CONTEXT_SCOPES:
            raise NWAiApiError(
                f"Invalid context scope '{scope}'. Valid scopes: {', '.join(_CONTEXT_SCOPES)}",
            )

        context_parts: list[str] = []

        # Collect based on scope
        if scope_key == "current_document":
            context_parts.append(self._collect_document_context(limit))
        elif scope_key == "project":
            context_parts.append(self._collect_project_context(limit))
        elif scope_key == "conversation":
            if include_memory:
                memory_context = self._format_memory_context(
                    scope_key,
                    max_turns=10,
                    include_cross_scope=True,
                )
                if memory_context:
                    context_parts.append(memory_context)

        combined_context = "\n\n".join(filter(None, context_parts))
        
        self._record_audit(
            None,
            f"context.collect.{scope_key}",
            summary=f"Collected {scope_key} context ({len(combined_context)} characters)",
            level="info",
        )
        
        return combined_context

    def _collect_document_context(self, limit: int) -> str:
        """Collect context for current document."""
        current_doc = self._api.getCurrentDocument()
        if not current_doc:
            return "[No document currently open]"
        
        try:
            doc_text = self._api.getDocText(current_doc["handle"])
            if len(doc_text) <= limit:
                return doc_text
            else:
                return f"{doc_text[:limit]}...\n[Document truncated due to length limit]"
        except Exception as exc:
            return f"[Error loading document: {exc}]"

    def _collect_project_context(self, limit: int) -> str:
        """Collect context for entire project."""
        meta = self._api.getProjectMeta()
        
        context_parts = [
            "# Project Context",
            f"**Title:** {meta.get('name', 'Untitled')}",
            f"**Author:** {meta.get('author', 'Unknown')}",
            f"**Language:** {meta.get('language', 'en_US')}",
            f"**Total Words:** {meta.get('totalWords', 0):,}",
            f"**Novel Words:** {meta.get('novelWords', 0):,}",
            "",
        ]

        def append_section(header: str) -> None:
            context_parts.append(header)
            context_parts.append("")

        try:
            novel_docs = self._api.listDocuments("novel")
        except Exception as exc:
            context_parts.append(f"[Error collecting project context: {exc}]")
            novel_docs = []

        if novel_docs:
            append_section("## Novel Content")
            for ref in novel_docs:
                try:
                    doc_text = self._api.getDocText(ref["handle"])
                except Exception:
                    context_parts.append(f"### {ref['name']}")
                    context_parts.append("[Content unavailable]")
                    context_parts.append("")
                    continue

                body = doc_text if len(doc_text) <= limit else f"{doc_text[:limit]}..."
                context_parts.append(f"### {ref['name']}")
                context_parts.append("")
                context_parts.append(body)
                context_parts.append("")

        project_text = "\n".join(context_parts).strip()
        if len(project_text) > limit:
            project_text = (
                f"{project_text[:limit]}\n\n[Project context truncated due to length limit. "
                "Use more specific context scopes for complete content.]"
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

    def proofreadDocument(self, handle: str) -> ProofreadResult:
        """Proofread a document using AI capabilities.
        
        This is AI-specific proofreading logic.
        """
        # Start a transaction
        txn_id = self.begin_transaction()
        
        try:
            # Get document text through API
            doc_text = self._api.getDocText(handle)
            
            # Use AI provider to proofread
            provider = self._ensure_provider()
            
            # Create proofreading prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional proofreader. Check for grammar, spelling, and style issues."
                },
                {
                    "role": "user",
                    "content": f"Please proofread the following text:\n\n{doc_text}"
                }
            ]
            
            # Stream the proofreading response
            response_parts = []
            for chunk in self.streamChatCompletion(messages, stream=True):
                response_parts.append(chunk)
            
            full_response = "".join(response_parts)
            
            # Create a suggestion for the proofread text
            # In a real implementation, this would parse the AI response
            suggestion = Suggestion(
                id=str(uuid4()),
                handle=handle,
                preview=full_response,  # This would be the corrected text
                diff=""  # Would contain the diff
            )
            
            # Create ProofreadResult with correct fields
            result = ProofreadResult(
                transaction_id=txn_id,
                suggestion=suggestion,
                original_text=doc_text,
                diff_stats={"lines_changed": 0}  # Would be calculated
            )
            
            self._record_audit(
                txn_id,
                "document.proofread",
                target=handle,
                summary=f"Proofread document '{handle}'",
                level="info",
            )
            
            # Commit the transaction
            self.commit_transaction(txn_id)
            
            return result
            
        except Exception as e:
            # Rollback on error
            self.rollback_transaction(txn_id)
            raise

    # ------------------------------------------------------------------
    # Transaction management (needed for AI operations)
    # ------------------------------------------------------------------
    def begin_transaction(self) -> str:
        """Open a new AI-controlled transaction scope and return its identifier."""

        transaction_id = (
            f"txn_{uuid4().hex[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        depth = len(self._transaction_stack) + 1
        context = _TransactionContext(transaction_id=transaction_id)
        self._transaction_stack.append(context)
        self._record_audit(transaction_id, "transaction.begin", summary=f"depth={depth}")
        return transaction_id

    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit the given transaction and persist all pending changes."""

        context = self._pop_transaction_frame(transaction_id, action="commit")
        if not context.pending_operations:
            self._record_audit(
                transaction_id,
                "transaction.commit.empty",
                summary="No operations to commit.",
                level="warning",
            )
            return True

        try:
            self._record_pending_operations(
                transaction_id,
                context.pending_operations,
                success=True,
            )
        except Exception as exc:
            logger.exception("Failed to record pending operations during commit")
            raise NWAiApiError(f"Transaction commit failed: {exc}") from exc

        self._record_audit(
            transaction_id,
            "transaction.commit",
            summary=f"Committed {len(context.pending_operations)} operation(s).",
        )
        return True

    def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback the given transaction and discard all pending changes."""

        context = self._pop_transaction_frame(transaction_id, action="rollback")
        if not context.pending_operations:
            self._record_audit(
                transaction_id,
                "transaction.rollback.empty",
                summary="No operations to rollback.",
                level="warning",
            )
            return True

        self._rollback_pending_operations(transaction_id, context.pending_operations)
        self._record_audit(
            transaction_id,
            "transaction.rollback",
            summary=f"Rolled back {len(context.pending_operations)} operation(s).",
            level="warning",
        )
        return True

    def _pop_transaction_frame(self, transaction_id: str, *, action: str) -> _TransactionContext:
        """Validate transaction_id and pop the current stack frame."""

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

    def _record_audit(
        self,
        transaction_id: Optional[str],
        operation: str,
        *,
        target: Optional[str] = None,
        summary: Optional[str] = None,
        level: str = "info",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Store an audit entry with the given metadata."""

        payload = dict(metadata) if metadata else {}
        event_id = uuid4().hex
        entry = _AuditRecord(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            transaction_id=transaction_id,
            operation=operation,
            target=target,
            summary=summary,
            level=level,
            metadata=MappingProxyType(payload),
        )
        self._audit_log.append(entry)
        self._history.add_event(
            event_id=event_id,
            timestamp=entry.timestamp,
            transaction_id=transaction_id,
            operation=operation,
            target=target,
            summary=summary,
            level=level,
            metadata=payload,
        )

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
        operation_list = list(operations)
        saved_operations: list[HistoryOperation] = []
        for operation in operation_list:
            summary = operation.summary or f"{operation.operation} {status}"
            self._record_audit(
                transaction_id,
                f"transaction.operation.{status}",
                target=operation.target,
                summary=summary,
                level=level,
                metadata=operation.metadata,
            )
            if success:
                saved_operations.append(
                    HistoryOperation(
                        operation=operation.operation,
                        target=operation.target,
                        summary=summary,
                        metadata=dict(operation.metadata) if operation.metadata else {},
                        undo=operation.undo,
                    )
                )
        if success and saved_operations:
            self._history.register_operations(transaction_id, saved_operations)

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
                        metadata=operation.metadata,
                    )
            self._record_audit(
                transaction_id,
                "transaction.operation.rolled_back",
                target=operation.target,
                summary=operation.summary or f"{operation.operation} rolled back",
                level="warning",
                metadata=operation.metadata,
            )
