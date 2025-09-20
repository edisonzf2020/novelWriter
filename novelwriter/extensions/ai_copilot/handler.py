"""Background request handling for the AI Copilot dock."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from novelwriter import CONFIG, SHARED
from novelwriter.ai import NWAiApi, NWAiApiError
from novelwriter.ai.threading import AiCancellationToken, AiTaskHandle, get_ai_executor

DEFAULT_SYSTEM_PROMPT = (
    "You are novelWriter's AI Copilot. Provide concise, safe and actionable "
    "writing assistance. Maintain the author's voice and respond in British English."
)


@dataclass(frozen=True)
class CopilotRequest:
    """Payload describing a single AI Copilot interaction."""

    scope: str
    user_prompt: str
    system_prompt: str
    quick_action: Optional[str]
    selection_text: str
    selection_range: Optional[Tuple[int, int]]
    document_handle: Optional[str]
    include_memory: bool
    stream: bool
    max_output_tokens: Optional[int]
    temperature: Optional[float]
    timeout: Optional[float]
    context_budget: Optional[int]


class _RequestCancelled(RuntimeError):
    """Internal signal used to abort a running request."""


class _RequestTimeout(RuntimeError):
    """Raised when a request exceeds its configured timeout."""


class CopilotWorker(QObject):
    """Execute a Copilot request on a background thread."""

    chunkProduced = pyqtSignal(str)
    requestFinished = pyqtSignal(dict)
    requestFailed = pyqtSignal(str)
    requestCancelled = pyqtSignal()
    statusChanged = pyqtSignal(str)

    def __init__(self, api: NWAiApi, request: CopilotRequest) -> None:
        super().__init__()
        self._api = api
        self._request = request
        self._token: Optional[AiCancellationToken] = None
        self._deadline: Optional[float] = None
        self._started_at: float = 0.0
        self._stream_lock = Lock()
        self._active_stream: Optional[Any] = None

    def attach_execution(self, token: AiCancellationToken, deadline: Optional[float]) -> None:
        """Configure cancellation semantics for the worker."""

        self._token = token
        self._deadline = deadline
        self._started_at = time.monotonic()

    def cancel(self) -> None:
        """Mark the running request as cancelled."""

        if self._token is not None:
            self._token.cancel()
        stream = self._get_active_stream()
        if stream is not None:
            closer = getattr(stream, "close", None)
            if callable(closer):
                with suppress(Exception):
                    closer()
        self._set_active_stream(None)

    @pyqtSlot()
    def run(self) -> None:
        """Execute the Copilot request and emit progress signals."""

        iterator: Optional[Any] = None
        try:
            self._check_interrupted()
            self.statusChanged.emit("collecting_context")
            context = self._api.collectContext(
                self._request.scope,
                selection_text=self._request.selection_text,
                include_memory=self._request.include_memory,
                max_length=self._request.context_budget,
            )

            self._check_interrupted()

            messages = self._build_messages(context)
            extra: Dict[str, Any] = {}
            if self._request.max_output_tokens is not None:
                extra["max_output_tokens"] = self._request.max_output_tokens
            if self._request.temperature is not None:
                extra["temperature"] = self._request.temperature

            timeout_override = self._request.timeout
            remaining = self._remaining_timeout()
            if remaining is not None:
                if timeout_override is not None:
                    extra["timeout"] = min(timeout_override, remaining)
                else:
                    extra["timeout"] = remaining
            elif timeout_override is not None:
                extra["timeout"] = timeout_override

            self.statusChanged.emit("requesting_completion")
            output_chunks: list[str] = []
            iterator = self._api.streamChatCompletion(
                messages,
                stream=self._request.stream,
                extra=extra,
            )
            self._set_active_stream(iterator)

            try:
                for chunk in iterator:
                    self._check_interrupted()
                    if not chunk:
                        continue
                    output_chunks.append(chunk)
                    self.chunkProduced.emit(chunk)
            finally:
                self._set_active_stream(None)
                close_iter = getattr(iterator, "close", None)
                if callable(close_iter):
                    with suppress(Exception):
                        close_iter()

            self._check_interrupted()

            response_text = "".join(output_chunks).strip()
            self.statusChanged.emit("logging_conversation")
            self._api.logConversationTurn(
                self._request.user_prompt,
                response_text,
                context_scope=self._request.scope,
                context_summary=context[:200],
                metadata={"quick_action": self._request.quick_action} if self._request.quick_action else None,
            )

            payload = {
                "response": response_text,
                "scope": self._request.scope,
                "quick_action": self._request.quick_action,
                "user_prompt": self._request.user_prompt,
                "selection_text": self._request.selection_text,
                "selection_range": self._request.selection_range,
                "document_handle": self._request.document_handle,
                "context_used": context,
            }
            self.statusChanged.emit("completed")
            self.requestFinished.emit(payload)
        except _RequestCancelled:
            self.requestCancelled.emit()
        except _RequestTimeout:
            self.cancel()
            elapsed = 0.0
            if self._started_at:
                elapsed = max(0.0, time.monotonic() - self._started_at)
            self.requestFailed.emit(f"Request timed out after {elapsed:.1f}s.")
        except NWAiApiError as exc:
            if self._token is not None and self._token.cancelled():
                self.requestCancelled.emit()
            else:
                self.requestFailed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - propagate failure
            if self._token is not None and self._token.cancelled():
                self.requestCancelled.emit()
            else:
                self.requestFailed.emit(str(exc))

    def _set_active_stream(self, stream: Optional[Any]) -> None:
        """Store the currently active streaming iterator in a threadsafe way."""

        with self._stream_lock:
            self._active_stream = stream

    def _get_active_stream(self) -> Optional[Any]:
        """Return the active streaming iterator, if any."""

        with self._stream_lock:
            return self._active_stream

    def _check_interrupted(self) -> None:
        """Raise if the worker should stop processing."""

        if self._token is not None and self._token.cancelled():
            raise _RequestCancelled()
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise _RequestTimeout()

    def _remaining_timeout(self) -> Optional[float]:
        if self._deadline is None:
            return None
        remaining = self._deadline - time.monotonic()
        return max(0.0, remaining)

    def _build_messages(self, context: str) -> list[Mapping[str, Any]]:
        """Compose provider messages for the current request."""

        selection_block = (
            f"# Selected Text\n{self._request.selection_text.strip()}"
            if self._request.selection_text.strip()
            else ""
        )
        context_block_parts = []
        if selection_block:
            context_block_parts.append(selection_block)
        if context.strip():
            context_block_parts.append(f"# Context ({self._request.scope})\n{context.strip()}")
        context_block = "\n\n".join(context_block_parts)

        user_content = self._request.user_prompt.strip()
        if context_block:
            user_content = f"{user_content}\n\n{context_block}"

        return [
            {"role": "system", "content": self._request.system_prompt},
            {"role": "user", "content": user_content},
        ]


class CopilotRequestManager(QObject):
    """Orchestrate Copilot workers and expose Qt signals to the UI."""

    chunkProduced = pyqtSignal(str)
    requestFinished = pyqtSignal(dict)
    requestFailed = pyqtSignal(str)
    requestCancelled = pyqtSignal()
    statusChanged = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._api = NWAiApi(SHARED.project)
        self._worker: Optional[CopilotWorker] = None
        self._task: Optional[AiTaskHandle] = None
        self._provider_id: Optional[str] = getattr(CONFIG.ai, "provider", None)
        self._executor = get_ai_executor()

    @property
    def api(self) -> NWAiApi:
        """Expose the underlying NWAiApi instance for downstream tasks."""

        return self._api

    def has_active_request(self) -> bool:
        """Return ``True`` when a worker thread is currently running."""

        return bool(self._task is not None and self._task.is_running())

    def start_request(self, payload: CopilotRequest) -> None:
        """Launch a background worker for the given request."""

        self._ensure_provider_synced()

        if self._task is not None and self._task.is_running():
            raise RuntimeError("A Copilot request is already running.")

        self._worker = CopilotWorker(self._api, payload)
        self._worker.chunkProduced.connect(self.chunkProduced)
        self._worker.requestFinished.connect(self._handle_finished)
        self._worker.requestFailed.connect(self._handle_failed)
        self._worker.requestCancelled.connect(self._handle_cancelled)
        self._worker.statusChanged.connect(self.statusChanged)

        timeout = payload.timeout if payload.timeout is not None else None
        self._task = self._executor.submit_worker(self._worker, timeout=timeout)

    def cancel_request(self) -> None:
        """Attempt to cancel the active worker."""

        if self._task is not None and self._task.is_running():
            self._task.cancel()
        if self._worker is not None:
            self._worker.cancel()

    def on_provider_changed(self, provider_id: str | None) -> None:
        """Reset the API bridge when the configured provider changes."""

        if provider_id == self._provider_id:
            return
        self._provider_id = provider_id
        with suppress(Exception):
            self._api.resetProvider()

    def _ensure_provider_synced(self) -> None:
        current = getattr(CONFIG.ai, "provider", None)
        if current != self._provider_id:
            self.on_provider_changed(current)

    def build_request(
        self,
        *,
        scope: str,
        user_prompt: str,
        quick_action: Optional[str],
        selection_text: str,
        selection_range: Optional[Tuple[int, int]],
        document_handle: Optional[str],
        include_memory: bool,
        context_budget: Optional[int] = None,
    ) -> CopilotRequest:
        """Create a CopilotRequest using current configuration defaults."""

        self._ensure_provider_synced()
        ai_config = getattr(CONFIG, "ai", None)
        max_tokens = getattr(ai_config, "max_tokens", None) if ai_config else None
        timeout = getattr(ai_config, "timeout", None) if ai_config else None
        temperature = getattr(ai_config, "temperature", None) if ai_config else None

        return CopilotRequest(
            scope=scope,
            user_prompt=user_prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            quick_action=quick_action,
            selection_text=selection_text,
            selection_range=selection_range,
            document_handle=document_handle,
            include_memory=include_memory,
            stream=True,
            max_output_tokens=max_tokens,
            temperature=temperature,
            timeout=float(timeout) if timeout is not None else None,
            context_budget=context_budget,
        )

    def _handle_finished(self, payload: dict) -> None:
        self._cleanup()
        self.requestFinished.emit(payload)

    def _handle_failed(self, message: str) -> None:
        self._cleanup()
        self.requestFailed.emit(message)

    def _handle_cancelled(self) -> None:
        self._cleanup()
        self.requestCancelled.emit()

    def _cleanup(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
