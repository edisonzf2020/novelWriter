"""Dialog for AI-powered document proofreading prior to export."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from contextlib import suppress
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from novelwriter import SHARED
from novelwriter.ai import NWAiApi, NWAiApiError, ProofreadResult
from novelwriter.ai.performance import log_metric_event
from novelwriter.ai.threading import AiCancellationToken, AiTaskHandle, get_ai_executor
from novelwriter.extensions.ai_copilot.diff_viewer import (
    DiffPreviewRequest,
    DiffPreviewResult,
    DiffPreviewWidget,
    build_diff_outputs,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingSuggestion:
    transaction_id: str
    suggestion_id: str
    handle: str
    preview_text: str


class _ProofreadWorker(QObject):
    """Execute a proofreading request on a background thread."""

    proofreadReady = pyqtSignal(ProofreadResult)
    failed = pyqtSignal(str)
    statusChanged = pyqtSignal(str)

    def __init__(self, api: NWAiApi, handle: str) -> None:
        super().__init__()
        self._api = api
        self._handle = handle
        self._token: Optional[AiCancellationToken] = None

    def attach_execution(self, token: AiCancellationToken, deadline: Optional[float]) -> None:  # noqa: ARG002
        self._token = token

    def cancel(self) -> None:
        if self._token is not None:
            self._token.cancel()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.statusChanged.emit("collecting_context")
            result = self._api.proofreadDocument(self._handle)
            self.statusChanged.emit("completed")
            self.proofreadReady.emit(result)
        except NWAiApiError as exc:
            if self._token is not None and self._token.cancelled():
                self.failed.emit("Request cancelled.")
            else:
                self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Proofreading worker failed")
            if self._token is not None and self._token.cancelled():
                self.failed.emit("Request cancelled.")
            else:
                self.failed.emit(str(exc))


class AICopilotProofreadDialog(QDialog):
    """Modal dialog guiding the user through AI proofreading of build documents."""

    def __init__(self, handles: Iterable[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("AI Proofreading"))
        self.setObjectName("AICopilotProofreadDialog")
        self.resize(900, 640)

        self._handles: List[str] = list(handles)
        self._current_index: int = 0
        self._api = NWAiApi(SHARED.project)
        self._executor = get_ai_executor()
        self._task: Optional[AiTaskHandle] = None
        self._worker: Optional[_ProofreadWorker] = None
        self._pending: Optional[_PendingSuggestion] = None
        self._processed: set[str] = set()
        self.completed_all: bool = False

        self._build_ui()
        self._apply_theme()
        self._load_document(0)

        if self._handles:
            QTimer.singleShot(0, self._start_proofread)
        else:
            self._statusLabel.setText(self.tr("No documents available for proofreading."))
            self._proofreadButton.setEnabled(False)
            self._applyButton.setEnabled(False)
            self._skipButton.setEnabled(False)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = QVBoxLayout()
        container.setContentsMargins(16, 16, 16, 16)
        container.setSpacing(12)

        self._titleLabel = QLabel(self)
        self._titleLabel.setWordWrap(True)
        container.addWidget(self._titleLabel)

        self._diffWidget = DiffPreviewWidget(self)
        container.addWidget(self._diffWidget, stretch=1)

        self._statusLabel = QLabel(self)
        self._statusLabel.setText(self.tr("Preparing..."))
        container.addWidget(self._statusLabel)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        self._proofreadButton = QPushButton(self.tr("Proofread"), self)
        self._proofreadButton.clicked.connect(self._start_proofread)
        button_row.addWidget(self._proofreadButton)

        self._applyButton = QPushButton(self.tr("Apply"), self)
        self._applyButton.clicked.connect(self._apply_current)
        self._applyButton.setEnabled(False)
        button_row.addWidget(self._applyButton)

        self._skipButton = QPushButton(self.tr("Skip"), self)
        self._skipButton.clicked.connect(self._skip_current)
        button_row.addWidget(self._skipButton)

        button_row.addStretch(1)
        container.addLayout(button_row)

        self._buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._buttonBox.rejected.connect(self.reject)
        container.addWidget(self._buttonBox)

        self.setLayout(container)

    def _apply_theme(self) -> None:
        theme = SHARED.theme
        self._titleLabel.setFont(theme.guiFontB)
        self._diffWidget.apply_theme(theme)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _document_name(self, handle: str) -> str:
        tree = SHARED.project.tree
        item = tree[handle]
        if item is None:
            return handle
        return item.itemName or handle

    def _load_document(self, index: int) -> None:
        self._clear_pending(rollback=True)
        if not self._handles:
            self._current_index = -1
            self._titleLabel.setText(self.tr("No documents to proofread."))
            return

        self._current_index = max(0, min(index, len(self._handles) - 1))
        handle = self._handles[self._current_index]
        title = self._document_name(handle)
        self._titleLabel.setText(
            self.tr("Document {0} of {1}: {2}").format(
                self._current_index + 1,
                len(self._handles),
                title,
            )
        )
        self._diffWidget.show_placeholder(self.tr("Preparing proofreading output."))
        self._statusLabel.setText(self.tr("Idle."))
        self._proofreadButton.setEnabled(True)
        self._applyButton.setEnabled(False)

    def _cleanup_worker(self) -> None:
        if self._task is not None and self._task.is_running():
            self._task.cancel()
        if self._worker is not None:
            self._worker.cancel()
            self._worker.deleteLater()
        self._task = None
        self._worker = None

    def _clear_pending(self, rollback: bool) -> None:
        if self._pending is None:
            return
        if rollback and self._pending.transaction_id:
            with suppress(Exception):
                self._api.rollback_transaction(self._pending.transaction_id)
        self._pending = None
        self._applyButton.setEnabled(False)

    # ------------------------------------------------------------------
    # Proofreading workflow
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _start_proofread(self) -> None:
        if self._current_index < 0 or self._current_index >= len(self._handles):
            return
        self._cleanup_worker()
        self._clear_pending(rollback=True)

        handle = self._handles[self._current_index]
        self._statusLabel.setText(self.tr("Proofreading in progress..."))
        self._proofreadButton.setEnabled(False)
        self._applyButton.setEnabled(False)

        worker = _ProofreadWorker(self._api, handle)
        worker.proofreadReady.connect(self._handle_ready)
        worker.failed.connect(self._handle_failed)
        worker.statusChanged.connect(self._handle_status)
        self._worker = worker
        self._task = self._executor.submit_worker(worker)

    def _handle_status(self, status: str) -> None:
        mapping = {
            "collecting_context": self.tr("Collecting context..."),
            "completed": self.tr("Proofreading complete."),
        }
        self._statusLabel.setText(mapping.get(status, status))

    def _handle_ready(self, result: ProofreadResult) -> None:
        self._cleanup_worker()
        handle = self._handles[self._current_index]

        inline_html, side_html, stats = build_diff_outputs(
            result.original_text,
            result.suggestion.preview,
            result.suggestion.diff or "",
        )
        preview = DiffPreviewResult(
            suggestion=result.suggestion,
            transaction_id=result.transaction_id,
            request=DiffPreviewRequest(
                handle=handle,
                selection_range=(0, len(result.original_text)),
                new_text=result.suggestion.preview,
            ),
            inline_html=inline_html,
            side_by_side_html=side_html,
            stats=stats,
        )
        self._diffWidget.display_result(preview)
        self._pending = _PendingSuggestion(
            transaction_id=result.transaction_id,
            suggestion_id=result.suggestion.id,
            handle=handle,
            preview_text=result.suggestion.preview,
        )
        self._statusLabel.setText(self.tr("Proofreading complete."))
        self._applyButton.setEnabled(True)
        self._proofreadButton.setEnabled(True)

    def _handle_failed(self, message: str) -> None:
        self._cleanup_worker()
        self._statusLabel.setText(self.tr("Proofreading failed: {0}").format(message))
        self._proofreadButton.setEnabled(True)
        self._applyButton.setEnabled(False)

    @pyqtSlot()
    def _apply_current(self) -> None:
        if self._pending is None:
            return
        data = self._pending
        try:
            applied = self._api.applySuggestion(data.suggestion_id)
            if not applied:
                raise NWAiApiError("Suggestion could not be applied.")
            self._api.commit_transaction(data.transaction_id)
        except NWAiApiError as exc:
            self._statusLabel.setText(str(exc))
            with suppress(Exception):
                self._api.rollback_transaction(data.transaction_id)
            self._pending = None
            return

        self._processed.add(data.handle)
        self._statusLabel.setText(self.tr("Changes applied."))
        self._pending = None
        self._advance()

    @pyqtSlot()
    def _skip_current(self) -> None:
        if self._pending is not None:
            with suppress(Exception):
                self._api.rollback_transaction(self._pending.transaction_id)
            self._pending = None
        handle = self._handles[self._current_index] if self._handles else ""
        self._processed.add(handle)
        self._statusLabel.setText(self.tr("Skipped."))
        self._advance()

    def _advance(self) -> None:
        if len(self._processed) >= len(self._handles):
            self.completed_all = True
            self._proofreadButton.setEnabled(False)
            self._applyButton.setEnabled(False)
            self._skipButton.setEnabled(False)
            self._statusLabel.setText(self.tr("All documents processed."))
            return

        next_index = self._current_index + 1
        if next_index < len(self._handles):
            self._load_document(next_index)
            QTimer.singleShot(150, self._start_proofread)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def accept(self) -> None:  # pragma: no cover - dialog closing
        self._cleanup_worker()
        self._clear_pending(rollback=True)
        super().accept()

    def reject(self) -> None:  # pragma: no cover - dialog closing
        self._cleanup_worker()
        self._clear_pending(rollback=True)
        super().reject()
