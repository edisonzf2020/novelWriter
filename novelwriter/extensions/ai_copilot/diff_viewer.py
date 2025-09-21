"""Asynchronous diff preview utilities for the AI Copilot dock."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import logging
import time
from typing import Dict, Mapping, Optional, Tuple

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from novelwriter.ai import NWAiApi, NWAiApiError, Suggestion, TextRange
from novelwriter.ai.performance import log_metric_event
from novelwriter.ai.threading import AiCancellationToken, AiTaskHandle, get_ai_executor

logger = logging.getLogger(__name__)

_DIFF_INLINE_STYLE = (
    "<style>"
    ".diff-inline {font-family: 'Courier New', monospace;}"
    ".diff-inline pre {background-color: #f6f8fa; padding: 12px; border-radius: 4px;}"
    ".diff-inline .diff-header {color: #6a737d;}"
    ".diff-inline .diff-added {color: #22863a;}"
    ".diff-inline .diff-removed {color: #cb2431;}"
    "</style>"
)

_DIFF_TABLE_STYLE = (
    "<style>"
    ".diff-table {font-family: 'Courier New', monospace; border-collapse: collapse;}"
    ".diff-table td {padding: 2px 6px; vertical-align: top;}"
    ".diff-table .diff_header {background: #f6f8fa; color: #6a737d;}"
    ".diff-table .diff_next {background: #e2e2e2;}"
    ".diff-table .diff_add {background: #e6ffed;}"
    ".diff-table .diff_sub {background: #ffeef0;}"
    "</style>"
)


@dataclass(frozen=True)
class DiffPreviewRequest:
    """Parameters describing a diff preview computation."""

    handle: str
    selection_range: Tuple[int, int]
    new_text: str

    def cache_key(self) -> str:
        """Return a stable cache key for this request."""

        digest = hashlib.sha256(self.new_text.encode("utf-8")).hexdigest()[:12]
        return f"{self.handle}:{self.selection_range[0]}:{self.selection_range[1]}:{digest}"


@dataclass
class DiffPreviewResult:
    """Computed diff artefacts returned to the UI."""

    suggestion: Suggestion
    transaction_id: str
    request: DiffPreviewRequest
    inline_html: str
    side_by_side_html: str
    stats: Dict[str, int]


class DiffPreviewWidget(QWidget):
    """Widget capable of rendering inline and side-by-side diff previews."""

    modeChanged = pyqtSignal(str)

    _INLINE_MODE = "inline"
    _SIDE_BY_SIDE_MODE = "side-by-side"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = self._INLINE_MODE
        self._result: Optional[DiffPreviewResult] = None

        self._statusLabel = QLabel(self)
        self._statusLabel.setObjectName("aiCopilotDiffStatus")
        self._statusLabel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._modeSelector = self._build_mode_selector()
        self._stack = QStackedWidget(self)
        self._inlineView = self._create_text_browser()
        self._sideBySideView = self._create_text_browser()
        self._sideBySideView.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)
        self._stack.addWidget(self._inlineView)
        self._stack.addWidget(self._sideBySideView)

        container = QFrame(self)
        container.setFrameShape(QFrame.Shape.NoFrame)
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self._statusLabel, stretch=1)
        header_layout.addWidget(self._modeSelector, stretch=0)

        container_layout.addLayout(header_layout)
        container_layout.addWidget(self._stack, stretch=1)
        container.setLayout(container_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(container)
        self.setLayout(root_layout)
        self.show_placeholder(self.tr("Diff preview is not available."))

    def mode(self) -> str:
        """Return the currently selected diff rendering mode."""

        return self._mode

    def set_mode(self, mode: str) -> None:
        """Update the rendering mode and re-render the view."""

        if mode not in {self._INLINE_MODE, self._SIDE_BY_SIDE_MODE}:
            return
        if self._mode == mode:
            return
        self._mode = mode
        self._update_view()
        index = 0 if mode == self._INLINE_MODE else 1
        self._stack.setCurrentIndex(index)
        self.modeChanged.emit(mode)

    def show_placeholder(self, message: str) -> None:
        """Render a placeholder message in the inline view."""

        safe_message = html.escape(message or "")
        html_payload = f"<p style='color:#6a737d;'>{safe_message}</p>"
        self._inlineView.setHtml(html_payload)
        self._sideBySideView.setHtml(html_payload)
        self._statusLabel.setText(message)
        self._result = None

    def show_progress(self, message: str) -> None:
        """Display a busy indicator style message."""

        self._statusLabel.setText(message)
        placeholder = html.escape(message or "")
        self._inlineView.setHtml(f"<p>{placeholder}</p>")
        self._sideBySideView.setHtml(f"<p>{placeholder}</p>")

    def show_error(self, message: str) -> None:
        """Display an error message."""

        safe_message = html.escape(message or "")
        self._statusLabel.setText(message)
        html_payload = f"<p style='color:#cb2431;'>{safe_message}</p>"
        self._inlineView.setHtml(html_payload)
        self._sideBySideView.setHtml(html_payload)
        self._result = None

    def display_result(self, result: DiffPreviewResult) -> None:
        """Render the supplied diff preview result."""

        self._result = result
        stats = result.stats
        additions = stats.get("additions", 0)
        deletions = stats.get("deletions", 0)
        summary_parts = [self.tr("Diff ready.")]
        summary_parts.append(self.tr("+{0} / -{1}").format(additions, deletions))
        self._statusLabel.setText(" ".join(summary_parts))
        self._update_view()

    def apply_theme(self, theme: object) -> None:
        """Update internal document fonts to match the application theme."""

        fixed_font = getattr(theme, "guiFontFixed", None)
        body_font = getattr(theme, "guiFont", None)
        for browser in (self._inlineView, self._sideBySideView):
            document = browser.document()
            if document is not None and fixed_font is not None:
                document.setDefaultFont(fixed_font)
            if body_font is not None:
                browser.setFont(body_font)

    def _build_mode_selector(self) -> QWidget:
        selector = QWidget(self)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)

        inline_button = self._build_mode_button(self.tr("Inline"), self._INLINE_MODE)
        side_button = self._build_mode_button(self.tr("Side-by-side"), self._SIDE_BY_SIDE_MODE)
        button_group.addButton(inline_button, 0)
        button_group.addButton(side_button, 1)
        inline_button.setChecked(True)

        button_group.idClicked.connect(self._handle_mode_clicked)

        layout.addWidget(inline_button)
        layout.addWidget(side_button)
        selector.setLayout(layout)
        return selector

    def _build_mode_button(self, label: str, mode: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(label)
        button.setCheckable(True)
        button.setProperty("diffMode", mode)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        if mode == self._INLINE_MODE:
            button.setToolTip(self.tr("Show unified inline diff view."))
        else:
            button.setToolTip(self.tr("Show side-by-side diff view."))
        return button

    def _create_text_browser(self) -> QTextBrowser:
        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setReadOnly(True)
        browser.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        doc = browser.document()
        if doc is not None:
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.NoWrap)
            doc.setDefaultTextOption(option)
        return browser

    @pyqtSlot(int)
    def _handle_mode_clicked(self, button_id: int) -> None:  # pragma: no cover - direct Qt slot
        mode = self._INLINE_MODE if button_id == 0 else self._SIDE_BY_SIDE_MODE
        button = self.sender()
        if isinstance(button, QButtonGroup):  # pragma: no cover - defensive
            checked = button.checkedButton()
            if checked is not None:
                mode = checked.property("diffMode") or mode
        self.set_mode(str(mode))

    def _update_view(self) -> None:
        if self._result is None:
            return
        if self._mode == self._INLINE_MODE:
            self._inlineView.setHtml(self._result.inline_html)
            self._stack.setCurrentIndex(0)
        else:
            self._sideBySideView.setHtml(self._result.side_by_side_html)
            self._stack.setCurrentIndex(1)


class _DiffPreviewWorker(QObject):
    """Background worker responsible for computing diff previews."""

    previewReady = pyqtSignal(DiffPreviewResult)
    previewFailed = pyqtSignal(str)
    previewCancelled = pyqtSignal()
    statusChanged = pyqtSignal(str)

    def __init__(self, api: NWAiApi, request: DiffPreviewRequest) -> None:
        super().__init__()
        self._api = api
        self._request = request
        self._token: Optional[AiCancellationToken] = None
        self._transaction_id: Optional[str] = None

    def attach_execution(self, token: AiCancellationToken, deadline: Optional[float]) -> None:
        self._token = token
        # deadline currently unused but kept for API parity

    def cancel(self) -> None:
        if self._token is not None:
            self._token.cancel()
        if self._transaction_id is not None:
            try:
                self._api.rollback_transaction(self._transaction_id)
            except Exception:  # noqa: BLE001 - defensive cleanup
                logger.debug("Failed to rollback diff transaction", exc_info=True)
            self._transaction_id = None

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._check_cancelled()
            self.statusChanged.emit("preparing")
            original_text = self._api.getDocText(self._request.handle)

            self._check_cancelled()
            self._transaction_id = self._api.begin_transaction()
            text_range = TextRange(
                start=int(self._request.selection_range[0]),
                end=int(self._request.selection_range[1]),
            )
            suggestion = self._api.previewSuggestion(
                self._request.handle,
                text_range,
                self._request.new_text,
            )

            self._check_cancelled()
            inline_html, table_html, stats = build_diff_outputs(
                original_text,
                suggestion.preview,
                suggestion.diff or "",
            )
            result = DiffPreviewResult(
                suggestion=suggestion,
                transaction_id=str(self._transaction_id),
                request=self._request,
                inline_html=inline_html,
                side_by_side_html=table_html,
                stats=stats,
            )
            self.previewReady.emit(result)
        except _DiffCancelled:
            self.previewCancelled.emit()
        except NWAiApiError as exc:
            self._rollback_if_needed()
            self.previewFailed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures
            logger.exception("Diff preview worker failed")
            self._rollback_if_needed()
            self.previewFailed.emit(str(exc))

    def _check_cancelled(self) -> None:
        if self._token is not None and self._token.cancelled():
            raise _DiffCancelled()

    def _rollback_if_needed(self) -> None:
        if self._transaction_id is None:
            return
        try:
            self._api.rollback_transaction(self._transaction_id)
        except Exception:  # noqa: BLE001 - defensive cleanup
            logger.debug("Rollback after diff failure failed", exc_info=True)
        finally:
            self._transaction_id = None


class DiffPreviewController(QObject):
    """Controller orchestrating diff preview computations."""

    previewReady = pyqtSignal(DiffPreviewResult)
    previewFailed = pyqtSignal(str)
    previewCancelled = pyqtSignal()
    statusChanged = pyqtSignal(str)

    def __init__(self, api: NWAiApi, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._executor = get_ai_executor()
        self._worker: Optional[_DiffPreviewWorker] = None
        self._handle: Optional[AiTaskHandle] = None
        self._request_started_at: Optional[float] = None
        self._last_request_details: Optional[DiffPreviewRequest] = None

    def request_preview(self, request: DiffPreviewRequest) -> None:
        """Submit a preview computation for the supplied request."""

        self.cancel_pending()
        self._request_started_at = time.perf_counter()
        self._last_request_details = request

        worker = _DiffPreviewWorker(self._api, request)
        worker.previewReady.connect(self._handle_ready)
        worker.previewFailed.connect(self._handle_failed)
        worker.previewCancelled.connect(self._handle_cancelled)
        worker.statusChanged.connect(self.statusChanged)
        self._worker = worker
        self._handle = self._executor.submit_worker(worker)

    def cancel_pending(self) -> None:
        """Cancel any running diff computation."""

        if self._handle is not None and self._handle.is_running():
            self._handle.cancel()
        if self._worker is not None:
            self._worker.cancel()
            self._worker.deleteLater()
        self._handle = None
        self._worker = None
        self._request_started_at = None
        self._last_request_details = None

    def _handle_ready(self, result: DiffPreviewResult) -> None:
        self.previewReady.emit(result)
        duration_ms = None
        if self._request_started_at is not None:
            duration_ms = (time.perf_counter() - self._request_started_at) * 1000.0
        request = self._last_request_details
        log_metric_event(
            "diff.preview.success",
            {
                "handle": getattr(request, "handle", ""),
                "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
                "additions": result.stats.get("additions", 0),
                "deletions": result.stats.get("deletions", 0),
            },
        )
        self._cleanup_worker()

    def _handle_failed(self, message: str) -> None:
        self.previewFailed.emit(message)
        duration_ms = None
        if self._request_started_at is not None:
            duration_ms = (time.perf_counter() - self._request_started_at) * 1000.0
        request = self._last_request_details
        log_metric_event(
            "diff.preview.failed",
            {
                "handle": getattr(request, "handle", ""),
                "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
                "error": message,
            },
        )
        self._cleanup_worker()

    def _handle_cancelled(self) -> None:
        self.previewCancelled.emit()
        duration_ms = None
        if self._request_started_at is not None:
            duration_ms = (time.perf_counter() - self._request_started_at) * 1000.0
        request = self._last_request_details
        log_metric_event(
            "diff.preview.cancelled",
            {
                "handle": getattr(request, "handle", ""),
                "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
            },
        )
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._handle = None
        self._request_started_at = None
        self._last_request_details = None


class _DiffCancelled(RuntimeError):
    """Internal marker exception signalling cancellation."""

    pass


def build_diff_outputs(original: str, updated: str, diff_text: str) -> Tuple[str, str, Dict[str, int]]:
    """Return inline and tabular diff renderings along with change statistics."""

    original_lines = original.splitlines()
    updated_lines = updated.splitlines()

    inline_html = _render_inline_diff(diff_text)
    side_by_side_html = _render_table_diff(original_lines, updated_lines)
    stats = {
        "additions": sum(1 for line in diff_text.splitlines() if line.startswith("+")),
        "deletions": sum(1 for line in diff_text.splitlines() if line.startswith("-")),
    }
    return inline_html, side_by_side_html, stats


def _render_inline_diff(diff_text: str) -> str:
    """Render a unified diff string as coloured inline HTML."""

    lines = []
    for raw_line in diff_text.splitlines():
        escaped = html.escape(raw_line)
        if raw_line.startswith("+++ ") or raw_line.startswith("--- "):
            lines.append(f"<span class='diff-header'>{escaped}</span>")
        elif raw_line.startswith("@@"):
            lines.append(f"<span class='diff-header'>{escaped}</span>")
        elif raw_line.startswith("+"):
            lines.append(f"<span class='diff-added'>{escaped}</span>")
        elif raw_line.startswith("-"):
            lines.append(f"<span class='diff-removed'>{escaped}</span>")
        else:
            lines.append(escaped)
    pretty = "\n".join(lines)
    return f"{_DIFF_INLINE_STYLE}<div class='diff-inline'><pre>{pretty}</pre></div>"


def _render_table_diff(old_lines: list[str], new_lines: list[str]) -> str:
    """Render a side-by-side diff view using difflib's HTML utilities."""

    import difflib

    diff = difflib.HtmlDiff(wrapcolumn=80)
    table = diff.make_table(old_lines, new_lines, context=True, numlines=3)
    return f"{_DIFF_TABLE_STYLE}<div class='diff-inline'>{table}</div>"
