"""Dialog for browsing AI transaction history and initiating rollbacks."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QTextBrowser,
    QWidget,
)

from novelwriter import SHARED
from novelwriter.ai import NWAiApi, NWAiApiError, Suggestion, Suggestion

from .diff_viewer import DiffPreviewRequest, DiffPreviewResult, DiffPreviewWidget, build_diff_outputs


class AICopilotHistoryDialog(QDialog):
    """Modal dialog presenting committed AI transactions and events."""

    def __init__(self, api: NWAiApi, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self.setObjectName("AICopilotHistoryDialog")
        self.setWindowTitle(self.tr("AI History"))
        self.resize(960, 640)

        self._timelineTree = QTreeWidget(self)
        self._timelineTree.setColumnCount(3)
        self._timelineTree.setHeaderLabels(
            [
                self.tr("Transaction / Event"),
                self.tr("Status"),
                self.tr("Timestamp"),
            ]
        )
        self._timelineTree.itemSelectionChanged.connect(self._handle_selection_changed)

        self._diffWidget = DiffPreviewWidget(self)
        self._diffWidget.show_placeholder(self.tr("Select an entry to preview its diff."))

        theme = SHARED.theme
        self._diffWidget.apply_theme(theme)

        self._metadataView = QTextBrowser(self)
        self._metadataView.setObjectName("aiHistoryMetadataView")
        self._metadataView.setReadOnly(True)

        detail_splitter = QSplitter(Qt.Orientation.Vertical, self)
        detail_splitter.addWidget(self._diffWidget)
        detail_splitter.addWidget(self._metadataView)
        detail_splitter.setStretchFactor(0, 3)
        detail_splitter.setStretchFactor(1, 2)

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.addWidget(self._timelineTree)
        main_splitter.addWidget(detail_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        self._statusLabel = QLabel(self)
        self._statusLabel.setObjectName("aiHistoryStatusLabel")
        self._statusLabel.setText("")

        button_box = QDialogButtonBox(self)
        self._rollbackButton = QPushButton(self.tr("Rollback"), self)
        self._rollbackButton.setEnabled(False)
        button_box.addButton(self._rollbackButton, QDialogButtonBox.ButtonRole.ActionRole)
        self._refreshButton = QPushButton(self.tr("Refresh"), self)
        button_box.addButton(self._refreshButton, QDialogButtonBox.ButtonRole.ActionRole)
        button_box.addButton(QDialogButtonBox.StandardButton.Close)

        layout = QVBoxLayout()
        layout.addWidget(main_splitter, stretch=1)
        layout.addWidget(self._statusLabel)
        layout.addWidget(button_box)
        self.setLayout(layout)

        self._rollbackButton.clicked.connect(self._handle_rollback)
        self._refreshButton.clicked.connect(self._load_snapshot)
        button_box.rejected.connect(self.reject)

        self._snapshot: Dict[str, Any] = {}
        self._load_snapshot()

        self._timelineTree.setAlternatingRowColors(True)
        header = self._timelineTree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self._metadataView.setFont(theme.guiFontFixed)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_snapshot(self) -> None:
        try:
            snapshot = self._api.getHistorySnapshot(transaction_limit=50, event_limit=200)
        except NWAiApiError as exc:
            QMessageBox.critical(self, self.tr("History Error"), str(exc))
            return

        self._snapshot = snapshot
        self._timelineTree.clear()
        self._diffWidget.show_placeholder(self.tr("Select an entry to preview its diff."))
        self._metadataView.clear()
        transaction_count = len(snapshot.get("transactions", []))
        self._statusLabel.setText(self.tr("Loaded {0} transactions.").format(transaction_count))

        for transaction in snapshot.get("transactions", []):
            self._add_transaction_item(transaction)

        events = snapshot.get("events", [])
        if events:
            system_root = QTreeWidgetItem(self._timelineTree)
            system_root.setText(0, self.tr("System Events"))
            system_root.setFirstColumnSpanned(True)
            system_root.setExpanded(True)
            system_root.setData(0, Qt.ItemDataRole.UserRole, {"type": "system-root"})
            for event in events:
                self._add_event_item(event, system_root)

        self._timelineTree.expandAll()
        self._update_button_state()

    def _add_transaction_item(self, transaction: Dict[str, Any]) -> None:
        item = QTreeWidgetItem(self._timelineTree)
        transaction_id = transaction.get("transaction_id", "?")
        status = transaction.get("status", "pending")
        started_at = self._format_timestamp(transaction.get("started_at"))
        item.setText(0, self.tr("Transaction {0}").format(transaction_id))
        item.setText(1, status.title())
        item.setText(2, started_at)
        item.setData(0, Qt.ItemDataRole.UserRole, {"type": "transaction", "data": transaction})
        rollback_available = bool(transaction.get("rollback_available"))
        item.setData(0, Qt.ItemDataRole.UserRole + 1, rollback_available)

        for operation in transaction.get("operations", []):
            child = QTreeWidgetItem(item)
            summary = operation.get("summary") or operation.get("operation") or self.tr("Operation")
            child.setText(0, self.tr("Operation: {0}").format(summary))
            child.setText(1, self.tr("Committed"))
            child.setText(2, started_at)
            child.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "operation",
                "transaction": transaction,
                "data": operation,
            })

        for event in transaction.get("events", []):
            child = self._add_event_item(event, item)
            if child is not None:
                child.setData(0, Qt.ItemDataRole.UserRole + 1, rollback_available)

    def _add_event_item(self, event: Dict[str, Any], parent: QTreeWidgetItem) -> Optional[QTreeWidgetItem]:
        timestamp = self._format_timestamp(event.get("timestamp"))
        operation = event.get("operation", "event")
        summary = event.get("summary") or operation
        item = QTreeWidgetItem(parent)
        item.setText(0, self.tr("Event: {0}").format(operation))
        level = event.get("level", "info").title()
        item.setText(1, self.tr(level))
        item.setText(2, timestamp)
        item.setData(0, Qt.ItemDataRole.UserRole, {"type": "event", "data": event})
        return item

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------
    def _handle_selection_changed(self) -> None:
        item = self._timelineTree.currentItem()
        if item is None:
            self._diffWidget.show_placeholder(self.tr("Select an entry to preview its diff."))
            self._metadataView.clear()
            self._update_button_state()
            return

        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            self._diffWidget.show_placeholder(self.tr("Select an entry to preview its diff."))
            self._metadataView.clear()
            self._update_button_state()
            return

        entry_type = payload.get("type")
        data = payload.get("data", {})
        metadata = data.get("metadata", {})

        diff_preview = metadata.get("diff_preview") if isinstance(metadata, dict) else None
        if diff_preview:
            self._diffWidget.display_result(
                self._build_diff_result(diff_preview, metadata.get("diff"))
            )
        else:
            self._diffWidget.show_placeholder(self.tr("No diff preview available."))

        pretty_metadata = json.dumps(data, indent=2, ensure_ascii=False)
        self._metadataView.setPlainText(pretty_metadata)
        self._update_button_state()

    def _build_diff_result(self, diff_text: str, stats: Any) -> DiffPreviewResult:
        """Construct a DiffPreviewResult for the diff widget."""

        stats_dict: Dict[str, int] = {}
        if isinstance(stats, dict):
            stats_dict = {
                "additions": int(stats.get("additions", 0)),
                "deletions": int(stats.get("deletions", 0)),
            }

        html_payload = diff_text.replace("\n", "<br />")
        suggestion = Suggestion(id="history", handle="", preview=diff_text, diff=diff_text)
        dummy_request = DiffPreviewRequest(handle="", selection_range=(0, 0), new_text="")
        return DiffPreviewResult(
            suggestion=suggestion,
            transaction_id="history",
            request=dummy_request,
            inline_html=html_payload,
            side_by_side_html=html_payload,
            stats=stats_dict,
        )

        html_payload = diff_text.replace("\n", "<br />")

    # ------------------------------------------------------------------
    # Rollback handling
    # ------------------------------------------------------------------
    def _handle_rollback(self) -> None:
        item = self._timelineTree.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "transaction":
            parent = item.parent()
            if parent is None:
                return
            parent_payload = parent.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(parent_payload, dict) or parent_payload.get("type") != "transaction":
                return
            payload = parent_payload
        transaction = payload.get("data", {})
        transaction_id = transaction.get("transaction_id")
        if not transaction_id:
            return
        if not transaction.get("rollback_available"):
            QMessageBox.information(
                self,
                self.tr("Rollback"),
                self.tr("Rollback data is not available for this transaction."),
            )
            return
        confirmation = QMessageBox.question(
            self,
            self.tr("Confirm Rollback"),
            self.tr(
                "Rollback transaction {0}? This will revert committed AI changes."
            ).format(transaction_id),
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return
        try:
            self._api.rollbackHistoryTransaction(str(transaction_id))
        except NWAiApiError as exc:
            QMessageBox.critical(self, self.tr("Rollback Failed"), str(exc))
            return
        self._load_snapshot()
        QMessageBox.information(
            self,
            self.tr("Rollback"),
            self.tr("Transaction {0} was rolled back.").format(transaction_id),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_button_state(self) -> None:
        item = self._timelineTree.currentItem()
        rollback_enabled = False
        if item is not None:
            payload = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(payload, dict):
                if payload.get("type") == "transaction":
                    rollback_enabled = bool(payload.get("data", {}).get("rollback_available"))
                elif payload.get("type") in {"operation", "event"}:
                    parent = item.parent()
                    if parent is not None:
                        parent_payload = parent.data(0, Qt.ItemDataRole.UserRole)
                        if isinstance(parent_payload, dict) and parent_payload.get("type") == "transaction":
                            rollback_enabled = bool(parent_payload.get("data", {}).get("rollback_available"))
        self._rollbackButton.setEnabled(rollback_enabled)

    def _format_timestamp(self, value: Optional[str]) -> str:
        if not value:
            return ""
        try:
            parsed = datetime.fromisoformat(value)
        except Exception:  # noqa: BLE001 - fallback
            return value
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
