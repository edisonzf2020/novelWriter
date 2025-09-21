"""
novelWriter – AI Copilot Dock GUI Tests
======================================

This file is a part of novelWriter
Copyright (C) 2025 Veronica Berglyd Olsen and novelWriter contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""  # noqa
from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTextEdit,
)

from novelwriter import SHARED

from novelwriter.extensions.ai_copilot import AICopilotDock, MainWindowIntegration


@pytest.mark.gui
def test_ai_copilot_dock_added_by_default(nwGUI):
    """The AI dock is created and attached during GUI construction."""
    dock = nwGUI.findChild(QDockWidget, "AICopilotDock")

    assert dock is not None
    assert dock.windowTitle() == dock.tr("AI Copilot")
    assert dock.features() & QDockWidget.DockWidgetFeature.DockWidgetFloatable
    assert dock.features() & QDockWidget.DockWidgetFeature.DockWidgetMovable

    status_label = dock.findChild(QLabel, "aiCopilotStatusLabel")
    message_label = dock.findChild(QLabel, "aiCopilotMessageLabel")
    scope_selector = dock.findChild(QComboBox, "aiContextScopeSelector")

    assert status_label is not None
    assert status_label.text() == dock.tr("AI Copilot is temporarily unavailable")
    assert message_label is not None
    assert message_label.text() == dock.tr("AI features are disabled in the preferences.")
    assert scope_selector is not None
    assert scope_selector.count() == 4
    assert scope_selector.currentData() == "current_document"

class _CancellableStream:
    """Streaming stub that supports cooperative cancellation."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> "_CancellableStream":
        return self

    def __next__(self) -> str:
        self.started.set()
        if self.closed.is_set():
            raise StopIteration
        if self.closed.wait(0.01):
            raise StopIteration
        return "chunk"

    def close(self) -> None:
        self.closed.set()


class _SlowStream(_CancellableStream):
    """Stream stub that delays subsequent chunks to trigger timeouts."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self._delay = delay
        self._first = True

    def __next__(self) -> str:
        self.started.set()
        if self.closed.is_set():
            raise StopIteration
        if self._first:
            self._first = False
            if self.closed.wait(0.005):
                raise StopIteration
            return "chunk"
        if self.closed.wait(self._delay):
            raise StopIteration
        return "chunk"


class _StubAiApi:
    """Minimal AI API facade used to drive background workers in tests."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def collectContext(self, scope: str, **_: object) -> str:  # noqa: D401 - signature matches prod
        return f"context:{scope}"

    def streamChatCompletion(self, messages, *, stream=True, extra=None):  # noqa: D401
        return self._stream

    def logConversationTurn(self, *args, **kwargs) -> None:  # noqa: D401
        return None


@pytest.mark.gui
def test_ai_copilot_dock_shows_disabled_message(monkeypatch, nwGUI):
    """A disabled AI configuration renders a friendly placeholder."""

    dummy_config = SimpleNamespace(ai=SimpleNamespace(enabled=False))
    monkeypatch.setattr("novelwriter.extensions.ai_copilot.dock.CONFIG", dummy_config, raising=False)

    dock = AICopilotDock(nwGUI)
    message_label = dock.findChild(QLabel, "aiCopilotMessageLabel")

    assert message_label is not None
    assert message_label.text() == dock.tr("AI features are disabled in the preferences.")

    dock.deleteLater()


@pytest.mark.gui
def test_ai_copilot_integration_idempotent(nwGUI):
    """Calling the integration multiple times must not duplicate the dock."""
    assert MainWindowIntegration.integrate_ai_dock(nwGUI) is True

    docks = nwGUI.findChildren(QDockWidget, "AICopilotDock")
    assert len(docks) == 1


@pytest.mark.gui
def test_ai_copilot_dock_updates_with_theme_change(monkeypatch, nwGUI):
    """Theme refresh should update dock label colours."""
    dock = nwGUI.findChild(QDockWidget, "AICopilotDock")
    message_label = dock.findChild(QLabel, "aiCopilotMessageLabel")
    assert message_label is not None

    original_style = message_label.styleSheet()
    original_error = SHARED.theme.errorText
    original_help = SHARED.theme.helpText
    original_available = dock._ai_available  # type: ignore[attr-defined]

    new_error = QColor("#1a73e8")
    new_help = QColor("#34a853")

    try:
        SHARED.theme.errorText = new_error
        SHARED.theme.helpText = new_help

        dock._ai_available = False  # type: ignore[attr-defined]
        dock.updateTheme()
        assert message_label.styleSheet() == f"color: {new_error.name()};"
        assert message_label.styleSheet() != original_style

        dock._ai_available = True  # type: ignore[attr-defined]
        dock.updateTheme()
        assert message_label.styleSheet() == f"color: {new_help.name()};"
    finally:
        SHARED.theme.errorText = original_error
        SHARED.theme.helpText = original_help
        dock._ai_available = original_available  # type: ignore[attr-defined]
        dock.updateTheme()


@pytest.mark.gui
def test_ai_copilot_dock_refresh_after_write_operations(monkeypatch, nwGUI):
    """When AI is enabled the interactive controls should be visible."""

    dummy_config = SimpleNamespace(
        ai=SimpleNamespace(
            enabled=True,
            api_key="token",
            api_key_from_env=False,
            dry_run_default=False,
            ask_before_apply=True,
            max_tokens=512,
            timeout=30,
        )
    )

    monkeypatch.setattr("novelwriter.extensions.ai_copilot.dock.CONFIG", dummy_config, raising=False)

    dock = AICopilotDock(nwGUI)
    stacked = dock.widget()
    assert isinstance(stacked, QStackedWidget)
    assert stacked.currentIndex() == dock._INTERACTIVE_INDEX  # type: ignore[attr-defined]

    send_button = dock.findChild(QPushButton, "aiCopilotSendButton")
    quick_rewrite = dock.findChild(QPushButton, "aiQuickAction_rewrite")

    assert send_button is not None
    assert quick_rewrite is not None
    assert send_button.isEnabled()
    assert quick_rewrite.isEnabled()

    dock.deleteLater()


@pytest.mark.gui
def test_ai_copilot_scope_selector_emits_signal(monkeypatch, qtbot, nwGUI):
    """Changing the scope selector should emit the dedicated signal."""

    dummy_config = SimpleNamespace(
        ai=SimpleNamespace(enabled=True, api_key="token", api_key_from_env=False)
    )
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.dock.CONFIG",
        dummy_config,
        raising=False,
    )

    dock = AICopilotDock(nwGUI)
    selector = dock.findChild(QComboBox, "aiContextScopeSelector")
    assert selector is not None

    with qtbot.waitSignal(dock.contextScopeChanged) as blocker:
        selector.setCurrentIndex(0)

    assert blocker.args[0] == "selection"
    assert dock.getCurrentScope() == "selection"
    dock.deleteLater()


@pytest.mark.gui
def test_ai_copilot_set_context_scope_updates_selector(monkeypatch, nwGUI):
    """Programmatic scope changes should update the selector widget."""

    dummy_config = SimpleNamespace(
        ai=SimpleNamespace(enabled=True, api_key="token", api_key_from_env=False)
    )
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.dock.CONFIG",
        dummy_config,
        raising=False,
    )

    dock = AICopilotDock(nwGUI)
    selector = dock.findChild(QComboBox, "aiContextScopeSelector")
    assert selector is not None

    dock.setContextScope("project")
    assert selector.currentData() == "project"
    assert dock.getCurrentScope() == "project"
    dock.deleteLater()


@pytest.mark.gui
def test_ai_copilot_quick_actions_present(monkeypatch, nwGUI):
    """Quick action buttons should be available when AI is enabled."""

    dummy_config = SimpleNamespace(
        ai=SimpleNamespace(enabled=True, api_key="token", api_key_from_env=False)
    )
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.dock.CONFIG",
        dummy_config,
        raising=False,
    )

    dock = AICopilotDock(nwGUI)
    for key in ("summarise", "continue", "rewrite"):
        button = dock.findChild(QPushButton, f"aiQuickAction_{key}")
        assert button is not None
        assert button.isEnabled()
    dock.deleteLater()



@pytest.mark.gui
def test_ai_copilot_cancel_button_aborts_request(monkeypatch, qtbot, nwGUI):
    """Pressing the cancel button should signal the worker and restore the UI."""

    config = SimpleNamespace(
        ai=SimpleNamespace(
            enabled=True,
            api_key="token",
            api_key_from_env=False,
            provider="openai",
            max_tokens=128,
            timeout=5.0,
        )
    )
    stream = _CancellableStream()
    stub_api = _StubAiApi(stream)

    monkeypatch.setattr("novelwriter.extensions.ai_copilot.dock.CONFIG", config, raising=False)
    monkeypatch.setattr("novelwriter.extensions.ai_copilot.handler.CONFIG", config, raising=False)
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.handler.NWAiApi",
        lambda project: stub_api,
        raising=False,
    )

    dock = AICopilotDock(nwGUI)
    try:
        input_edit = dock.findChild(QTextEdit, "aiCopilotInput")
        send_button = dock.findChild(QPushButton, "aiCopilotSendButton")
        cancel_button = dock.findChild(QPushButton, "aiCopilotCancelButton")
        status_label = dock.findChild(QLabel, "aiCopilotStatusMessage")

        assert input_edit is not None and send_button is not None
        assert cancel_button is not None and status_label is not None

        input_edit.setPlainText("Hello AI")
        qtbot.mouseClick(send_button, Qt.MouseButton.LeftButton)

        qtbot.waitUntil(stream.started.is_set, timeout=2000)
        qtbot.wait(50)

        cancel_button.click()

        qtbot.waitUntil(lambda: stream.closed.is_set(), timeout=2000)
        qtbot.waitUntil(
            lambda: not dock._request_manager.has_active_request(),  # type: ignore[attr-defined]
            timeout=2000,
        )
        qtbot.waitUntil(
            lambda: dock._messages and dock._messages[-1]["role"] == "system",  # type: ignore[attr-defined]
            timeout=2000,
        )

        assert dock._messages[-1]["content"] == dock.tr("Request cancelled.")  # type: ignore[attr-defined]
        assert status_label.text().startswith(dock.tr("Cancelled."))
    finally:
        manager = getattr(dock, "_request_manager", None)
        if manager is not None and manager.has_active_request():
            manager.cancel_request()
            qtbot.wait(50)
        dock.deleteLater()


@pytest.mark.gui
def test_ai_copilot_timeout_reports_error(monkeypatch, qtbot, nwGUI):
    """Requests exceeding the configured timeout should surface an error state."""

    timeout_seconds = 0.05
    config = SimpleNamespace(
        ai=SimpleNamespace(
            enabled=True,
            api_key="token",
            api_key_from_env=False,
            provider="openai",
            max_tokens=128,
            timeout=timeout_seconds,
        )
    )
    stream = _SlowStream(delay=0.2)
    stub_api = _StubAiApi(stream)

    monkeypatch.setattr("novelwriter.extensions.ai_copilot.dock.CONFIG", config, raising=False)
    monkeypatch.setattr("novelwriter.extensions.ai_copilot.handler.CONFIG", config, raising=False)
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.handler.NWAiApi",
        lambda project: stub_api,
        raising=False,
    )

    dock = AICopilotDock(nwGUI)
    try:
        input_edit = dock.findChild(QTextEdit, "aiCopilotInput")
        send_button = dock.findChild(QPushButton, "aiCopilotSendButton")
        status_label = dock.findChild(QLabel, "aiCopilotStatusMessage")

        assert input_edit is not None and send_button is not None
        assert status_label is not None

        input_edit.setPlainText("Trigger timeout")
        qtbot.mouseClick(send_button, Qt.MouseButton.LeftButton)

        qtbot.waitUntil(stream.started.is_set, timeout=2000)
        qtbot.waitUntil(
            lambda: dock._messages and dock._messages[-1]["role"] == "error",  # type: ignore[attr-defined]
            timeout=3000,
        )
        qtbot.waitUntil(
            lambda: not dock._request_manager.has_active_request(),  # type: ignore[attr-defined]
            timeout=2000,
        )

        error_message = dock._messages[-1]["content"]  # type: ignore[attr-defined]
        assert "timed out" in error_message.lower()
        assert status_label.text().startswith(dock.tr("An error occurred."))
    finally:
        manager = getattr(dock, "_request_manager", None)
        if manager is not None and manager.has_active_request():
            manager.cancel_request()
            qtbot.wait(50)
        dock.deleteLater()

@pytest.mark.gui
def test_ai_copilot_status_shows_provider(monkeypatch, nwGUI) -> None:
    dummy_config = SimpleNamespace(
        ai=SimpleNamespace(
            enabled=True,
            api_key="token",
            api_key_from_env=False,
            provider="openai",
        )
    )
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.dock.CONFIG",
        dummy_config,
        raising=False,
    )

    dock = AICopilotDock(nwGUI)

    status_label = dock.findChild(QLabel, "aiCopilotStatusMessage")
    assert status_label is not None
    assert "OpenAI (SDK)" in status_label.text()

    dock.deleteLater()
