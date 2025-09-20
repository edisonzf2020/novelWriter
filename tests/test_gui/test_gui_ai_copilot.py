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

import pytest

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QDockWidget, QLabel, QPushButton, QStackedWidget

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
def test_ai_copilot_status_shows_provider(monkeypatch, nwGUI) -> None:
    dummy_config = SimpleNamespace(
        ai=SimpleNamespace(
            enabled=True,
            api_key="token",
            api_key_from_env=False,
            provider="openai-sdk",
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
    assert "OpenAI Official SDK" in status_label.text()

    dock.deleteLater()
