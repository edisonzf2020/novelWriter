"""GUI tests for the AI preferences page."""
from __future__ import annotations

import pytest

from PyQt6.QtWidgets import QLabel, QDockWidget

from novelwriter import CONFIG
from novelwriter.ai.config import AIConfig
from novelwriter.common import NWConfigParser
from novelwriter.dialogs.preferences import GuiPreferences
from novelwriter.extensions.ai_copilot import AICopilotDock


@pytest.mark.gui
def test_preferences_ai_updates_configuration(qtbot, monkeypatch, nwGUI) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(CONFIG, "_ai_config", AIConfig(), raising=False)

    ai_config = CONFIG.ai
    ai_config.enabled = False

    dialog = GuiPreferences(nwGUI)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert dialog.aiEnabled.isChecked() is False
    assert dialog.aiProvider.isEnabled() is False
    assert dialog.aiDisabledMessage.isVisible() is True

    dialog.aiEnabled.setChecked(True)
    assert dialog.aiProvider.isEnabled() is True
    assert dialog.aiDisabledMessage.isVisible() is False

    dialog.aiBaseUrl.setText("https://example.test/v1")
    dialog.aiTimeout.setValue(55)
    dialog.aiMaxTokens.setValue(8192)
    dialog.aiDryRunDefault.setChecked(False)
    dialog.aiAskBeforeApply.setChecked(False)
    dialog.aiApiKey.setText("secret-key")

    dialog._doSave()

    updated = CONFIG.ai
    assert updated.enabled is True
    assert updated.openai_base_url == "https://example.test/v1"
    assert updated.timeout == 55
    assert updated.max_tokens == 8192
    assert updated.dry_run_default is False
    assert updated.ask_before_apply is False
    assert updated.api_key == "secret-key"

    dock = nwGUI.findChild(QDockWidget, "AICopilotDock")
    assert dock is not None
    assert isinstance(dock, AICopilotDock)

    message_label = dock.findChild(QLabel, "aiCopilotMessageLabel")
    assert message_label is not None

    qtbot.waitUntil(
        lambda: message_label.text()
        == dock.tr("Interactive Copilot features will appear here in a later release."),
        timeout=1000,
    )


@pytest.mark.gui
def test_preferences_ai_env_override_disables_key(qtbot, monkeypatch, nwGUI) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setattr(CONFIG, "_ai_config", AIConfig(), raising=False)

    ai_config = CONFIG.ai
    ai_config.load_from_main_config(NWConfigParser())
    ai_config.enabled = True

    dialog = GuiPreferences(nwGUI)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert dialog.aiEnabled.isChecked() is True
    assert dialog.aiApiKey.isEnabled() is False
    assert dialog.aiApiKey.isReadOnly() is True
    assert "OPENAI_API_KEY" in dialog.aiApiKey.placeholderText()

    dialog.close()
