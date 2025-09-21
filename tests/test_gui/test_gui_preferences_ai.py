"""GUI tests for the AI preferences page."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QDockWidget, QPushButton

from novelwriter import CONFIG, SHARED
from novelwriter.ai.config import AIConfig
from novelwriter.ai.models import ModelInfo
from novelwriter.common import NWConfigParser
from novelwriter.dialogs.preferences import GuiPreferences
from novelwriter.extensions.ai_copilot import AICopilotDock


@pytest.mark.gui
def test_preferences_ai_updates_configuration(qtbot, monkeypatch, nwGUI) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(CONFIG, "_ai_config", AIConfig(), raising=False)
    monkeypatch.setattr(
        "novelwriter.dialogs.preferences.importlib.util.find_spec",
        lambda name: None,
        raising=False,
    )

    ai_config = CONFIG.ai
    ai_config.enabled = False

    dialog = GuiPreferences(nwGUI)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert dialog.aiEnabled.isChecked() is False
    assert dialog.aiProvider.isEnabled() is False
    assert dialog.aiDisabledMessage.isVisible() is True
    assert dialog.aiProviderAvailability.isVisible() is False

    sdk_index = dialog.aiProvider.findData("openai")
    assert sdk_index >= 0
    model = dialog.aiProvider.model()
    item_getter = getattr(model, "item", None)
    if callable(item_getter):
        item = model.item(sdk_index)
        assert item is not None and item.isEnabled() is False
    else:
        assert dialog.aiProvider.itemData(sdk_index, Qt.ItemDataRole.UserRole) is False

    dialog.aiEnabled.setChecked(True)
    assert dialog.aiProvider.isEnabled() is True
    assert dialog.aiDisabledMessage.isVisible() is False
    assert dialog.aiProviderAvailability.isVisible() is True

    dialog.aiProvider.setCurrentIndex(sdk_index)
    qtbot.wait(10)
    assert dialog.aiProviderAvailability.isVisible() is True
    assert "OpenAI" in dialog.aiProviderAvailability.text()
    assert dialog.aiBaseUrl.isEnabled() is False

    dialog.aiProvider.setCurrentIndex(dialog.aiProvider.findData("openai"))
    qtbot.wait(10)
    # OpenAI SDK is not available, so availability message should remain visible
    assert dialog.aiProviderAvailability.isVisible() is True

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

    send_button = dock.findChild(QPushButton, "aiCopilotSendButton")
    assert send_button is not None
    qtbot.waitUntil(lambda: send_button.isEnabled(), timeout=1000)


@pytest.mark.gui
def test_preferences_ai_env_override_disables_key(qtbot, monkeypatch, nwGUI) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setattr(CONFIG, "_ai_config", AIConfig(), raising=False)
    monkeypatch.setattr(
        "novelwriter.dialogs.preferences.importlib.util.find_spec",
        lambda name: None,
        raising=False,
    )

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


@pytest.mark.gui
def test_preferences_ai_test_connection_populates_models(qtbot, monkeypatch, nwGUI) -> None:
    """The connection test should refresh the available models list."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(CONFIG, "_ai_config", AIConfig(), raising=False)
    monkeypatch.setattr(
        "novelwriter.dialogs.preferences.importlib.util.find_spec",
        lambda name: None,
        raising=False,
    )

    ai_config = CONFIG.ai
    ai_config.enabled = True
    ai_config.model = ""
    ai_config.api_key = "test-key"
    ai_config.openai_base_url = "https://example.test/v1"

    dummy_models = [
        ModelInfo(
            id="test-model",
            display_name="Test Model",
            description="",
            input_token_limit=4096,
            output_token_limit=2048,
            owned_by="organization",
            capabilities={},
            metadata={},
        )
    ]

    class DummyApi:
        def __init__(self, project):
            self.project = project

        def listAvailableModels(self, refresh: bool = False):
            return list(dummy_models)

    monkeypatch.setattr("novelwriter.ai.NWAiApi", DummyApi, raising=False)
    monkeypatch.setattr("novelwriter.dialogs.preferences.NWAiApi", DummyApi, raising=False)
    monkeypatch.setattr(SHARED, "_project", SimpleNamespace(isValid=False), raising=False)

    dialog = GuiPreferences(nwGUI)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    dialog.aiEnabled.setChecked(True)
    dialog._test_ai_connection()

    assert dialog.aiModelSelector.count() == len(dummy_models)
    assert dialog.aiModelSelector.currentData() == "test-model"
    assert "Connected successfully" in dialog.aiTestStatusLabel.text()

    dialog.close()
