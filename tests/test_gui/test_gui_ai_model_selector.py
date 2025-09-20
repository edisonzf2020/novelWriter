"""GUI tests for the AI model selector dialog."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from PyQt6.QtWidgets import QDialogButtonBox, QTreeView
from PyQt6.QtCore import QItemSelectionModel

from novelwriter.ai.models import ModelInfo
from novelwriter.extensions.ai_copilot.model_selector import ModelSelectorDialog


@pytest.fixture
def _patch_ai_environment(monkeypatch):
    """Provide a predictable AI configuration and provider."""
    dummy_config = SimpleNamespace(
        ai=SimpleNamespace(
            enabled=True,
            model="test-model",
            temperature=0.5,
            max_tokens=1024,
            set_availability_reason=lambda *_: None,
        )
    )
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.model_selector.CONFIG",
        dummy_config,
        raising=False,
    )

    dummy_shared = SimpleNamespace(project=object())
    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.model_selector.SHARED",
        dummy_shared,
        raising=False,
    )

    models = [
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
            return list(models)

    monkeypatch.setattr(
        "novelwriter.extensions.ai_copilot.model_selector.NWAiApi",
        DummyApi,
        raising=False,
    )
    monkeypatch.setattr(
        "novelwriter.ai.NWAiApi",
        DummyApi,
        raising=False,
    )

    yield models


@pytest.mark.gui
def test_model_selector_enables_ok_when_row_selected(qtbot, _patch_ai_environment):
    """Selecting a model row enables the OK button."""
    dialog = ModelSelectorDialog()
    qtbot.addWidget(dialog)

    tree = dialog.findChild(QTreeView, "modelsTreeView")
    assert tree is not None

    button_box = dialog.findChild(QDialogButtonBox)
    assert button_box is not None
    ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button is not None
    assert not ok_button.isEnabled()

    model_index = tree.model().index(0, 0)
    tree.selectionModel().select(
        model_index,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    tree.setCurrentIndex(model_index)
    qtbot.waitUntil(lambda: ok_button.isEnabled(), timeout=1000)

    tree.selectionModel().clearSelection()
    qtbot.waitUntil(lambda: not ok_button.isEnabled(), timeout=1000)

    dialog.close()
