"""AI Copilot dock widget implementation."""
from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from novelwriter import CONFIG, SHARED
from novelwriter.ai import NWAiApiError, TextRange

from .handler import CopilotRequestManager
from .diff_viewer import DiffPreviewController, DiffPreviewRequest, DiffPreviewWidget
from .history_dialog import AICopilotHistoryDialog

logger = logging.getLogger(__name__)


_QUICK_ACTION_TEMPLATES: Dict[str, Dict[str, object]] = {
    "summarise": {
        "label": "Summarise",
        "prompt": (
            "Provide a concise summary (no more than five bullet points) of the supplied material. "
            "Focus on key plot beats and character developments."
        ),
        "requires_selection": False,
        "yields_suggestion": False,
        "context_budget": 6000,
    },
    "continue": {
        "label": "Continue",
        "prompt": (
            "Write the next short paragraph that naturally follows the supplied context. Maintain "
            "the established tone and point of view."
        ),
        "requires_selection": False,
        "yields_suggestion": False,
        "context_budget": 5000,
    },
    "rewrite": {
        "label": "Rewrite",
        "prompt": (
            "Rewrite the selected passage to improve clarity and pacing while preserving voice and "
            "meaning. Return only the rewritten passage without commentary."
        ),
        "requires_selection": True,
        "yields_suggestion": True,
        "context_budget": 2000,
    },
}

_STATUS_TEXT = {
    "ready": "Ready.",
    "collecting_context": "Collecting context...",
    "requesting_completion": "Contacting provider...",
    "logging_conversation": "Recording conversation...",
    "completed": "Completed.",
    "cancelled": "Cancelled.",
    "error": "An error occurred.",
    "preparing": "Preparing diff preview...",
}


@dataclass
class QuickActionDefinition:
    """Metadata describing a quick action button."""

    key: str
    label: str
    prompt: str
    requires_selection: bool
    yields_suggestion: bool
    context_budget: Optional[int]


class AICopilotDock(QDockWidget):
    """Dock widget hosting the AI Copilot UI with context selection capability."""

    contextScopeChanged = pyqtSignal(str)

    _SCOPE_KEYS: Tuple[str, ...] = (
        "selection",
        "current_document",
        "outline",
        "project",
    )

    _PLACEHOLDER_INDEX = 0
    _INTERACTIVE_INDEX = 1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("AICopilotDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        self._currentScope: str = "current_document"
        self._ai_available: bool = False
        self._availability_reason: str | None = None
        self._current_provider_id: str | None = None

        self._stack = QStackedWidget(self)
        self._placeholderPage = self._build_placeholder_page()
        self._interactivePage = self._build_interactive_page()
        self._stack.addWidget(self._placeholderPage)
        self._stack.addWidget(self._interactivePage)
        self.setWidget(self._stack)

        self._messages: list[Dict[str, str]] = []
        self._streaming_index: Optional[int] = None
        self._quick_actions: Dict[str, QuickActionDefinition] = {}
        self._request_manager: Optional[CopilotRequestManager] = None
        self._diff_controller: Optional[DiffPreviewController] = None
        self._pending_suggestion: Optional[Dict[str, object]] = None
        self._active_action: Optional[str] = None
        self._request_in_progress: bool = False

        self._rebuild_quick_actions()
        self.refresh_from_config()
        self._applyTranslations()

    ##
    #  Qt Events
    ##

    def changeEvent(self, event: QEvent) -> None:  # pragma: no cover - Qt runtime behaviour
        super().changeEvent(event)
        if event.type() == QEvent.Type.LanguageChange:
            self._rebuild_quick_actions()
            self._applyTranslations()

    ##
    #  Public API
    ##

    def getCurrentScope(self) -> str:
        """Return the currently selected context scope."""

        return self._currentScope

    def setContextScope(self, scope: str) -> None:
        """Update the context scope programmatically without emitting signals."""

        if scope not in self._SCOPE_KEYS or scope == self._currentScope:
            return

        self._currentScope = scope
        self._populateScopeSelector()

    def refresh_from_config(self) -> None:
        """Refresh availability state and update the UI accordingly."""

        ai_config = getattr(CONFIG, "ai", None)
        provider_id = getattr(ai_config, "provider", None)

        self._ai_available, self._availability_reason = self._resolve_availability()
        provider_changed = provider_id != getattr(self, "_current_provider_id", None)
        self._current_provider_id = provider_id

        self._updateScopeSelectorState()
        self._updateModelStatusState()
        if self._ai_available:
            self._stack.setCurrentIndex(self._INTERACTIVE_INDEX)
            self._ensure_request_manager()
            if provider_changed and self._request_manager is not None and provider_id is not None:
                self._request_manager.on_provider_changed(provider_id)
            self._toggle_interaction_enabled(not self._request_in_progress)
            if hasattr(self, "_historyButton"):
                self._historyButton.setEnabled(True)
                self._historyButton.setToolTip("")
            self._set_status_message("ready")
        else:
            self._stack.setCurrentIndex(self._PLACEHOLDER_INDEX)
            disabled_reason = (
                self._availability_reason
                or self.tr("AI features are currently disabled or missing optional dependencies.")
            )
            self._placeholderDetailLabel.setText(disabled_reason)
            if hasattr(self, "_historyButton"):
                self._historyButton.setEnabled(False)
                self._historyButton.setToolTip(disabled_reason)
            self._clear_pending_suggestion(rollback=True)
            self._previewFrame.setVisible(False)

    def updateTheme(self) -> None:
        """Update fonts and colours to match the current theme."""

        theme = SHARED.theme
        self._placeholderStatusLabel.setFont(theme.guiFontB)
        self._placeholderDetailLabel.setFont(theme.guiFont)
        colour = theme.helpText.name() if self._ai_available else theme.errorText.name()
        self._placeholderDetailLabel.setStyleSheet(f"color: {colour};")
        if self._messagesView.document():
            self._messagesView.document().setDefaultFont(theme.guiFont)
        self._inputEdit.setFont(theme.guiFont)
        self._runStatusLabel.setFont(theme.guiFontSmall)
        self._previewTitleLabel.setFont(theme.guiFontB)
        if hasattr(self, "_diffWidget"):
            self._diffWidget.apply_theme(theme)
        if hasattr(self, "_modelLabel"):
            self._modelLabel.setFont(theme.guiFont)
        if hasattr(self, "_currentModelButton"):
            self._currentModelButton.setFont(theme.guiFont)
        if hasattr(self, "_historyButton"):
            self._historyButton.setFont(theme.guiFont)
        self._applyTranslations()

    ##
    #  Private helpers
    ##

    def _build_placeholder_page(self) -> QWidget:
        theme = SHARED.theme
        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._placeholderStatusLabel = QLabel(container)
        self._placeholderStatusLabel.setObjectName("aiCopilotStatusLabel")
        self._placeholderStatusLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._placeholderStatusLabel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self._placeholderStatusLabel.setFont(theme.guiFontB)
        self._placeholderStatusLabel.setText(self.tr("AI Copilot is temporarily unavailable"))

        self._placeholderDetailLabel = QLabel(container)
        self._placeholderDetailLabel.setObjectName("aiCopilotMessageLabel")
        self._placeholderDetailLabel.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._placeholderDetailLabel.setWordWrap(True)
        self._placeholderDetailLabel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        self._placeholderDetailLabel.setFont(theme.guiFont)
        self._placeholderDetailLabel.setStyleSheet(f"color: {theme.fadedText.name()};")
        self._placeholderDetailLabel.setText(
            self.tr("AI features are currently disabled or missing optional dependencies.")
        )

        layout.addStretch(1)
        layout.addWidget(self._placeholderStatusLabel)
        layout.addWidget(self._placeholderDetailLabel)
        layout.addStretch(2)

        container.setLayout(layout)
        return container

    def _build_interactive_page(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._scopeFrame = self._buildScopeSelectorRow()
        layout.addWidget(self._scopeFrame)

        self._messagesView = QTextBrowser(container)
        self._messagesView.setObjectName("aiCopilotConversation")
        self._messagesView.setOpenExternalLinks(False)
        self._messagesView.setReadOnly(True)
        self._messagesView.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self._messagesView, stretch=1)

        self._runStatusLabel = QLabel(container)
        self._runStatusLabel.setObjectName("aiCopilotStatusMessage")
        self._runStatusLabel.setText(self.tr("Ready."))
        layout.addWidget(self._runStatusLabel)

        self._quickActionRow = QFrame(container)
        quick_layout = QHBoxLayout()
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.setSpacing(8)
        self._quickActionButtons: Dict[str, QPushButton] = {}
        for key, template in _QUICK_ACTION_TEMPLATES.items():
            button = QPushButton(self.tr(str(template["label"])), self._quickActionRow)
            button.setObjectName(f"aiQuickAction_{key}")
            button.clicked.connect(lambda _, action=key: self._trigger_quick_action(action))
            quick_layout.addWidget(button)
            self._quickActionButtons[key] = button
        quick_layout.addStretch(1)
        self._quickActionRow.setLayout(quick_layout)
        layout.addWidget(self._quickActionRow)

        self._inputEdit = QTextEdit(container)
        self._inputEdit.setObjectName("aiCopilotInput")
        self._inputEdit.setPlaceholderText(self.tr("Type a request or choose a quick action"))
        self._inputEdit.setFixedHeight(96)
        layout.addWidget(self._inputEdit)

        button_row = QFrame(container)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self._sendButton = QPushButton(self.tr("Send"), button_row)
        self._sendButton.setObjectName("aiCopilotSendButton")
        self._sendButton.clicked.connect(self._handleSendClicked)
        self._cancelButton = QPushButton(self.tr("Cancel"), button_row)
        self._cancelButton.setObjectName("aiCopilotCancelButton")
        self._cancelButton.clicked.connect(self._handleCancel)
        self._cancelButton.setVisible(False)

        button_layout.addStretch(1)
        button_layout.addWidget(self._cancelButton)
        button_layout.addWidget(self._sendButton)
        button_row.setLayout(button_layout)
        layout.addWidget(button_row)

        self._previewFrame = QFrame(container)
        self._previewFrame.setObjectName("aiCopilotPreviewFrame")
        self._previewFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self._previewFrame.setVisible(False)

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)
        self._previewTitleLabel = QLabel(self.tr("Suggestion preview"), self._previewFrame)
        self._diffWidget = DiffPreviewWidget(self._previewFrame)
        preview_button_row = QHBoxLayout()
        preview_button_row.setContentsMargins(0, 0, 0, 0)
        preview_button_row.setSpacing(8)
        self._applyButton = QPushButton(self.tr("Apply"), self._previewFrame)
        self._applyButton.clicked.connect(self._apply_suggestion)
        self._dismissButton = QPushButton(self.tr("Dismiss"), self._previewFrame)
        self._dismissButton.clicked.connect(self._dismiss_suggestion)
        preview_button_row.addStretch(1)
        preview_button_row.addWidget(self._dismissButton)
        preview_button_row.addWidget(self._applyButton)

        preview_layout.addWidget(self._previewTitleLabel)
        preview_layout.addWidget(self._diffWidget)
        preview_layout.addLayout(preview_button_row)
        self._previewFrame.setLayout(preview_layout)
        layout.addWidget(self._previewFrame)

        # Model status bar at the bottom
        self._modelStatusFrame = self._buildModelStatusRow()
        layout.addWidget(self._modelStatusFrame)

        container.setLayout(layout)
        return container

    def _rebuild_quick_actions(self) -> None:
        self._quick_actions = {
            key: QuickActionDefinition(
                key=key,
                label=self.tr(template["label"]),
                prompt=str(template["prompt"]),
                requires_selection=bool(template["requires_selection"]),
                yields_suggestion=bool(template["yields_suggestion"]),
                context_budget=template.get("context_budget"),
            )
            for key, template in _QUICK_ACTION_TEMPLATES.items()
        }
        if hasattr(self, "_quickActionButtons"):
            for action_key, button in self._quickActionButtons.items():
                button.setText(self._quick_actions[action_key].label)

    def _buildScopeSelectorRow(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("aiContextScopeRow")
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Plain)

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(8)

        self._scopeLabel = QLabel(self.tr("Context:"), frame)
        self._scopeLabel.setObjectName("aiContextScopeLabel")
        self._scopeLabel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._contextScopeSelector = QComboBox(frame)
        self._contextScopeSelector.setObjectName("aiContextScopeSelector")
        self._contextScopeSelector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._contextScopeSelector.currentIndexChanged.connect(self._handleScopeChanged)

        row_layout.addWidget(self._scopeLabel)
        row_layout.addWidget(self._contextScopeSelector, stretch=1)
        row_layout.addStretch(1)
        self._historyButton = QPushButton(self.tr("History"), frame)
        self._historyButton.setObjectName("aiHistoryButton")
        self._historyButton.clicked.connect(self._show_history_dialog)
        row_layout.addWidget(self._historyButton)
        frame.setLayout(row_layout)
        self._populateScopeSelector()
        return frame

    def _buildModelStatusRow(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("aiModelStatusRow")
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Plain)

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(8)

        self._modelLabel = QLabel(self.tr("Model:"), frame)
        self._modelLabel.setObjectName("aiModelLabel")
        self._modelLabel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._modelSettingsButton = QPushButton(frame)
        self._modelSettingsButton.setObjectName("aiModelSettingsButton")
        self._modelSettingsButton.setText("⚙")
        self._modelSettingsButton.setToolTip(self.tr("Model Settings"))
        self._modelSettingsButton.setFixedSize(24, 24)
        self._modelSettingsButton.clicked.connect(self._handleModelSettingsClicked)

        self._currentModelButton = QPushButton(frame)
        self._currentModelButton.setObjectName("aiCurrentModelButton")
        self._currentModelButton.setText(self.tr("Loading..."))
        self._currentModelButton.setToolTip(self.tr("Click to change model"))
        self._currentModelButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._currentModelButton.clicked.connect(self._handleModelSelectionClicked)

        row_layout.addWidget(self._modelLabel)
        row_layout.addWidget(self._modelSettingsButton)
        row_layout.addWidget(self._currentModelButton, stretch=1)
        frame.setLayout(row_layout)
        return frame

    def _scopeOptions(self) -> list[Tuple[str, str]]:
        return [
            ("selection", self.tr("Selection")),
            ("current_document", self.tr("Current Document")),
            ("outline", self.tr("Outline")),
            ("project", self.tr("Entire Project")),
        ]

    def _populateScopeSelector(self) -> None:
        if not hasattr(self, "_contextScopeSelector") or self._contextScopeSelector is None:
            return

        selector = self._contextScopeSelector
        options = self._scopeOptions()

        selector.blockSignals(True)
        selector.clear()

        current_index = 0
        for index, (key, label) in enumerate(options):
            selector.addItem(label, key)
            if key == self._currentScope:
                current_index = index

        selector.setCurrentIndex(current_index)
        selector.blockSignals(False)

    def _handleScopeChanged(self, index: int) -> None:
        if not hasattr(self, "_contextScopeSelector") or self._contextScopeSelector is None:
            return

        scope_key = self._contextScopeSelector.itemData(index, role=Qt.ItemDataRole.UserRole)
        if not isinstance(scope_key, str) or scope_key == self._currentScope:
            return

        self._currentScope = scope_key
        self.contextScopeChanged.emit(scope_key)
        logger.debug("Context scope changed to: %s", scope_key)

    def _handleModelSelectionClicked(self) -> None:
        """Handle model selection button click."""
        from .model_selector import ModelSelectorDialog
        
        dialog = ModelSelectorDialog(self)
        dialog.modelSelected.connect(self._onModelSelected)
        dialog.exec()

    def _handleModelSettingsClicked(self) -> None:
        """Handle model settings button click."""
        from .model_selector import ModelParametersDialog
        
        dialog = ModelParametersDialog(self)
        dialog.parametersChanged.connect(self._onParametersChanged)
        dialog.exec()

    def _onModelSelected(self, model_id: str, model_data: dict) -> None:
        """Handle model selection from dialog."""
        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            logger.warning("AI config not available for model selection")
            return

        try:
            # Update configuration
            ai_config.model = model_id
            
            # Update model metadata if provided
            model_metadata = model_data.get("model_metadata")
            if isinstance(model_metadata, dict):
                ai_config.default_model_metadata = model_metadata
            
            # Update parameters if provided
            parameters = model_data.get("parameters", {})
            if "temperature" in parameters:
                ai_config.temperature = float(parameters["temperature"])
            if "max_tokens" in parameters:
                ai_config.max_tokens = int(parameters["max_tokens"])
            
            # Save configuration
            CONFIG.saveConfig()
            
            # Reset provider to use new model
            if self._request_manager and hasattr(self._request_manager, "api"):
                self._request_manager.api.resetProvider()
            
            # Update UI
            self._updateModelStatusState()
            logger.info("Selected model '%s'", model_id)
            
        except Exception as exc:
            logger.error("Failed to update model configuration: %s", exc)
            self._display_error(self.tr("Failed to update model: {0}").format(str(exc)))

    def _onParametersChanged(self, parameters: dict) -> None:
        """Handle parameter changes from dialog."""
        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            logger.warning("AI config not available for parameter update")
            return

        try:
            # Update parameters
            if "temperature" in parameters:
                ai_config.temperature = float(parameters["temperature"])
            if "max_tokens" in parameters:
                ai_config.max_tokens = int(parameters["max_tokens"])
            
            # Save configuration
            CONFIG.saveConfig()
            logger.info("Updated model parameters: %s", parameters)
            
        except Exception as exc:
            logger.error("Failed to update model parameters: %s", exc)
            self._display_error(self.tr("Failed to update parameters: {0}").format(str(exc)))

    def _updateScopeSelectorState(self) -> None:
        enabled = self._ai_available and not self._request_in_progress
        if hasattr(self, "_contextScopeSelector") and self._contextScopeSelector is not None:
            self._contextScopeSelector.setEnabled(enabled)
        if hasattr(self, "_scopeLabel") and self._scopeLabel is not None:
            self._scopeLabel.setEnabled(enabled)
        if hasattr(self, "_scopeFrame") and self._scopeFrame is not None:
            self._scopeFrame.setEnabled(enabled)

    def _updateModelStatusState(self) -> None:
        """Update model status bar enabled state and current model display."""
        enabled = self._ai_available and not self._request_in_progress
        if hasattr(self, "_currentModelButton") and self._currentModelButton is not None:
            self._currentModelButton.setEnabled(enabled)
        if hasattr(self, "_modelSettingsButton") and self._modelSettingsButton is not None:
            self._modelSettingsButton.setEnabled(enabled)
        if hasattr(self, "_modelLabel") and self._modelLabel is not None:
            self._modelLabel.setEnabled(enabled)
            provider_name = self._provider_display_name()
            label_text = self.tr("Model:") if not provider_name else self.tr("Model ({0}):").format(provider_name)
            self._modelLabel.setText(label_text)
        if hasattr(self, "_modelStatusFrame") and self._modelStatusFrame is not None:
            self._modelStatusFrame.setEnabled(enabled)

        # Update current model display
        if hasattr(self, "_currentModelButton") and self._currentModelButton is not None:
            current_model = self._getCurrentModelName()
            self._currentModelButton.setText(current_model)

    def _getCurrentModelName(self) -> str:
        """Get the current model name from configuration."""
        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            return self.tr("No model")
        
        model = getattr(ai_config, "model", "")
        if not model:
            return self.tr("No model")
        
        # Try to get display name from metadata if available
        metadata = getattr(ai_config, "default_model_metadata", None)
        if isinstance(metadata, dict):
            display_name = metadata.get("display_name")
            if display_name and isinstance(display_name, str):
                return display_name
        
        return model

    def _resolve_availability(self) -> Tuple[bool, Optional[str]]:
        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            return False, self.tr("AI configuration is not enabled yet. Enable AI options in Preferences.")

        enabled = bool(getattr(ai_config, "enabled", False))
        if not enabled:
            reason = getattr(ai_config, "availability_reason", None)
            return False, reason or self.tr("AI features are disabled in the preferences.")

        if not getattr(ai_config, "api_key", "") and not getattr(ai_config, "api_key_from_env", False):
            return False, self.tr("Configure an AI API key in Preferences to enable the Copilot.")

        try:
            __import__("novelwriter.ai")
        except Exception as exc:  # pragma: no cover - environment dependent
            message = str(exc) or exc.__class__.__name__
            logger.warning("AI Copilot dependencies unavailable: %s", message)
            return False, self.tr("AI module could not be loaded: {0}").format(message)

        return True, None

    def _ensure_request_manager(self) -> None:
        if self._request_manager is not None:
            return
        self._request_manager = CopilotRequestManager()
        self._request_manager.chunkProduced.connect(self._handle_chunk)
        self._request_manager.requestFinished.connect(self._handle_finished)
        self._request_manager.requestFailed.connect(self._handle_failed)
        self._request_manager.requestCancelled.connect(self._handle_cancelled)
        self._request_manager.statusChanged.connect(self._handle_status_changed)

    def _ensure_diff_controller(self) -> None:
        """Initialise the diff preview controller when required."""

        if self._diff_controller is not None:
            return
        if self._request_manager is None:
            return

        controller = DiffPreviewController(self._request_manager.api, self)
        controller.previewReady.connect(self._handle_diff_ready)
        controller.previewFailed.connect(self._handle_diff_failed)
        controller.previewCancelled.connect(self._handle_diff_cancelled)
        controller.statusChanged.connect(self._handle_diff_status)
        self._diff_controller = controller

    def _show_history_dialog(self) -> None:
        """Open the modal dialog showing AI transaction history."""

        self._ensure_request_manager()
        if self._request_manager is None:
            self._display_error(self.tr("History is unavailable."))
            return

        dialog = AICopilotHistoryDialog(self._request_manager.api, self)
        dialog.exec()

    def _toggle_interaction_enabled(self, enabled: bool) -> None:
        if hasattr(self, "_quickActionButtons"):
            for button in self._quickActionButtons.values():
                button.setEnabled(enabled)
        if hasattr(self, "_sendButton"):
            self._sendButton.setEnabled(enabled)
        if hasattr(self, "_inputEdit"):
            self._inputEdit.setEnabled(enabled)
        self._updateScopeSelectorState()
        self._updateModelStatusState()

    def _handleSendClicked(self) -> None:
        text = self._inputEdit.toPlainText().strip()
        if not text:
            self._display_error(self.tr("Enter a message or choose a quick action."))
            return
        self._start_request(user_text=text, action_key=None)

    def _trigger_quick_action(self, action_key: str) -> None:
        self._start_request(user_text=self._inputEdit.toPlainText().strip(), action_key=action_key)

    def _start_request(self, *, user_text: str, action_key: Optional[str]) -> None:
        if not self._ai_available:
            self._display_error(self.tr("AI Copilot is disabled."))
            return

        self._ensure_request_manager()
        if self._request_manager is None:
            self._display_error(self.tr("Unable to initialise AI Copilot."))
            return
        if self._request_manager.has_active_request():
            self._display_error(self.tr("Wait for the current request to finish."))
            return

        selection_text, selection_range, document_handle = self._capture_selection_state()
        action = self._quick_actions.get(action_key) if action_key else None
        if action and action.requires_selection and (not selection_text or not selection_range):
            self._display_error(self.tr("Select some text before using rewrite."))
            return

        prompt = self._compose_prompt(action_key, user_text, selection_text)
        if not prompt:
            self._display_error(self.tr("Nothing to send."))
            return

        self._clear_pending_suggestion(rollback=True)
        self._active_action = action_key
        self._request_in_progress = True
        self._toggle_interaction_enabled(False)
        self._cancelButton.setVisible(True)
        self._inputEdit.clear()

        label_text = getattr(action, 'label', '') if action else ''
        self._append_message("user", prompt if action_key is None else user_text or label_text)
        self._append_message("assistant", "")
        self._streaming_index = len(self._messages) - 1
        self._render_messages()

        context_budget = getattr(action, 'context_budget', None) if action else None
        request = self._request_manager.build_request(
            scope=self._currentScope,
            user_prompt=prompt,
            quick_action=action_key,
            selection_text=selection_text,
            selection_range=selection_range,
            document_handle=document_handle,
            include_memory=True,
            context_budget=context_budget,
        )
        try:
            self._request_manager.start_request(request)
        except Exception as exc:  # noqa: BLE001 - defensive guard
            logger.exception("Failed to start AI request")
            self._request_in_progress = False
            self._toggle_interaction_enabled(True)
            self._cancelButton.setVisible(False)
            self._display_error(str(exc))

    def _handleCancel(self) -> None:
        if self._request_manager is not None:
            self._request_manager.cancel_request()

    def _handle_chunk(self, chunk: str) -> None:
        if self._streaming_index is None:
            return
        self._messages[self._streaming_index]["content"] += chunk
        self._render_messages()

    def _handle_finished(self, payload: dict) -> None:
        self._request_in_progress = False
        self._toggle_interaction_enabled(True)
        self._cancelButton.setVisible(False)
        self._streaming_index = None

        response = str(payload.get("response", "")).strip()
        if not response:
            response = self.tr("(No response received)")
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages[-1]["content"] = response
        else:
            self._append_message("assistant", response)
        self._render_messages()
        self._set_status_message("completed")

        action_key = payload.get("quick_action")
        if action_key == "rewrite" and response:
            selection_range = payload.get("selection_range")
            handle = payload.get("document_handle")
            if isinstance(selection_range, tuple) and handle:
                self._show_suggestion_preview(handle, selection_range, response)

    def _handle_failed(self, message: str) -> None:
        self._request_in_progress = False
        self._toggle_interaction_enabled(True)
        self._cancelButton.setVisible(False)
        self._streaming_index = None
        self._append_message("error", message)
        self._render_messages()
        self._set_status_message("error")

    def _handle_cancelled(self) -> None:
        self._request_in_progress = False
        self._toggle_interaction_enabled(True)
        self._cancelButton.setVisible(False)
        self._streaming_index = None
        self._append_message("system", self.tr("Request cancelled."))
        self._render_messages()
        self._set_status_message("cancelled")

    def _handle_status_changed(self, status_key: str) -> None:
        self._set_status_message(status_key)

    def _set_status_message(self, status_key: str) -> None:
        text = _STATUS_TEXT.get(status_key, status_key)
        base_text = self.tr(text)
        suffix = self._provider_status_suffix()
        if suffix:
            base_text = f"{base_text} {suffix}"
        self._runStatusLabel.setText(base_text)

    def _provider_status_suffix(self) -> str:
        provider_label = self._provider_display_name()
        if not provider_label:
            return ""
        return self.tr("(Provider: {0})").format(provider_label)

    def _provider_display_name(self) -> str:
        provider_id = self._current_provider_id or getattr(getattr(CONFIG, "ai", None), "provider", None)
        mapping = {
            "openai": self.tr("OpenAI (SDK)"),
            "openai": self.tr("OpenAI (SDK)"),
        }
        if not provider_id:
            return ""
        return mapping.get(
            provider_id,
            provider_id.replace("_", " ").replace("-", " ").title(),
        )

    def _compose_prompt(
        self,
        action_key: Optional[str],
        user_text: str,
        selection_text: str,
    ) -> str:
        if action_key is None:
            return user_text.strip()

        action = self._quick_actions[action_key]
        prompt_parts = [action.prompt]
        if selection_text.strip():
            prompt_parts.append(self.tr("Selected passage:\n{0}").format(selection_text.strip()))
        if user_text.strip():
            prompt_parts.append(
                self.tr("Additional instructions from the author:\n{0}").format(user_text.strip())
            )
        return "\n\n".join(prompt_parts)

    def _append_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def _render_messages(self) -> None:
        if not self._messages:
            self._messagesView.setHtml("")
            return

        parts: list[str] = []
        for message in self._messages:
            role = message["role"]
            label = {
                "user": self.tr("You"),
                "assistant": self.tr("Copilot"),
                "system": self.tr("System"),
                "error": self.tr("Error"),
            }.get(role, role.title())
            classes = {
                "user": "role-user",
                "assistant": "role-assistant",
                "system": "role-system",
                "error": "role-error",
            }.get(role, "role-other")
            escaped = html.escape(message["content"]).replace("\n", "<br />")
            parts.append(f"<div class='{classes}'><strong>{label}:</strong> {escaped}</div>")

        html_output = "\n".join(parts)
        self._messagesView.setHtml(html_output)
        self._messagesView.moveCursor(QTextCursor.MoveOperation.End)

    def _display_error(self, message: str) -> None:
        self._append_message("error", message)
        self._render_messages()
        self._set_status_message("error")

    def _capture_selection_state(self) -> Tuple[str, Optional[Tuple[int, int]], Optional[str]]:
        editor = SHARED.mainGui.docEditor
        handle = editor.docHandle
        selection_text = editor.getSelectedText() if handle else ""
        selection_range: Optional[Tuple[int, int]] = None
        cursor = editor.textCursor()
        if cursor.hasSelection():
            selection_range = (cursor.selectionStart(), cursor.selectionEnd())
        return selection_text, selection_range, handle

    def _show_suggestion_preview(
        self,
        handle: str,
        selection_range: Tuple[int, int],
        suggestion_text: str,
    ) -> None:
        if self._request_manager is None:
            return
        self._ensure_diff_controller()
        if self._diff_controller is None:
            self._display_error(self.tr("Unable to prepare diff preview."))
            return

        self._clear_pending_suggestion(rollback=True)
        self._pending_suggestion = {
            "transaction_id": None,
            "suggestion_id": None,
            "handle": handle,
            "range": (int(selection_range[0]), int(selection_range[1])),
            "new_text": suggestion_text,
        }
        self._applyButton.setEnabled(False)
        self._previewFrame.setVisible(True)
        self._diffWidget.show_progress(self.tr("Generating diff preview..."))

        request = DiffPreviewRequest(
            handle=handle,
            selection_range=(int(selection_range[0]), int(selection_range[1])),
            new_text=suggestion_text,
        )
        self._diff_controller.request_preview(request)

    def _handle_diff_ready(self, result) -> None:
        if self._pending_suggestion is None:
            if self._request_manager is not None:
                try:
                    self._request_manager.api.rollback_transaction(result.transaction_id)
                except Exception:  # noqa: BLE001 - defensive cleanup
                    logger.debug("Rollback failed for orphaned diff result")
            return

        self._pending_suggestion["transaction_id"] = result.transaction_id
        self._pending_suggestion["suggestion_id"] = result.suggestion.id
        self._pending_suggestion["diff_stats"] = result.stats
        self._pending_suggestion["request"] = result.request
        self._diffWidget.display_result(result)
        self._applyButton.setEnabled(True)
        self._set_status_message("ready")

    def _handle_diff_failed(self, message: str) -> None:
        if self._pending_suggestion and self._pending_suggestion.get("transaction_id"):
            transaction_id = str(self._pending_suggestion.get("transaction_id"))
            if self._request_manager is not None and transaction_id:
                try:
                    self._request_manager.api.rollback_transaction(transaction_id)
                except Exception:  # noqa: BLE001 - defensive cleanup
                    logger.debug("Rollback failed after diff error")
        self._pending_suggestion = None
        self._applyButton.setEnabled(False)
        self._diffWidget.show_error(self.tr("Diff preview failed: {0}").format(message))
        self._set_status_message("error")

    def _handle_diff_cancelled(self) -> None:
        if self._pending_suggestion is None:
            return
        self._applyButton.setEnabled(False)
        self._diffWidget.show_placeholder(self.tr("Diff preview cancelled."))
        self._set_status_message("cancelled")

    def _handle_diff_status(self, status: str) -> None:
        self._set_status_message(status)

    def _apply_suggestion(self) -> None:
        if not self._pending_suggestion or self._request_manager is None:
            return
        api = self._request_manager.api
        suggestion_id = self._pending_suggestion.get("suggestion_id")
        transaction_id = self._pending_suggestion.get("transaction_id")
        if not suggestion_id or not transaction_id:
            self._display_error(self.tr("Diff preview is still running."))
            return
        suggestion_id = str(suggestion_id)
        transaction_id = str(transaction_id)
        handle = str(self._pending_suggestion["handle"])
        selection_range = self._pending_suggestion["range"]
        new_text = str(self._pending_suggestion["new_text"])

        try:
            applied = api.applySuggestion(suggestion_id)
            if not applied:
                raise NWAiApiError("Suggestion could not be applied.")
            api.commit_transaction(transaction_id)
        except NWAiApiError as exc:
            logger.error("Failed to apply suggestion: %s", exc)
            self._display_error(str(exc))
            try:
                api.rollback_transaction(transaction_id)
            except Exception:  # noqa: BLE001
                logger.debug("Rollback failed after apply error")
            self._pending_suggestion = None
            self._previewFrame.setVisible(False)
            return

        self._append_message("system", self.tr("Suggestion applied."))
        self._render_messages()
        self._previewFrame.setVisible(False)
        self._pending_suggestion = None
        if isinstance(selection_range, (tuple, list)) and len(selection_range) == 2:
            self._refresh_editor_after_apply(handle, (int(selection_range[0]), int(selection_range[1])), new_text)

    def _dismiss_suggestion(self) -> None:
        self._clear_pending_suggestion(rollback=True)
        self._previewFrame.setVisible(False)
        self._append_message("system", self.tr("Suggestion dismissed."))
        self._render_messages()

    def _clear_pending_suggestion(self, rollback: bool) -> None:
        if self._diff_controller is not None:
            self._diff_controller.cancel_pending()
        if not self._pending_suggestion or self._request_manager is None:
            self._pending_suggestion = None
            if hasattr(self, "_applyButton"):
                self._applyButton.setEnabled(False)
            return

        if rollback:
            transaction_id = self._pending_suggestion.get("transaction_id")
            if transaction_id:
                try:
                    self._request_manager.api.rollback_transaction(str(transaction_id))
                except Exception:  # noqa: BLE001 - defensive guard
                    logger.debug("Rollback failed while clearing pending suggestion")
        self._pending_suggestion = None
        self._applyButton.setEnabled(False)

    def _refresh_editor_after_apply(
        self,
        handle: str,
        selection_range: Tuple[int, int],
        new_text: str,
    ) -> None:
        gui = SHARED.mainGui
        editor = gui.docEditor
        if editor.docHandle != handle:
            gui.openDocument(handle, changeFocus=False)
            editor = gui.docEditor
        if not editor.loadText(handle, None):
            logger.warning("Failed to reload document '%s' after applying suggestion", handle)
            return

        cursor = editor.textCursor()
        start = int(selection_range[0])
        cursor.setPosition(start)
        cursor.setPosition(start + len(new_text), QTextCursor.MoveMode.KeepAnchor)
        editor.setTextCursor(cursor)
        editor.ensureCursorVisibleNoCentre()

    def _applyTranslations(self) -> None:
        self.setWindowTitle(self.tr("AI Copilot"))
        if hasattr(self, "_scopeLabel") and self._scopeLabel is not None:
            self._scopeLabel.setText(self.tr("Context:"))
        self._populateScopeSelector()
        if hasattr(self, "_placeholderStatusLabel"):
            self._placeholderStatusLabel.setText(self.tr("AI Copilot is temporarily unavailable"))
        if hasattr(self, "_placeholderDetailLabel") and self._availability_reason:
            self._placeholderDetailLabel.setText(self._availability_reason)
        if hasattr(self, "_sendButton"):
            self._sendButton.setText(self.tr("Send"))
        if hasattr(self, "_cancelButton"):
            self._cancelButton.setText(self.tr("Cancel"))
        if hasattr(self, "_previewTitleLabel"):
            self._previewTitleLabel.setText(self.tr("Suggestion preview"))
        if hasattr(self, "_applyButton"):
            self._applyButton.setText(self.tr("Apply"))
        if hasattr(self, "_dismissButton"):
            self._dismissButton.setText(self.tr("Dismiss"))
        if hasattr(self, "_historyButton"):
            self._historyButton.setText(self.tr("History"))
        if hasattr(self, "_inputEdit"):
            self._inputEdit.setPlaceholderText(
                self.tr("Type a request or choose a quick action")
            )
        for action_key, button in getattr(self, "_quickActionButtons", {}).items():
            button.setText(self._quick_actions[action_key].label)
        if hasattr(self, "_modelLabel"):
            self._modelLabel.setText(self.tr("Model:"))
        if hasattr(self, "_modelSettingsButton"):
            self._modelSettingsButton.setToolTip(self.tr("Model Settings"))
        if hasattr(self, "_currentModelButton"):
            self._currentModelButton.setToolTip(self.tr("Click to change model"))

