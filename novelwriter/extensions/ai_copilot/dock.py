"""AI Copilot dock widget implementation."""
from __future__ import annotations

import importlib
import logging


from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from novelwriter import CONFIG, SHARED

logger = logging.getLogger(__name__)


class AICopilotDock(QDockWidget):
    """Dock widget hosting the AI Copilot UI with context selection capability."""

    contextScopeChanged = pyqtSignal(str)

    _SCOPE_KEYS: tuple[str, ...] = (
        "selection",
        "current_document",
        "outline",
        "project",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("AICopilotDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        self._statusLabel = QLabel(self)
        self._statusLabel.setObjectName("aiCopilotStatusLabel")
        self._detailLabel = QLabel(self)
        self._detailLabel.setObjectName("aiCopilotMessageLabel")
        self._contextScopeSelector: QComboBox | None = None
        self._scopeLabel: QLabel | None = None
        self._scopeFrame: QFrame | None = None
        self._currentScope: str = "current_document"
        self._ai_available: bool = False
        self._availability_reason: str | None = None

        self._setupPlaceholder()
        self.refresh_from_config()

    ##
    #  Qt Events
    ##

    def changeEvent(self, event: QEvent) -> None:  # pragma: no cover - Qt runtime behaviour
        super().changeEvent(event)
        if event.type() == QEvent.Type.LanguageChange:
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

    ##
    #  Internal helpers
    ##

    def _setupPlaceholder(self) -> None:
        """Initialise the placeholder UI elements and scope selector."""

        theme = SHARED.theme

        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scope_row = self._buildScopeSelectorRow()
        layout.addWidget(scope_row)
        layout.addSpacing(8)

        self._statusLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._statusLabel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._statusLabel.setFont(theme.guiFontB)

        self._detailLabel.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._detailLabel.setWordWrap(True)
        self._detailLabel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        self._detailLabel.setFont(theme.guiFont)
        self._detailLabel.setStyleSheet(f"color: {theme.fadedText.name()};")

        layout.addStretch(1)
        layout.addWidget(self._statusLabel)
        layout.addWidget(self._detailLabel)
        layout.addStretch(2)

        container.setLayout(layout)
        self.setWidget(container)

    def _buildScopeSelectorRow(self) -> QWidget:
        """Create the scope selector row shown at the top of the dock."""

        frame = QFrame(self)
        frame.setObjectName("aiContextScopeRow")
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Plain)

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(8)

        label = QLabel(self.tr("Context:"), frame)
        label.setObjectName("aiContextScopeLabel")
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        selector = QComboBox(frame)
        selector.setObjectName("aiContextScopeSelector")
        selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        selector.currentIndexChanged.connect(self._handleScopeChanged)

        self._scopeLabel = label
        self._contextScopeSelector = selector
        self._scopeFrame = frame
        self._populateScopeSelector()

        row_layout.addWidget(label)
        row_layout.addWidget(selector, stretch=1)
        row_layout.addStretch(1)
        frame.setLayout(row_layout)
        return frame

    def _scopeOptions(self) -> list[tuple[str, str]]:
        """Return the translated scope options for the selector."""

        return [
            ("selection", self.tr("Selection")),
            ("current_document", self.tr("Current Document")),
            ("outline", self.tr("Outline")),
            ("project", self.tr("Entire Project")),
        ]

    def _populateScopeSelector(self) -> None:
        """Populate the scope selector with translated entries."""

        if self._contextScopeSelector is None:
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
        """React to user-triggered scope changes and emit the signal."""

        if self._contextScopeSelector is None:
            return

        scope_key = self._contextScopeSelector.itemData(
            index,
            role=Qt.ItemDataRole.UserRole,
        )
        if not isinstance(scope_key, str) or scope_key == self._currentScope:
            return

        self._currentScope = scope_key
        self.contextScopeChanged.emit(scope_key)
        logger.debug("Context scope changed to: %s", scope_key)

    def refresh_from_config(self) -> None:
        """Refresh availability state and update the UI accordingly."""

        self._ai_available, self._availability_reason = self._resolve_availability()
        self._updateScopeSelectorState()
        self.updateTheme()

    def updateTheme(self) -> None:
        """Update fonts and colours to match the current theme."""

        theme = SHARED.theme
        self._statusLabel.setFont(theme.guiFontB)
        self._detailLabel.setFont(theme.guiFont)
        self._applyTranslations()

    def _applyTranslations(self) -> None:
        """Refresh translated texts based on availability state."""

        self.setWindowTitle(self.tr("AI Copilot"))
        if self._scopeLabel is not None:
            self._scopeLabel.setText(self.tr("Context:"))
        self._populateScopeSelector()

        if self._ai_available:
            self._statusLabel.setText(self.tr("AI Copilot is ready"))
            self._detailLabel.setText(
                self.tr("Interactive Copilot features will appear here in a later release.")
            )
            colour = SHARED.theme.helpText.name()
        else:
            self._statusLabel.setText(self.tr("AI Copilot is temporarily unavailable"))
            fallback = (
                self._availability_reason
                or self.tr("AI features are currently disabled or missing optional dependencies.")
            )
            self._detailLabel.setText(fallback)
            colour = SHARED.theme.errorText.name()

        self._detailLabel.setStyleSheet(f"color: {colour};")

    def _updateScopeSelectorState(self) -> None:
        """Enable or disable the scope selector depending on availability."""

        enabled = self._ai_available
        if self._contextScopeSelector is not None:
            self._contextScopeSelector.setEnabled(enabled)
        if self._scopeLabel is not None:
            self._scopeLabel.setEnabled(enabled)
        if self._scopeFrame is not None:
            self._scopeFrame.setEnabled(enabled)

    def _resolve_availability(self) -> tuple[bool, str | None]:
        """Determine whether AI dependencies/configuration are available."""

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
            importlib.import_module("novelwriter.ai")
        except Exception as exc:  # pragma: no cover - environment dependent
            message = str(exc) or exc.__class__.__name__
            logger.warning("AI Copilot dependencies unavailable: %s", message)
            return False, self.tr("AI module could not be loaded: {0}").format(message)

        return True, None
