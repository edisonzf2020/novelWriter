"""AI Copilot dock widget implementation."""
from __future__ import annotations

import importlib
import logging
from typing import Tuple

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QLabel, QDockWidget, QSizePolicy, QVBoxLayout, QWidget

from novelwriter import CONFIG, SHARED

logger = logging.getLogger(__name__)


class AICopilotDock(QDockWidget):
    """Dock widget hosting the AI Copilot UI placeholder.

    The widget is currently a themed placeholder until later stories
    provide the interactive UI. It still honours theme/i18n updates and
    degrades gracefully when AI dependencies are disabled or missing.
    """

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
    #  Internal Helpers
    ##

    def _setupPlaceholder(self) -> None:
        """Initialise the placeholder UI elements."""
        theme = SHARED.theme

        container = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self._statusLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._statusLabel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._statusLabel.setFont(theme.guiFontB)

        self._detailLabel.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._detailLabel.setWordWrap(True)
        self._detailLabel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._detailLabel.setFont(theme.guiFont)
        self._detailLabel.setStyleSheet(f"color: {theme.fadedText.name()};")

        layout.addStretch(1)
        layout.addWidget(self._statusLabel)
        layout.addWidget(self._detailLabel)
        layout.addStretch(2)

        container.setLayout(layout)
        self.setWidget(container)

    def refresh_from_config(self) -> None:
        """Refresh availability and messaging from the shared configuration."""

        self._ai_available, self._availability_reason = self._resolve_availability()
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

        if self._ai_available:
            self._statusLabel.setText(self.tr("AI Copilot is ready"))
            self._detailLabel.setText(
                self.tr("Interactive Copilot features will appear here in a later release.")
            )
            self._detailLabel.setStyleSheet(f"color: {SHARED.theme.helpText.name()};")
        else:
            self._statusLabel.setText(self.tr("AI Copilot is temporarily unavailable"))
            fallback = (
                self._availability_reason
                or self.tr("AI features are currently disabled or missing optional dependencies.")
            )
            self._detailLabel.setText(fallback)
            self._detailLabel.setStyleSheet(f"color: {SHARED.theme.errorText.name()};")

    def _resolve_availability(self) -> Tuple[bool, str | None]:
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
