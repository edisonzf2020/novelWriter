"""Integration helpers for the AI Copilot dock."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QMainWindow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from PyQt6.QtWidgets import QWidget


class MainWindowIntegration:
    """Utility methods for wiring the AI Copilot dock into the main UI."""

    _FLAG_ATTR = "_ai_dock_integrated"
    _DOCK_ATTR = "_ai_copilot_dock"

    @staticmethod
    def integrate_ai_dock(main_window: QMainWindow) -> bool:
        """Attach the AI Copilot dock to the main window safely.

        The integration is idempotent and will log-and-skip failures to
        avoid blocking the rest of the UI during start-up.
        """
        try:
            if not isinstance(main_window, QMainWindow):
                raise TypeError("main_window must be a QMainWindow instance")

            if getattr(main_window, MainWindowIntegration._FLAG_ATTR, False):
                dock = getattr(main_window, MainWindowIntegration._DOCK_ATTR, None)
                if isinstance(dock, QDockWidget) and dock.parent() is None:
                    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
                return True

            from .dock import AICopilotDock

            dock = AICopilotDock(main_window)
            dock.setObjectName("AICopilotDock")
            main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            setattr(main_window, MainWindowIntegration._DOCK_ATTR, dock)
            setattr(main_window, MainWindowIntegration._FLAG_ATTR, True)
            return True
        except Exception:  # pragma: no cover - log-only path
            logger.exception("Failed to integrate AI Copilot dock")
            return False
