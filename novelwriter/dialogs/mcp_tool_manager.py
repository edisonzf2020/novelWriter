"""
novelWriter – MCP Tool Manager Dialog
======================================

File History:
Created: 2025-01-01 [x.x.x] MCPToolManagerDialog

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
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialogButtonBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget
)

from novelwriter import CONFIG, SHARED
from novelwriter.extensions.configlayout import NColorLabel
from novelwriter.extensions.modified import NDialog, NSpinBox
from novelwriter.extensions.statusled import StatusLED
from novelwriter.extensions.switch import NSwitch
from novelwriter.types import QtDialogClose

logger = logging.getLogger(__name__)


class MCPToolManagerDialog(NDialog):
    """MCP Tool Manager Dialog for managing external MCP connections."""

    # Signals
    connectionsChanged = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent=parent)

        logger.debug("Create: MCPToolManagerDialog")
        self.setObjectName("MCPToolManagerDialog")
        self.setWindowTitle(self.tr("MCP Tool Manager"))
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        # State
        self._connections: List[Dict[str, Any]] = []
        self._selectedRow = -1
        self._updateTimer = QTimer(self)
        self._updateTimer.timeout.connect(self._refreshStatus)
        self._updateTimer.setInterval(5000)  # Update every 5 seconds

        # Build UI
        self._buildUI()
        
        # Load connections
        self._loadConnections()
        
        # Start status updates
        self._updateTimer.start()

        logger.debug("Ready: MCPToolManagerDialog")

    def __del__(self) -> None:  # pragma: no cover
        logger.debug("Delete: MCPToolManagerDialog")

    def _buildUI(self) -> None:
        """Build the dialog UI."""
        # Title
        self.titleLabel = NColorLabel(
            self.tr("External MCP Tool Connections"),
            self,
            color=SHARED.theme.helpText,
            scale=NColorLabel.HEADER_SCALE,
            indent=4,
        )

        # Connection Table
        self.connectionTable = QTableWidget(self)
        self.connectionTable.setObjectName("MCPConnectionTable")
        self.connectionTable.setColumnCount(5)
        self.connectionTable.setHorizontalHeaderLabels([
            self.tr("Status"),
            self.tr("Name"),
            self.tr("URL"),
            self.tr("Tools"),
            self.tr("Enabled"),
        ])
        self.connectionTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.connectionTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.connectionTable.horizontalHeader().setStretchLastSection(False)
        self.connectionTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.connectionTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.connectionTable.itemSelectionChanged.connect(self._onSelectionChanged)

        # Connection Controls
        self.addButton = QPushButton(self.tr("Add Connection"), self)
        self.addButton.clicked.connect(self._addConnection)
        
        self.editButton = QPushButton(self.tr("Edit"), self)
        self.editButton.clicked.connect(self._editConnection)
        self.editButton.setEnabled(False)
        
        self.removeButton = QPushButton(self.tr("Remove"), self)
        self.removeButton.clicked.connect(self._removeConnection)
        self.removeButton.setEnabled(False)
        
        self.testButton = QPushButton(self.tr("Test Connection"), self)
        self.testButton.clicked.connect(self._testConnection)
        self.testButton.setEnabled(False)

        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.addButton)
        controlLayout.addWidget(self.editButton)
        controlLayout.addWidget(self.removeButton)
        controlLayout.addStretch()
        controlLayout.addWidget(self.testButton)

        # Connection Details Group
        self.detailsGroup = QGroupBox(self.tr("Connection Details"), self)
        detailsLayout = QVBoxLayout()

        # Name field
        nameLayout = QHBoxLayout()
        nameLayout.addWidget(QLabel(self.tr("Name:"), self))
        self.nameEdit = QLineEdit(self)
        self.nameEdit.setPlaceholderText(self.tr("Connection name"))
        self.nameEdit.setEnabled(False)
        nameLayout.addWidget(self.nameEdit)
        detailsLayout.addLayout(nameLayout)

        # URL field
        urlLayout = QHBoxLayout()
        urlLayout.addWidget(QLabel(self.tr("URL:"), self))
        self.urlEdit = QLineEdit(self)
        self.urlEdit.setPlaceholderText(self.tr("http://localhost:3000"))
        self.urlEdit.setEnabled(False)
        urlLayout.addWidget(self.urlEdit)
        detailsLayout.addLayout(urlLayout)

        # Transport type
        transportLayout = QHBoxLayout()
        transportLayout.addWidget(QLabel(self.tr("Transport:"), self))
        self.transportLabel = QLabel("streamable-http", self)
        transportLayout.addWidget(self.transportLabel)
        transportLayout.addStretch()
        detailsLayout.addLayout(transportLayout)

        # Available tools
        self.toolsLabel = QLabel(self.tr("Available Tools:"), self)
        detailsLayout.addWidget(self.toolsLabel)
        
        self.toolsList = QTextEdit(self)
        self.toolsList.setReadOnly(True)
        self.toolsList.setMaximumHeight(100)
        detailsLayout.addWidget(self.toolsList)

        self.detailsGroup.setLayout(detailsLayout)

        # Status Group
        self.statusGroup = QGroupBox(self.tr("Connection Status"), self)
        statusLayout = QVBoxLayout()

        # Health status
        healthLayout = QHBoxLayout()
        healthLayout.addWidget(QLabel(self.tr("Health:"), self))
        self.healthLED = StatusLED(16, 16, self)
        healthLayout.addWidget(self.healthLED)
        self.healthLabel = QLabel(self.tr("Unknown"), self)
        healthLayout.addWidget(self.healthLabel)
        healthLayout.addStretch()
        statusLayout.addLayout(healthLayout)

        # Last check time
        checkLayout = QHBoxLayout()
        checkLayout.addWidget(QLabel(self.tr("Last Check:"), self))
        self.lastCheckLabel = QLabel(self.tr("Never"), self)
        checkLayout.addWidget(self.lastCheckLabel)
        checkLayout.addStretch()
        statusLayout.addLayout(checkLayout)

        # Response time
        responseLayout = QHBoxLayout()
        responseLayout.addWidget(QLabel(self.tr("Response Time:"), self))
        self.responseTimeLabel = QLabel(self.tr("N/A"), self)
        responseLayout.addWidget(self.responseTimeLabel)
        responseLayout.addStretch()
        statusLayout.addLayout(responseLayout)

        self.statusGroup.setLayout(statusLayout)

        # Buttons
        self.buttonBox = QDialogButtonBox(QtDialogClose, self)
        self.buttonBox.rejected.connect(self.reject)

        # Main Layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.titleLabel)
        mainLayout.addWidget(self.connectionTable, 1)
        mainLayout.addLayout(controlLayout)
        
        detailsStatusLayout = QHBoxLayout()
        detailsStatusLayout.addWidget(self.detailsGroup, 2)
        detailsStatusLayout.addWidget(self.statusGroup, 1)
        mainLayout.addLayout(detailsStatusLayout)
        
        mainLayout.addWidget(self.buttonBox)

        self.setLayout(mainLayout)

    ##
    #  Public Methods
    ##

    def addExternalConnection(self, name: str, url: str, enabled: bool = True) -> bool:
        """Add a new external MCP connection.
        
        Args:
            name: Connection name
            url: Server URL
            enabled: Whether connection is enabled
            
        Returns:
            True if connection was added successfully
        """
        # Validate URL
        if not url.startswith(("http://", "https://")):
            QMessageBox.warning(
                self,
                self.tr("Invalid URL"),
                self.tr("URL must start with http:// or https://")
            )
            return False
        
        # Check for duplicate
        for conn in self._connections:
            if conn["url"] == url:
                QMessageBox.warning(
                    self,
                    self.tr("Duplicate Connection"),
                    self.tr("A connection to this URL already exists")
                )
                return False
        
        # Add connection
        connection = {
            "name": name,
            "url": url,
            "transport": "streamable-http",
            "enabled": enabled,
            "status": "unknown",
            "tools": [],
            "last_check": None,
            "response_time": None,
        }
        
        self._connections.append(connection)
        self._addTableRow(connection)
        self._saveConnections()
        
        # Test the new connection
        self._testConnectionByIndex(len(self._connections) - 1)
        
        self.connectionsChanged.emit()
        return True

    ##
    #  Private Methods
    ##

    def _loadConnections(self) -> None:
        """Load saved connections from configuration."""
        try:
            from novelwriter.api.base.config import MCP_CONFIG
            connections = MCP_CONFIG.externalMCPConfig.get("connections", [])
            
            for conn_data in connections:
                connection = {
                    "name": conn_data.get("name", "Unnamed"),
                    "url": conn_data.get("url", ""),
                    "transport": conn_data.get("transport", "streamable-http"),
                    "enabled": conn_data.get("enabled", True),
                    "status": "unknown",
                    "tools": conn_data.get("tools", []),
                    "last_check": None,
                    "response_time": None,
                }
                self._connections.append(connection)
                self._addTableRow(connection)
        except ImportError:
            logger.debug("MCP configuration not available")

    def _saveConnections(self) -> None:
        """Save connections to configuration."""
        try:
            from novelwriter.api.base.config import MCP_CONFIG
            
            # Convert connections to saveable format
            connections_data = []
            for conn in self._connections:
                connections_data.append({
                    "name": conn["name"],
                    "url": conn["url"],
                    "transport": conn["transport"],
                    "enabled": conn["enabled"],
                    "tools": conn["tools"],
                })
            
            MCP_CONFIG.setValue("externalMCP.connections", connections_data)
        except ImportError:
            logger.debug("MCP configuration not available")

    def _addTableRow(self, connection: Dict[str, Any]) -> None:
        """Add a connection to the table."""
        row = self.connectionTable.rowCount()
        self.connectionTable.insertRow(row)
        
        # Status LED
        statusLED = StatusLED(16, 16, self)
        self._updateStatusLED(statusLED, connection["status"])
        self.connectionTable.setCellWidget(row, 0, statusLED)
        
        # Name
        nameItem = QTableWidgetItem(connection["name"])
        self.connectionTable.setItem(row, 1, nameItem)
        
        # URL
        urlItem = QTableWidgetItem(connection["url"])
        self.connectionTable.setItem(row, 2, urlItem)
        
        # Tools count
        toolsItem = QTableWidgetItem(str(len(connection["tools"])))
        toolsItem.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connectionTable.setItem(row, 3, toolsItem)
        
        # Enabled switch
        enabledSwitch = NSwitch(self)
        enabledSwitch.setChecked(connection["enabled"])
        enabledSwitch.toggled.connect(lambda checked, r=row: self._onEnabledToggled(r, checked))
        self.connectionTable.setCellWidget(row, 4, enabledSwitch)

    def _updateStatusLED(self, led: StatusLED, status: str) -> None:
        """Update status LED based on connection status."""
        if status == "healthy":
            led.setState(True)  # Green
        elif status == "degraded":
            led.setState(None)  # Yellow/neutral
        elif status == "offline":
            led.setState(False)  # Red
        else:
            led.setState(None)  # Neutral

    def _refreshStatus(self) -> None:
        """Refresh status of all enabled connections."""
        for i, conn in enumerate(self._connections):
            if conn["enabled"]:
                self._testConnectionByIndex(i, silent=True)

    def _testConnectionByIndex(self, index: int, silent: bool = False) -> None:
        """Test a specific connection by index."""
        if 0 <= index < len(self._connections):
            connection = self._connections[index]
            
            # Simulate connection test (in real implementation, this would call the MCP client)
            import random
            import datetime
            
            # Simulate test results
            if random.random() > 0.2:  # 80% success rate
                connection["status"] = "healthy"
                connection["response_time"] = random.randint(10, 200)
                connection["tools"] = [
                    "get_current_time",
                    "convert_timezone",
                    "weather_info",
                ]
            else:
                connection["status"] = "offline"
                connection["response_time"] = None
                connection["tools"] = []
            
            connection["last_check"] = datetime.datetime.now()
            
            # Update table
            if index < self.connectionTable.rowCount():
                led = self.connectionTable.cellWidget(index, 0)
                if isinstance(led, StatusLED):
                    self._updateStatusLED(led, connection["status"])
                
                toolsItem = self.connectionTable.item(index, 3)
                if toolsItem:
                    toolsItem.setText(str(len(connection["tools"])))
            
            # Update details if this connection is selected
            if index == self._selectedRow:
                self._updateDetails()
            
            if not silent and connection["status"] == "offline":
                QMessageBox.warning(
                    self,
                    self.tr("Connection Failed"),
                    self.tr("Failed to connect to {0}").format(connection["name"])
                )

    ##
    #  Private Slots
    ##

    @pyqtSlot()
    def _onSelectionChanged(self) -> None:
        """Handle table selection change."""
        selected = self.connectionTable.selectedItems()
        if selected:
            self._selectedRow = selected[0].row()
            self.editButton.setEnabled(True)
            self.removeButton.setEnabled(True)
            self.testButton.setEnabled(True)
            self._updateDetails()
        else:
            self._selectedRow = -1
            self.editButton.setEnabled(False)
            self.removeButton.setEnabled(False)
            self.testButton.setEnabled(False)
            self._clearDetails()

    def _updateDetails(self) -> None:
        """Update the details panel with selected connection info."""
        if 0 <= self._selectedRow < len(self._connections):
            conn = self._connections[self._selectedRow]
            
            # Update details
            self.nameEdit.setText(conn["name"])
            self.urlEdit.setText(conn["url"])
            self.transportLabel.setText(conn["transport"])
            
            # Update tools list
            if conn["tools"]:
                self.toolsList.setPlainText("\n".join(f"• {tool}" for tool in conn["tools"]))
            else:
                self.toolsList.setPlainText(self.tr("No tools available"))
            
            # Update status
            self._updateStatusDisplay(conn)

    def _updateStatusDisplay(self, conn: Dict[str, Any]) -> None:
        """Update status display for a connection."""
        # Health status
        status = conn["status"]
        if status == "healthy":
            self.healthLED.setState(True)  # Green
            self.healthLabel.setText(self.tr("Healthy"))
            self.healthLabel.setStyleSheet(f"color: {SHARED.theme.getBaseColor('green').name()};")
        elif status == "degraded":
            self.healthLED.setState(None)  # Yellow/neutral
            self.healthLabel.setText(self.tr("Degraded"))
            self.healthLabel.setStyleSheet(f"color: {SHARED.theme.getBaseColor('yellow').name()};")
        elif status == "offline":
            self.healthLED.setState(False)  # Red
            self.healthLabel.setText(self.tr("Offline"))
            self.healthLabel.setStyleSheet(f"color: {SHARED.theme.getBaseColor('red').name()};")
        else:
            self.healthLED.setState(None)  # Neutral
            self.healthLabel.setText(self.tr("Unknown"))
            self.healthLabel.setStyleSheet("")
        
        # Last check time
        if conn["last_check"]:
            self.lastCheckLabel.setText(conn["last_check"].strftime("%H:%M:%S"))
        else:
            self.lastCheckLabel.setText(self.tr("Never"))
        
        # Response time
        if conn["response_time"] is not None:
            self.responseTimeLabel.setText(f"{conn['response_time']} ms")
        else:
            self.responseTimeLabel.setText(self.tr("N/A"))

    def _clearDetails(self) -> None:
        """Clear the details panel."""
        self.nameEdit.clear()
        self.urlEdit.clear()
        self.transportLabel.setText("streamable-http")
        self.toolsList.clear()
        self.healthLED.setState(None)  # Neutral
        self.healthLabel.setText(self.tr("Unknown"))
        self.healthLabel.setStyleSheet("")
        self.lastCheckLabel.setText(self.tr("Never"))
        self.responseTimeLabel.setText(self.tr("N/A"))

    @pyqtSlot()
    def _addConnection(self) -> None:
        """Show dialog to add a new connection."""
        dialog = MCPConnectionEditDialog(self)
        if dialog.exec() == NDialog.DialogCode.Accepted:
            name, url = dialog.getConnectionData()
            self.addExternalConnection(name, url)

    @pyqtSlot()
    def _editConnection(self) -> None:
        """Edit the selected connection."""
        if 0 <= self._selectedRow < len(self._connections):
            conn = self._connections[self._selectedRow]
            dialog = MCPConnectionEditDialog(self, conn["name"], conn["url"])
            if dialog.exec() == NDialog.DialogCode.Accepted:
                name, url = dialog.getConnectionData()
                conn["name"] = name
                conn["url"] = url
                
                # Update table
                nameItem = self.connectionTable.item(self._selectedRow, 1)
                if nameItem:
                    nameItem.setText(name)
                urlItem = self.connectionTable.item(self._selectedRow, 2)
                if urlItem:
                    urlItem.setText(url)
                
                self._updateDetails()
                self._saveConnections()
                self.connectionsChanged.emit()

    @pyqtSlot()
    def _removeConnection(self) -> None:
        """Remove the selected connection."""
        if 0 <= self._selectedRow < len(self._connections):
            conn = self._connections[self._selectedRow]
            
            reply = QMessageBox.question(
                self,
                self.tr("Remove Connection"),
                self.tr("Are you sure you want to remove the connection '{0}'?").format(conn["name"]),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._connections.pop(self._selectedRow)
                self.connectionTable.removeRow(self._selectedRow)
                self._selectedRow = -1
                self._clearDetails()
                self._saveConnections()
                self.connectionsChanged.emit()

    @pyqtSlot()
    def _testConnection(self) -> None:
        """Test the selected connection."""
        if 0 <= self._selectedRow < len(self._connections):
            self._testConnectionByIndex(self._selectedRow)

    @pyqtSlot(int, bool)
    def _onEnabledToggled(self, row: int, checked: bool) -> None:
        """Handle enabled switch toggle."""
        if 0 <= row < len(self._connections):
            self._connections[row]["enabled"] = checked
            self._saveConnections()
            
            if checked:
                # Test connection when enabled
                self._testConnectionByIndex(row, silent=True)
            
            self.connectionsChanged.emit()

    ##
    #  Events
    ##

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle close event."""
        self._updateTimer.stop()
        event.accept()


class MCPConnectionEditDialog(NDialog):
    """Dialog for editing MCP connection details."""

    def __init__(self, parent: QWidget, name: str = "", url: str = "") -> None:
        super().__init__(parent=parent)
        
        self.setWindowTitle(self.tr("Edit MCP Connection"))
        self.setMinimumWidth(400)
        
        # Build UI
        layout = QVBoxLayout()
        
        # Name field
        nameLayout = QHBoxLayout()
        nameLayout.addWidget(QLabel(self.tr("Name:"), self))
        self.nameEdit = QLineEdit(self)
        self.nameEdit.setText(name)
        self.nameEdit.setPlaceholderText(self.tr("Connection name"))
        nameLayout.addWidget(self.nameEdit)
        layout.addLayout(nameLayout)
        
        # URL field
        urlLayout = QHBoxLayout()
        urlLayout.addWidget(QLabel(self.tr("URL:"), self))
        self.urlEdit = QLineEdit(self)
        self.urlEdit.setText(url)
        self.urlEdit.setPlaceholderText(self.tr("http://localhost:3000"))
        urlLayout.addWidget(self.urlEdit)
        layout.addLayout(urlLayout)
        
        # Buttons
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)
        
        self.setLayout(layout)
    
    def getConnectionData(self) -> tuple[str, str]:
        """Get the connection name and URL."""
        return self.nameEdit.text().strip(), self.urlEdit.text().strip()
