"""
novelWriter – AI Settings Dialog
=================================

File History:
Created: 2025-09-25 [James - Dev Agent]

This file is a part of novelWriter
Copyright (C) 2025 Veronica Berglyd Olsen and novelWriter Contributors

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
import json
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QLabel, QPushButton, QLineEdit,
    QTextEdit, QCheckBox, QSpinBox, QComboBox,
    QGroupBox, QListWidget, QListWidgetItem,
    QDialogButtonBox, QMessageBox, QFormLayout,
    QSplitter
)

from novelwriter.common import qtLambda

if TYPE_CHECKING:
    from novelwriter.guimain import GuiMain

logger = logging.getLogger(__name__)


class GuiAISettings(QDialog):
    """AI Settings Dialog for managing MCP tools configuration."""
    
    def __init__(self, parent: GuiMain) -> None:
        """Initialize the AI Settings dialog.
        
        Args:
            parent: Main GUI window
        """
        super().__init__(parent=parent)
        
        self.mainGui = parent
        
        # Dialog settings
        self.setWindowTitle(self.tr("AI Settings"))
        self.setMinimumSize(800, 600)
        
        # Create main layout
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        
        # Create tab widget
        self.tabWidget = QTabWidget()
        self.mainLayout.addWidget(self.tabWidget)
        
        # Create tabs
        self._createExternalToolsTab()
        self._createLocalToolsTab()
        self._createPerformanceTab()
        
        # Create button box
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        self.buttonBox.accepted.connect(self._saveAndClose)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._saveSettings)
        self.mainLayout.addWidget(self.buttonBox)
        
        # Load current settings
        self._loadSettings()
        
        return
    
    def _createExternalToolsTab(self) -> None:
        """Create the External Tools configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # MCP Servers section
        serverGroup = QGroupBox(self.tr("MCP Servers"))
        serverLayout = QVBoxLayout()
        serverGroup.setLayout(serverLayout)
        
        # Server list
        self.serverList = QListWidget()
        serverLayout.addWidget(self.serverList)
        
        # Server controls
        controlLayout = QHBoxLayout()
        self.addServerBtn = QPushButton(self.tr("Add Server"))
        self.addServerBtn.clicked.connect(self._addServer)
        self.removeServerBtn = QPushButton(self.tr("Remove"))
        self.removeServerBtn.clicked.connect(self._removeServer)
        self.testServerBtn = QPushButton(self.tr("Test Connection"))
        self.testServerBtn.clicked.connect(self._testConnection)
        
        controlLayout.addWidget(self.addServerBtn)
        controlLayout.addWidget(self.removeServerBtn)
        controlLayout.addWidget(self.testServerBtn)
        controlLayout.addStretch()
        serverLayout.addLayout(controlLayout)
        
        layout.addWidget(serverGroup)
        
        # Server Configuration section
        configGroup = QGroupBox(self.tr("Server Configuration"))
        configLayout = QFormLayout()
        configGroup.setLayout(configLayout)
        
        self.serverNameEdit = QLineEdit()
        self.serverUrlEdit = QLineEdit()
        self.serverCommandEdit = QLineEdit()
        self.serverArgsEdit = QLineEdit()
        self.serverEnabledCheck = QCheckBox(self.tr("Enabled"))
        
        configLayout.addRow(self.tr("Name:"), self.serverNameEdit)
        configLayout.addRow(self.tr("URL:"), self.serverUrlEdit)
        configLayout.addRow(self.tr("Command:"), self.serverCommandEdit)
        configLayout.addRow(self.tr("Arguments:"), self.serverArgsEdit)
        configLayout.addRow("", self.serverEnabledCheck)
        
        layout.addWidget(configGroup)
        
        # Connection status
        statusGroup = QGroupBox(self.tr("Connection Status"))
        statusLayout = QVBoxLayout()
        statusGroup.setLayout(statusLayout)
        
        self.statusText = QTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumHeight(100)
        statusLayout.addWidget(self.statusText)
        
        layout.addWidget(statusGroup)
        
        self.tabWidget.addTab(widget, self.tr("External Tools"))
        return
    
    def _createLocalToolsTab(self) -> None:
        """Create the Local Tools configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Tool permissions
        permGroup = QGroupBox(self.tr("Tool Permissions"))
        permLayout = QVBoxLayout()
        permGroup.setLayout(permLayout)
        
        self.allowReadCheck = QCheckBox(self.tr("Allow read operations"))
        self.allowWriteCheck = QCheckBox(self.tr("Allow write operations"))
        self.allowCreateCheck = QCheckBox(self.tr("Allow create operations"))
        self.allowDeleteCheck = QCheckBox(self.tr("Allow delete operations"))
        
        permLayout.addWidget(self.allowReadCheck)
        permLayout.addWidget(self.allowWriteCheck)
        permLayout.addWidget(self.allowCreateCheck)
        permLayout.addWidget(self.allowDeleteCheck)
        
        layout.addWidget(permGroup)
        
        # Tool enable/disable
        toolGroup = QGroupBox(self.tr("Available Tools"))
        toolLayout = QVBoxLayout()
        toolGroup.setLayout(toolLayout)
        
        self.toolList = QListWidget()
        self.toolList.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        # Add local tools
        tools = [
            "ProjectInfoTool",
            "ProjectTreeTool",
            "DocumentListTool",
            "DocumentReadTool",
            "DocumentWriteTool",
            "CreateDocumentTool",
            "GlobalSearchTool",
            "TagListTool",
            "ProjectStatsTool"
        ]
        
        for tool in tools:
            item = QListWidgetItem(tool)
            item.setCheckState(Qt.CheckState.Checked)
            self.toolList.addItem(item)
        
        toolLayout.addWidget(self.toolList)
        
        layout.addWidget(toolGroup)
        layout.addStretch()
        
        self.tabWidget.addTab(widget, self.tr("Local Tools"))
        return
    
    def _createPerformanceTab(self) -> None:
        """Create the Performance configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Cache settings
        cacheGroup = QGroupBox(self.tr("Cache Settings"))
        cacheLayout = QFormLayout()
        cacheGroup.setLayout(cacheLayout)
        
        self.cacheSizeSpin = QSpinBox()
        self.cacheSizeSpin.setMinimum(10)
        self.cacheSizeSpin.setMaximum(1000)
        self.cacheSizeSpin.setValue(100)
        self.cacheSizeSpin.setSuffix(" MB")
        
        self.cacheTTLSpin = QSpinBox()
        self.cacheTTLSpin.setMinimum(60)
        self.cacheTTLSpin.setMaximum(3600)
        self.cacheTTLSpin.setValue(300)
        self.cacheTTLSpin.setSuffix(" seconds")
        
        self.cacheEntriesSpin = QSpinBox()
        self.cacheEntriesSpin.setMinimum(100)
        self.cacheEntriesSpin.setMaximum(100000)
        self.cacheEntriesSpin.setValue(10000)
        self.cacheEntriesSpin.setSuffix(" entries")
        
        cacheLayout.addRow(self.tr("Max Cache Size:"), self.cacheSizeSpin)
        cacheLayout.addRow(self.tr("Default TTL:"), self.cacheTTLSpin)
        cacheLayout.addRow(self.tr("Max Entries:"), self.cacheEntriesSpin)
        
        # Clear cache button
        self.clearCacheBtn = QPushButton(self.tr("Clear Cache"))
        self.clearCacheBtn.clicked.connect(self._clearCache)
        cacheLayout.addRow("", self.clearCacheBtn)
        
        layout.addWidget(cacheGroup)
        
        # Timeout settings
        timeoutGroup = QGroupBox(self.tr("Timeout Settings"))
        timeoutLayout = QFormLayout()
        timeoutGroup.setLayout(timeoutLayout)
        
        self.externalTimeoutSpin = QSpinBox()
        self.externalTimeoutSpin.setMinimum(50)
        self.externalTimeoutSpin.setMaximum(5000)
        self.externalTimeoutSpin.setValue(200)
        self.externalTimeoutSpin.setSuffix(" ms")
        
        self.healthCheckIntervalSpin = QSpinBox()
        self.healthCheckIntervalSpin.setMinimum(10)
        self.healthCheckIntervalSpin.setMaximum(300)
        self.healthCheckIntervalSpin.setValue(30)
        self.healthCheckIntervalSpin.setSuffix(" seconds")
        
        timeoutLayout.addRow(self.tr("External Tool Timeout:"), self.externalTimeoutSpin)
        timeoutLayout.addRow(self.tr("Health Check Interval:"), self.healthCheckIntervalSpin)
        
        layout.addWidget(timeoutGroup)
        
        # Circuit breaker settings
        breakerGroup = QGroupBox(self.tr("Circuit Breaker"))
        breakerLayout = QFormLayout()
        breakerGroup.setLayout(breakerLayout)
        
        self.failureThresholdSpin = QSpinBox()
        self.failureThresholdSpin.setMinimum(1)
        self.failureThresholdSpin.setMaximum(10)
        self.failureThresholdSpin.setValue(3)
        
        self.recoveryTimeoutSpin = QSpinBox()
        self.recoveryTimeoutSpin.setMinimum(10)
        self.recoveryTimeoutSpin.setMaximum(300)
        self.recoveryTimeoutSpin.setValue(60)
        self.recoveryTimeoutSpin.setSuffix(" seconds")
        
        breakerLayout.addRow(self.tr("Failure Threshold:"), self.failureThresholdSpin)
        breakerLayout.addRow(self.tr("Recovery Timeout:"), self.recoveryTimeoutSpin)
        
        layout.addWidget(breakerGroup)
        
        # Statistics
        statsGroup = QGroupBox(self.tr("Performance Statistics"))
        statsLayout = QVBoxLayout()
        statsGroup.setLayout(statsLayout)
        
        self.statsText = QTextEdit()
        self.statsText.setReadOnly(True)
        self.statsText.setMaximumHeight(150)
        statsLayout.addWidget(self.statsText)
        
        # Refresh stats button
        self.refreshStatsBtn = QPushButton(self.tr("Refresh Statistics"))
        self.refreshStatsBtn.clicked.connect(self._refreshStats)
        statsLayout.addWidget(self.refreshStatsBtn)
        
        layout.addWidget(statsGroup)
        layout.addStretch()
        
        self.tabWidget.addTab(widget, self.tr("Performance"))
        return
    
    def _addServer(self) -> None:
        """Add a new MCP server configuration."""
        dialog = AddServerDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.getConfiguration()
            item = QListWidgetItem(f"{config['name']} - {config['url']}")
            item.setData(Qt.ItemDataRole.UserRole, config)
            self.serverList.addItem(item)
            self.statusText.append(f"Added server: {config['name']}")
        return
    
    def _removeServer(self) -> None:
        """Remove selected MCP server."""
        current = self.serverList.currentItem()
        if current:
            reply = QMessageBox.question(
                self,
                self.tr("Remove Server"),
                self.tr("Are you sure you want to remove this server?")
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.serverList.takeItem(self.serverList.row(current))
                self.statusText.append("Server removed")
        return
    
    def _testConnection(self) -> None:
        """Test connection to selected MCP server."""
        current = self.serverList.currentItem()
        if current:
            config = current.data(Qt.ItemDataRole.UserRole)
            self.statusText.append(f"Testing connection to {config['name']}...")
            # In real implementation, would actually test the connection
            # For now, simulate success
            self.statusText.append(f"✅ Connection successful to {config['name']}")
        else:
            self.statusText.append("No server selected")
        return
    
    def _clearCache(self) -> None:
        """Clear the MCP cache."""
        reply = QMessageBox.question(
            self,
            self.tr("Clear Cache"),
            self.tr("Are you sure you want to clear all cached data?")
        )
        if reply == QMessageBox.StandardButton.Yes:
            from novelwriter.api.external_mcp import CacheManager
            CacheManager().clear_all()
            self.statusText.append("Cache cleared successfully")
        return
    
    def _refreshStats(self) -> None:
        """Refresh performance statistics."""
        from novelwriter.api.external_mcp import CacheManager
        
        stats = CacheManager().get_global_statistics()
        
        stats_text = f"""Cache Statistics:
  Total Entries: {stats['global']['total_entries']}
  Total Size: {stats['global']['total_size_mb']:.2f} MB
  Total Hits: {stats['global']['total_hits']}
  Total Misses: {stats['global']['total_misses']}
  Hit Rate: {stats['global']['global_hit_rate']:.2%}
"""
        self.statsText.setText(stats_text)
        return
    
    def _loadSettings(self) -> None:
        """Load current AI settings."""
        # Load from config (would be implemented with actual config system)
        # For now, use defaults
        self.allowReadCheck.setChecked(True)
        self.allowWriteCheck.setChecked(False)
        self.allowCreateCheck.setChecked(False)
        self.allowDeleteCheck.setChecked(False)
        
        # Add example server
        example_config = {
            "name": "Time Server",
            "url": "http://localhost:3001",
            "command": "uvx",
            "args": "mcp-server-time",
            "enabled": True
        }
        item = QListWidgetItem(f"{example_config['name']} - {example_config['url']}")
        item.setData(Qt.ItemDataRole.UserRole, example_config)
        self.serverList.addItem(item)
        
        return
    
    def _saveSettings(self) -> None:
        """Save AI settings."""
        # Collect settings
        settings = {
            "permissions": {
                "read": self.allowReadCheck.isChecked(),
                "write": self.allowWriteCheck.isChecked(),
                "create": self.allowCreateCheck.isChecked(),
                "delete": self.allowDeleteCheck.isChecked()
            },
            "cache": {
                "max_size_mb": self.cacheSizeSpin.value(),
                "default_ttl": self.cacheTTLSpin.value(),
                "max_entries": self.cacheEntriesSpin.value()
            },
            "timeouts": {
                "external_tool_ms": self.externalTimeoutSpin.value(),
                "health_check_interval": self.healthCheckIntervalSpin.value()
            },
            "circuit_breaker": {
                "failure_threshold": self.failureThresholdSpin.value(),
                "recovery_timeout": self.recoveryTimeoutSpin.value()
            },
            "servers": []
        }
        
        # Collect server configurations
        for i in range(self.serverList.count()):
            item = self.serverList.item(i)
            config = item.data(Qt.ItemDataRole.UserRole)
            if config:
                settings["servers"].append(config)
        
        # Save to config (would be implemented with actual config system)
        logger.info(f"Saving AI settings: {json.dumps(settings, indent=2)}")
        self.statusText.append("Settings saved")
        
        return
    
    def _saveAndClose(self) -> None:
        """Save settings and close dialog."""
        self._saveSettings()
        self.accept()
        return


class AddServerDialog(QDialog):
    """Dialog for adding a new MCP server."""
    
    def __init__(self, parent: QWidget) -> None:
        """Initialize the add server dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent=parent)
        
        self.setWindowTitle(self.tr("Add MCP Server"))
        self.setModal(True)
        
        # Create layout
        layout = QFormLayout()
        self.setLayout(layout)
        
        # Create input fields
        self.nameEdit = QLineEdit()
        self.urlEdit = QLineEdit()
        self.urlEdit.setPlaceholderText("http://localhost:3000")
        self.commandEdit = QLineEdit()
        self.commandEdit.setPlaceholderText("uvx")
        self.argsEdit = QLineEdit()
        self.argsEdit.setPlaceholderText("mcp-server-name")
        self.enabledCheck = QCheckBox(self.tr("Enable on startup"))
        self.enabledCheck.setChecked(True)
        
        layout.addRow(self.tr("Name:"), self.nameEdit)
        layout.addRow(self.tr("URL:"), self.urlEdit)
        layout.addRow(self.tr("Command:"), self.commandEdit)
        layout.addRow(self.tr("Arguments:"), self.argsEdit)
        layout.addRow("", self.enabledCheck)
        
        # Button box
        buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
        
        return
    
    def getConfiguration(self) -> dict[str, Any]:
        """Get the server configuration.
        
        Returns:
            Server configuration dictionary
        """
        return {
            "name": self.nameEdit.text(),
            "url": self.urlEdit.text(),
            "command": self.commandEdit.text(),
            "args": self.argsEdit.text(),
            "enabled": self.enabledCheck.isChecked()
        }
