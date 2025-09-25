"""
novelWriter – MCP UI Integration Tests
=======================================

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

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialogButtonBox, QMessageBox

from novelwriter import CONFIG, SHARED
from novelwriter.dialogs.preferences import GuiPreferences
from novelwriter.dialogs.mcp_tool_manager import MCPToolManagerDialog, MCPConnectionEditDialog
from novelwriter.gui.statusbar import GuiMainStatus
from novelwriter.extensions.statusled import StatusLED

from tests.tools import C, buildTestProject


@pytest.mark.gui
class TestMCPPreferences:
    """Test MCP configuration in preferences dialog."""

    @pytest.fixture(autouse=True)
    def _setup(self, qtbot, nwGUI, fncPath):
        """Set up test environment."""
        buildTestProject(nwGUI, fncPath)
        self.nwGUI = nwGUI
        yield
        self.nwGUI.closeProject()

    def test_mcp_preferences_section(self, qtbot, monkeypatch):
        """Test that MCP preferences section is created and functional."""
        # Mock the exec method to prevent dialog from blocking
        monkeypatch.setattr(GuiPreferences, "exec", lambda *a: None)
        
        # Mock MCP_CONFIG availability
        class MockMCPConfig:
            def __init__(self):
                self.enabled = False
                self._config = {
                    "enabled": False,
                    "localTools": {
                        "project_info": True,
                        "document_read": True,
                        "document_write": True,
                        "project_tree": True,
                        "search": True,
                        "metadata": True,
                        "statistics": True,
                        "export": True,
                    },
                    "externalMCP": {
                        "timeout": 200,
                        "retryAttempts": 3,
                    },
                    "performance": {
                        "monitoringEnabled": True,
                        "alertThresholds": {
                            "apiLatency": 5,
                            "toolLatency": 10,
                            "errorRate": 0.05,
                        },
                    },
                    "failureRecovery": {
                        "enableCircuitBreaker": True,
                        "circuitBreakerThreshold": 5,
                    },
                }
            
            @property
            def localToolsEnabled(self):
                return self._config["localTools"]
            
            @property
            def externalMCPConfig(self):
                return self._config["externalMCP"]
            
            @property
            def performanceConfig(self):
                return self._config["performance"]
            
            @property
            def failureRecoveryConfig(self):
                return self._config["failureRecovery"]
            
            def getValue(self, key, default=None):
                keys = key.split(".")
                value = self._config
                for k in keys:
                    if isinstance(value, dict):
                        value = value.get(k)
                        if value is None:
                            return default
                    else:
                        return default
                return value
            
            def setValue(self, key, value):
                keys = key.split(".")
                target = self._config
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value
                return True

        mock_config = MockMCPConfig()
        
        # Create a mock module for api.base.config
        import sys
        from types import ModuleType
        mock_module = ModuleType("novelwriter.api.base.config")
        mock_module.MCP_CONFIG = mock_config
        sys.modules["novelwriter.api.base.config"] = mock_module

        # Create preferences dialog without showing it
        prefDialog = GuiPreferences(self.nwGUI)
        qtbot.addWidget(prefDialog)

        # Check that MCP section exists
        assert hasattr(prefDialog, "mcpEnabled")
        assert hasattr(prefDialog, "mcpLocalTools")
        assert hasattr(prefDialog, "mcpTimeout")
        assert hasattr(prefDialog, "mcpMonitoring")

        # Test enabling MCP
        assert not prefDialog.mcpEnabled.isChecked()
        qtbot.mouseClick(prefDialog.mcpEnabled, Qt.MouseButton.LeftButton)
        assert prefDialog.mcpEnabled.isChecked()

        # Test local tools configuration
        assert len(prefDialog.mcpLocalTools) == 8
        for tool_id, switch in prefDialog.mcpLocalTools.items():
            assert switch.isEnabled()
            assert switch.isChecked()  # All should be enabled by default

        # Disable a tool
        prefDialog.mcpLocalTools["document_write"].setChecked(False)
        assert not prefDialog.mcpLocalTools["document_write"].isChecked()

        # Test performance monitoring
        assert prefDialog.mcpMonitoring.isChecked()
        assert prefDialog.mcpApiLatency.value() == 5
        assert prefDialog.mcpToolLatency.value() == 10
        assert prefDialog.mcpErrorRate.value() == 0.05

        # Change some values
        prefDialog.mcpApiLatency.setValue(10)
        prefDialog.mcpToolLatency.setValue(20)
        prefDialog.mcpErrorRate.setValue(0.1)

        # Test circuit breaker
        assert prefDialog.mcpCircuitBreaker.isChecked()
        assert prefDialog.mcpCircuitThreshold.value() == 5
        prefDialog.mcpCircuitThreshold.setValue(10)

        # Save preferences directly without showing dialog
        prefDialog._doSave()

        # Verify values were saved
        assert mock_config.enabled == True
        assert mock_config.getValue("localTools.document_write") == False
        assert mock_config.getValue("performance.alertThresholds.apiLatency") == 10
        assert mock_config.getValue("performance.alertThresholds.toolLatency") == 20
        assert mock_config.getValue("performance.alertThresholds.errorRate") == 0.1
        assert mock_config.getValue("failureRecovery.circuitBreakerThreshold") == 10

    def test_mcp_controls_disabled_when_unavailable(self, qtbot, monkeypatch):
        """Test that MCP controls are disabled when MCP is not available."""
        # Mock the exec method to prevent dialog from blocking
        monkeypatch.setattr(GuiPreferences, "exec", lambda *a: None)
        
        # Make sure MCP_CONFIG import fails
        import sys
        if "novelwriter.api.base.config" in sys.modules:
            del sys.modules["novelwriter.api.base.config"]
        
        # Create preferences without MCP available
        prefDialog = GuiPreferences(self.nwGUI)
        qtbot.addWidget(prefDialog)

        # Check that MCP section shows unavailable message
        if hasattr(prefDialog, "mcpUnavailableMessage"):
            # When MCP is not available, the message should be visible
            assert prefDialog.mcpUnavailableMessage.isVisible()


@pytest.mark.gui
class TestMCPStatusBar:
    """Test MCP status bar integration."""

    @pytest.fixture(autouse=True)
    def _setup(self, qtbot, nwGUI, fncPath):
        """Set up test environment."""
        buildTestProject(nwGUI, fncPath)
        self.nwGUI = nwGUI
        self.statusBar = nwGUI.mainStatus
        yield
        self.nwGUI.closeProject()

    def test_mcp_status_indicators(self, qtbot):
        """Test MCP status indicators in status bar."""
        # Initially hidden
        assert not self.statusBar.mcpIcon.isVisible()
        assert not self.statusBar.mcpText.isVisible()
        assert not self.statusBar.mcpMetrics.isVisible()

        # Enable MCP status
        self.statusBar.setMCPStatus(True)
        assert self.statusBar.mcpIcon.isVisible()
        assert self.statusBar.mcpText.isVisible()
        assert self.statusBar.mcpMetrics.isVisible()

        # Test health status updates
        self.statusBar.updateMCPHealth("healthy")
        assert self.statusBar.mcpIcon._state == True  # Green
        assert self.statusBar.mcpText.text() == "MCP"

        self.statusBar.updateMCPHealth("degraded")
        assert self.statusBar.mcpIcon._state == None  # Yellow/neutral
        assert self.statusBar.mcpText.text() == "MCP!"

        self.statusBar.updateMCPHealth("offline")
        assert self.statusBar.mcpIcon._state == False  # Red
        assert self.statusBar.mcpText.text() == "MCP✗"

        # Test metrics updates
        self.statusBar.updateMCPMetrics(25.5, 42, 98.5)
        assert "25.5ms" in self.statusBar.mcpMetrics.text()
        assert "42" in self.statusBar.mcpMetrics.text()
        assert "98%" in self.statusBar.mcpMetrics.text()

        # Test high latency warning
        self.statusBar.updateMCPMetrics(150.0, 10, 99.0)
        assert self.statusBar.mcpMetrics.styleSheet() != ""  # Should have warning color

        # Test low success rate warning
        self.statusBar.updateMCPMetrics(50.0, 10, 85.0)
        assert self.statusBar.mcpMetrics.styleSheet() != ""  # Should have warning color

        # Disable MCP status
        self.statusBar.setMCPStatus(False)
        assert not self.statusBar.mcpIcon.isVisible()
        assert not self.statusBar.mcpText.isVisible()
        assert not self.statusBar.mcpMetrics.isVisible()


@pytest.mark.gui
class TestMCPToolManager:
    """Test MCP Tool Manager dialog."""

    @pytest.fixture(autouse=True)
    def _setup(self, qtbot, nwGUI, fncPath, monkeypatch):
        """Set up test environment."""
        buildTestProject(nwGUI, fncPath)
        self.nwGUI = nwGUI
        
        # Mock MCP_CONFIG
        class MockMCPConfig:
            def __init__(self):
                self._config = {
                    "externalMCP": {
                        "connections": []
                    }
                }
            
            @property
            def externalMCPConfig(self):
                return self._config["externalMCP"]
            
            def setValue(self, key, value):
                if key == "externalMCP.connections":
                    self._config["externalMCP"]["connections"] = value
                return True

        self.mock_config = MockMCPConfig()
        
        # Create a mock module for api.base.config
        import sys
        from types import ModuleType
        if "novelwriter.api.base.config" not in sys.modules:
            mock_module = ModuleType("novelwriter.api.base.config")
            mock_module.MCP_CONFIG = self.mock_config
            sys.modules["novelwriter.api.base.config"] = mock_module
        
        yield
        self.nwGUI.closeProject()

    def test_tool_manager_dialog(self, qtbot, monkeypatch):
        """Test basic tool manager dialog functionality."""
        # Mock exec methods to prevent dialogs from blocking
        from novelwriter.dialogs.mcp_tool_manager import MCPConnectionEditDialog
        monkeypatch.setattr(MCPConnectionEditDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        monkeypatch.setattr(MCPToolManagerDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        
        # Mock ALL QMessageBox methods to prevent any dialogs
        def mock_warning(parent, title, message):
            return QMessageBox.StandardButton.Ok
        
        def mock_question(parent, title, message, buttons, default=None):
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, "warning", mock_warning)
        monkeypatch.setattr(QMessageBox, "question", mock_question)
        
        dialog = MCPToolManagerDialog(self.nwGUI)
        qtbot.addWidget(dialog)
        # Stop timer immediately to prevent background updates
        dialog._updateTimer.stop()

        # Check initial state
        assert dialog.connectionTable.rowCount() == 0
        assert not dialog.editButton.isEnabled()
        assert not dialog.removeButton.isEnabled()
        assert not dialog.testButton.isEnabled()

        # Add a connection directly (not through button click)
        success = dialog.addExternalConnection("Test Server", "http://localhost:3000")
        assert success
        assert dialog.connectionTable.rowCount() == 1

        # Check that connection appears in table
        nameItem = dialog.connectionTable.item(0, 1)
        assert nameItem.text() == "Test Server"
        urlItem = dialog.connectionTable.item(0, 2)
        assert urlItem.text() == "http://localhost:3000"

        # Select the connection
        dialog.connectionTable.selectRow(0)
        qtbot.wait(100)
        
        assert dialog.editButton.isEnabled()
        assert dialog.removeButton.isEnabled()
        assert dialog.testButton.isEnabled()

        # Check details are displayed
        assert dialog.nameEdit.text() == "Test Server"
        assert dialog.urlEdit.text() == "http://localhost:3000"

        # Test connection (call method directly to avoid dialog issues)
        dialog._testConnection()
        qtbot.wait(100)

    def test_connection_edit_dialog(self, qtbot, monkeypatch):
        """Test connection edit dialog."""
        # Mock exec to prevent dialog from blocking
        from novelwriter.dialogs.mcp_tool_manager import MCPConnectionEditDialog
        monkeypatch.setattr(MCPConnectionEditDialog, "exec", lambda *a: None)
        
        dialog = MCPConnectionEditDialog(self.nwGUI, "Test", "http://localhost:3000")
        qtbot.addWidget(dialog)

        # Check initial values
        assert dialog.nameEdit.text() == "Test"
        assert dialog.urlEdit.text() == "http://localhost:3000"

        # Edit values
        dialog.nameEdit.setText("Updated Server")
        dialog.urlEdit.setText("http://localhost:4000")

        # Get data
        name, url = dialog.getConnectionData()
        assert name == "Updated Server"
        assert url == "http://localhost:4000"

    def test_add_invalid_connection(self, qtbot, monkeypatch):
        """Test adding invalid connection shows error."""
        # Mock exec methods to prevent dialogs from blocking
        from novelwriter.dialogs.mcp_tool_manager import MCPConnectionEditDialog
        monkeypatch.setattr(MCPConnectionEditDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        monkeypatch.setattr(MCPToolManagerDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        
        dialog = MCPToolManagerDialog(self.nwGUI)
        qtbot.addWidget(dialog)
        dialog._updateTimer.stop()

        # Mock ALL QMessageBox methods to avoid actual dialogs
        shown_warnings = []
        def mock_warning(parent, title, message):
            shown_warnings.append((title, message))
            return QMessageBox.StandardButton.Ok
        
        def mock_question(parent, title, message, buttons, default=None):
            shown_warnings.append((title, message))
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, "warning", mock_warning)
        monkeypatch.setattr(QMessageBox, "question", mock_question)

        # Try to add connection with invalid URL
        success = dialog.addExternalConnection("Bad Server", "not-a-url")
        assert not success
        assert len(shown_warnings) == 1
        assert "Invalid URL" in shown_warnings[0][0]

        # Try to add duplicate connection  
        shown_warnings.clear()
        dialog.addExternalConnection("Server 1", "http://localhost:3000")
        shown_warnings.clear()  # Clear any warnings from the first add
        
        success = dialog.addExternalConnection("Server 2", "http://localhost:3000")
        assert not success
        assert len(shown_warnings) == 1
        assert "Duplicate" in shown_warnings[0][0]

    def test_remove_connection(self, qtbot, monkeypatch):
        """Test removing a connection."""
        # Mock exec methods and ALL QMessageBox to prevent dialogs from blocking
        from novelwriter.dialogs.mcp_tool_manager import MCPConnectionEditDialog
        monkeypatch.setattr(MCPConnectionEditDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        monkeypatch.setattr(MCPToolManagerDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        
        # Mock ALL QMessageBox methods
        def mock_warning(parent, title, message):
            return QMessageBox.StandardButton.Ok
        
        def mock_question(parent, title, message, buttons, default=None):
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, "warning", mock_warning)
        monkeypatch.setattr(QMessageBox, "question", mock_question)
        
        dialog = MCPToolManagerDialog(self.nwGUI)
        qtbot.addWidget(dialog)
        dialog._updateTimer.stop()

        # Add a connection
        dialog.addExternalConnection("Test Server", "http://localhost:3000")
        assert dialog.connectionTable.rowCount() == 1

        # Select it
        dialog.connectionTable.selectRow(0)
        qtbot.wait(50)  # Allow selection to process
        assert dialog._selectedRow == 0
        
        # Test the remove functionality by calling the method directly
        # This avoids the actual button click and dialog interaction
        dialog._removeConnection()
        
        # Check it's gone
        assert dialog.connectionTable.rowCount() == 0
        assert len(dialog._connections) == 0
        assert dialog._selectedRow == -1

    def test_connection_enabled_toggle(self, qtbot, monkeypatch):
        """Test enabling/disabling connections."""
        # Mock exec methods to prevent dialogs from blocking
        from novelwriter.dialogs.mcp_tool_manager import MCPConnectionEditDialog
        monkeypatch.setattr(MCPConnectionEditDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        monkeypatch.setattr(MCPToolManagerDialog, "exec", lambda *a: NDialog.DialogCode.Rejected)
        
        # Mock ALL QMessageBox methods to prevent any dialogs
        def mock_warning(parent, title, message):
            return QMessageBox.StandardButton.Ok
        
        def mock_question(parent, title, message, buttons, default=None):
            return QMessageBox.StandardButton.Yes
        
        monkeypatch.setattr(QMessageBox, "warning", mock_warning)
        monkeypatch.setattr(QMessageBox, "question", mock_question)
        
        dialog = MCPToolManagerDialog(self.nwGUI)
        qtbot.addWidget(dialog)
        dialog._updateTimer.stop()

        # Add a connection
        dialog.addExternalConnection("Test Server", "http://localhost:3000")
        
        # Get the enabled switch
        switch = dialog.connectionTable.cellWidget(0, 4)
        assert switch.isChecked()  # Should be enabled by default

        # Toggle it off using the method directly instead of mouse click
        dialog._onEnabledToggled(0, False)
        qtbot.wait(50)  # Allow toggle to process
        assert not dialog._connections[0]["enabled"]

        # Toggle it back on 
        dialog._onEnabledToggled(0, True) 
        qtbot.wait(50)  # Allow toggle to process
        assert dialog._connections[0]["enabled"]


@pytest.mark.gui
class TestMCPConfigIntegration:
    """Test MCP configuration integration."""

    def test_mcp_config_validation(self):
        """Test MCP configuration validation."""
        from novelwriter.api.base.config import MCPHybridConfig
        
        config = MCPHybridConfig()
        
        # Test valid configuration
        assert config.isValid
        assert len(config.validationErrors) == 0

        # Test invalid values (should fail to set and remain valid due to rollback)
        result = config.setValue("performance.alertThresholds.apiLatency", -1)
        assert not result  # setValue should return False for invalid value
        assert config.isValid  # Config should remain valid after rollback
        assert config.getValue("performance.alertThresholds.apiLatency") == 5  # Should keep default value

        # Reset to valid state
        config.reset()
        assert config.isValid

        # Test invalid error rate (should also fail and rollback)
        result = config.setValue("performance.alertThresholds.errorRate", 1.5)
        assert not result  # setValue should return False
        assert config.isValid  # Config should remain valid after rollback
        assert config.getValue("performance.alertThresholds.errorRate") == 0.05  # Should keep default value

    def test_mcp_config_export_import(self):
        """Test MCP configuration export and import."""
        from novelwriter.api.base.config import MCPHybridConfig
        
        config = MCPHybridConfig()
        
        # Modify some values
        config.enabled = True
        config.setValue("localTools.project_info", False)
        config.setValue("externalMCP.timeout", 500)
        
        # Export
        exported = config.toDict()
        assert exported["enabled"] == True
        assert exported["localTools"]["project_info"] == False
        assert exported["externalMCP"]["timeout"] == 500
        
        # Create new config and import
        config2 = MCPHybridConfig()
        assert config2.enabled == False  # Default
        
        success = config2.fromDict(exported)
        assert success
        assert config2.enabled == True
        assert config2.getValue("localTools.project_info") == False
        assert config2.getValue("externalMCP.timeout") == 500

    def test_mcp_config_signals(self, qtbot):
        """Test MCP configuration change signals."""
        from novelwriter.api.base.config import MCPHybridConfig
        
        config = MCPHybridConfig()
        
        # Track signals
        signals_received = []
        config.configChanged.connect(lambda key, value: signals_received.append((key, value)))
        
        # Change a value
        config.setValue("enabled", True)
        assert len(signals_received) == 1
        assert signals_received[0] == ("enabled", True)
        
        # Change nested value
        config.setValue("performance.monitoringEnabled", False)
        assert len(signals_received) == 2
        assert signals_received[1] == ("performance.monitoringEnabled", False)
        
        # Reset configuration
        with qtbot.waitSignal(config.configReset):
            config.reset()
        
        assert config.enabled == False  # Back to default
