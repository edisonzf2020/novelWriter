"""
novelWriter – AI Settings Dialog Tests
=======================================

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

import pytest
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QListWidgetItem, QMessageBox

from novelwriter.dialogs.ai_settings import GuiAISettings, AddServerDialog

from tests.tools import C, buildTestProject


@pytest.mark.gui
class TestGuiAISettings:
    """Test the AI Settings dialog."""
    
    @pytest.fixture(autouse=True)
    def _setup(self, qtbot, nwGUI, projPath, mockRnd):
        """Set up test environment."""
        buildTestProject(nwGUI, projPath)
        yield
        nwGUI.closeProject()
    
    def test_dialog_creation(self, qtbot, nwGUI):
        """Test dialog creation and basic structure."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Check dialog properties
        assert dialog.windowTitle() == "AI Settings"
        assert dialog.minimumWidth() >= 800
        assert dialog.minimumHeight() >= 600
        
        # Check tabs exist
        assert dialog.tabWidget.count() == 3
        assert dialog.tabWidget.tabText(0) == "External Tools"
        assert dialog.tabWidget.tabText(1) == "Local Tools"
        assert dialog.tabWidget.tabText(2) == "Performance"
        
        # Check buttons exist
        assert dialog.buttonBox is not None
        buttons = dialog.buttonBox.buttons()
        assert len(buttons) == 3  # Ok, Cancel, Apply
    
    def test_external_tools_tab(self, qtbot, nwGUI):
        """Test External Tools tab functionality."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Check server list
        assert dialog.serverList is not None
        assert dialog.serverList.count() == 1  # Example server
        
        # Check server configuration fields
        assert dialog.serverNameEdit is not None
        assert dialog.serverUrlEdit is not None
        assert dialog.serverCommandEdit is not None
        assert dialog.serverArgsEdit is not None
        assert dialog.serverEnabledCheck is not None
        
        # Check buttons
        assert dialog.addServerBtn is not None
        assert dialog.removeServerBtn is not None
        assert dialog.testServerBtn is not None
    
    def test_local_tools_tab(self, qtbot, nwGUI):
        """Test Local Tools tab functionality."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Switch to Local Tools tab
        dialog.tabWidget.setCurrentIndex(1)
        
        # Check permission checkboxes
        assert dialog.allowReadCheck is not None
        assert dialog.allowWriteCheck is not None
        assert dialog.allowCreateCheck is not None
        assert dialog.allowDeleteCheck is not None
        
        # Check default permissions
        assert dialog.allowReadCheck.isChecked() is True
        assert dialog.allowWriteCheck.isChecked() is False
        assert dialog.allowCreateCheck.isChecked() is False
        assert dialog.allowDeleteCheck.isChecked() is False
        
        # Check tool list
        assert dialog.toolList is not None
        assert dialog.toolList.count() == 9  # 9 local tools
        
        # Check all tools are enabled by default
        for i in range(dialog.toolList.count()):
            item = dialog.toolList.item(i)
            assert item.checkState() == Qt.CheckState.Checked
    
    def test_performance_tab(self, qtbot, nwGUI):
        """Test Performance tab functionality."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Switch to Performance tab
        dialog.tabWidget.setCurrentIndex(2)
        
        # Check cache settings
        assert dialog.cacheSizeSpin is not None
        assert dialog.cacheSizeSpin.value() == 100
        assert dialog.cacheTTLSpin is not None
        assert dialog.cacheTTLSpin.value() == 300
        assert dialog.cacheEntriesSpin is not None
        assert dialog.cacheEntriesSpin.value() == 10000
        
        # Check timeout settings
        assert dialog.externalTimeoutSpin is not None
        assert dialog.externalTimeoutSpin.value() == 200
        assert dialog.healthCheckIntervalSpin is not None
        assert dialog.healthCheckIntervalSpin.value() == 30
        
        # Check circuit breaker settings
        assert dialog.failureThresholdSpin is not None
        assert dialog.failureThresholdSpin.value() == 3
        assert dialog.recoveryTimeoutSpin is not None
        assert dialog.recoveryTimeoutSpin.value() == 60
    
    def test_add_server(self, qtbot, nwGUI, monkeypatch):
        """Test adding a new MCP server."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Mock the AddServerDialog
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.getConfiguration.return_value = {
            "name": "Test Server",
            "url": "http://localhost:3002",
            "command": "uvx",
            "args": "test-server",
            "enabled": True
        }
        
        with patch("novelwriter.dialogs.ai_settings.AddServerDialog", return_value=mock_dialog):
            # Click add server button
            initial_count = dialog.serverList.count()
            dialog._addServer()
            
            # Check server was added
            assert dialog.serverList.count() == initial_count + 1
            
            # Check last item
            last_item = dialog.serverList.item(dialog.serverList.count() - 1)
            assert "Test Server" in last_item.text()
            assert "http://localhost:3002" in last_item.text()
    
    def test_remove_server(self, qtbot, nwGUI, monkeypatch):
        """Test removing an MCP server."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Select first server
        dialog.serverList.setCurrentRow(0)
        initial_count = dialog.serverList.count()
        
        # Mock confirmation dialog
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        
        # Remove server
        dialog._removeServer()
        
        # Check server was removed
        assert dialog.serverList.count() == initial_count - 1
    
    def test_test_connection(self, qtbot, nwGUI):
        """Test connection testing functionality."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Select first server
        dialog.serverList.setCurrentRow(0)
        
        # Test connection
        dialog._testConnection()
        
        # Check status text updated
        assert "Testing connection" in dialog.statusText.toPlainText()
        assert "Connection successful" in dialog.statusText.toPlainText()
    
    def test_clear_cache(self, qtbot, nwGUI, monkeypatch):
        """Test cache clearing functionality."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Switch to Performance tab
        dialog.tabWidget.setCurrentIndex(2)
        
        # Mock confirmation dialog
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        
        # Mock CacheManager
        with patch("novelwriter.api.external_mcp.CacheManager") as mock_cm:
            mock_instance = MagicMock()
            mock_cm.return_value = mock_instance
            
            # Clear cache
            dialog._clearCache()
            
            # Check clear_all was called
            mock_instance.clear_all.assert_called_once()
    
    def test_refresh_stats(self, qtbot, nwGUI):
        """Test statistics refresh functionality."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Switch to Performance tab
        dialog.tabWidget.setCurrentIndex(2)
        
        # Mock CacheManager
        with patch("novelwriter.api.external_mcp.CacheManager") as mock_cm:
            mock_instance = MagicMock()
            mock_cm.return_value = mock_instance
            mock_instance.get_global_statistics.return_value = {
                "global": {
                    "total_entries": 100,
                    "total_size_mb": 5.5,
                    "total_hits": 500,
                    "total_misses": 100,
                    "global_hit_rate": 0.833
                }
            }
            
            # Refresh stats
            dialog._refreshStats()
            
            # Check stats displayed
            stats_text = dialog.statsText.toPlainText()
            assert "Total Entries: 100" in stats_text
            assert "Total Size: 5.50 MB" in stats_text
            assert "Hit Rate: 83.30%" in stats_text
    
    def test_save_settings(self, qtbot, nwGUI):
        """Test saving settings."""
        dialog = GuiAISettings(nwGUI)
        qtbot.addWidget(dialog)
        
        # Modify some settings
        dialog.allowWriteCheck.setChecked(True)
        dialog.cacheSizeSpin.setValue(200)
        dialog.externalTimeoutSpin.setValue(300)
        
        # Save settings
        dialog._saveSettings()
        
        # Check status updated
        assert "Settings saved" in dialog.statusText.toPlainText()


@pytest.mark.gui
class TestAddServerDialog:
    """Test the Add Server dialog."""
    
    def test_dialog_creation(self, qtbot):
        """Test dialog creation."""
        dialog = AddServerDialog(None)
        qtbot.addWidget(dialog)
        
        assert dialog.windowTitle() == "Add MCP Server"
        assert dialog.isModal() is True
        
        # Check fields exist
        assert dialog.nameEdit is not None
        assert dialog.urlEdit is not None
        assert dialog.commandEdit is not None
        assert dialog.argsEdit is not None
        assert dialog.enabledCheck is not None
    
    def test_get_configuration(self, qtbot):
        """Test getting server configuration."""
        dialog = AddServerDialog(None)
        qtbot.addWidget(dialog)
        
        # Set values
        dialog.nameEdit.setText("Test Server")
        dialog.urlEdit.setText("http://localhost:3000")
        dialog.commandEdit.setText("uvx")
        dialog.argsEdit.setText("test-mcp-server")
        dialog.enabledCheck.setChecked(False)
        
        # Get configuration
        config = dialog.getConfiguration()
        
        assert config["name"] == "Test Server"
        assert config["url"] == "http://localhost:3000"
        assert config["command"] == "uvx"
        assert config["args"] == "test-mcp-server"
        assert config["enabled"] is False
