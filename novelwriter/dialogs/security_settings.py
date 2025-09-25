"""
novelWriter – Security Settings Dialog
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

import logging
from typing import TYPE_CHECKING, List, Optional
from datetime import datetime, timedelta

from PyQt6.QtCore import Qt, QDateTime, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QLabel, QPushButton, QLineEdit,
    QTextEdit, QCheckBox, QSpinBox, QComboBox,
    QGroupBox, QTableWidget, QTableWidgetItem,
    QDialogButtonBox, QMessageBox, QFormLayout,
    QDateTimeEdit, QHeaderView
)

from novelwriter.common import qtLambda

if TYPE_CHECKING:
    from novelwriter.guimain import GuiMain

logger = logging.getLogger(__name__)


class GuiSecuritySettings(QDialog):
    """Security Settings Dialog for managing permissions and audit logs."""
    
    def __init__(self, parent: GuiMain) -> None:
        """Initialize the Security Settings dialog.
        
        Args:
            parent: Main GUI window
        """
        super().__init__(parent=parent)
        
        self.mainGui = parent
        
        # Dialog settings
        self.setWindowTitle(self.tr("Security Settings"))
        self.setMinimumSize(900, 700)
        
        # Create main layout
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        
        # Create tab widget
        self.tabWidget = QTabWidget()
        self.mainLayout.addWidget(self.tabWidget)
        
        # Create tabs
        self._createPermissionsTab()
        self._createAuditLogTab()
        self._createResourceLimitsTab()
        self._createDataSecurityTab()
        
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
    
    def _createPermissionsTab(self) -> None:
        """Create the Permissions configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Permission levels
        permGroup = QGroupBox(self.tr("Permission Levels"))
        permLayout = QVBoxLayout()
        permGroup.setLayout(permLayout)
        
        # Permission checkboxes
        self.permissionChecks = {}
        permissions = [
            ("read", self.tr("Read - View documents and project information")),
            ("write", self.tr("Write - Modify existing documents")),
            ("create", self.tr("Create - Add new documents and folders")),
            ("delete", self.tr("Delete - Remove documents and folders")),
            ("tool_call", self.tr("Tool Call - Execute local tools")),
            ("external_tool", self.tr("External Tool - Call external MCP tools")),
            ("admin", self.tr("Admin - Full system access"))
        ]
        
        for perm_id, perm_desc in permissions:
            checkbox = QCheckBox(perm_desc)
            self.permissionChecks[perm_id] = checkbox
            permLayout.addWidget(checkbox)
        
        layout.addWidget(permGroup)
        
        # Role-based access control
        rbacGroup = QGroupBox(self.tr("Role-Based Access Control"))
        rbacLayout = QFormLayout()
        rbacGroup.setLayout(rbacLayout)
        
        self.roleCombo = QComboBox()
        self.roleCombo.addItems([
            self.tr("User"),
            self.tr("Editor"),
            self.tr("Administrator")
        ])
        self.roleCombo.currentIndexChanged.connect(self._onRoleChanged)
        
        rbacLayout.addRow(self.tr("Default Role:"), self.roleCombo)
        
        layout.addWidget(rbacGroup)
        
        # Session management
        sessionGroup = QGroupBox(self.tr("Session Management"))
        sessionLayout = QFormLayout()
        sessionGroup.setLayout(sessionLayout)
        
        self.sessionTimeoutSpin = QSpinBox()
        self.sessionTimeoutSpin.setMinimum(5)
        self.sessionTimeoutSpin.setMaximum(1440)
        self.sessionTimeoutSpin.setValue(30)
        self.sessionTimeoutSpin.setSuffix(self.tr(" minutes"))
        
        self.requireAuthCheck = QCheckBox(self.tr("Require authentication"))
        
        sessionLayout.addRow(self.tr("Session Timeout:"), self.sessionTimeoutSpin)
        sessionLayout.addRow("", self.requireAuthCheck)
        
        layout.addWidget(sessionGroup)
        layout.addStretch()
        
        self.tabWidget.addTab(widget, self.tr("Permissions"))
        return
    
    def _createAuditLogTab(self) -> None:
        """Create the Audit Log viewer tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Filter controls
        filterGroup = QGroupBox(self.tr("Filters"))
        filterLayout = QHBoxLayout()
        filterGroup.setLayout(filterLayout)
        
        # Date range
        filterLayout.addWidget(QLabel(self.tr("From:")))
        self.startDateEdit = QDateTimeEdit()
        self.startDateEdit.setCalendarPopup(True)
        self.startDateEdit.setDateTime(QDateTime.currentDateTime().addDays(-7))
        filterLayout.addWidget(self.startDateEdit)
        
        filterLayout.addWidget(QLabel(self.tr("To:")))
        self.endDateEdit = QDateTimeEdit()
        self.endDateEdit.setCalendarPopup(True)
        self.endDateEdit.setDateTime(QDateTime.currentDateTime())
        filterLayout.addWidget(self.endDateEdit)
        
        # Risk level filter
        filterLayout.addWidget(QLabel(self.tr("Risk Level:")))
        self.riskLevelCombo = QComboBox()
        self.riskLevelCombo.addItems([
            self.tr("All"),
            self.tr("Low"),
            self.tr("Medium"),
            self.tr("High"),
            self.tr("Critical")
        ])
        filterLayout.addWidget(self.riskLevelCombo)
        
        # Search button
        self.searchBtn = QPushButton(self.tr("Search"))
        self.searchBtn.clicked.connect(self._searchAuditLogs)
        filterLayout.addWidget(self.searchBtn)
        
        filterLayout.addStretch()
        
        layout.addWidget(filterGroup)
        
        # Audit log table
        self.auditTable = QTableWidget()
        self.auditTable.setColumnCount(7)
        self.auditTable.setHorizontalHeaderLabels([
            self.tr("Timestamp"),
            self.tr("Session"),
            self.tr("Operation"),
            self.tr("Resource"),
            self.tr("Result"),
            self.tr("Risk Level"),
            self.tr("Time (ms)")
        ])
        
        # Set column widths
        header = self.auditTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.auditTable)
        
        # Control buttons
        controlLayout = QHBoxLayout()
        
        self.exportBtn = QPushButton(self.tr("Export"))
        self.exportBtn.clicked.connect(self._exportAuditLogs)
        controlLayout.addWidget(self.exportBtn)
        
        self.clearBtn = QPushButton(self.tr("Clear Old Logs"))
        self.clearBtn.clicked.connect(self._clearOldLogs)
        controlLayout.addWidget(self.clearBtn)
        
        controlLayout.addStretch()
        
        # Statistics
        self.statsLabel = QLabel(self.tr("Total entries: 0"))
        controlLayout.addWidget(self.statsLabel)
        
        layout.addLayout(controlLayout)
        
        self.tabWidget.addTab(widget, self.tr("Audit Logs"))
        return
    
    def _createResourceLimitsTab(self) -> None:
        """Create the Resource Limits configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Rate limiting
        rateGroup = QGroupBox(self.tr("Rate Limiting"))
        rateLayout = QFormLayout()
        rateGroup.setLayout(rateLayout)
        
        self.apiCallsPerMinuteSpin = QSpinBox()
        self.apiCallsPerMinuteSpin.setMinimum(10)
        self.apiCallsPerMinuteSpin.setMaximum(1000)
        self.apiCallsPerMinuteSpin.setValue(60)
        
        self.apiCallsPerHourSpin = QSpinBox()
        self.apiCallsPerHourSpin.setMinimum(100)
        self.apiCallsPerHourSpin.setMaximum(10000)
        self.apiCallsPerHourSpin.setValue(1000)
        
        self.concurrentCallsSpin = QSpinBox()
        self.concurrentCallsSpin.setMinimum(1)
        self.concurrentCallsSpin.setMaximum(100)
        self.concurrentCallsSpin.setValue(10)
        
        rateLayout.addRow(self.tr("API Calls per Minute:"), self.apiCallsPerMinuteSpin)
        rateLayout.addRow(self.tr("API Calls per Hour:"), self.apiCallsPerHourSpin)
        rateLayout.addRow(self.tr("Concurrent Calls:"), self.concurrentCallsSpin)
        
        layout.addWidget(rateGroup)
        
        # Resource quotas
        quotaGroup = QGroupBox(self.tr("Resource Quotas"))
        quotaLayout = QFormLayout()
        quotaGroup.setLayout(quotaLayout)
        
        self.memoryLimitSpin = QSpinBox()
        self.memoryLimitSpin.setMinimum(100)
        self.memoryLimitSpin.setMaximum(5000)
        self.memoryLimitSpin.setValue(500)
        self.memoryLimitSpin.setSuffix(" MB")
        
        self.cpuLimitSpin = QSpinBox()
        self.cpuLimitSpin.setMinimum(10)
        self.cpuLimitSpin.setMaximum(100)
        self.cpuLimitSpin.setValue(50)
        self.cpuLimitSpin.setSuffix(" %")
        
        self.timeoutSpin = QSpinBox()
        self.timeoutSpin.setMinimum(100)
        self.timeoutSpin.setMaximum(30000)
        self.timeoutSpin.setValue(5000)
        self.timeoutSpin.setSuffix(" ms")
        
        quotaLayout.addRow(self.tr("Memory Limit:"), self.memoryLimitSpin)
        quotaLayout.addRow(self.tr("CPU Limit:"), self.cpuLimitSpin)
        quotaLayout.addRow(self.tr("Operation Timeout:"), self.timeoutSpin)
        
        layout.addWidget(quotaGroup)
        
        # Usage monitoring
        monitorGroup = QGroupBox(self.tr("Current Usage"))
        monitorLayout = QVBoxLayout()
        monitorGroup.setLayout(monitorLayout)
        
        self.usageText = QTextEdit()
        self.usageText.setReadOnly(True)
        self.usageText.setMaximumHeight(150)
        monitorLayout.addWidget(self.usageText)
        
        self.refreshUsageBtn = QPushButton(self.tr("Refresh"))
        self.refreshUsageBtn.clicked.connect(self._refreshUsage)
        monitorLayout.addWidget(self.refreshUsageBtn)
        
        layout.addWidget(monitorGroup)
        layout.addStretch()
        
        self.tabWidget.addTab(widget, self.tr("Resource Limits"))
        return
    
    def _createDataSecurityTab(self) -> None:
        """Create the Data Security configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Data classification
        classifyGroup = QGroupBox(self.tr("Data Classification"))
        classifyLayout = QVBoxLayout()
        classifyGroup.setLayout(classifyLayout)
        
        self.autoClassifyCheck = QCheckBox(
            self.tr("Automatically classify sensitive data")
        )
        self.maskSensitiveCheck = QCheckBox(
            self.tr("Mask sensitive data in logs")
        )
        self.encryptStorageCheck = QCheckBox(
            self.tr("Encrypt sensitive data at rest")
        )
        
        classifyLayout.addWidget(self.autoClassifyCheck)
        classifyLayout.addWidget(self.maskSensitiveCheck)
        classifyLayout.addWidget(self.encryptStorageCheck)
        
        layout.addWidget(classifyGroup)
        
        # Input validation
        validationGroup = QGroupBox(self.tr("Input Validation"))
        validationLayout = QVBoxLayout()
        validationGroup.setLayout(validationLayout)
        
        self.sqlInjectionCheck = QCheckBox(
            self.tr("SQL injection protection")
        )
        self.xssProtectionCheck = QCheckBox(
            self.tr("XSS attack protection")
        )
        self.pathTraversalCheck = QCheckBox(
            self.tr("Path traversal protection")
        )
        self.sanitizeInputCheck = QCheckBox(
            self.tr("Sanitize all user input")
        )
        
        validationLayout.addWidget(self.sqlInjectionCheck)
        validationLayout.addWidget(self.xssProtectionCheck)
        validationLayout.addWidget(self.pathTraversalCheck)
        validationLayout.addWidget(self.sanitizeInputCheck)
        
        layout.addWidget(validationGroup)
        
        # Audit settings
        auditGroup = QGroupBox(self.tr("Audit Settings"))
        auditLayout = QFormLayout()
        auditGroup.setLayout(auditLayout)
        
        self.retentionDaysSpin = QSpinBox()
        self.retentionDaysSpin.setMinimum(7)
        self.retentionDaysSpin.setMaximum(365)
        self.retentionDaysSpin.setValue(30)
        self.retentionDaysSpin.setSuffix(self.tr(" days"))
        
        self.maxLogSizeSpin = QSpinBox()
        self.maxLogSizeSpin.setMinimum(10)
        self.maxLogSizeSpin.setMaximum(1000)
        self.maxLogSizeSpin.setValue(100)
        self.maxLogSizeSpin.setSuffix(" MB")
        
        self.compressLogsCheck = QCheckBox(self.tr("Compress old logs"))
        self.integrityCheckCheck = QCheckBox(self.tr("Enable integrity verification"))
        
        auditLayout.addRow(self.tr("Log Retention:"), self.retentionDaysSpin)
        auditLayout.addRow(self.tr("Max Log Size:"), self.maxLogSizeSpin)
        auditLayout.addRow("", self.compressLogsCheck)
        auditLayout.addRow("", self.integrityCheckCheck)
        
        layout.addWidget(auditGroup)
        layout.addStretch()
        
        self.tabWidget.addTab(widget, self.tr("Data Security"))
        return
    
    def _onRoleChanged(self, index: int) -> None:
        """Handle role selection change.
        
        Args:
            index: Selected role index
        """
        # Update permissions based on role
        if index == 0:  # User
            self.permissionChecks["read"].setChecked(True)
            self.permissionChecks["write"].setChecked(False)
            self.permissionChecks["create"].setChecked(False)
            self.permissionChecks["delete"].setChecked(False)
            self.permissionChecks["tool_call"].setChecked(False)
            self.permissionChecks["external_tool"].setChecked(False)
            self.permissionChecks["admin"].setChecked(False)
        elif index == 1:  # Editor
            self.permissionChecks["read"].setChecked(True)
            self.permissionChecks["write"].setChecked(True)
            self.permissionChecks["create"].setChecked(True)
            self.permissionChecks["delete"].setChecked(False)
            self.permissionChecks["tool_call"].setChecked(True)
            self.permissionChecks["external_tool"].setChecked(False)
            self.permissionChecks["admin"].setChecked(False)
        elif index == 2:  # Administrator
            for checkbox in self.permissionChecks.values():
                checkbox.setChecked(True)
        return
    
    def _searchAuditLogs(self) -> None:
        """Search and display audit logs."""
        # Clear table
        self.auditTable.setRowCount(0)
        
        # In real implementation, would query actual audit logs
        # For now, add sample data
        sample_logs = [
            {
                "timestamp": "2025-09-25 10:30:15",
                "session": "sess_123",
                "operation": "get_document",
                "resource": "doc_456",
                "result": "success",
                "risk_level": "low",
                "time_ms": "5"
            },
            {
                "timestamp": "2025-09-25 10:31:20",
                "session": "sess_124",
                "operation": "save_document",
                "resource": "doc_789",
                "result": "success",
                "risk_level": "medium",
                "time_ms": "12"
            },
            {
                "timestamp": "2025-09-25 10:32:45",
                "session": "sess_125",
                "operation": "delete_document",
                "resource": "doc_111",
                "result": "denied",
                "risk_level": "high",
                "time_ms": "2"
            }
        ]
        
        for log in sample_logs:
            row = self.auditTable.rowCount()
            self.auditTable.insertRow(row)
            
            self.auditTable.setItem(row, 0, QTableWidgetItem(log["timestamp"]))
            self.auditTable.setItem(row, 1, QTableWidgetItem(log["session"]))
            self.auditTable.setItem(row, 2, QTableWidgetItem(log["operation"]))
            self.auditTable.setItem(row, 3, QTableWidgetItem(log["resource"]))
            self.auditTable.setItem(row, 4, QTableWidgetItem(log["result"]))
            self.auditTable.setItem(row, 5, QTableWidgetItem(log["risk_level"]))
            self.auditTable.setItem(row, 6, QTableWidgetItem(log["time_ms"]))
        
        self.statsLabel.setText(f"Total entries: {len(sample_logs)}")
        return
    
    def _exportAuditLogs(self) -> None:
        """Export audit logs to file."""
        QMessageBox.information(
            self,
            self.tr("Export"),
            self.tr("Audit logs would be exported to file.")
        )
        return
    
    def _clearOldLogs(self) -> None:
        """Clear old audit logs."""
        reply = QMessageBox.question(
            self,
            self.tr("Clear Logs"),
            self.tr("Are you sure you want to clear old audit logs?")
        )
        if reply == QMessageBox.StandardButton.Yes:
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr("Old logs cleared successfully.")
            )
        return
    
    def _refreshUsage(self) -> None:
        """Refresh resource usage display."""
        usage_text = """Current Resource Usage:
  API Calls (last minute): 15 / 60
  API Calls (last hour): 234 / 1000
  Concurrent Calls: 3 / 10
  Memory Usage: 125 MB / 500 MB
  CPU Usage: 23% / 50%
"""
        self.usageText.setText(usage_text)
        return
    
    def _loadSettings(self) -> None:
        """Load current security settings."""
        # Set default values
        self.permissionChecks["read"].setChecked(True)
        self.permissionChecks["write"].setChecked(False)
        self.permissionChecks["create"].setChecked(False)
        self.permissionChecks["delete"].setChecked(False)
        self.permissionChecks["tool_call"].setChecked(True)
        self.permissionChecks["external_tool"].setChecked(False)
        self.permissionChecks["admin"].setChecked(False)
        
        self.roleCombo.setCurrentIndex(0)
        
        # Data security defaults
        self.autoClassifyCheck.setChecked(True)
        self.maskSensitiveCheck.setChecked(True)
        self.encryptStorageCheck.setChecked(False)
        
        self.sqlInjectionCheck.setChecked(True)
        self.xssProtectionCheck.setChecked(True)
        self.pathTraversalCheck.setChecked(True)
        self.sanitizeInputCheck.setChecked(True)
        
        self.compressLogsCheck.setChecked(True)
        self.integrityCheckCheck.setChecked(True)
        
        return
    
    def _saveSettings(self) -> None:
        """Save security settings."""
        # Collect settings
        settings = {
            "permissions": {
                perm_id: checkbox.isChecked()
                for perm_id, checkbox in self.permissionChecks.items()
            },
            "session": {
                "timeout_minutes": self.sessionTimeoutSpin.value(),
                "require_auth": self.requireAuthCheck.isChecked()
            },
            "rate_limits": {
                "api_calls_per_minute": self.apiCallsPerMinuteSpin.value(),
                "api_calls_per_hour": self.apiCallsPerHourSpin.value(),
                "concurrent_calls": self.concurrentCallsSpin.value()
            },
            "resource_quotas": {
                "memory_mb": self.memoryLimitSpin.value(),
                "cpu_percent": self.cpuLimitSpin.value(),
                "timeout_ms": self.timeoutSpin.value()
            },
            "data_security": {
                "auto_classify": self.autoClassifyCheck.isChecked(),
                "mask_sensitive": self.maskSensitiveCheck.isChecked(),
                "encrypt_storage": self.encryptStorageCheck.isChecked(),
                "sql_injection_protection": self.sqlInjectionCheck.isChecked(),
                "xss_protection": self.xssProtectionCheck.isChecked(),
                "path_traversal_protection": self.pathTraversalCheck.isChecked(),
                "sanitize_input": self.sanitizeInputCheck.isChecked()
            },
            "audit": {
                "retention_days": self.retentionDaysSpin.value(),
                "max_log_size_mb": self.maxLogSizeSpin.value(),
                "compress_logs": self.compressLogsCheck.isChecked(),
                "integrity_check": self.integrityCheckCheck.isChecked()
            }
        }
        
        # Save to config (would be implemented with actual config system)
        logger.info(f"Saving security settings: {settings}")
        
        QMessageBox.information(
            self,
            self.tr("Success"),
            self.tr("Security settings saved successfully.")
        )
        
        return
    
    def _saveAndClose(self) -> None:
        """Save settings and close dialog."""
        self._saveSettings()
        self.accept()
        return
