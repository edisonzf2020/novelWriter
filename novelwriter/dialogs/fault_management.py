"""
novelWriter – Fault Management Settings Dialog
===============================================

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
from datetime import datetime
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from novelwriter import CONFIG
from novelwriter.api.base.circuit_breaker import (
    CircuitBreakerState,
    get_circuit_breaker_manager,
)
from novelwriter.api.base.degradation_service import (
    ServiceDegradationLevel,
    get_degradation_service,
)
from novelwriter.api.base.fault_handling import (
    AlertLevel,
    get_fault_handling_system,
)
from novelwriter.api.base.retry_manager import get_retry_manager
from novelwriter.common import formatFileFilter, formatTimeStamp
from novelwriter.constants import nwLabels
from novelwriter.types import QtDialogClose

if TYPE_CHECKING:
    from novelwriter.guimain import GuiMain

logger = logging.getLogger(__name__)


class GuiFaultManagement(QDialog):
    """Fault Management Settings Dialog."""
    
    def __init__(self, parent: GuiMain) -> None:
        super().__init__(parent=parent)
        
        logger.debug("Create: GuiFaultManagement")
        
        self.mainGui = parent
        
        # Get fault handling components
        self.fault_system = get_fault_handling_system()
        self.circuit_manager = get_circuit_breaker_manager()
        self.degradation_service = get_degradation_service()
        self.retry_manager = get_retry_manager()
        
        # Dialog settings
        self.setObjectName("GuiFaultManagement")
        self.setWindowTitle(self.tr("Fault Management"))
        self.setMinimumSize(CONFIG.pxInt(800), CONFIG.pxInt(600))
        
        # Create UI
        self._buildUI()
        
        # Load current state
        self._loadCurrentState()
        
        # Setup auto-refresh
        self._setupAutoRefresh()
        
        logger.debug("Ready: GuiFaultManagement")
    
    def _buildUI(self) -> None:
        """Build the dialog UI."""
        # Main layout
        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)
        
        # Tab widget
        self.tabWidget = QTabWidget()
        mainLayout.addWidget(self.tabWidget)
        
        # Create tabs
        self._createSystemStatusTab()
        self._createCircuitBreakersTab()
        self._createDegradationTab()
        self._createAlertsTab()
        self._createMetricsTab()
        
        # Button box
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close
        )
        self.buttonBox.rejected.connect(self.close)
        mainLayout.addWidget(self.buttonBox)
    
    def _createSystemStatusTab(self) -> None:
        """Create system status tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Health status group
        healthGroup = QGroupBox(self.tr("System Health"))
        healthLayout = QGridLayout()
        healthGroup.setLayout(healthLayout)
        
        # Health score
        self.healthScoreLabel = QLabel("100")
        self.healthScoreLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.healthScoreLabel.setFont(font)
        healthLayout.addWidget(QLabel(self.tr("Health Score:")), 0, 0)
        healthLayout.addWidget(self.healthScoreLabel, 0, 1)
        
        # Health status
        self.healthStatusLabel = QLabel(self.tr("Healthy"))
        self.healthStatusLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        healthLayout.addWidget(QLabel(self.tr("Status:")), 1, 0)
        healthLayout.addWidget(self.healthStatusLabel, 1, 1)
        
        # Error rate
        self.errorRateLabel = QLabel("0.0/min")
        healthLayout.addWidget(QLabel(self.tr("Error Rate:")), 2, 0)
        healthLayout.addWidget(self.errorRateLabel, 2, 1)
        
        # Degradation level
        self.degradationLabel = QLabel(self.tr("Full Service"))
        healthLayout.addWidget(QLabel(self.tr("Service Level:")), 3, 0)
        healthLayout.addWidget(self.degradationLabel, 3, 1)
        
        layout.addWidget(healthGroup)
        
        # Quick actions group
        actionsGroup = QGroupBox(self.tr("Quick Actions"))
        actionsLayout = QHBoxLayout()
        actionsGroup.setLayout(actionsLayout)
        
        # Reset circuit breakers button
        self.resetBreakersBtn = QPushButton(self.tr("Reset All Circuit Breakers"))
        self.resetBreakersBtn.clicked.connect(self._resetAllBreakers)
        actionsLayout.addWidget(self.resetBreakersBtn)
        
        # Force recovery button
        self.forceRecoveryBtn = QPushButton(self.tr("Force Service Recovery"))
        self.forceRecoveryBtn.clicked.connect(self._forceRecovery)
        actionsLayout.addWidget(self.forceRecoveryBtn)
        
        # Clear alerts button
        self.clearAlertsBtn = QPushButton(self.tr("Acknowledge All Alerts"))
        self.clearAlertsBtn.clicked.connect(self._acknowledgeAllAlerts)
        actionsLayout.addWidget(self.clearAlertsBtn)
        
        layout.addWidget(actionsGroup)
        
        # System info text
        self.systemInfoText = QTextEdit()
        self.systemInfoText.setReadOnly(True)
        layout.addWidget(self.systemInfoText)
        
        self.tabWidget.addTab(widget, self.tr("System Status"))
    
    def _createCircuitBreakersTab(self) -> None:
        """Create circuit breakers tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Circuit breakers table
        self.breakersTable = QTableWidget()
        self.breakersTable.setColumnCount(5)
        self.breakersTable.setHorizontalHeaderLabels([
            self.tr("Name"),
            self.tr("State"),
            self.tr("Total Calls"),
            self.tr("Failed Calls"),
            self.tr("Actions")
        ])
        self.breakersTable.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.breakersTable)
        
        # Configuration group
        configGroup = QGroupBox(self.tr("Circuit Breaker Configuration"))
        configLayout = QFormLayout()
        configGroup.setLayout(configLayout)
        
        # Failure threshold
        self.failureThresholdSpin = QSpinBox()
        self.failureThresholdSpin.setRange(1, 100)
        self.failureThresholdSpin.setValue(5)
        configLayout.addRow(self.tr("Failure Threshold:"), self.failureThresholdSpin)
        
        # Recovery timeout
        self.recoveryTimeoutSpin = QSpinBox()
        self.recoveryTimeoutSpin.setRange(10, 600)
        self.recoveryTimeoutSpin.setValue(60)
        self.recoveryTimeoutSpin.setSuffix(" s")
        configLayout.addRow(self.tr("Recovery Timeout:"), self.recoveryTimeoutSpin)
        
        # Success threshold
        self.successThresholdSpin = QSpinBox()
        self.successThresholdSpin.setRange(1, 20)
        self.successThresholdSpin.setValue(3)
        configLayout.addRow(self.tr("Success Threshold:"), self.successThresholdSpin)
        
        layout.addWidget(configGroup)
        
        self.tabWidget.addTab(widget, self.tr("Circuit Breakers"))
    
    def _createDegradationTab(self) -> None:
        """Create degradation service tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Current degradation state
        stateGroup = QGroupBox(self.tr("Current Degradation State"))
        stateLayout = QFormLayout()
        stateGroup.setLayout(stateLayout)
        
        # Service level
        self.serviceLevelCombo = QComboBox()
        self.serviceLevelCombo.addItems([
            self.tr("Full Service"),
            self.tr("Limited Service"),
            self.tr("Offline Mode"),
            self.tr("Emergency Mode")
        ])
        self.serviceLevelCombo.setEnabled(False)  # Read-only
        stateLayout.addRow(self.tr("Service Level:"), self.serviceLevelCombo)
        
        # Degradation reason
        self.degradationReasonLabel = QLabel(self.tr("System Normal"))
        stateLayout.addRow(self.tr("Reason:"), self.degradationReasonLabel)
        
        # Features affected
        self.featuresAffectedLabel = QLabel("0")
        stateLayout.addRow(self.tr("Features Affected:"), self.featuresAffectedLabel)
        
        layout.addWidget(stateGroup)
        
        # Available features list
        featuresGroup = QGroupBox(self.tr("Feature Availability"))
        featuresLayout = QVBoxLayout()
        featuresGroup.setLayout(featuresLayout)
        
        self.featuresList = QListWidget()
        featuresLayout.addWidget(self.featuresList)
        
        layout.addWidget(featuresGroup)
        
        # Degradation thresholds
        thresholdsGroup = QGroupBox(self.tr("Degradation Thresholds"))
        thresholdsLayout = QFormLayout()
        thresholdsGroup.setLayout(thresholdsLayout)
        
        # Error rate threshold
        self.errorRateSlider = QSlider(Qt.Orientation.Horizontal)
        self.errorRateSlider.setRange(10, 90)
        self.errorRateSlider.setValue(30)
        self.errorRateSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.errorRateSlider.setTickInterval(10)
        self.errorRateLabel = QLabel("30%")
        self.errorRateSlider.valueChanged.connect(
            lambda v: self.errorRateLabel.setText(f"{v}%")
        )
        errorRateLayout = QHBoxLayout()
        errorRateLayout.addWidget(self.errorRateSlider)
        errorRateLayout.addWidget(self.errorRateLabel)
        thresholdsLayout.addRow(self.tr("Error Rate:"), errorRateLayout)
        
        # CPU threshold
        self.cpuSlider = QSlider(Qt.Orientation.Horizontal)
        self.cpuSlider.setRange(50, 100)
        self.cpuSlider.setValue(80)
        self.cpuSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cpuSlider.setTickInterval(10)
        self.cpuLabel = QLabel("80%")
        self.cpuSlider.valueChanged.connect(
            lambda v: self.cpuLabel.setText(f"{v}%")
        )
        cpuLayout = QHBoxLayout()
        cpuLayout.addWidget(self.cpuSlider)
        cpuLayout.addWidget(self.cpuLabel)
        thresholdsLayout.addRow(self.tr("CPU Usage:"), cpuLayout)
        
        layout.addWidget(thresholdsGroup)
        
        self.tabWidget.addTab(widget, self.tr("Degradation Service"))
    
    def _createAlertsTab(self) -> None:
        """Create alerts tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Active alerts count
        self.activeAlertsLabel = QLabel(self.tr("Active Alerts: 0"))
        font = QFont()
        font.setBold(True)
        self.activeAlertsLabel.setFont(font)
        layout.addWidget(self.activeAlertsLabel)
        
        # Alerts table
        self.alertsTable = QTableWidget()
        self.alertsTable.setColumnCount(6)
        self.alertsTable.setHorizontalHeaderLabels([
            self.tr("Time"),
            self.tr("Level"),
            self.tr("Component"),
            self.tr("Title"),
            self.tr("Message"),
            self.tr("Actions")
        ])
        self.alertsTable.horizontalHeader().setStretchLastSection(False)
        self.alertsTable.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.alertsTable)
        
        # Alert actions
        actionsLayout = QHBoxLayout()
        
        self.acknowledgeBtn = QPushButton(self.tr("Acknowledge Selected"))
        self.acknowledgeBtn.clicked.connect(self._acknowledgeSelectedAlert)
        actionsLayout.addWidget(self.acknowledgeBtn)
        
        self.viewDetailsBtn = QPushButton(self.tr("View Details"))
        self.viewDetailsBtn.clicked.connect(self._viewAlertDetails)
        actionsLayout.addWidget(self.viewDetailsBtn)
        
        actionsLayout.addStretch()
        layout.addLayout(actionsLayout)
        
        self.tabWidget.addTab(widget, self.tr("Alerts"))
    
    def _createMetricsTab(self) -> None:
        """Create metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Metrics summary
        summaryGroup = QGroupBox(self.tr("Fault Handling Metrics"))
        summaryLayout = QGridLayout()
        summaryGroup.setLayout(summaryLayout)
        
        # Total errors
        self.totalErrorsLabel = QLabel("0")
        summaryLayout.addWidget(QLabel(self.tr("Total Errors:")), 0, 0)
        summaryLayout.addWidget(self.totalErrorsLabel, 0, 1)
        
        # Recovered errors
        self.recoveredErrorsLabel = QLabel("0")
        summaryLayout.addWidget(QLabel(self.tr("Recovered:")), 0, 2)
        summaryLayout.addWidget(self.recoveredErrorsLabel, 0, 3)
        
        # Recovery rate
        self.recoveryRateLabel = QLabel("0.0%")
        summaryLayout.addWidget(QLabel(self.tr("Recovery Rate:")), 1, 0)
        summaryLayout.addWidget(self.recoveryRateLabel, 1, 1)
        
        # Mean time to recovery
        self.mttrLabel = QLabel("0.0s")
        summaryLayout.addWidget(QLabel(self.tr("MTTR:")), 1, 2)
        summaryLayout.addWidget(self.mttrLabel, 1, 3)
        
        # Circuit breakers open
        self.breakersOpenLabel = QLabel("0")
        summaryLayout.addWidget(QLabel(self.tr("Breakers Open:")), 2, 0)
        summaryLayout.addWidget(self.breakersOpenLabel, 2, 1)
        
        # Active alerts
        self.activeAlertsMetricLabel = QLabel("0")
        summaryLayout.addWidget(QLabel(self.tr("Active Alerts:")), 2, 2)
        summaryLayout.addWidget(self.activeAlertsMetricLabel, 2, 3)
        
        layout.addWidget(summaryGroup)
        
        # Retry statistics
        retryGroup = QGroupBox(self.tr("Retry Statistics"))
        retryLayout = QFormLayout()
        retryGroup.setLayout(retryLayout)
        
        self.totalAttemptsLabel = QLabel("0")
        retryLayout.addRow(self.tr("Total Attempts:"), self.totalAttemptsLabel)
        
        self.successfulRetriesLabel = QLabel("0")
        retryLayout.addRow(self.tr("Successful Retries:"), self.successfulRetriesLabel)
        
        self.retrySuccessRateLabel = QLabel("0.0%")
        retryLayout.addRow(self.tr("Success Rate:"), self.retrySuccessRateLabel)
        
        layout.addWidget(retryGroup)
        
        # Error classification distribution
        classificationGroup = QGroupBox(self.tr("Error Classification"))
        classificationLayout = QVBoxLayout()
        classificationGroup.setLayout(classificationLayout)
        
        self.classificationTable = QTableWidget()
        self.classificationTable.setColumnCount(2)
        self.classificationTable.setHorizontalHeaderLabels([
            self.tr("Classification"),
            self.tr("Count")
        ])
        self.classificationTable.horizontalHeader().setStretchLastSection(True)
        classificationLayout.addWidget(self.classificationTable)
        
        layout.addWidget(classificationGroup)
        
        self.tabWidget.addTab(widget, self.tr("Metrics"))
    
    def _loadCurrentState(self) -> None:
        """Load current fault handling state."""
        try:
            # Update system status
            self._updateSystemStatus()
            
            # Update circuit breakers
            self._updateCircuitBreakers()
            
            # Update degradation state
            self._updateDegradationState()
            
            # Update alerts
            self._updateAlerts()
            
            # Update metrics
            self._updateMetrics()
            
        except Exception as e:
            logger.error(f"Failed to load fault management state: {e}")
    
    def _updateSystemStatus(self) -> None:
        """Update system status display."""
        try:
            health = self.fault_system.perform_health_check()
            
            # Update health score
            score = health.get("score", 100)
            self.healthScoreLabel.setText(str(int(score)))
            
            # Set color based on score
            if score >= 80:
                color = QColor(0, 128, 0)  # Green
            elif score >= 60:
                color = QColor(255, 165, 0)  # Orange
            else:
                color = QColor(255, 0, 0)  # Red
            
            self.healthScoreLabel.setStyleSheet(
                f"color: {color.name()};"
            )
            
            # Update status
            status = health.get("status", "unknown")
            self.healthStatusLabel.setText(status.title())
            
            # Update error rate
            metrics = health.get("metrics", {})
            error_rate = metrics.get("error_rate_per_minute", 0.0)
            self.errorRateLabel.setText(f"{error_rate:.1f}/min")
            
            # Update degradation level
            degradation = health.get("degradation", {})
            level = degradation.get("level", "full")
            self.degradationLabel.setText(self._formatDegradationLevel(level))
            
            # Update system info
            info_text = self._formatHealthInfo(health)
            self.systemInfoText.setPlainText(info_text)
            
        except Exception as e:
            logger.error(f"Failed to update system status: {e}")
    
    def _updateCircuitBreakers(self) -> None:
        """Update circuit breakers display."""
        try:
            all_metrics = self.circuit_manager.get_all_metrics()
            
            self.breakersTable.setRowCount(len(all_metrics))
            
            for row, (name, metrics) in enumerate(all_metrics.items()):
                # Get breaker
                breaker = self.circuit_manager.get(name)
                if not breaker:
                    continue
                
                # Name
                self.breakersTable.setItem(row, 0, QTableWidgetItem(name))
                
                # State
                state = breaker.get_state()
                state_item = QTableWidgetItem(state.value)
                if state == CircuitBreakerState.OPEN:
                    state_item.setForeground(QColor(255, 0, 0))
                elif state == CircuitBreakerState.HALF_OPEN:
                    state_item.setForeground(QColor(255, 165, 0))
                else:
                    state_item.setForeground(QColor(0, 128, 0))
                self.breakersTable.setItem(row, 1, state_item)
                
                # Total calls
                self.breakersTable.setItem(
                    row, 2, QTableWidgetItem(str(metrics.total_calls))
                )
                
                # Failed calls
                self.breakersTable.setItem(
                    row, 3, QTableWidgetItem(str(metrics.failed_calls))
                )
                
                # Actions button
                reset_btn = QPushButton(self.tr("Reset"))
                reset_btn.clicked.connect(lambda _, n=name: self._resetBreaker(n))
                self.breakersTable.setCellWidget(row, 4, reset_btn)
            
        except Exception as e:
            logger.error(f"Failed to update circuit breakers: {e}")
    
    def _updateDegradationState(self) -> None:
        """Update degradation state display."""
        try:
            state = self.degradation_service.get_state()
            
            # Update service level
            level_index = {
                ServiceDegradationLevel.FULL: 0,
                ServiceDegradationLevel.LIMITED: 1,
                ServiceDegradationLevel.OFFLINE: 2,
                ServiceDegradationLevel.EMERGENCY: 3
            }.get(state.level, 0)
            self.serviceLevelCombo.setCurrentIndex(level_index)
            
            # Update reason
            self.degradationReasonLabel.setText(state.reason)
            
            # Update features affected
            affected = len(state.features_disabled) + len(state.features_limited)
            self.featuresAffectedLabel.setText(str(affected))
            
            # Update features list
            self.featuresList.clear()
            available_features = self.degradation_service.get_available_features()
            
            for feature in available_features:
                item = QListWidgetItem(feature)
                item.setForeground(QColor(0, 128, 0))
                self.featuresList.addItem(item)
            
            # Add disabled features
            for feature in state.features_disabled:
                item = QListWidgetItem(f"{feature} (Disabled)")
                item.setForeground(QColor(255, 0, 0))
                self.featuresList.addItem(item)
            
            # Add limited features
            for feature in state.features_limited:
                item = QListWidgetItem(f"{feature} (Limited)")
                item.setForeground(QColor(255, 165, 0))
                self.featuresList.addItem(item)
            
        except Exception as e:
            logger.error(f"Failed to update degradation state: {e}")
    
    def _updateAlerts(self) -> None:
        """Update alerts display."""
        try:
            alerts = self.fault_system.get_active_alerts()
            
            # Update count
            self.activeAlertsLabel.setText(
                self.tr("Active Alerts: %d") % len(alerts)
            )
            
            # Update table
            self.alertsTable.setRowCount(len(alerts))
            
            for row, alert in enumerate(alerts):
                # Time
                time_str = alert.timestamp.strftime("%H:%M:%S")
                self.alertsTable.setItem(row, 0, QTableWidgetItem(time_str))
                
                # Level
                level_item = QTableWidgetItem(alert.level.value)
                if alert.level == AlertLevel.CRITICAL:
                    level_item.setForeground(QColor(255, 0, 0))
                elif alert.level == AlertLevel.ERROR:
                    level_item.setForeground(QColor(255, 100, 0))
                elif alert.level == AlertLevel.WARNING:
                    level_item.setForeground(QColor(255, 165, 0))
                self.alertsTable.setItem(row, 1, level_item)
                
                # Component
                self.alertsTable.setItem(row, 2, QTableWidgetItem(alert.component))
                
                # Title
                self.alertsTable.setItem(row, 3, QTableWidgetItem(alert.title))
                
                # Message (truncated)
                msg = alert.message[:50] + "..." if len(alert.message) > 50 else alert.message
                self.alertsTable.setItem(row, 4, QTableWidgetItem(msg))
                
                # Acknowledge button
                ack_btn = QPushButton(self.tr("Ack"))
                ack_btn.clicked.connect(lambda _, aid=alert.id: self._acknowledgeAlert(aid))
                self.alertsTable.setCellWidget(row, 5, ack_btn)
            
        except Exception as e:
            logger.error(f"Failed to update alerts: {e}")
    
    def _updateMetrics(self) -> None:
        """Update metrics display."""
        try:
            # Fault handling metrics
            metrics = self.fault_system.get_metrics()
            
            self.totalErrorsLabel.setText(str(metrics.total_errors))
            self.recoveredErrorsLabel.setText(str(metrics.recovered_errors))
            self.recoveryRateLabel.setText(f"{metrics.recovery_success_rate * 100:.1f}%")
            self.mttrLabel.setText(f"{metrics.mean_time_to_recovery:.1f}s")
            self.breakersOpenLabel.setText(str(metrics.circuit_breakers_open))
            self.activeAlertsMetricLabel.setText(str(metrics.active_alerts))
            
            # Retry statistics
            retry_stats = self.retry_manager.get_statistics()
            
            self.totalAttemptsLabel.setText(str(retry_stats.get("total_attempts", 0)))
            self.successfulRetriesLabel.setText(str(retry_stats.get("successful_retries", 0)))
            
            success_rate = retry_stats.get("success_rate", 0.0)
            self.retrySuccessRateLabel.setText(f"{success_rate * 100:.1f}%")
            
            # Error classification
            error_metrics = self.fault_system.error_classifier.get_metrics()
            classifications = error_metrics.errors_by_classification
            
            self.classificationTable.setRowCount(len(classifications))
            
            for row, (classification, count) in enumerate(classifications.items()):
                self.classificationTable.setItem(
                    row, 0, QTableWidgetItem(classification)
                )
                self.classificationTable.setItem(
                    row, 1, QTableWidgetItem(str(count))
                )
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _setupAutoRefresh(self) -> None:
        """Setup auto-refresh timer."""
        self.refreshTimer = QTimer(self)
        self.refreshTimer.timeout.connect(self._loadCurrentState)
        self.refreshTimer.start(5000)  # Refresh every 5 seconds
    
    def _resetAllBreakers(self) -> None:
        """Reset all circuit breakers."""
        try:
            self.circuit_manager.reset_all()
            self._updateCircuitBreakers()
            logger.info("All circuit breakers reset")
        except Exception as e:
            logger.error(f"Failed to reset circuit breakers: {e}")
    
    def _resetBreaker(self, name: str) -> None:
        """Reset specific circuit breaker."""
        try:
            breaker = self.circuit_manager.get(name)
            if breaker:
                breaker.reset()
                self._updateCircuitBreakers()
                logger.info(f"Circuit breaker '{name}' reset")
        except Exception as e:
            logger.error(f"Failed to reset circuit breaker: {e}")
    
    def _forceRecovery(self) -> None:
        """Force service recovery."""
        try:
            self.degradation_service.attempt_recovery()
            self._updateDegradationState()
            logger.info("Service recovery attempted")
        except Exception as e:
            logger.error(f"Failed to force recovery: {e}")
    
    def _acknowledgeAllAlerts(self) -> None:
        """Acknowledge all alerts."""
        try:
            alerts = self.fault_system.get_active_alerts()
            for alert in alerts:
                self.fault_system.acknowledge_alert(alert.id, "user")
            self._updateAlerts()
            logger.info(f"Acknowledged {len(alerts)} alerts")
        except Exception as e:
            logger.error(f"Failed to acknowledge alerts: {e}")
    
    def _acknowledgeAlert(self, alert_id: str) -> None:
        """Acknowledge specific alert."""
        try:
            self.fault_system.acknowledge_alert(alert_id, "user")
            self._updateAlerts()
            logger.info(f"Alert '{alert_id}' acknowledged")
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
    
    def _acknowledgeSelectedAlert(self) -> None:
        """Acknowledge selected alert."""
        row = self.alertsTable.currentRow()
        if row >= 0:
            alerts = self.fault_system.get_active_alerts()
            if row < len(alerts):
                self._acknowledgeAlert(alerts[row].id)
    
    def _viewAlertDetails(self) -> None:
        """View alert details."""
        row = self.alertsTable.currentRow()
        if row >= 0:
            alerts = self.fault_system.get_active_alerts()
            if row < len(alerts):
                alert = alerts[row]
                # Show details in system info text
                details = f"Alert Details:\n"
                details += f"ID: {alert.id}\n"
                details += f"Time: {alert.timestamp}\n"
                details += f"Level: {alert.level.value}\n"
                details += f"Component: {alert.component}\n"
                details += f"Title: {alert.title}\n"
                details += f"Message: {alert.message}\n"
                if alert.metadata:
                    details += f"Metadata: {alert.metadata}\n"
                self.systemInfoText.setPlainText(details)
                self.tabWidget.setCurrentIndex(0)  # Switch to status tab
    
    def _formatDegradationLevel(self, level: str) -> str:
        """Format degradation level for display."""
        levels = {
            "full": self.tr("Full Service"),
            "limited": self.tr("Limited Service"),
            "offline": self.tr("Offline Mode"),
            "emergency": self.tr("Emergency Mode")
        }
        return levels.get(level, level.title())
    
    def _formatHealthInfo(self, health: dict) -> str:
        """Format health information for display."""
        info = []
        info.append(f"System Health Report")
        info.append("=" * 40)
        info.append(f"Status: {health.get('status', 'unknown').title()}")
        info.append(f"Score: {health.get('score', 0):.1f}/100")
        info.append("")
        
        # Error analysis
        error_analysis = health.get("error_analysis", {})
        if error_analysis:
            info.append("Error Analysis:")
            info.append(f"  Error Count: {error_analysis.get('error_count', 0)}")
            info.append(f"  Error Rate: {error_analysis.get('error_rate', 0):.2f}/min")
            info.append(f"  Failure Prediction: {error_analysis.get('failure_prediction', 'none')}")
            info.append("")
        
        # Cascade risk
        cascade_risk = health.get("cascade_risk", {})
        if cascade_risk:
            info.append("Cascade Risk:")
            info.append(f"  Risk Level: {cascade_risk.get('risk_level', 'unknown')}")
            info.append(f"  Open Breakers: {cascade_risk.get('open_breakers', 0)}")
            info.append(f"  Total Breakers: {cascade_risk.get('total_breakers', 0)}")
            info.append("")
        
        # Degradation
        degradation = health.get("degradation", {})
        if degradation:
            info.append("Service Degradation:")
            info.append(f"  Level: {self._formatDegradationLevel(degradation.get('level', 'full'))}")
            info.append(f"  Reason: {degradation.get('reason', 'N/A')}")
            info.append(f"  Features Affected: {degradation.get('features_affected', 0)}")
        
        return "\n".join(info)
