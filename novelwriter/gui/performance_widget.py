"""
novelWriter – Performance Monitoring Widget
============================================

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
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from novelwriter import CONFIG
from novelwriter.api.base.performance import (
    MetricType,
    PerformanceAlert,
    PerformanceMonitor,
    PerformanceStatistics,
    get_performance_monitor,
)
from novelwriter.api.base.performance_stats import (
    PerformanceHotspotDetector,
    TrendAnalyzer,
)
from novelwriter.common import formatTimeStamp

if TYPE_CHECKING:
    from novelwriter.guimain import GuiMain

logger = logging.getLogger(__name__)


class PerformanceStatusWidget(QWidget):
    """Status bar widget for performance monitoring."""
    
    clicked = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # Setup UI
        self.setMaximumHeight(20)
        self.setMinimumWidth(200)
        
        # Create layout
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)
        self.setLayout(layout)
        
        # Status indicator
        self.statusIndicator = QLabel("●")
        self.statusIndicator.setFixedWidth(12)
        self.setStatus("normal")
        layout.addWidget(self.statusIndicator)
        
        # Latency display
        self.latencyLabel = QLabel("Latency:")
        layout.addWidget(self.latencyLabel)
        
        self.latencyValue = QLabel("0ms")
        self.latencyValue.setMinimumWidth(50)
        layout.addWidget(self.latencyValue)
        
        # Success rate display
        self.successLabel = QLabel("Success:")
        layout.addWidget(self.successLabel)
        
        self.successValue = QLabel("100%")
        self.successValue.setMinimumWidth(40)
        layout.addWidget(self.successValue)
        
        # Active operations
        self.activeLabel = QLabel("Active:")
        layout.addWidget(self.activeLabel)
        
        self.activeValue = QLabel("0")
        self.activeValue.setMinimumWidth(20)
        layout.addWidget(self.activeValue)
        
        # Update timer
        self.updateTimer = QTimer(self)
        self.updateTimer.timeout.connect(self.updateMetrics)
        self.updateTimer.start(1000)  # Update every second
        
        # Click to open details
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def setStatus(self, status: str) -> None:
        """Set the status indicator color.
        
        Args:
            status: Status level (normal, warning, critical)
        """
        if status == "normal":
            color = QColor(0, 200, 0)  # Green
        elif status == "warning":
            color = QColor(255, 200, 0)  # Yellow
        elif status == "critical":
            color = QColor(255, 0, 0)  # Red
        else:
            color = QColor(128, 128, 128)  # Gray
        
        self.statusIndicator.setStyleSheet(f"color: {color.name()};")
    
    def updateMetrics(self) -> None:
        """Update displayed metrics."""
        monitor = get_performance_monitor()
        if not monitor:
            return
        
        # Get overall statistics
        all_stats = monitor.get_all_statistics(window_minutes=1)
        
        if all_stats:
            # Calculate aggregated metrics
            total_latency = 0.0
            total_count = 0
            total_success = 0
            total_errors = 0
            
            for stats in all_stats:
                if stats.count > 0:
                    total_latency += stats.mean * stats.count
                    total_count += stats.count
                    total_success += int(stats.count * stats.success_rate)
                    total_errors += stats.error_count
            
            # Update latency
            if total_count > 0:
                avg_latency = total_latency / total_count
                self.latencyValue.setText(f"{avg_latency:.0f}ms")
                
                # Set status based on latency
                if avg_latency > 5000:
                    self.setStatus("critical")
                elif avg_latency > 1000:
                    self.setStatus("warning")
                else:
                    self.setStatus("normal")
            else:
                self.latencyValue.setText("0ms")
                self.setStatus("normal")
            
            # Update success rate
            if total_count > 0:
                success_rate = (total_success / total_count) * 100
                self.successValue.setText(f"{success_rate:.1f}%")
            else:
                self.successValue.setText("100%")
        else:
            self.latencyValue.setText("0ms")
            self.successValue.setText("100%")
            self.setStatus("normal")
        
        # Update active operations count
        active_count = sum(monitor.active_operations.values())
        self.activeValue.setText(str(active_count))
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PerformanceDetailsDialog(QDialog):
    """Detailed performance monitoring dialog."""
    
    def __init__(self, parent: GuiMain) -> None:
        super().__init__(parent)
        
        self.mainGui = parent
        self.monitor = get_performance_monitor()
        self.hotspot_detector = PerformanceHotspotDetector()
        self.trend_analyzers = {}
        
        # Dialog settings
        self.setWindowTitle(self.tr("Performance Monitor"))
        self.setMinimumSize(CONFIG.pxInt(900), CONFIG.pxInt(600))
        
        # Build UI
        self._buildUI()
        
        # Load initial data
        self._loadData()
        
        # Setup auto-refresh
        self.refreshTimer = QTimer(self)
        self.refreshTimer.timeout.connect(self._loadData)
        self.refreshTimer.start(2000)  # Refresh every 2 seconds
    
    def _buildUI(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tab widget
        self.tabWidget = QTabWidget()
        layout.addWidget(self.tabWidget)
        
        # Create tabs
        self._createOverviewTab()
        self._createStatisticsTab()
        self._createHotspotsTab()
        self._createAlertsTab()
        self._createTrendsTab()
        
        # Button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttonBox.rejected.connect(self.close)
        layout.addWidget(buttonBox)
    
    def _createOverviewTab(self) -> None:
        """Create overview tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Summary group
        summaryGroup = QGroupBox(self.tr("Performance Summary"))
        summaryLayout = QGridLayout()
        summaryGroup.setLayout(summaryLayout)
        
        # Overall metrics
        self.overallLatencyLabel = QLabel("0ms")
        summaryLayout.addWidget(QLabel(self.tr("Average Latency:")), 0, 0)
        summaryLayout.addWidget(self.overallLatencyLabel, 0, 1)
        
        self.overallP95Label = QLabel("0ms")
        summaryLayout.addWidget(QLabel(self.tr("P95 Latency:")), 0, 2)
        summaryLayout.addWidget(self.overallP95Label, 0, 3)
        
        self.overallP99Label = QLabel("0ms")
        summaryLayout.addWidget(QLabel(self.tr("P99 Latency:")), 1, 0)
        summaryLayout.addWidget(self.overallP99Label, 1, 1)
        
        self.overallSuccessLabel = QLabel("100%")
        summaryLayout.addWidget(QLabel(self.tr("Success Rate:")), 1, 2)
        summaryLayout.addWidget(self.overallSuccessLabel, 1, 3)
        
        self.totalCallsLabel = QLabel("0")
        summaryLayout.addWidget(QLabel(self.tr("Total Calls:")), 2, 0)
        summaryLayout.addWidget(self.totalCallsLabel, 2, 1)
        
        self.errorCountLabel = QLabel("0")
        summaryLayout.addWidget(QLabel(self.tr("Error Count:")), 2, 2)
        summaryLayout.addWidget(self.errorCountLabel, 2, 3)
        
        layout.addWidget(summaryGroup)
        
        # Component breakdown
        breakdownGroup = QGroupBox(self.tr("Component Breakdown"))
        breakdownLayout = QVBoxLayout()
        breakdownGroup.setLayout(breakdownLayout)
        
        self.componentTable = QTableWidget()
        self.componentTable.setColumnCount(6)
        self.componentTable.setHorizontalHeaderLabels([
            self.tr("Component"),
            self.tr("Calls"),
            self.tr("Mean (ms)"),
            self.tr("P95 (ms)"),
            self.tr("P99 (ms)"),
            self.tr("Success Rate")
        ])
        self.componentTable.horizontalHeader().setStretchLastSection(True)
        breakdownLayout.addWidget(self.componentTable)
        
        layout.addWidget(breakdownGroup)
        
        self.tabWidget.addTab(widget, self.tr("Overview"))
    
    def _createStatisticsTab(self) -> None:
        """Create statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Time window selection
        windowLayout = QHBoxLayout()
        windowLayout.addWidget(QLabel(self.tr("Time Window:")))
        
        self.window1MinBtn = QPushButton(self.tr("1 Minute"))
        self.window1MinBtn.clicked.connect(lambda: self._updateStatistics(1))
        windowLayout.addWidget(self.window1MinBtn)
        
        self.window5MinBtn = QPushButton(self.tr("5 Minutes"))
        self.window5MinBtn.clicked.connect(lambda: self._updateStatistics(5))
        windowLayout.addWidget(self.window5MinBtn)
        
        self.window15MinBtn = QPushButton(self.tr("15 Minutes"))
        self.window15MinBtn.clicked.connect(lambda: self._updateStatistics(15))
        windowLayout.addWidget(self.window15MinBtn)
        
        windowLayout.addStretch()
        layout.addLayout(windowLayout)
        
        # Statistics table
        self.statsTable = QTableWidget()
        self.statsTable.setColumnCount(9)
        self.statsTable.setHorizontalHeaderLabels([
            self.tr("Component"),
            self.tr("Operation"),
            self.tr("Count"),
            self.tr("Mean"),
            self.tr("Min"),
            self.tr("Max"),
            self.tr("P95"),
            self.tr("P99"),
            self.tr("Std Dev")
        ])
        layout.addWidget(self.statsTable)
        
        self.tabWidget.addTab(widget, self.tr("Statistics"))
    
    def _createHotspotsTab(self) -> None:
        """Create hotspots tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Hotspots info
        infoLabel = QLabel(self.tr(
            "Performance hotspots show which operations consume the most time."
        ))
        layout.addWidget(infoLabel)
        
        # Hotspots table
        self.hotspotsTable = QTableWidget()
        self.hotspotsTable.setColumnCount(5)
        self.hotspotsTable.setHorizontalHeaderLabels([
            self.tr("Component"),
            self.tr("Operation"),
            self.tr("Total Time (ms)"),
            self.tr("Call Count"),
            self.tr("Avg Time (ms)")
        ])
        layout.addWidget(self.hotspotsTable)
        
        # Reset button
        resetBtn = QPushButton(self.tr("Reset Hotspot Data"))
        resetBtn.clicked.connect(self._resetHotspots)
        layout.addWidget(resetBtn)
        
        self.tabWidget.addTab(widget, self.tr("Hotspots"))
    
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
            self.tr("Metric"),
            self.tr("Value"),
            self.tr("Actions")
        ])
        layout.addWidget(self.alertsTable)
        
        # Clear all button
        clearBtn = QPushButton(self.tr("Acknowledge All Alerts"))
        clearBtn.clicked.connect(self._acknowledgeAllAlerts)
        layout.addWidget(clearBtn)
        
        self.tabWidget.addTab(widget, self.tr("Alerts"))
    
    def _createTrendsTab(self) -> None:
        """Create trends tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Trend analysis info
        infoLabel = QLabel(self.tr(
            "Trend analysis shows performance changes over time."
        ))
        layout.addWidget(infoLabel)
        
        # Trends table
        self.trendsTable = QTableWidget()
        self.trendsTable.setColumnCount(5)
        self.trendsTable.setHorizontalHeaderLabels([
            self.tr("Component"),
            self.tr("Operation"),
            self.tr("Trend"),
            self.tr("Confidence"),
            self.tr("Prediction")
        ])
        layout.addWidget(self.trendsTable)
        
        # Baseline actions
        baselineLayout = QHBoxLayout()
        
        self.saveBaselineBtn = QPushButton(self.tr("Save as Baseline"))
        self.saveBaselineBtn.clicked.connect(self._saveBaseline)
        baselineLayout.addWidget(self.saveBaselineBtn)
        
        self.compareBaselineBtn = QPushButton(self.tr("Compare to Baseline"))
        self.compareBaselineBtn.clicked.connect(self._compareToBaseline)
        baselineLayout.addWidget(self.compareBaselineBtn)
        
        baselineLayout.addStretch()
        layout.addLayout(baselineLayout)
        
        # Comparison results
        self.comparisonText = QTextEdit()
        self.comparisonText.setReadOnly(True)
        self.comparisonText.setMaximumHeight(150)
        layout.addWidget(self.comparisonText)
        
        self.tabWidget.addTab(widget, self.tr("Trends"))
    
    def _loadData(self) -> None:
        """Load and display performance data."""
        if not self.monitor:
            return
        
        # Update overview
        self._updateOverview()
        
        # Update statistics (default 5 minutes)
        self._updateStatistics(5)
        
        # Update hotspots
        self._updateHotspots()
        
        # Update alerts
        self._updateAlerts()
        
        # Update trends
        self._updateTrends()
    
    def _updateOverview(self) -> None:
        """Update overview tab."""
        all_stats = self.monitor.get_all_statistics(window_minutes=5)
        
        if not all_stats:
            return
        
        # Calculate overall metrics
        total_latency = 0.0
        total_p95 = 0.0
        total_p99 = 0.0
        total_count = 0
        total_success = 0
        total_errors = 0
        
        # Group by component
        component_stats = {}
        
        for stats in all_stats:
            if stats.count > 0:
                # Overall metrics
                total_latency += stats.mean * stats.count
                total_p95 = max(total_p95, stats.p95)
                total_p99 = max(total_p99, stats.p99)
                total_count += stats.count
                total_success += int(stats.count * stats.success_rate)
                total_errors += stats.error_count
                
                # Component grouping
                if stats.component not in component_stats:
                    component_stats[stats.component] = {
                        "count": 0,
                        "total_latency": 0.0,
                        "max_p95": 0.0,
                        "max_p99": 0.0,
                        "total_success": 0,
                        "total_errors": 0
                    }
                
                comp = component_stats[stats.component]
                comp["count"] += stats.count
                comp["total_latency"] += stats.mean * stats.count
                comp["max_p95"] = max(comp["max_p95"], stats.p95)
                comp["max_p99"] = max(comp["max_p99"], stats.p99)
                comp["total_success"] += int(stats.count * stats.success_rate)
                comp["total_errors"] += stats.error_count
        
        # Update overall labels
        if total_count > 0:
            avg_latency = total_latency / total_count
            success_rate = (total_success / total_count) * 100
            
            self.overallLatencyLabel.setText(f"{avg_latency:.1f}ms")
            self.overallP95Label.setText(f"{total_p95:.1f}ms")
            self.overallP99Label.setText(f"{total_p99:.1f}ms")
            self.overallSuccessLabel.setText(f"{success_rate:.1f}%")
        
        self.totalCallsLabel.setText(str(total_count))
        self.errorCountLabel.setText(str(total_errors))
        
        # Update component table
        self.componentTable.setRowCount(len(component_stats))
        
        for row, (component, stats) in enumerate(component_stats.items()):
            # Component name
            self.componentTable.setItem(row, 0, QTableWidgetItem(component))
            
            # Call count
            self.componentTable.setItem(row, 1, QTableWidgetItem(str(stats["count"])))
            
            # Mean latency
            mean = stats["total_latency"] / stats["count"] if stats["count"] > 0 else 0
            self.componentTable.setItem(row, 2, QTableWidgetItem(f"{mean:.1f}"))
            
            # P95
            self.componentTable.setItem(row, 3, QTableWidgetItem(f"{stats['max_p95']:.1f}"))
            
            # P99
            self.componentTable.setItem(row, 4, QTableWidgetItem(f"{stats['max_p99']:.1f}"))
            
            # Success rate
            success_rate = (stats["total_success"] / stats["count"] * 100) if stats["count"] > 0 else 100
            self.componentTable.setItem(row, 5, QTableWidgetItem(f"{success_rate:.1f}%"))
    
    def _updateStatistics(self, window_minutes: int) -> None:
        """Update statistics tab.
        
        Args:
            window_minutes: Time window in minutes
        """
        all_stats = self.monitor.get_all_statistics(window_minutes=window_minutes)
        
        self.statsTable.setRowCount(len(all_stats))
        
        for row, stats in enumerate(all_stats):
            self.statsTable.setItem(row, 0, QTableWidgetItem(stats.component))
            self.statsTable.setItem(row, 1, QTableWidgetItem(stats.operation))
            self.statsTable.setItem(row, 2, QTableWidgetItem(str(stats.count)))
            self.statsTable.setItem(row, 3, QTableWidgetItem(f"{stats.mean:.1f}"))
            self.statsTable.setItem(row, 4, QTableWidgetItem(f"{stats.min:.1f}"))
            self.statsTable.setItem(row, 5, QTableWidgetItem(f"{stats.max:.1f}"))
            self.statsTable.setItem(row, 6, QTableWidgetItem(f"{stats.p95:.1f}"))
            self.statsTable.setItem(row, 7, QTableWidgetItem(f"{stats.p99:.1f}"))
            self.statsTable.setItem(row, 8, QTableWidgetItem(f"{stats.std_dev:.1f}"))
    
    def _updateHotspots(self) -> None:
        """Update hotspots tab."""
        # Collect hotspot data from recent metrics
        for metric in self.monitor.metrics:
            if metric.metric_type == MetricType.LATENCY:
                self.hotspot_detector.record_call(
                    metric.component,
                    metric.operation,
                    metric.value
                )
        
        # Get top hotspots
        hotspots = self.hotspot_detector.get_hotspots(top_n=20)
        
        self.hotspotsTable.setRowCount(len(hotspots))
        
        for row, hotspot in enumerate(hotspots):
            self.hotspotsTable.setItem(row, 0, QTableWidgetItem(hotspot["component"]))
            self.hotspotsTable.setItem(row, 1, QTableWidgetItem(hotspot["operation"]))
            self.hotspotsTable.setItem(row, 2, QTableWidgetItem(f"{hotspot['total_time_ms']:.1f}"))
            self.hotspotsTable.setItem(row, 3, QTableWidgetItem(str(hotspot["call_count"])))
            self.hotspotsTable.setItem(row, 4, QTableWidgetItem(f"{hotspot['avg_time_ms']:.1f}"))
    
    def _updateAlerts(self) -> None:
        """Update alerts tab."""
        alerts = self.monitor.get_active_alerts()
        
        self.activeAlertsLabel.setText(self.tr("Active Alerts: %d") % len(alerts))
        self.alertsTable.setRowCount(len(alerts))
        
        for row, alert in enumerate(alerts):
            # Time
            time_str = alert.timestamp.strftime("%H:%M:%S")
            self.alertsTable.setItem(row, 0, QTableWidgetItem(time_str))
            
            # Level
            level_item = QTableWidgetItem(alert.level)
            if alert.level == "CRITICAL":
                level_item.setForeground(QColor(255, 0, 0))
            elif alert.level == "WARNING":
                level_item.setForeground(QColor(255, 165, 0))
            self.alertsTable.setItem(row, 1, level_item)
            
            # Component
            self.alertsTable.setItem(row, 2, QTableWidgetItem(alert.component))
            
            # Metric
            self.alertsTable.setItem(row, 3, QTableWidgetItem(alert.metric_type.value))
            
            # Value
            self.alertsTable.setItem(row, 4, QTableWidgetItem(f"{alert.current_value:.1f}"))
            
            # Acknowledge button
            ackBtn = QPushButton(self.tr("Ack"))
            ackBtn.clicked.connect(lambda _, aid=alert.id: self._acknowledgeAlert(aid))
            self.alertsTable.setCellWidget(row, 5, ackBtn)
    
    def _updateTrends(self) -> None:
        """Update trends tab."""
        # Analyze trends for each component:operation
        all_stats = self.monitor.get_all_statistics(window_minutes=15)
        
        trends_data = []
        
        for stats in all_stats:
            key = f"{stats.component}:{stats.operation}"
            
            # Create or get trend analyzer
            if key not in self.trend_analyzers:
                self.trend_analyzers[key] = TrendAnalyzer()
            
            analyzer = self.trend_analyzers[key]
            
            # Add recent data points
            analyzer.add_point(stats.mean)
            
            # Analyze trend
            trend = analyzer.analyze_trend()
            
            trends_data.append({
                "component": stats.component,
                "operation": stats.operation,
                "trend": trend.trend,
                "confidence": trend.confidence,
                "prediction": trend.prediction_next
            })
        
        # Update table
        self.trendsTable.setRowCount(len(trends_data))
        
        for row, data in enumerate(trends_data):
            self.trendsTable.setItem(row, 0, QTableWidgetItem(data["component"]))
            self.trendsTable.setItem(row, 1, QTableWidgetItem(data["operation"]))
            
            # Trend with color
            trend_item = QTableWidgetItem(data["trend"])
            if data["trend"] == "improving":
                trend_item.setForeground(QColor(0, 200, 0))
            elif data["trend"] == "degrading":
                trend_item.setForeground(QColor(255, 0, 0))
            self.trendsTable.setItem(row, 2, trend_item)
            
            # Confidence
            self.trendsTable.setItem(row, 3, QTableWidgetItem(f"{data['confidence']:.1%}"))
            
            # Prediction
            self.trendsTable.setItem(row, 4, QTableWidgetItem(f"{data['prediction']:.1f}ms"))
    
    def _resetHotspots(self) -> None:
        """Reset hotspot data."""
        self.hotspot_detector.reset()
        self._updateHotspots()
    
    def _acknowledgeAlert(self, alert_id: str) -> None:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
        """
        self.monitor.acknowledge_alert(alert_id)
        self._updateAlerts()
    
    def _acknowledgeAllAlerts(self) -> None:
        """Acknowledge all alerts."""
        for alert in self.monitor.get_active_alerts():
            self.monitor.acknowledge_alert(alert.id)
        self._updateAlerts()
    
    def _saveBaseline(self) -> None:
        """Save current performance as baseline."""
        all_stats = self.monitor.get_all_statistics(window_minutes=5)
        
        for stats in all_stats:
            self.monitor.save_baseline(stats.component, stats.operation)
        
        self.comparisonText.setPlainText(
            self.tr("Baseline saved for %d operations") % len(all_stats)
        )
    
    def _compareToBaseline(self) -> None:
        """Compare current performance to baseline."""
        all_stats = self.monitor.get_all_statistics(window_minutes=5)
        
        comparison_text = []
        degraded_count = 0
        
        for stats in all_stats:
            comparison = self.monitor.compare_to_baseline(stats.component, stats.operation)
            
            if comparison:
                comparison_text.append(
                    f"{stats.component}:{stats.operation}:\n"
                    f"  Latency change: {comparison['latency_change']:.1f}%\n"
                    f"  P95 change: {comparison['p95_change']:.1f}%\n"
                    f"  Success rate change: {comparison['success_rate_change']:.1%}\n"
                )
                
                # Check for degradation
                if (comparison["latency_change"] > 20 or
                    comparison["p95_change"] > 30 or
                    comparison["success_rate_change"] < -0.05):
                    degraded_count += 1
        
        if comparison_text:
            summary = self.tr("Performance Comparison:\n")
            if degraded_count > 0:
                summary += self.tr("⚠️ %d operations have degraded performance\n\n") % degraded_count
            
            self.comparisonText.setPlainText(summary + "\n".join(comparison_text))
        else:
            self.comparisonText.setPlainText(self.tr("No baseline data available for comparison"))
