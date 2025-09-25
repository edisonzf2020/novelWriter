"""
novelWriter – MCP Hybrid Architecture Configuration
====================================================

File History:
Created: 2025-01-01 [x.x.x] MCPHybridConfig

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

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class MCPHybridConfig(QObject):
    """MCP Hybrid Architecture Configuration Manager."""

    # Signals for configuration changes
    configChanged = pyqtSignal(str, object)  # key, value
    configReset = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        
        # Initialize default configuration
        self._config: Dict[str, Any] = {
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
                "connections": [],
                "timeout": 200,  # ms
                "retryAttempts": 3,
                "enableAutoDiscovery": False,
            },
            "performance": {
                "monitoringEnabled": True,
                "alertThresholds": {
                    "apiLatency": 5,    # ms
                    "toolLatency": 10,  # ms
                    "errorRate": 0.05,  # 5%
                },
                "collectMetrics": True,
                "metricsRetention": 3600,  # seconds
            },
            "security": {
                "requireAuth": False,
                "allowedOrigins": ["localhost"],
                "maxRequestSize": 10485760,  # 10MB
                "rateLimiting": {
                    "enabled": True,
                    "requestsPerMinute": 100,
                },
            },
            "failureRecovery": {
                "enableCircuitBreaker": True,
                "circuitBreakerThreshold": 5,
                "circuitBreakerTimeout": 60,  # seconds
                "enableFallback": True,
            },
        }
        
        # Track configuration validity
        self._valid = True
        self._validationErrors: List[str] = []
        
        # Configuration history for rollback
        self._history: List[Dict[str, Any]] = []
        self._maxHistory = 10
        
        logger.debug("MCPHybridConfig initialized")
    
    ##
    # Properties
    ##
    
    @property
    def enabled(self) -> bool:
        """Check if MCP hybrid architecture is enabled."""
        return self._config.get("enabled", False)
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable MCP hybrid architecture."""
        self.setValue("enabled", value)
    
    @property
    def localToolsEnabled(self) -> Dict[str, bool]:
        """Get local tools enable status."""
        return self._config.get("localTools", {})
    
    @property
    def externalMCPConfig(self) -> Dict[str, Any]:
        """Get external MCP configuration."""
        return self._config.get("externalMCP", {})
    
    @property
    def performanceConfig(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self._config.get("performance", {})
    
    @property
    def securityConfig(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self._config.get("security", {})
    
    @property
    def failureRecoveryConfig(self) -> Dict[str, Any]:
        """Get failure recovery configuration."""
        return self._config.get("failureRecovery", {})
    
    @property
    def isValid(self) -> bool:
        """Check if current configuration is valid."""
        return self._valid
    
    @property
    def validationErrors(self) -> List[str]:
        """Get validation errors if configuration is invalid."""
        return self._validationErrors
    
    ##
    # Methods
    ##
    
    def getValue(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key path.
        
        Args:
            key: Dot-separated key path (e.g., "performance.monitoringEnabled")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
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
    
    def setValue(self, key: str, value: Any) -> bool:
        """Set a configuration value by key path.
        
        Args:
            key: Dot-separated key path (e.g., "performance.monitoringEnabled")
            value: Value to set
            
        Returns:
            True if value was set successfully
        """
        # Save current state for potential rollback
        self._saveHistory()
        
        keys = key.split(".")
        target = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the value
        old_value = target.get(keys[-1])
        target[keys[-1]] = value
        
        # Validate the new configuration
        if not self._validateConfig():
            # Rollback on validation failure
            self._rollback()
            return False
        
        # Emit change signal
        if old_value != value:
            self.configChanged.emit(key, value)
        
        return True
    
    def setValues(self, updates: Dict[str, Any]) -> bool:
        """Set multiple configuration values atomically.
        
        Args:
            updates: Dictionary of key paths and values
            
        Returns:
            True if all values were set successfully
        """
        # Save current state for potential rollback
        self._saveHistory()
        
        # Apply all updates
        for key, value in updates.items():
            keys = key.split(".")
            target = self._config
            
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            target[keys[-1]] = value
        
        # Validate the new configuration
        if not self._validateConfig():
            # Rollback on validation failure
            self._rollback()
            return False
        
        # Emit change signals
        for key, value in updates.items():
            self.configChanged.emit(key, value)
        
        return True
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._saveHistory()
        
        # Reset to default values without calling __init__
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
                "connections": [],
                "timeout": 200,
                "retryAttempts": 3,
                "enableAutoDiscovery": False,
            },
            "performance": {
                "monitoringEnabled": True,
                "alertThresholds": {
                    "apiLatency": 5,
                    "toolLatency": 10,
                    "errorRate": 0.05,
                },
                "collectMetrics": True,
                "metricsRetention": 3600,
            },
            "security": {
                "requireAuth": False,
                "allowedOrigins": ["localhost"],
                "maxRequestSize": 10485760,
                "rateLimiting": {
                    "enabled": True,
                    "requestsPerMinute": 100,
                },
            },
            "failureRecovery": {
                "enableCircuitBreaker": True,
                "circuitBreakerThreshold": 5,
                "circuitBreakerTimeout": 60,
                "enableFallback": True,
            },
        }
        
        self._valid = True
        self._validationErrors = []
        
        self.configReset.emit()
    
    def toDict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        import copy
        return copy.deepcopy(self._config)
    
    def fromDict(self, config: Dict[str, Any]) -> bool:
        """Import configuration from dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if import was successful
        """
        self._saveHistory()
        
        import copy
        self._config = copy.deepcopy(config)
        
        if not self._validateConfig():
            self._rollback()
            return False
        
        self.configReset.emit()
        return True
    
    ##
    # Private Methods
    ##
    
    def _validateConfig(self) -> bool:
        """Validate current configuration.
        
        Returns:
            True if configuration is valid
        """
        self._validationErrors = []
        
        # Check required fields
        if not isinstance(self._config.get("enabled"), bool):
            self._validationErrors.append("'enabled' must be a boolean")
        
        # Validate performance thresholds
        perf = self._config.get("performance", {})
        thresholds = perf.get("alertThresholds", {})
        
        if thresholds.get("apiLatency", 0) < 0:
            self._validationErrors.append("API latency threshold must be >= 0")
        
        if thresholds.get("toolLatency", 0) < 0:
            self._validationErrors.append("Tool latency threshold must be >= 0")
        
        error_rate = thresholds.get("errorRate", 0)
        if not (0 <= error_rate <= 1):
            self._validationErrors.append("Error rate threshold must be between 0 and 1")
        
        # Validate external MCP config
        ext = self._config.get("externalMCP", {})
        
        if ext.get("timeout", 0) < 0:
            self._validationErrors.append("External MCP timeout must be >= 0")
        
        if ext.get("retryAttempts", 0) < 0:
            self._validationErrors.append("Retry attempts must be >= 0")
        
        # Validate security config
        sec = self._config.get("security", {})
        
        if sec.get("maxRequestSize", 0) < 0:
            self._validationErrors.append("Max request size must be >= 0")
        
        rate_limit = sec.get("rateLimiting", {})
        if rate_limit.get("requestsPerMinute", 0) < 0:
            self._validationErrors.append("Rate limit must be >= 0")
        
        # Validate failure recovery config
        recovery = self._config.get("failureRecovery", {})
        
        if recovery.get("circuitBreakerThreshold", 0) < 0:
            self._validationErrors.append("Circuit breaker threshold must be >= 0")
        
        if recovery.get("circuitBreakerTimeout", 0) < 0:
            self._validationErrors.append("Circuit breaker timeout must be >= 0")
        
        self._valid = len(self._validationErrors) == 0
        
        if not self._valid:
            logger.warning(f"Configuration validation failed: {self._validationErrors}")
        
        return self._valid
    
    def _saveHistory(self) -> None:
        """Save current configuration to history."""
        import copy
        self._history.append(copy.deepcopy(self._config))
        
        # Limit history size
        if len(self._history) > self._maxHistory:
            self._history.pop(0)
    
    def _rollback(self) -> None:
        """Rollback to previous configuration."""
        if self._history:
            import copy
            self._config = copy.deepcopy(self._history.pop())
            self._validationErrors = []  # Clear errors before revalidation
            self._validateConfig()
            logger.info("Configuration rolled back to previous state")


# Global instance
MCP_CONFIG = MCPHybridConfig()
