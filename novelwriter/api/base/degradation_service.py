"""
novelWriter – Service Degradation Management
=============================================

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
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ServiceDegradationLevel(Enum):
    """Service degradation levels."""
    FULL = "full"                # Full functionality
    LIMITED = "limited"          # Limited functionality
    OFFLINE = "offline"          # Offline mode
    EMERGENCY = "emergency"      # Emergency mode


class DegradableFeature(BaseModel):
    """Degradable feature configuration."""
    name: str
    component: str
    priority: int = 50  # 0-100, higher is more important
    dependencies: List[str] = Field(default_factory=list)
    fallback_handler: Optional[str] = None
    offline_capable: bool = False
    degradation_levels: List[ServiceDegradationLevel] = Field(
        default_factory=lambda: [
            ServiceDegradationLevel.FULL,
            ServiceDegradationLevel.LIMITED,
            ServiceDegradationLevel.OFFLINE
        ]
    )


class DegradationPolicy(BaseModel):
    """Degradation policy configuration."""
    error_rate_threshold: float = 0.3      # Error rate to trigger degradation
    latency_threshold: float = 5.0         # Seconds
    cpu_threshold: float = 80.0            # Percentage
    memory_threshold: float = 80.0         # Percentage
    auto_recovery_enabled: bool = True
    recovery_check_interval: int = 60      # Seconds
    min_degradation_duration: int = 300    # Minimum seconds in degraded state


class DegradationState(BaseModel):
    """Current degradation state."""
    level: ServiceDegradationLevel
    reason: str
    started_at: datetime = Field(default_factory=datetime.now)
    features_disabled: List[str] = Field(default_factory=list)
    features_limited: List[str] = Field(default_factory=list)
    estimated_recovery: Optional[datetime] = None


class DegradationMetrics(BaseModel):
    """Degradation service metrics."""
    current_level: ServiceDegradationLevel = ServiceDegradationLevel.FULL
    degradation_count: int = 0
    recovery_count: int = 0
    time_degraded: float = 0.0  # Total seconds
    features_affected: int = 0
    last_degradation: Optional[datetime] = None
    last_recovery: Optional[datetime] = None


class DegradationDecision(BaseModel):
    """Degradation decision result."""
    should_degrade: bool
    target_level: ServiceDegradationLevel
    reason: str
    affected_features: List[str]
    recovery_actions: List[str] = Field(default_factory=list)


class DegradationService:
    """Manages service degradation."""
    
    # Core features that should remain available
    CORE_FEATURES = {
        "project_access",
        "document_read",
        "document_write",
        "save_project",
        "export_basic"
    }
    
    def __init__(self, policy: Optional[DegradationPolicy] = None):
        """Initialize degradation service.
        
        Args:
            policy: Degradation policy
        """
        self.policy = policy or DegradationPolicy()
        self.current_state = DegradationState(
            level=ServiceDegradationLevel.FULL,
            reason="System normal"
        )
        self.metrics = DegradationMetrics()
        
        # Feature registry
        self._features: Dict[str, DegradableFeature] = {}
        self._feature_states: Dict[str, bool] = {}  # True = enabled
        
        # State management
        self._lock = threading.RLock()
        self._degradation_history: List[DegradationState] = []
        self._recovery_callbacks: List[Callable] = []
        
        # Register default features
        self._register_default_features()
        
    def register_feature(self, feature: DegradableFeature) -> None:
        """Register degradable feature.
        
        Args:
            feature: Feature configuration
        """
        with self._lock:
            self._features[feature.name] = feature
            self._feature_states[feature.name] = True
            logger.info(f"Registered degradable feature: {feature.name}")
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if feature is available.
        
        Args:
            feature_name: Feature name
            
        Returns:
            True if feature is available
        """
        with self._lock:
            # Core features always available unless emergency
            if feature_name in self.CORE_FEATURES:
                return self.current_state.level != ServiceDegradationLevel.EMERGENCY
            
            # Check feature state
            return self._feature_states.get(feature_name, False)
    
    def get_fallback_handler(self, feature_name: str) -> Optional[str]:
        """Get fallback handler for feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Fallback handler name or None
        """
        feature = self._features.get(feature_name)
        return feature.fallback_handler if feature else None
    
    def evaluate_degradation(self,
                            error_rate: float,
                            latency: float,
                            cpu_usage: float,
                            memory_usage: float) -> DegradationDecision:
        """Evaluate if degradation is needed.
        
        Args:
            error_rate: Current error rate (0.0-1.0)
            latency: Current latency in seconds
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            
        Returns:
            Degradation decision
        """
        with self._lock:
            # Check thresholds
            should_degrade = False
            reasons = []
            
            if error_rate > self.policy.error_rate_threshold:
                should_degrade = True
                reasons.append(f"High error rate: {error_rate:.1%}")
            
            if latency > self.policy.latency_threshold:
                should_degrade = True
                reasons.append(f"High latency: {latency:.1f}s")
            
            if cpu_usage > self.policy.cpu_threshold:
                should_degrade = True
                reasons.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            if memory_usage > self.policy.memory_threshold:
                should_degrade = True
                reasons.append(f"High memory usage: {memory_usage:.1f}%")
            
            # Determine target level
            if not should_degrade:
                return DegradationDecision(
                    should_degrade=False,
                    target_level=ServiceDegradationLevel.FULL,
                    reason="System within normal parameters",
                    affected_features=[]
                )
            
            # Calculate severity
            severity_score = 0
            if error_rate > 0.5:
                severity_score += 3
            elif error_rate > self.policy.error_rate_threshold:
                severity_score += 1
            
            if latency > self.policy.latency_threshold * 2:
                severity_score += 2
            elif latency > self.policy.latency_threshold:
                severity_score += 1
            
            if cpu_usage > 90:
                severity_score += 2
            elif cpu_usage > self.policy.cpu_threshold:
                severity_score += 1
            
            if memory_usage > 90:
                severity_score += 2
            elif memory_usage > self.policy.memory_threshold:
                severity_score += 1
            
            # Map severity to degradation level
            if severity_score >= 6:
                target_level = ServiceDegradationLevel.EMERGENCY
            elif severity_score >= 4:
                target_level = ServiceDegradationLevel.OFFLINE
            elif severity_score >= 2:
                target_level = ServiceDegradationLevel.LIMITED
            else:
                target_level = ServiceDegradationLevel.FULL
            
            # Determine affected features
            affected_features = self._get_features_to_disable(target_level)
            
            return DegradationDecision(
                should_degrade=True,
                target_level=target_level,
                reason="; ".join(reasons),
                affected_features=affected_features,
                recovery_actions=self._get_recovery_actions(target_level)
            )
    
    def apply_degradation(self, decision: DegradationDecision) -> None:
        """Apply degradation decision.
        
        Args:
            decision: Degradation decision
        """
        with self._lock:
            if not decision.should_degrade:
                return
            
            # Check minimum duration
            if self.current_state.level != ServiceDegradationLevel.FULL:
                elapsed = (datetime.now() - self.current_state.started_at).total_seconds()
                if elapsed < self.policy.min_degradation_duration:
                    logger.info(
                        f"Skipping degradation change, minimum duration not met "
                        f"({elapsed:.0f}s < {self.policy.min_degradation_duration}s)"
                    )
                    return
            
            # Create new state
            old_level = self.current_state.level
            new_state = DegradationState(
                level=decision.target_level,
                reason=decision.reason,
                features_disabled=[],
                features_limited=[]
            )
            
            # Apply feature changes
            for feature_name in decision.affected_features:
                feature = self._features.get(feature_name)
                if not feature:
                    continue
                
                if decision.target_level == ServiceDegradationLevel.OFFLINE:
                    if feature.offline_capable:
                        new_state.features_limited.append(feature_name)
                        self._feature_states[feature_name] = True  # Limited
                    else:
                        new_state.features_disabled.append(feature_name)
                        self._feature_states[feature_name] = False
                elif decision.target_level == ServiceDegradationLevel.LIMITED:
                    new_state.features_limited.append(feature_name)
                    self._feature_states[feature_name] = True  # Limited
                elif decision.target_level == ServiceDegradationLevel.EMERGENCY:
                    if feature_name not in self.CORE_FEATURES:
                        new_state.features_disabled.append(feature_name)
                        self._feature_states[feature_name] = False
            
            # Estimate recovery time
            if self.policy.auto_recovery_enabled:
                new_state.estimated_recovery = datetime.now() + timedelta(
                    seconds=self.policy.recovery_check_interval * 2
                )
            
            # Update state
            self.current_state = new_state
            self._degradation_history.append(new_state)
            
            # Update metrics
            self.metrics.current_level = decision.target_level
            self.metrics.degradation_count += 1
            self.metrics.last_degradation = datetime.now()
            self.metrics.features_affected = len(
                new_state.features_disabled + new_state.features_limited
            )
            
            logger.warning(
                f"Service degraded from {old_level.value} to {decision.target_level.value}: "
                f"{decision.reason}"
            )
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover from degradation.
        
        Returns:
            True if recovery successful
        """
        with self._lock:
            if self.current_state.level == ServiceDegradationLevel.FULL:
                return True
            
            # Check if minimum duration has passed
            elapsed = (datetime.now() - self.current_state.started_at).total_seconds()
            if elapsed < self.policy.min_degradation_duration:
                return False
            
            # Re-enable all features
            for feature_name in self._features:
                self._feature_states[feature_name] = True
            
            # Update state
            old_level = self.current_state.level
            self.current_state = DegradationState(
                level=ServiceDegradationLevel.FULL,
                reason="System recovered"
            )
            
            # Update metrics
            self.metrics.current_level = ServiceDegradationLevel.FULL
            self.metrics.recovery_count += 1
            self.metrics.last_recovery = datetime.now()
            self.metrics.time_degraded += elapsed
            self.metrics.features_affected = 0
            
            # Notify callbacks
            for callback in self._recovery_callbacks:
                try:
                    callback(old_level, ServiceDegradationLevel.FULL)
                except Exception as e:
                    logger.error(f"Error in recovery callback: {e}")
            
            logger.info(f"Service recovered from {old_level.value} to FULL")
            return True
    
    def get_state(self) -> DegradationState:
        """Get current degradation state.
        
        Returns:
            Current state
        """
        with self._lock:
            return self.current_state.model_copy()
    
    def get_metrics(self) -> DegradationMetrics:
        """Get degradation metrics.
        
        Returns:
            Metrics
        """
        with self._lock:
            return self.metrics.model_copy()
    
    def get_available_features(self) -> List[str]:
        """Get list of available features.
        
        Returns:
            Available feature names
        """
        with self._lock:
            return [
                name for name, enabled in self._feature_states.items()
                if enabled
            ]
    
    def add_recovery_callback(self, callback: Callable) -> None:
        """Add recovery callback.
        
        Args:
            callback: Callback function(old_level, new_level)
        """
        self._recovery_callbacks.append(callback)
    
    def _register_default_features(self) -> None:
        """Register default degradable features."""
        default_features = [
            # Core features (high priority)
            DegradableFeature(
                name="project_access",
                component="core",
                priority=100,
                offline_capable=True
            ),
            DegradableFeature(
                name="document_read",
                component="core",
                priority=95,
                offline_capable=True
            ),
            DegradableFeature(
                name="document_write",
                component="core",
                priority=90,
                offline_capable=True
            ),
            
            # AI features (medium priority)
            DegradableFeature(
                name="ai_suggestions",
                component="ai",
                priority=50,
                offline_capable=False,
                fallback_handler="offline_ai_fallback"
            ),
            DegradableFeature(
                name="ai_completion",
                component="ai",
                priority=45,
                offline_capable=False
            ),
            
            # External features (low priority)
            DegradableFeature(
                name="external_mcp_tools",
                component="external",
                priority=30,
                offline_capable=False,
                fallback_handler="local_tools_fallback"
            ),
            DegradableFeature(
                name="cloud_sync",
                component="external",
                priority=25,
                offline_capable=False
            ),
            
            # Analytics features (lowest priority)
            DegradableFeature(
                name="analytics",
                component="monitoring",
                priority=10,
                offline_capable=False
            ),
            DegradableFeature(
                name="telemetry",
                component="monitoring",
                priority=5,
                offline_capable=False
            )
        ]
        
        for feature in default_features:
            self.register_feature(feature)
    
    def _get_features_to_disable(self, level: ServiceDegradationLevel) -> List[str]:
        """Get features to disable for degradation level.
        
        Args:
            level: Target degradation level
            
        Returns:
            Feature names to disable
        """
        if level == ServiceDegradationLevel.FULL:
            return []
        
        # Sort features by priority (ascending, so lowest priority first)
        sorted_features = sorted(
            self._features.values(),
            key=lambda f: f.priority
        )
        
        features_to_affect = []
        
        if level == ServiceDegradationLevel.LIMITED:
            # Disable lowest 30% priority features
            cutoff = len(sorted_features) * 0.3
            features_to_affect = [
                f.name for f in sorted_features[:int(cutoff)]
            ]
        
        elif level == ServiceDegradationLevel.OFFLINE:
            # Disable all non-offline-capable features
            features_to_affect = [
                f.name for f in sorted_features
                if not f.offline_capable and f.name not in self.CORE_FEATURES
            ]
        
        elif level == ServiceDegradationLevel.EMERGENCY:
            # Disable everything except core features
            features_to_affect = [
                f.name for f in sorted_features
                if f.name not in self.CORE_FEATURES
            ]
        
        return features_to_affect
    
    def _get_recovery_actions(self, level: ServiceDegradationLevel) -> List[str]:
        """Get recommended recovery actions.
        
        Args:
            level: Degradation level
            
        Returns:
            Recovery action descriptions
        """
        actions = []
        
        if level == ServiceDegradationLevel.LIMITED:
            actions.append("Monitor system resources")
            actions.append("Consider scaling resources")
        
        elif level == ServiceDegradationLevel.OFFLINE:
            actions.append("Check network connectivity")
            actions.append("Verify external service status")
            actions.append("Review error logs")
        
        elif level == ServiceDegradationLevel.EMERGENCY:
            actions.append("Immediate investigation required")
            actions.append("Check system health")
            actions.append("Review recent changes")
            actions.append("Consider rollback")
        
        return actions


# Global degradation service instance
_degradation_service: Optional[DegradationService] = None


def get_degradation_service() -> DegradationService:
    """Get global degradation service instance.
    
    Returns:
        Degradation service
    """
    global _degradation_service
    if _degradation_service is None:
        _degradation_service = DegradationService()
    return _degradation_service
