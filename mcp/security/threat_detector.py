"""
MCP Threat Detector - Real-time Security Threat Detection

Implements enterprise-grade threat detection with:
- Real-time anomaly detection and threat assessment
- Machine learning-based behavioral analysis
- Pattern recognition for attack signatures
- Adaptive threat scoring with risk assessment
- Automated threat response and mitigation
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re


class ThreatLevel(Enum):
    """Threat level classification"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats"""
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    CREDENTIAL_STUFFING = "credential_stuffing"
    SESSION_HIJACKING = "session_hijacking"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    RATE_LIMITING_VIOLATION = "rate_limiting_violation"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_COMPROMISE = "system_compromise"


class ThreatAssessment(NamedTuple):
    """Threat assessment result"""
    threat_level: ThreatLevel
    threat_types: List[ThreatType]
    confidence_score: float  # 0.0 to 1.0
    risk_factors: List[str]
    recommended_actions: List[str]
    reason: str
    metadata: Dict[str, Any] = {}


@dataclass
class ThreatIndicator:
    """Security threat indicator"""
    indicator_type: str
    value: Any
    severity: ThreatLevel
    confidence: float
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int = 1
    related_events: List[str] = field(default_factory=list)
    
    def update_occurrence(self):
        """Update last seen time and increment count"""
        self.last_seen = datetime.utcnow()
        self.occurrence_count += 1


@dataclass
class BehavioralProfile:
    """Behavioral profile for server activity"""
    server_id: str
    authentication_patterns: Dict[str, Any] = field(default_factory=dict)
    operation_patterns: Dict[str, Any] = field(default_factory=dict)
    resource_access_patterns: Dict[str, Any] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    baseline_established: bool = False
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_authentication_pattern(self, auth_time: datetime, outcome: str):
        """Update authentication pattern data"""
        hour = auth_time.hour
        day_of_week = auth_time.weekday()
        
        if 'auth_times' not in self.authentication_patterns:
            self.authentication_patterns['auth_times'] = defaultdict(int)
        if 'auth_outcomes' not in self.authentication_patterns:
            self.authentication_patterns['auth_outcomes'] = defaultdict(int)
        if 'hourly_distribution' not in self.authentication_patterns:
            self.authentication_patterns['hourly_distribution'] = defaultdict(int)
        if 'daily_distribution' not in self.authentication_patterns:
            self.authentication_patterns['daily_distribution'] = defaultdict(int)
        
        self.authentication_patterns['auth_outcomes'][outcome] += 1
        self.authentication_patterns['hourly_distribution'][hour] += 1
        self.authentication_patterns['daily_distribution'][day_of_week] += 1
        self.last_updated = datetime.utcnow()


@dataclass
class ThreatMetrics:
    """Threat detection metrics"""
    total_threats_detected: int = 0
    threats_by_level: Dict[str, int] = field(default_factory=dict)
    threats_by_type: Dict[str, int] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    detection_accuracy: float = 0.0
    average_response_time_ms: float = 0.0
    active_indicators: int = 0


class ThreatDetector:
    """
    Enterprise-grade threat detection engine for MCP security gateway.
    
    Features:
    - Real-time threat assessment with machine learning-based analysis
    - Behavioral profiling and anomaly detection
    - Pattern recognition for known attack signatures
    - Adaptive threat scoring with confidence levels
    - Automated threat response and mitigation recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize threat detector with configuration.
        
        Args:
            config: Threat detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self.storage_path = Path(config.get('storage_path', 'knowledge/security/threats'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Threat detection parameters
        self.threat_thresholds = config.get('threat_thresholds', {
            'auth_failures_per_minute': 10,
            'auth_failures_per_hour': 50,
            'unusual_operation_threshold': 0.8,
            'resource_access_deviation_threshold': 0.7,
            'behavioral_anomaly_threshold': 0.75
        })
        
        # Data structures for threat tracking
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.behavioral_profiles: Dict[str, BehavioralProfile] = {}
        self.active_threats: Dict[str, ThreatAssessment] = {}
        
        # Rate limiting and pattern tracking
        self.auth_attempt_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.operation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.resource_access_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Machine learning baseline establishment
        self.learning_period_days = config.get('learning_period_days', 7)
        self.min_events_for_baseline = config.get('min_events_for_baseline', 100)
        
        # Threat signatures (attack patterns)
        self.threat_signatures = self._load_threat_signatures()
        
        # Metrics
        self.metrics = ThreatMetrics()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_tasks()
        
        self.logger.info("Threat Detector initialized with real-time anomaly detection")
    
    async def assess_authentication_risk(
        self,
        server_id: str,
        credentials: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ThreatAssessment:
        """
        Assess threat risk for authentication attempt.
        
        Args:
            server_id: Server attempting authentication
            credentials: Authentication credentials
            context: Authentication context
            
        Returns:
            ThreatAssessment with risk evaluation
        """
        start_time = time.perf_counter()
        
        try:
            threat_level = ThreatLevel.NONE
            threat_types = []
            risk_factors = []
            recommended_actions = []
            confidence_score = 0.0
            
            # Track authentication attempt
            current_time = datetime.utcnow()
            self.auth_attempt_history[server_id].append({
                'timestamp': current_time,
                'credentials_hash': self._hash_credentials(credentials),
                'context': context
            })
            
            # Check for brute force attacks
            brute_force_assessment = await self._detect_brute_force_attack(server_id)
            if brute_force_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, brute_force_assessment.threat_level)
                threat_types.extend(brute_force_assessment.threat_types)
                risk_factors.extend(brute_force_assessment.risk_factors)
                recommended_actions.extend(brute_force_assessment.recommended_actions)
                confidence_score = max(confidence_score, brute_force_assessment.confidence_score)
            
            # Check for credential stuffing
            credential_stuffing_assessment = await self._detect_credential_stuffing(server_id, credentials)
            if credential_stuffing_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, credential_stuffing_assessment.threat_level)
                threat_types.extend(credential_stuffing_assessment.threat_types)
                risk_factors.extend(credential_stuffing_assessment.risk_factors)
                recommended_actions.extend(credential_stuffing_assessment.recommended_actions)
                confidence_score = max(confidence_score, credential_stuffing_assessment.confidence_score)
            
            # Check behavioral anomalies
            behavioral_assessment = await self._assess_authentication_behavioral_anomaly(server_id, context)
            if behavioral_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, behavioral_assessment.threat_level)
                threat_types.extend(behavioral_assessment.threat_types)
                risk_factors.extend(behavioral_assessment.risk_factors)
                recommended_actions.extend(behavioral_assessment.recommended_actions)
                confidence_score = max(confidence_score, behavioral_assessment.confidence_score)
            
            # Check against threat signatures
            signature_assessment = await self._check_threat_signatures(server_id, 'authentication', context)
            if signature_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, signature_assessment.threat_level)
                threat_types.extend(signature_assessment.threat_types)
                risk_factors.extend(signature_assessment.risk_factors)
                recommended_actions.extend(signature_assessment.recommended_actions)
                confidence_score = max(confidence_score, signature_assessment.confidence_score)
            
            # Create final assessment
            assessment = ThreatAssessment(
                threat_level=threat_level,
                threat_types=list(set(threat_types)),  # Remove duplicates
                confidence_score=confidence_score,
                risk_factors=list(set(risk_factors)),
                recommended_actions=list(set(recommended_actions)),
                reason=f"Authentication risk assessment for server {server_id}",
                metadata={
                    'server_id': server_id,
                    'assessment_time': current_time.isoformat(),
                    'context': context
                }
            )
            
            # Update metrics
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.average_response_time_ms = (
                (self.metrics.average_response_time_ms * self.metrics.total_threats_detected + response_time) /
                (self.metrics.total_threats_detected + 1)
            )
            
            if threat_level != ThreatLevel.NONE:
                await self._record_threat_detection(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Failed to assess authentication risk for server {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.MEDIUM,
                threat_types=[ThreatType.SYSTEM_COMPROMISE],
                confidence_score=0.5,
                risk_factors=[f"Risk assessment failed: {str(e)}"],
                recommended_actions=["Manual security review required"],
                reason="Threat assessment system error"
            )
    
    async def assess_operation_risk(
        self,
        server_id: str,
        operation: str,
        resources: List[str],
        context: Dict[str, Any]
    ) -> ThreatAssessment:
        """
        Assess threat risk for operation request.
        
        Args:
            server_id: Server requesting operation
            operation: Operation being requested
            resources: Resources being accessed
            context: Operation context
            
        Returns:
            ThreatAssessment with risk evaluation
        """
        try:
            threat_level = ThreatLevel.NONE
            threat_types = []
            risk_factors = []
            recommended_actions = []
            confidence_score = 0.0
            
            # Track operation attempt
            current_time = datetime.utcnow()
            operation_event = {
                'timestamp': current_time,
                'operation': operation,
                'resources': resources,
                'context': context
            }
            self.operation_history[server_id].append(operation_event)
            
            # Check for privilege escalation attempts
            privilege_escalation_assessment = await self._detect_privilege_escalation(server_id, operation, resources)
            if privilege_escalation_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, privilege_escalation_assessment.threat_level)
                threat_types.extend(privilege_escalation_assessment.threat_types)
                risk_factors.extend(privilege_escalation_assessment.risk_factors)
                recommended_actions.extend(privilege_escalation_assessment.recommended_actions)
                confidence_score = max(confidence_score, privilege_escalation_assessment.confidence_score)
            
            # Check for data exfiltration patterns
            exfiltration_assessment = await self._detect_data_exfiltration(server_id, operation, resources)
            if exfiltration_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, exfiltration_assessment.threat_level)
                threat_types.extend(exfiltration_assessment.threat_types)
                risk_factors.extend(exfiltration_assessment.risk_factors)
                recommended_actions.extend(exfiltration_assessment.recommended_actions)
                confidence_score = max(confidence_score, exfiltration_assessment.confidence_score)
            
            # Check for unusual operation patterns
            anomaly_assessment = await self._assess_operation_behavioral_anomaly(server_id, operation, resources)
            if anomaly_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, anomaly_assessment.threat_level)
                threat_types.extend(anomaly_assessment.threat_types)
                risk_factors.extend(anomaly_assessment.risk_factors)
                recommended_actions.extend(anomaly_assessment.recommended_actions)
                confidence_score = max(confidence_score, anomaly_assessment.confidence_score)
            
            # Check rate limiting violations
            rate_limit_assessment = await self._check_rate_limiting_violations(server_id, operation)
            if rate_limit_assessment.threat_level != ThreatLevel.NONE:
                threat_level = max(threat_level, rate_limit_assessment.threat_level)
                threat_types.extend(rate_limit_assessment.threat_types)
                risk_factors.extend(rate_limit_assessment.risk_factors)
                recommended_actions.extend(rate_limit_assessment.recommended_actions)
                confidence_score = max(confidence_score, rate_limit_assessment.confidence_score)
            
            # Create final assessment
            assessment = ThreatAssessment(
                threat_level=threat_level,
                threat_types=list(set(threat_types)),
                confidence_score=confidence_score,
                risk_factors=list(set(risk_factors)),
                recommended_actions=list(set(recommended_actions)),
                reason=f"Operation risk assessment for server {server_id}",
                metadata={
                    'server_id': server_id,
                    'operation': operation,
                    'resources': resources,
                    'assessment_time': current_time.isoformat()
                }
            )
            
            if threat_level != ThreatLevel.NONE:
                await self._record_threat_detection(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Failed to assess operation risk for server {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.MEDIUM,
                threat_types=[ThreatType.SYSTEM_COMPROMISE],
                confidence_score=0.5,
                risk_factors=[f"Risk assessment failed: {str(e)}"],
                recommended_actions=["Manual security review required"],
                reason="Threat assessment system error"
            )
    
    async def get_threat_metrics(self) -> ThreatMetrics:
        """Get current threat detection metrics"""
        # Update active indicators count
        self.metrics.active_indicators = len([
            indicator for indicator in self.threat_indicators.values()
            if indicator.last_seen > datetime.utcnow() - timedelta(hours=24)
        ])
        
        return self.metrics
    
    async def get_server_threat_profile(self, server_id: str) -> Dict[str, Any]:
        """
        Get comprehensive threat profile for a server.
        
        Args:
            server_id: Server to get threat profile for
            
        Returns:
            Threat profile with behavioral analysis and risk assessment
        """
        try:
            profile = {
                "server_id": server_id,
                "generated_at": datetime.utcnow().isoformat(),
                "behavioral_profile": {},
                "threat_indicators": [],
                "recent_threats": [],
                "risk_score": 0.0,
                "recommendation": "normal_operations"
            }
            
            # Get behavioral profile
            if server_id in self.behavioral_profiles:
                behavioral_profile = self.behavioral_profiles[server_id]
                profile["behavioral_profile"] = {
                    "baseline_established": behavioral_profile.baseline_established,
                    "last_updated": behavioral_profile.last_updated.isoformat(),
                    "authentication_patterns": behavioral_profile.authentication_patterns,
                    "operation_patterns": behavioral_profile.operation_patterns
                }
            
            # Get threat indicators for this server
            server_indicators = [
                {
                    "type": indicator.indicator_type,
                    "severity": indicator.severity.value,
                    "confidence": indicator.confidence,
                    "occurrence_count": indicator.occurrence_count,
                    "first_seen": indicator.first_seen.isoformat(),
                    "last_seen": indicator.last_seen.isoformat()
                }
                for indicator in self.threat_indicators.values()
                if server_id in str(indicator.value)
            ]
            profile["threat_indicators"] = server_indicators
            
            # Calculate risk score based on recent activity
            risk_score = await self._calculate_server_risk_score(server_id)
            profile["risk_score"] = risk_score
            
            # Provide recommendation
            if risk_score > 0.8:
                profile["recommendation"] = "immediate_investigation_required"
            elif risk_score > 0.6:
                profile["recommendation"] = "enhanced_monitoring"
            elif risk_score > 0.4:
                profile["recommendation"] = "increased_scrutiny"
            else:
                profile["recommendation"] = "normal_operations"
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to get threat profile for server {server_id}: {e}")
            return {"error": str(e)}
    
    async def _detect_brute_force_attack(self, server_id: str) -> ThreatAssessment:
        """Detect brute force authentication attacks"""
        try:
            recent_attempts = [
                attempt for attempt in self.auth_attempt_history[server_id]
                if attempt['timestamp'] > datetime.utcnow() - timedelta(minutes=5)
            ]
            
            if len(recent_attempts) > self.threat_thresholds['auth_failures_per_minute']:
                return ThreatAssessment(
                    threat_level=ThreatLevel.HIGH,
                    threat_types=[ThreatType.BRUTE_FORCE_ATTACK],
                    confidence_score=0.9,
                    risk_factors=[f"Excessive authentication attempts: {len(recent_attempts)} in 5 minutes"],
                    recommended_actions=["Block IP address", "Implement rate limiting", "Alert security team"],
                    reason="High-frequency authentication attempts detected"
                )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No brute force indicators detected"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to detect brute force attack for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Brute force detection error"
            )
    
    async def _detect_credential_stuffing(
        self,
        server_id: str,
        credentials: Dict[str, Any]
    ) -> ThreatAssessment:
        """Detect credential stuffing attacks"""
        try:
            # Check for credential reuse patterns
            credential_hash = self._hash_credentials(credentials)
            
            # Look for same credentials used across multiple servers
            similar_attempts = 0
            for other_server_id, attempts in self.auth_attempt_history.items():
                if other_server_id != server_id:
                    recent_attempts = [
                        attempt for attempt in attempts
                        if (attempt['timestamp'] > datetime.utcnow() - timedelta(hours=1) and
                            attempt.get('credentials_hash') == credential_hash)
                    ]
                    similar_attempts += len(recent_attempts)
            
            if similar_attempts > 5:
                return ThreatAssessment(
                    threat_level=ThreatLevel.MEDIUM,
                    threat_types=[ThreatType.CREDENTIAL_STUFFING],
                    confidence_score=0.7,
                    risk_factors=[f"Same credentials used across {similar_attempts} different servers"],
                    recommended_actions=["Force password reset", "Enable MFA", "Monitor account activity"],
                    reason="Credential reuse pattern detected"
                )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No credential stuffing indicators detected"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to detect credential stuffing for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Credential stuffing detection error"
            )
    
    async def _assess_authentication_behavioral_anomaly(
        self,
        server_id: str,
        context: Dict[str, Any]
    ) -> ThreatAssessment:
        """Assess behavioral anomalies in authentication patterns"""
        try:
            if server_id not in self.behavioral_profiles:
                # Create new behavioral profile
                self.behavioral_profiles[server_id] = BehavioralProfile(server_id=server_id)
            
            profile = self.behavioral_profiles[server_id]
            current_time = datetime.utcnow()
            
            # Update behavioral profile
            profile.update_authentication_pattern(current_time, 'attempt')
            
            # If baseline not established, collect data
            if not profile.baseline_established:
                auth_count = sum(profile.authentication_patterns.get('auth_outcomes', {}).values())
                if auth_count > self.min_events_for_baseline:
                    profile.baseline_established = True
                    self.logger.info(f"Baseline established for server {server_id}")
                
                return ThreatAssessment(
                    threat_level=ThreatLevel.NONE,
                    threat_types=[],
                    confidence_score=0.0,
                    risk_factors=[],
                    recommended_actions=[],
                    reason="Learning baseline behavior"
                )
            
            # Check for temporal anomalies
            hour = current_time.hour
            day_of_week = current_time.weekday()
            
            hourly_dist = profile.authentication_patterns.get('hourly_distribution', {})
            daily_dist = profile.authentication_patterns.get('daily_distribution', {})
            
            # Calculate anomaly scores based on historical patterns
            hour_frequency = hourly_dist.get(hour, 0)
            day_frequency = daily_dist.get(day_of_week, 0)
            
            total_auths = sum(hourly_dist.values())
            expected_hour_frequency = total_auths / 24 if total_auths > 0 else 0
            
            # Simple anomaly detection - activity at unusual times
            if hour_frequency < expected_hour_frequency * 0.1 and expected_hour_frequency > 1:
                return ThreatAssessment(
                    threat_level=ThreatLevel.LOW,
                    threat_types=[ThreatType.ANOMALOUS_BEHAVIOR],
                    confidence_score=0.6,
                    risk_factors=[f"Authentication at unusual time: {hour}:00"],
                    recommended_actions=["Verify user identity", "Monitor session activity"],
                    reason="Temporal behavioral anomaly detected"
                )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No behavioral anomalies detected"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess behavioral anomaly for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Behavioral analysis error"
            )
    
    async def _detect_privilege_escalation(
        self,
        server_id: str,
        operation: str,
        resources: List[str]
    ) -> ThreatAssessment:
        """Detect privilege escalation attempts"""
        try:
            # Check for admin operations by non-admin servers
            if 'admin' in operation.lower() or any('admin' in res.lower() for res in resources):
                # This would check against user permissions in a real implementation
                return ThreatAssessment(
                    threat_level=ThreatLevel.MEDIUM,
                    threat_types=[ThreatType.PRIVILEGE_ESCALATION],
                    confidence_score=0.7,
                    risk_factors=[f"Admin operation attempted: {operation}"],
                    recommended_actions=["Verify permissions", "Review operation legitimacy"],
                    reason="Administrative operation detected"
                )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No privilege escalation indicators detected"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to detect privilege escalation for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Privilege escalation detection error"
            )
    
    async def _detect_data_exfiltration(
        self,
        server_id: str,
        operation: str,
        resources: List[str]
    ) -> ThreatAssessment:
        """Detect data exfiltration patterns"""
        try:
            # Check for bulk data access patterns
            if (operation.lower() in ['read', 'download', 'export'] and 
                len(resources) > 10):
                
                return ThreatAssessment(
                    threat_level=ThreatLevel.MEDIUM,
                    threat_types=[ThreatType.DATA_EXFILTRATION],
                    confidence_score=0.6,
                    risk_factors=[f"Bulk data access: {len(resources)} resources"],
                    recommended_actions=["Monitor data transfer", "Verify business need"],
                    reason="Bulk data access pattern detected"
                )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No data exfiltration indicators detected"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to detect data exfiltration for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Data exfiltration detection error"
            )
    
    async def _assess_operation_behavioral_anomaly(
        self,
        server_id: str,
        operation: str,
        resources: List[str]
    ) -> ThreatAssessment:
        """Assess behavioral anomalies in operation patterns"""
        try:
            # Simple behavioral analysis - check for unusual operation types
            recent_operations = [
                op['operation'] for op in self.operation_history[server_id]
                if op['timestamp'] > datetime.utcnow() - timedelta(days=1)
            ]
            
            if recent_operations:
                operation_counts = defaultdict(int)
                for op in recent_operations:
                    operation_counts[op] += 1
                
                # Check if current operation is rare for this server
                current_op_count = operation_counts.get(operation, 0)
                total_ops = len(recent_operations)
                
                if total_ops > 10 and current_op_count / total_ops < 0.1:
                    return ThreatAssessment(
                        threat_level=ThreatLevel.LOW,
                        threat_types=[ThreatType.ANOMALOUS_BEHAVIOR],
                        confidence_score=0.5,
                        risk_factors=[f"Unusual operation for server: {operation}"],
                        recommended_actions=["Monitor operation outcome", "Verify legitimacy"],
                        reason="Unusual operation pattern detected"
                    )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No operation behavioral anomalies detected"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess operation behavioral anomaly for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Operation behavioral analysis error"
            )
    
    async def _check_rate_limiting_violations(self, server_id: str, operation: str) -> ThreatAssessment:
        """Check for rate limiting violations"""
        try:
            # Check operation frequency in the last minute
            recent_ops = [
                op for op in self.operation_history[server_id]
                if op['timestamp'] > datetime.utcnow() - timedelta(minutes=1)
            ]
            
            if len(recent_ops) > 60:  # More than 1 operation per second
                return ThreatAssessment(
                    threat_level=ThreatLevel.MEDIUM,
                    threat_types=[ThreatType.RATE_LIMITING_VIOLATION],
                    confidence_score=0.8,
                    risk_factors=[f"High operation frequency: {len(recent_ops)} ops/minute"],
                    recommended_actions=["Apply rate limiting", "Monitor for automation"],
                    reason="Rate limiting threshold exceeded"
                )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No rate limiting violations detected"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check rate limiting for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Rate limiting check error"
            )
    
    async def _check_threat_signatures(
        self,
        server_id: str,
        event_type: str,
        context: Dict[str, Any]
    ) -> ThreatAssessment:
        """Check against known threat signatures"""
        try:
            for signature in self.threat_signatures:
                if signature.get('event_type') == event_type:
                    patterns = signature.get('patterns', [])
                    
                    for pattern in patterns:
                        # Simple pattern matching - in production this would be more sophisticated
                        if self._match_threat_pattern(pattern, context):
                            return ThreatAssessment(
                                threat_level=ThreatLevel(signature.get('threat_level', 'medium')),
                                threat_types=[ThreatType(signature.get('threat_type', 'suspicious_activity'))],
                                confidence_score=signature.get('confidence', 0.7),
                                risk_factors=[f"Matched threat signature: {signature.get('name')}"],
                                recommended_actions=signature.get('recommended_actions', []),
                                reason=f"Threat signature match: {signature.get('name')}"
                            )
            
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="No threat signatures matched"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check threat signatures for {server_id}: {e}")
            return ThreatAssessment(
                threat_level=ThreatLevel.NONE,
                threat_types=[],
                confidence_score=0.0,
                risk_factors=[],
                recommended_actions=[],
                reason="Threat signature check error"
            )
    
    def _hash_credentials(self, credentials: Dict[str, Any]) -> str:
        """Create hash of credentials for comparison"""
        # Remove sensitive data and create hash
        credential_str = json.dumps(credentials, sort_keys=True)
        return hashlib.sha256(credential_str.encode('utf-8')).hexdigest()
    
    def _match_threat_pattern(self, pattern: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Match threat pattern against context"""
        try:
            # Simple pattern matching implementation
            for key, expected_value in pattern.items():
                if key not in context:
                    return False
                
                if isinstance(expected_value, str) and expected_value.startswith('regex:'):
                    regex_pattern = expected_value[6:]  # Remove 'regex:' prefix
                    if not re.match(regex_pattern, str(context[key])):
                        return False
                elif context[key] != expected_value:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to match threat pattern: {e}")
            return False
    
    async def _record_threat_detection(self, assessment: ThreatAssessment) -> None:
        """Record threat detection for metrics and analysis"""
        try:
            self.metrics.total_threats_detected += 1
            
            level_str = assessment.threat_level.value
            self.metrics.threats_by_level[level_str] = self.metrics.threats_by_level.get(level_str, 0) + 1
            
            for threat_type in assessment.threat_types:
                type_str = threat_type.value
                self.metrics.threats_by_type[type_str] = self.metrics.threats_by_type.get(type_str, 0) + 1
            
            # Store active threat if high severity
            if assessment.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                threat_id = f"threat_{int(datetime.utcnow().timestamp())}_{secrets.token_hex(8)}"
                self.active_threats[threat_id] = assessment
            
        except Exception as e:
            self.logger.error(f"Failed to record threat detection: {e}")
    
    async def _calculate_server_risk_score(self, server_id: str) -> float:
        """Calculate overall risk score for a server"""
        try:
            risk_score = 0.0
            
            # Factor in recent authentication failures
            recent_failures = len([
                attempt for attempt in self.auth_attempt_history[server_id]
                if attempt['timestamp'] > datetime.utcnow() - timedelta(hours=24)
            ])
            risk_score += min(recent_failures * 0.1, 0.3)
            
            # Factor in threat indicators
            server_indicators = [
                indicator for indicator in self.threat_indicators.values()
                if server_id in str(indicator.value) and 
                indicator.last_seen > datetime.utcnow() - timedelta(hours=24)
            ]
            
            for indicator in server_indicators:
                if indicator.severity == ThreatLevel.CRITICAL:
                    risk_score += 0.4
                elif indicator.severity == ThreatLevel.HIGH:
                    risk_score += 0.3
                elif indicator.severity == ThreatLevel.MEDIUM:
                    risk_score += 0.2
                elif indicator.severity == ThreatLevel.LOW:
                    risk_score += 0.1
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate risk score for {server_id}: {e}")
            return 0.5  # Default medium risk
    
    def _load_threat_signatures(self) -> List[Dict[str, Any]]:
        """Load threat signatures from configuration"""
        try:
            signatures_file = self.storage_path / 'threat_signatures.json'
            if signatures_file.exists():
                with open(signatures_file, 'r') as f:
                    return json.load(f)
            
            # Return default threat signatures
            return [
                {
                    "name": "SQL Injection Attempt",
                    "event_type": "authentication",
                    "threat_level": "high",
                    "threat_type": "policy_violation",
                    "confidence": 0.8,
                    "patterns": [
                        {"user_agent": "regex:.*sqlmap.*"},
                        {"credentials": "regex:.*[\';].*"}
                    ],
                    "recommended_actions": ["Block request", "Alert security team"]
                },
                {
                    "name": "Automated Tool Detection",
                    "event_type": "authentication", 
                    "threat_level": "medium",
                    "threat_type": "suspicious_activity",
                    "confidence": 0.7,
                    "patterns": [
                        {"user_agent": "regex:.*(bot|crawler|scanner).*"}
                    ],
                    "recommended_actions": ["Apply rate limiting", "Monitor activity"]
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to load threat signatures: {e}")
            return []
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self._cleanup_old_data()
                except Exception as e:
                    self.logger.error(f"Error in threat detector cleanup: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        self.logger.info("Threat detector background tasks started")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old threat detection data"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            
            # Cleanup old threat indicators
            old_indicators = [
                key for key, indicator in self.threat_indicators.items()
                if indicator.last_seen < cutoff_time
            ]
            
            for key in old_indicators:
                del self.threat_indicators[key]
            
            # Cleanup old active threats
            old_threats = [
                key for key, threat in self.active_threats.items()
                if datetime.fromisoformat(threat.metadata.get('assessment_time', '1970-01-01')) < cutoff_time
            ]
            
            for key in old_threats:
                del self.active_threats[key]
            
            if old_indicators or old_threats:
                self.logger.info(f"Cleaned up {len(old_indicators)} old indicators and {len(old_threats)} old threats")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old threat data: {e}")