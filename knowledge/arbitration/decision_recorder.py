#!/usr/bin/env python3
"""
RIF Decision Recorder
Comprehensive audit trail system for arbitration decisions and processes.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from pathlib import Path

# Import arbitration components
from .arbitration_system import ArbitrationResult, ArbitrationDecision, ArbitrationStatus, ArbitrationType
from .conflict_detector import ConflictAnalysis
from .escalation_engine import EscalationResult, EscalationLevel

# Import consensus components
import sys
sys.path.insert(0, '/Users/cal/DEV/RIF/claude/commands')

from consensus_architecture import AgentVote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecordType(Enum):
    """Types of decision records"""
    ARBITRATION_DECISION = "arbitration_decision"
    CONFLICT_ANALYSIS = "conflict_analysis" 
    ESCALATION_PROCESS = "escalation_process"
    HUMAN_INTERVENTION = "human_intervention"
    AUDIT_EVENT = "audit_event"
    PERFORMANCE_METRIC = "performance_metric"

class AuditEventType(Enum):
    """Types of audit events"""
    DECISION_INITIATED = "decision_initiated"
    CONFLICT_DETECTED = "conflict_detected"
    ESCALATION_TRIGGERED = "escalation_triggered"
    RESOLUTION_ACHIEVED = "resolution_achieved"
    HUMAN_ESCALATED = "human_escalated"
    DECISION_OVERRIDDEN = "decision_overridden"
    SYSTEM_ERROR = "system_error"
    VALIDATION_FAILED = "validation_failed"

@dataclass
class ArbitrationRecord:
    """Complete record of arbitration process"""
    record_id: str
    record_type: RecordType
    arbitration_id: str
    timestamp: datetime
    arbitration_result: ArbitrationResult
    conflict_analysis: ConflictAnalysis
    escalation_results: List[EscalationResult]
    original_context: Dict[str, Any]
    performance_metrics: Dict[str, float]
    compliance_flags: List[str]
    verification_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditEvent:
    """Individual audit event in decision process"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    arbitration_id: str
    agent_id: Optional[str]
    event_data: Dict[str, Any]
    severity: str
    correlation_id: Optional[str] = None

@dataclass 
class ComplianceReport:
    """Compliance analysis report"""
    report_id: str
    arbitration_id: str
    generated_at: datetime
    compliance_score: float
    violations: List[str]
    recommendations: List[str]
    audit_trail_complete: bool
    decision_justified: bool
    escalation_appropriate: bool
    evidence_sufficient: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class DecisionRecorder:
    """
    Comprehensive audit trail and decision recording system
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """Initialize decision recorder"""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.arbitration_path = self.knowledge_base_path / "arbitration"
        
        # Create directory structure
        self.records_path = self.arbitration_path / "records"
        self.audit_path = self.arbitration_path / "audit_events"
        self.compliance_path = self.arbitration_path / "compliance"
        self.reports_path = self.arbitration_path / "reports"
        
        for path in [self.records_path, self.audit_path, self.compliance_path, self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Recording configuration
        self.config = {
            "retention_days": 365,      # Keep records for 1 year
            "compression_enabled": True,
            "encryption_enabled": False,  # Would enable for sensitive data
            "verification_enabled": True,
            "compliance_threshold": 0.8,
            "audit_level": "detailed"    # minimal, standard, detailed
        }
        
        # Performance tracking
        self.recording_metrics = {
            "total_records": 0,
            "records_by_type": {rt.value: 0 for rt in RecordType},
            "audit_events": 0,
            "compliance_violations": 0,
            "verification_failures": 0,
            "average_record_size_bytes": 0,
            "storage_used_mb": 0.0
        }

    def record_arbitration_decision(self, arbitration_result: ArbitrationResult,
                                  conflict_analysis: ConflictAnalysis,
                                  escalation_results: List[EscalationResult],
                                  context: Dict[str, Any]) -> ArbitrationRecord:
        """
        Record complete arbitration decision with full audit trail
        
        Args:
            arbitration_result: Complete arbitration result
            conflict_analysis: Conflict analysis that led to arbitration
            escalation_results: Results from escalation process
            context: Original decision context
            
        Returns:
            ArbitrationRecord: Complete record with verification
        """
        record_id = self._generate_record_id(arbitration_result.arbitration_id)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            arbitration_result, escalation_results
        )
        
        # Check compliance
        compliance_flags = self._check_compliance(
            arbitration_result, conflict_analysis, escalation_results, context
        )
        
        # Create complete record
        record = ArbitrationRecord(
            record_id=record_id,
            record_type=RecordType.ARBITRATION_DECISION,
            arbitration_id=arbitration_result.arbitration_id,
            timestamp=datetime.now(),
            arbitration_result=arbitration_result,
            conflict_analysis=conflict_analysis,
            escalation_results=escalation_results,
            original_context=context,
            performance_metrics=performance_metrics,
            compliance_flags=compliance_flags,
            verification_hash="",  # Will be calculated after serialization
            metadata={
                "recorder_version": "1.0.0",
                "record_size_estimate": 0,
                "retention_until": (datetime.now() + timedelta(days=self.config["retention_days"])).isoformat()
            }
        )
        
        # Generate verification hash
        record.verification_hash = self._generate_verification_hash(record)
        
        # Store record
        self._store_record(record)
        
        # Generate audit events for this recording
        self._generate_audit_events(record)
        
        # Update metrics
        self._update_recording_metrics(record)
        
        logger.info(f"Arbitration decision recorded: {record_id}")
        
        return record

    def record_audit_event(self, event_type: AuditEventType, arbitration_id: str,
                         agent_id: Optional[str] = None, event_data: Dict[str, Any] = None,
                         severity: str = "info", correlation_id: Optional[str] = None) -> AuditEvent:
        """Record individual audit event"""
        event_id = f"audit-{int(datetime.now().timestamp())}-{len(event_data or {})}"
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            arbitration_id=arbitration_id,
            agent_id=agent_id,
            event_data=event_data or {},
            severity=severity,
            correlation_id=correlation_id
        )
        
        # Store audit event
        self._store_audit_event(event)
        
        self.recording_metrics["audit_events"] += 1
        
        return event

    def generate_compliance_report(self, arbitration_id: str) -> ComplianceReport:
        """
        Generate compliance report for arbitration decision
        
        Args:
            arbitration_id: ID of arbitration to analyze
            
        Returns:
            ComplianceReport: Detailed compliance analysis
        """
        # Load arbitration record
        record = self._load_arbitration_record(arbitration_id)
        if not record:
            raise ValueError(f"No arbitration record found for ID: {arbitration_id}")
        
        report_id = f"compliance-{arbitration_id}-{int(datetime.now().timestamp())}"
        
        # Analyze compliance dimensions
        audit_trail_complete = self._verify_audit_trail_completeness(record)
        decision_justified = self._verify_decision_justification(record)
        escalation_appropriate = self._verify_escalation_appropriateness(record)
        evidence_sufficient = self._verify_evidence_sufficiency(record)
        
        # Calculate overall compliance score
        compliance_dimensions = [
            audit_trail_complete,
            decision_justified, 
            escalation_appropriate,
            evidence_sufficient
        ]
        compliance_score = sum(compliance_dimensions) / len(compliance_dimensions)
        
        # Identify violations
        violations = []
        if not audit_trail_complete:
            violations.append("Incomplete audit trail - missing process steps")
        if not decision_justified:
            violations.append("Insufficient justification for final decision")
        if not escalation_appropriate:
            violations.append("Escalation path not appropriate for conflict severity")
        if not evidence_sufficient:
            violations.append("Insufficient evidence quality for decision confidence")
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(record, violations)
        
        report = ComplianceReport(
            report_id=report_id,
            arbitration_id=arbitration_id,
            generated_at=datetime.now(),
            compliance_score=compliance_score,
            violations=violations,
            recommendations=recommendations,
            audit_trail_complete=audit_trail_complete,
            decision_justified=decision_justified,
            escalation_appropriate=escalation_appropriate,
            evidence_sufficient=evidence_sufficient,
            metadata={
                "compliance_threshold": self.config["compliance_threshold"],
                "analysis_version": "1.0.0",
                "record_hash": record.verification_hash
            }
        )
        
        # Store compliance report
        self._store_compliance_report(report)
        
        if compliance_score < self.config["compliance_threshold"]:
            self.recording_metrics["compliance_violations"] += 1
            logger.warning(f"Compliance violation detected: {report_id} (score: {compliance_score:.2f})")
        
        return report

    def query_arbitration_history(self, filters: Dict[str, Any] = None,
                                limit: int = 100) -> List[ArbitrationRecord]:
        """
        Query arbitration history with filters
        
        Args:
            filters: Filter criteria (e.g., date_range, agent_id, decision_type)
            limit: Maximum number of records to return
            
        Returns:
            List of matching arbitration records
        """
        records = []
        
        # Load all records (in production, this would be optimized with indexing)
        for record_file in self.records_path.glob("*.json"):
            try:
                record = self._load_record_from_file(record_file)
                if record and self._matches_filters(record, filters or {}):
                    records.append(record)
                    
                    if len(records) >= limit:
                        break
                        
            except Exception as e:
                logger.error(f"Error loading record {record_file}: {str(e)}")
                continue
        
        # Sort by timestamp (most recent first)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        
        return records

    def verify_record_integrity(self, record_id: str) -> Tuple[bool, List[str]]:
        """
        Verify integrity of stored record
        
        Args:
            record_id: ID of record to verify
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Load record
            record = self._load_arbitration_record_by_record_id(record_id)
            if not record:
                return False, ["Record not found"]
            
            # Verify hash
            expected_hash = self._generate_verification_hash(record)
            if record.verification_hash != expected_hash:
                issues.append("Verification hash mismatch - possible tampering")
            
            # Verify audit trail consistency
            if not self._verify_audit_trail_consistency(record):
                issues.append("Audit trail inconsistency detected")
            
            # Verify decision logic consistency
            if not self._verify_decision_logic_consistency(record):
                issues.append("Decision logic inconsistency detected")
            
            # Check for required fields
            required_fields = ['arbitration_id', 'timestamp', 'arbitration_result', 'conflict_analysis']
            for field in required_fields:
                if not hasattr(record, field) or getattr(record, field) is None:
                    issues.append(f"Missing required field: {field}")
            
        except Exception as e:
            issues.append(f"Verification error: {str(e)}")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.recording_metrics["verification_failures"] += 1
        
        return is_valid, issues

    def generate_summary_report(self, date_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """Generate summary report of arbitration activity"""
        if not date_range:
            # Default to last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_range = (start_date, end_date)
        
        start_date, end_date = date_range
        
        # Query records in date range
        filters = {"date_range": (start_date, end_date)}
        records = self.query_arbitration_history(filters, limit=1000)
        
        if not records:
            return {"message": "No arbitration records found in date range", "date_range": date_range}
        
        # Calculate summary statistics
        summary = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_days": (end_date - start_date).days
            },
            "activity_summary": {
                "total_arbitrations": len(records),
                "successful_resolutions": len([r for r in records if r.arbitration_result.status == ArbitrationStatus.RESOLVED]),
                "human_escalations": len([r for r in records if r.arbitration_result.status == ArbitrationStatus.HUMAN_REQUIRED]),
                "failed_arbitrations": len([r for r in records if r.arbitration_result.status == ArbitrationStatus.FAILED])
            },
            "resolution_methods": {},
            "escalation_patterns": {},
            "performance_metrics": {
                "average_resolution_time_seconds": 0.0,
                "average_confidence_score": 0.0,
                "compliance_rate": 0.0
            },
            "agent_participation": {},
            "conflict_patterns": {},
            "recommendations": []
        }
        
        # Analyze resolution methods
        resolution_methods = {}
        for record in records:
            if record.arbitration_result.final_decision:
                method = record.arbitration_result.final_decision.resolution_method
                resolution_methods[method] = resolution_methods.get(method, 0) + 1
        summary["resolution_methods"] = resolution_methods
        
        # Analyze escalation patterns
        escalation_patterns = {}
        for record in records:
            if record.escalation_results:
                max_level = max(r.step.level.value for r in record.escalation_results)
                escalation_patterns[max_level] = escalation_patterns.get(max_level, 0) + 1
        summary["escalation_patterns"] = escalation_patterns
        
        # Calculate performance metrics
        resolution_times = [r.arbitration_result.processing_time_seconds for r in records]
        confidence_scores = [
            r.arbitration_result.final_decision.confidence_score 
            for r in records 
            if r.arbitration_result.final_decision
        ]
        
        if resolution_times:
            summary["performance_metrics"]["average_resolution_time_seconds"] = sum(resolution_times) / len(resolution_times)
        if confidence_scores:
            summary["performance_metrics"]["average_confidence_score"] = sum(confidence_scores) / len(confidence_scores)
        
        # Calculate compliance rate (placeholder - would integrate with compliance reports)
        compliance_rate = len([r for r in records if not r.compliance_flags]) / len(records)
        summary["performance_metrics"]["compliance_rate"] = compliance_rate
        
        # Analyze agent participation
        agent_participation = {}
        for record in records:
            for vote in record.arbitration_result.original_votes:
                agent_id = vote.agent_id
                if agent_id not in agent_participation:
                    agent_participation[agent_id] = {"total_votes": 0, "conflicts_involved": 0}
                agent_participation[agent_id]["total_votes"] += 1
                agent_participation[agent_id]["conflicts_involved"] += 1
        summary["agent_participation"] = agent_participation
        
        # Analyze conflict patterns
        conflict_patterns = {}
        for record in records:
            for pattern in record.conflict_analysis.patterns_detected:
                pattern_name = pattern.value
                conflict_patterns[pattern_name] = conflict_patterns.get(pattern_name, 0) + 1
        summary["conflict_patterns"] = conflict_patterns
        
        # Generate recommendations
        recommendations = []
        if compliance_rate < 0.8:
            recommendations.append("Consider reviewing arbitration processes - compliance rate below threshold")
        if resolution_methods.get("human_escalation", 0) > len(records) * 0.3:
            recommendations.append("High rate of human escalation - consider improving automated resolution")
        if summary["performance_metrics"]["average_confidence_score"] < 0.7:
            recommendations.append("Low average confidence scores - review decision quality processes")
        
        summary["recommendations"] = recommendations
        
        return summary

    # Internal helper methods
    def _generate_record_id(self, arbitration_id: str) -> str:
        """Generate unique record ID"""
        timestamp = datetime.now().isoformat()
        content = f"{arbitration_id}-{timestamp}"
        return f"rec-{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def _generate_verification_hash(self, record: ArbitrationRecord) -> str:
        """Generate verification hash for record integrity"""
        # Create deterministic representation excluding the hash field
        record_copy = asdict(record)
        record_copy.pop('verification_hash', None)
        
        # Convert to JSON string with sorted keys for consistency
        content = json.dumps(record_copy, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_performance_metrics(self, arbitration_result: ArbitrationResult,
                                     escalation_results: List[EscalationResult]) -> Dict[str, float]:
        """Calculate performance metrics for arbitration"""
        metrics = {
            "total_processing_time_seconds": arbitration_result.processing_time_seconds,
            "escalation_steps_count": len(escalation_results),
            "final_confidence_score": 0.0,
            "efficiency_score": 0.0,
            "resolution_success": 0.0
        }
        
        if arbitration_result.final_decision:
            metrics["final_confidence_score"] = arbitration_result.final_decision.confidence_score
        
        # Calculate efficiency score (inverse of processing time and escalation steps)
        base_time = 60  # 1 minute baseline
        time_factor = base_time / max(arbitration_result.processing_time_seconds, 1)
        escalation_factor = 1.0 / max(len(escalation_results), 1)
        metrics["efficiency_score"] = min(1.0, (time_factor + escalation_factor) / 2)
        
        # Resolution success score
        if arbitration_result.status == ArbitrationStatus.RESOLVED:
            metrics["resolution_success"] = 1.0
        elif arbitration_result.status == ArbitrationStatus.HUMAN_REQUIRED:
            metrics["resolution_success"] = 0.5
        else:
            metrics["resolution_success"] = 0.0
        
        return metrics

    def _check_compliance(self, arbitration_result: ArbitrationResult,
                         conflict_analysis: ConflictAnalysis,
                         escalation_results: List[EscalationResult],
                         context: Dict[str, Any]) -> List[str]:
        """Check compliance flags for arbitration"""
        flags = []
        
        # Check escalation appropriateness
        if conflict_analysis.severity.value == "critical" and len(escalation_results) < 2:
            flags.append("INSUFFICIENT_ESCALATION_FOR_CRITICAL_CONFLICT")
        
        # Check confidence thresholds
        if arbitration_result.final_decision:
            confidence = arbitration_result.final_decision.confidence_score
            if context.get("security_critical", False) and confidence < 0.8:
                flags.append("LOW_CONFIDENCE_FOR_SECURITY_CRITICAL")
            elif confidence < 0.5:
                flags.append("VERY_LOW_CONFIDENCE_DECISION")
        
        # Check audit trail completeness
        if not arbitration_result.audit_trail:
            flags.append("MISSING_AUDIT_TRAIL")
        elif len(arbitration_result.audit_trail) < 2:
            flags.append("INCOMPLETE_AUDIT_TRAIL")
        
        # Check processing time reasonableness
        if arbitration_result.processing_time_seconds > 3600:  # 1 hour
            flags.append("EXCESSIVE_PROCESSING_TIME")
        
        return flags

    def _store_record(self, record: ArbitrationRecord):
        """Store arbitration record to persistent storage"""
        filename = f"{record.record_id}.json"
        filepath = self.records_path / filename
        
        try:
            record_data = asdict(record)
            
            # Convert datetime objects to ISO strings
            record_data = self._serialize_datetime_fields(record_data)
            
            with open(filepath, 'w') as f:
                json.dump(record_data, f, indent=2, default=str)
            
            # Update storage metrics
            file_size = filepath.stat().st_size
            record.metadata["record_size_bytes"] = file_size
            
        except Exception as e:
            logger.error(f"Failed to store record {record.record_id}: {str(e)}")
            raise

    def _store_audit_event(self, event: AuditEvent):
        """Store audit event"""
        filename = f"{event.event_id}.json"
        filepath = self.audit_path / filename
        
        try:
            event_data = asdict(event)
            event_data = self._serialize_datetime_fields(event_data)
            
            with open(filepath, 'w') as f:
                json.dump(event_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to store audit event {event.event_id}: {str(e)}")

    def _store_compliance_report(self, report: ComplianceReport):
        """Store compliance report"""
        filename = f"{report.report_id}.json"
        filepath = self.compliance_path / filename
        
        try:
            report_data = asdict(report)
            report_data = self._serialize_datetime_fields(report_data)
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to store compliance report {report.report_id}: {str(e)}")

    def _serialize_datetime_fields(self, data):
        """Convert datetime objects to ISO strings recursively"""
        if isinstance(data, dict):
            return {k: self._serialize_datetime_fields(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_datetime_fields(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data

    def _load_arbitration_record(self, arbitration_id: str) -> Optional[ArbitrationRecord]:
        """Load arbitration record by arbitration ID"""
        for record_file in self.records_path.glob("*.json"):
            try:
                with open(record_file, 'r') as f:
                    record_data = json.load(f)
                
                if record_data.get("arbitration_id") == arbitration_id:
                    return self._deserialize_arbitration_record(record_data)
                    
            except Exception as e:
                logger.error(f"Error loading record from {record_file}: {str(e)}")
                continue
        
        return None

    def _load_arbitration_record_by_record_id(self, record_id: str) -> Optional[ArbitrationRecord]:
        """Load arbitration record by record ID"""
        record_file = self.records_path / f"{record_id}.json"
        
        if not record_file.exists():
            return None
        
        try:
            with open(record_file, 'r') as f:
                record_data = json.load(f)
            return self._deserialize_arbitration_record(record_data)
            
        except Exception as e:
            logger.error(f"Error loading record {record_id}: {str(e)}")
            return None

    def _load_record_from_file(self, filepath: Path) -> Optional[ArbitrationRecord]:
        """Load record from file path"""
        try:
            with open(filepath, 'r') as f:
                record_data = json.load(f)
            return self._deserialize_arbitration_record(record_data)
            
        except Exception as e:
            logger.error(f"Error loading record from {filepath}: {str(e)}")
            return None

    def _deserialize_arbitration_record(self, record_data: Dict[str, Any]) -> ArbitrationRecord:
        """Deserialize record data back to ArbitrationRecord object"""
        # This is a simplified deserialization - in production would need proper type reconstruction
        # For now, return as a simplified object with the key fields
        
        # Convert ISO strings back to datetime objects
        if isinstance(record_data.get('timestamp'), str):
            record_data['timestamp'] = datetime.fromisoformat(record_data['timestamp'])
        
        # Create a minimal record object (would need full reconstruction in production)
        return ArbitrationRecord(
            record_id=record_data.get('record_id', ''),
            record_type=RecordType(record_data.get('record_type', 'arbitration_decision')),
            arbitration_id=record_data.get('arbitration_id', ''),
            timestamp=record_data.get('timestamp', datetime.now()),
            arbitration_result=record_data.get('arbitration_result'),  # Would need proper deserialization
            conflict_analysis=record_data.get('conflict_analysis'),     # Would need proper deserialization
            escalation_results=record_data.get('escalation_results', []),
            original_context=record_data.get('original_context', {}),
            performance_metrics=record_data.get('performance_metrics', {}),
            compliance_flags=record_data.get('compliance_flags', []),
            verification_hash=record_data.get('verification_hash', ''),
            metadata=record_data.get('metadata', {})
        )

    def _matches_filters(self, record: ArbitrationRecord, filters: Dict[str, Any]) -> bool:
        """Check if record matches filter criteria"""
        if 'date_range' in filters:
            start_date, end_date = filters['date_range']
            if not (start_date <= record.timestamp <= end_date):
                return False
        
        if 'agent_id' in filters:
            # Would check if agent participated in arbitration
            pass
        
        if 'decision_type' in filters:
            # Would check decision type
            pass
        
        return True

    def _generate_audit_events(self, record: ArbitrationRecord):
        """Generate audit events for record creation"""
        self.record_audit_event(
            AuditEventType.DECISION_INITIATED,
            record.arbitration_id,
            event_data={"record_id": record.record_id},
            severity="info"
        )
        
        if record.compliance_flags:
            self.record_audit_event(
                AuditEventType.VALIDATION_FAILED,
                record.arbitration_id,
                event_data={"compliance_flags": record.compliance_flags},
                severity="warning"
            )

    def _update_recording_metrics(self, record: ArbitrationRecord):
        """Update recording system metrics"""
        self.recording_metrics["total_records"] += 1
        self.recording_metrics["records_by_type"][record.record_type.value] += 1
        
        # Update average record size
        if "record_size_bytes" in record.metadata:
            total_records = self.recording_metrics["total_records"]
            current_avg = self.recording_metrics["average_record_size_bytes"]
            new_size = record.metadata["record_size_bytes"]
            
            self.recording_metrics["average_record_size_bytes"] = (
                (current_avg * (total_records - 1) + new_size) / total_records
            )

    # Compliance verification methods
    def _verify_audit_trail_completeness(self, record: ArbitrationRecord) -> bool:
        """Verify audit trail is complete"""
        if not record.arbitration_result.audit_trail:
            return False
        
        # Check for key audit events
        required_events = ["resolution_successful", "escalation_step"]
        events_found = [event.get("action", "") for event in record.arbitration_result.audit_trail]
        
        return any(req in event for req in required_events for event in events_found)

    def _verify_decision_justification(self, record: ArbitrationRecord) -> bool:
        """Verify decision has adequate justification"""
        if not record.arbitration_result.final_decision:
            return False
        
        decision = record.arbitration_result.final_decision
        
        # Check reasoning length and supporting evidence
        reasoning_adequate = len(decision.reasoning) > 20  # Minimum reasoning length
        evidence_present = bool(decision.supporting_evidence)
        
        return reasoning_adequate and evidence_present

    def _verify_escalation_appropriateness(self, record: ArbitrationRecord) -> bool:
        """Verify escalation was appropriate for conflict severity"""
        severity = record.conflict_analysis.severity.value
        escalation_count = len(record.escalation_results)
        
        # Define minimum escalation steps by severity
        min_escalation_steps = {
            "low": 1,
            "medium": 2, 
            "high": 3,
            "critical": 3
        }
        
        required_steps = min_escalation_steps.get(severity, 2)
        return escalation_count >= required_steps

    def _verify_evidence_sufficiency(self, record: ArbitrationRecord) -> bool:
        """Verify evidence is sufficient for decision confidence"""
        if not record.arbitration_result.final_decision:
            return False
        
        confidence = record.arbitration_result.final_decision.confidence_score
        evidence_quality = record.conflict_analysis.evidence_summary.get("vote_summary", {}).get("evidence_quality_stats", {}).get("avg", 0.5)
        
        # Evidence is sufficient if both confidence and evidence quality are reasonable
        return confidence > 0.6 and evidence_quality > 0.5

    def _verify_audit_trail_consistency(self, record: ArbitrationRecord) -> bool:
        """Verify audit trail is internally consistent"""
        # Check timestamp ordering
        audit_events = record.arbitration_result.audit_trail
        
        for i in range(1, len(audit_events)):
            if 'timestamp' in audit_events[i-1] and 'timestamp' in audit_events[i]:
                prev_time = datetime.fromisoformat(audit_events[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(audit_events[i]['timestamp'])
                
                if curr_time < prev_time:
                    return False
        
        return True

    def _verify_decision_logic_consistency(self, record: ArbitrationRecord) -> bool:
        """Verify decision logic is consistent with process"""
        # Placeholder for decision logic verification
        # Would check if final decision aligns with escalation results
        return True

    def _generate_compliance_recommendations(self, record: ArbitrationRecord, violations: List[str]) -> List[str]:
        """Generate recommendations based on compliance violations"""
        recommendations = []
        
        for violation in violations:
            if "audit trail" in violation.lower():
                recommendations.append("Implement more comprehensive audit trail logging")
            elif "justification" in violation.lower():
                recommendations.append("Require more detailed reasoning for arbitration decisions")
            elif "escalation" in violation.lower():
                recommendations.append("Review escalation path appropriateness for conflict severity")
            elif "evidence" in violation.lower():
                recommendations.append("Establish minimum evidence quality thresholds")
        
        return recommendations

    def get_recording_metrics(self) -> Dict[str, Any]:
        """Get decision recording system metrics"""
        # Calculate storage usage
        storage_used = sum(
            f.stat().st_size for f in self.records_path.glob("*.json")
            if f.is_file()
        ) / (1024 * 1024)  # Convert to MB
        
        self.recording_metrics["storage_used_mb"] = storage_used
        
        return {
            **self.recording_metrics,
            "configuration": self.config,
            "storage_paths": {
                "records": str(self.records_path),
                "audit_events": str(self.audit_path),
                "compliance": str(self.compliance_path),
                "reports": str(self.reports_path)
            }
        }


def main():
    """Demonstration of decision recording functionality"""
    print("=== RIF Decision Recorder Demo ===\n")
    
    # Initialize recorder
    recorder = DecisionRecorder()
    
    print(f"Decision Recorder initialized")
    print(f"Records path: {recorder.records_path}")
    print(f"Audit path: {recorder.audit_path}")
    print(f"Compliance path: {recorder.compliance_path}")
    print()
    
    # Show recording metrics
    print("=== Recording System Metrics ===")
    metrics = recorder.get_recording_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict) and len(value) <= 10:
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif not isinstance(value, dict):
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()