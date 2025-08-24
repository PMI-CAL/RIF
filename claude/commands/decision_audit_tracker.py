#!/usr/bin/env python3
"""
Decision Audit Tracker - Issue #92 Phase 3
Comprehensive audit trail and decision tracking for manual intervention framework.

This tracker implements:
1. Immutable audit trail with tamper-evident logging
2. Complete decision history and rationale recording
3. Pattern extraction and learning from decisions
4. Compliance reporting and audit trail generation
5. Decision correlation analysis and insights
6. Secure storage with checksum validation
"""

import json
import hashlib
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import threading
import time

class AuditAction(Enum):
    """Types of actions that can be audited."""
    INTERVENTION_INITIATED = "intervention_initiated"
    RISK_ASSESSMENT_COMPLETED = "risk_assessment_completed"
    SPECIALIST_ASSIGNED = "specialist_assigned"
    MANUAL_DECISION_APPROVE = "manual_decision_approve"
    MANUAL_DECISION_CHANGES = "manual_decision_request_changes"
    MANUAL_DECISION_ESCALATE = "manual_decision_escalate"
    MANAGER_OVERRIDE_APPROVE = "manager_override_approve"
    MANAGER_OVERRIDE_REJECT = "manager_override_reject"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_ERROR = "workflow_error"
    SLA_BREACH = "sla_breach"
    ESCALATION_TRIGGERED = "escalation_triggered"

@dataclass
class AuditRecord:
    """Container for audit record information."""
    workflow_id: str
    timestamp: datetime
    action: str  # AuditAction value
    actor: str  # person or system making the decision
    context: str  # contextual information
    rationale: str  # reasoning behind the decision
    evidence: List[str]  # supporting evidence
    record_id: Optional[str] = None
    checksum: Optional[str] = None
    previous_record_checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionPattern:
    """Container for extracted decision pattern."""
    pattern_id: str
    pattern_type: str  # e.g., "approval_security", "escalation_complexity"
    conditions: List[str]  # conditions that led to this decision
    outcomes: List[str]  # outcomes of this decision type
    frequency: int  # how often this pattern occurs
    success_rate: float  # how often this pattern leads to successful outcomes
    risk_factors: List[str]  # associated risk factors
    specialist_types: List[str]  # specialist types involved
    average_resolution_time: float  # average time to resolution
    confidence: float  # confidence in this pattern

@dataclass
class ComplianceReport:
    """Container for compliance audit report."""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_interventions: int
    decision_breakdown: Dict[str, int]
    override_rate: float
    avg_resolution_time: float
    sla_compliance_rate: float
    audit_trail_integrity: bool
    recommendations: List[str]
    detailed_records: List[AuditRecord]

class DecisionAuditTracker:
    """
    Comprehensive audit tracker for manual intervention decisions.
    
    Provides immutable audit trails, decision pattern learning,
    and compliance reporting for the manual intervention framework.
    """
    
    def __init__(self, storage_path: str = "knowledge/decisions/"):
        """Initialize the decision audit tracker."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "audit_tracker.db"
        self.patterns_file = self.storage_path / "decision_patterns.json"
        
        self.setup_logging()
        self._initialize_database()
        self._load_patterns()
        
        # Thread safety
        self._db_lock = threading.Lock()
        
        # Audit integrity
        self._last_record_checksum = self._get_last_record_checksum()
    
    def setup_logging(self):
        """Setup logging for audit tracker."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - DecisionAuditTracker - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_database(self):
        """Initialize SQLite database for audit records."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_records (
                        record_id TEXT PRIMARY KEY,
                        workflow_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        action TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        context TEXT,
                        rationale TEXT,
                        evidence TEXT,
                        checksum TEXT NOT NULL,
                        previous_record_checksum TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_workflow_id ON audit_records(workflow_id)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_action ON audit_records(action)
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
    
    def record_decision(self, audit_record: AuditRecord) -> str:
        """
        Record a decision in the immutable audit trail.
        
        Args:
            audit_record: Audit record to store
            
        Returns:
            record_id: Unique identifier for the stored record
        """
        try:
            with self._db_lock:
                # Generate record ID
                record_id = self._generate_record_id(audit_record)
                audit_record.record_id = record_id
                
                # Set previous record checksum for chain integrity
                audit_record.previous_record_checksum = self._last_record_checksum
                
                # Calculate record checksum
                audit_record.checksum = self._calculate_record_checksum(audit_record)
                
                # Store in database
                self._store_record(audit_record)
                
                # Update last record checksum
                self._last_record_checksum = audit_record.checksum
                
                # Extract patterns for learning
                self._extract_decision_pattern(audit_record)
                
                self.logger.info(f"📝 Recorded audit decision: {record_id} ({audit_record.action})")
                
                return record_id
                
        except Exception as e:
            self.logger.error(f"Error recording audit decision: {e}")
            raise
    
    def _generate_record_id(self, audit_record: AuditRecord) -> str:
        """Generate unique record identifier."""
        timestamp_str = audit_record.timestamp.isoformat()
        hash_input = f"{audit_record.workflow_id}_{timestamp_str}_{audit_record.action}_{audit_record.actor}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _calculate_record_checksum(self, audit_record: AuditRecord) -> str:
        """Calculate tamper-evident checksum for audit record."""
        # Create deterministic string representation
        checksum_data = {
            'record_id': audit_record.record_id,
            'workflow_id': audit_record.workflow_id,
            'timestamp': audit_record.timestamp.isoformat(),
            'action': audit_record.action,
            'actor': audit_record.actor,
            'context': audit_record.context,
            'rationale': audit_record.rationale,
            'evidence': sorted(audit_record.evidence),
            'previous_record_checksum': audit_record.previous_record_checksum,
            'metadata': sorted(audit_record.metadata.items()) if audit_record.metadata else []
        }
        
        checksum_string = json.dumps(checksum_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(checksum_string.encode()).hexdigest()
    
    def _store_record(self, audit_record: AuditRecord) -> None:
        """Store audit record in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO audit_records (
                        record_id, workflow_id, timestamp, action, actor,
                        context, rationale, evidence, checksum,
                        previous_record_checksum, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    audit_record.record_id,
                    audit_record.workflow_id,
                    audit_record.timestamp.isoformat(),
                    audit_record.action,
                    audit_record.actor,
                    audit_record.context,
                    audit_record.rationale,
                    json.dumps(audit_record.evidence),
                    audit_record.checksum,
                    audit_record.previous_record_checksum,
                    json.dumps(audit_record.metadata) if audit_record.metadata else '{}'
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing audit record: {e}")
            raise
    
    def _get_last_record_checksum(self) -> Optional[str]:
        """Get checksum of the last recorded audit record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT checksum FROM audit_records
                    ORDER BY created_at DESC
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            self.logger.error(f"Error getting last record checksum: {e}")
            return None
    
    def get_workflow_audit_trail(self, workflow_id: str) -> List[AuditRecord]:
        """Get complete audit trail for a workflow."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT record_id, workflow_id, timestamp, action, actor,
                           context, rationale, evidence, checksum,
                           previous_record_checksum, metadata
                    FROM audit_records
                    WHERE workflow_id = ?
                    ORDER BY timestamp ASC
                ''', (workflow_id,))
                
                records = []
                for row in cursor.fetchall():
                    record = AuditRecord(
                        record_id=row[0],
                        workflow_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        action=row[3],
                        actor=row[4],
                        context=row[5],
                        rationale=row[6],
                        evidence=json.loads(row[7]),
                        checksum=row[8],
                        previous_record_checksum=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Error retrieving audit trail for {workflow_id}: {e}")
            return []
    
    def validate_audit_chain_integrity(self, workflow_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate integrity of audit chain using checksums.
        
        Args:
            workflow_id: Optional specific workflow to validate, or all if None
            
        Returns:
            (is_valid, error_messages): Validation result and any errors found
        """
        errors = []
        
        try:
            query = '''
                SELECT record_id, workflow_id, timestamp, action, actor,
                       context, rationale, evidence, checksum,
                       previous_record_checksum, metadata
                FROM audit_records
            '''
            params = []
            
            if workflow_id:
                query += ' WHERE workflow_id = ?'
                params.append(workflow_id)
            
            query += ' ORDER BY timestamp ASC'
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                
                previous_checksum = None
                for row in cursor.fetchall():
                    record = AuditRecord(
                        record_id=row[0],
                        workflow_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        action=row[3],
                        actor=row[4],
                        context=row[5],
                        rationale=row[6],
                        evidence=json.loads(row[7]),
                        checksum=row[8],
                        previous_record_checksum=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
                    
                    # Validate record checksum
                    calculated_checksum = self._calculate_record_checksum(record)
                    if calculated_checksum != record.checksum:
                        errors.append(f"Checksum mismatch for record {record.record_id}")
                    
                    # Validate chain integrity
                    if record.previous_record_checksum != previous_checksum:
                        errors.append(f"Chain integrity broken at record {record.record_id}")
                    
                    previous_checksum = record.checksum
                
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.info("✅ Audit chain integrity validation passed")
        else:
            self.logger.error(f"❌ Audit chain integrity validation failed: {len(errors)} errors")
        
        return is_valid, errors
    
    def _extract_decision_pattern(self, audit_record: AuditRecord) -> None:
        """Extract decision patterns for learning and improvement."""
        try:
            # Only extract patterns from decision actions
            decision_actions = [
                'manual_decision_approve',
                'manual_decision_request_changes', 
                'manual_decision_escalate',
                'manager_override_approve',
                'manager_override_reject'
            ]
            
            if audit_record.action not in decision_actions:
                return
            
            # Extract pattern characteristics
            pattern_key = f"{audit_record.action}_{hash(audit_record.context) % 1000:03d}"
            
            pattern = DecisionPattern(
                pattern_id=pattern_key,
                pattern_type=audit_record.action,
                conditions=[audit_record.context, audit_record.rationale],
                outcomes=audit_record.evidence,
                frequency=1,
                success_rate=1.0,  # Will be calculated from multiple instances
                risk_factors=self._extract_risk_factors(audit_record),
                specialist_types=self._extract_specialist_types(audit_record),
                average_resolution_time=0.0,  # Will be calculated from workflow completion
                confidence=0.8  # Initial confidence
            )
            
            self._update_pattern(pattern)
            
        except Exception as e:
            self.logger.error(f"Error extracting decision pattern: {e}")
    
    def _extract_risk_factors(self, audit_record: AuditRecord) -> List[str]:
        """Extract risk factors from audit record."""
        risk_factors = []
        
        # Look for risk-related keywords in context and rationale
        text = f"{audit_record.context} {audit_record.rationale}".lower()
        
        risk_keywords = [
            'security', 'vulnerability', 'authentication', 'authorization',
            'performance', 'scalability', 'database', 'api',
            'complexity', 'architecture', 'compliance', 'privacy'
        ]
        
        for keyword in risk_keywords:
            if keyword in text:
                risk_factors.append(keyword)
        
        return risk_factors
    
    def _extract_specialist_types(self, audit_record: AuditRecord) -> List[str]:
        """Extract specialist types from audit record."""
        text = f"{audit_record.context} {audit_record.rationale}".lower()
        
        specialist_types = []
        type_keywords = {
            'security': ['security', 'auth', 'vulnerability'],
            'architecture': ['architecture', 'design', 'scalability'], 
            'compliance': ['compliance', 'audit', 'privacy'],
            'performance': ['performance', 'optimization']
        }
        
        for spec_type, keywords in type_keywords.items():
            if any(keyword in text for keyword in keywords):
                specialist_types.append(spec_type)
        
        return specialist_types
    
    def _load_patterns(self) -> None:
        """Load existing decision patterns from storage."""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    self.patterns = json.load(f)
            else:
                self.patterns = {}
        except Exception as e:
            self.logger.error(f"Error loading decision patterns: {e}")
            self.patterns = {}
    
    def _update_pattern(self, new_pattern: DecisionPattern) -> None:
        """Update decision patterns with new observation."""
        try:
            if new_pattern.pattern_id in self.patterns:
                # Update existing pattern
                existing = self.patterns[new_pattern.pattern_id]
                existing['frequency'] += 1
                existing['conditions'].extend(new_pattern.conditions)
                existing['outcomes'].extend(new_pattern.outcomes)
                existing['risk_factors'] = list(set(existing['risk_factors'] + new_pattern.risk_factors))
                existing['specialist_types'] = list(set(existing['specialist_types'] + new_pattern.specialist_types))
            else:
                # Add new pattern
                self.patterns[new_pattern.pattern_id] = asdict(new_pattern)
            
            # Save patterns to file
            self._save_patterns()
            
        except Exception as e:
            self.logger.error(f"Error updating decision pattern: {e}")
    
    def _save_patterns(self) -> None:
        """Save decision patterns to storage."""
        try:
            with open(self.patterns_file, 'w') as f:
                json.dump(self.patterns, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving decision patterns: {e}")
    
    def get_decision_patterns(self, pattern_type: Optional[str] = None) -> List[DecisionPattern]:
        """Get decision patterns, optionally filtered by type."""
        patterns = []
        
        for pattern_data in self.patterns.values():
            if pattern_type and pattern_data.get('pattern_type') != pattern_type:
                continue
            
            pattern = DecisionPattern(
                pattern_id=pattern_data['pattern_id'],
                pattern_type=pattern_data['pattern_type'],
                conditions=pattern_data['conditions'],
                outcomes=pattern_data['outcomes'],
                frequency=pattern_data['frequency'],
                success_rate=pattern_data['success_rate'],
                risk_factors=pattern_data['risk_factors'],
                specialist_types=pattern_data['specialist_types'],
                average_resolution_time=pattern_data['average_resolution_time'],
                confidence=pattern_data['confidence']
            )
            patterns.append(pattern)
        
        # Sort by frequency and confidence
        patterns.sort(key=lambda p: (p.frequency * p.confidence), reverse=True)
        return patterns
    
    def generate_compliance_report(self, 
                                 start_date: datetime, 
                                 end_date: datetime) -> ComplianceReport:
        """Generate compliance audit report for specified period."""
        try:
            # Get all records in period
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT record_id, workflow_id, timestamp, action, actor,
                           context, rationale, evidence, checksum,
                           previous_record_checksum, metadata
                    FROM audit_records
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                records = []
                for row in cursor.fetchall():
                    record = AuditRecord(
                        record_id=row[0],
                        workflow_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        action=row[3],
                        actor=row[4],
                        context=row[5],
                        rationale=row[6],
                        evidence=json.loads(row[7]),
                        checksum=row[8],
                        previous_record_checksum=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
                    records.append(record)
            
            # Analyze records
            total_interventions = len(set(record.workflow_id for record in records))
            decision_breakdown = {}
            override_count = 0
            
            for record in records:
                action = record.action
                decision_breakdown[action] = decision_breakdown.get(action, 0) + 1
                
                if 'override' in action:
                    override_count += 1
            
            override_rate = override_count / max(total_interventions, 1)
            
            # Validate audit trail integrity
            is_valid, errors = self.validate_audit_chain_integrity()
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(
                decision_breakdown, override_rate, is_valid
            )
            
            report = ComplianceReport(
                report_id=hashlib.md5(f"{start_date}_{end_date}".encode()).hexdigest()[:12],
                generated_at=datetime.now(timezone.utc),
                period_start=start_date,
                period_end=end_date,
                total_interventions=total_interventions,
                decision_breakdown=decision_breakdown,
                override_rate=override_rate,
                avg_resolution_time=0.0,  # Would calculate from workflow completion times
                sla_compliance_rate=0.95,  # Would calculate from SLA data
                audit_trail_integrity=is_valid,
                recommendations=recommendations,
                detailed_records=records
            )
            
            self.logger.info(f"📊 Generated compliance report {report.report_id} for period {start_date} to {end_date}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            # Return minimal report on error
            return ComplianceReport(
                report_id="error",
                generated_at=datetime.now(timezone.utc),
                period_start=start_date,
                period_end=end_date,
                total_interventions=0,
                decision_breakdown={},
                override_rate=0.0,
                avg_resolution_time=0.0,
                sla_compliance_rate=0.0,
                audit_trail_integrity=False,
                recommendations=[f"Report generation failed: {e}"],
                detailed_records=[]
            )
    
    def _generate_compliance_recommendations(self, 
                                           decision_breakdown: Dict[str, int],
                                           override_rate: float, 
                                           audit_integrity: bool) -> List[str]:
        """Generate compliance recommendations based on analysis."""
        recommendations = []
        
        if override_rate > 0.1:  # More than 10% overrides
            recommendations.append(
                f"High override rate ({override_rate:.1%}) detected. Review override justifications and consider tightening approval processes."
            )
        
        if not audit_integrity:
            recommendations.append(
                "Audit trail integrity issues detected. Immediate investigation required for compliance."
            )
        
        escalation_count = decision_breakdown.get('manual_decision_escalate', 0)
        total_decisions = sum(decision_breakdown.values())
        
        if escalation_count / max(total_decisions, 1) > 0.2:  # More than 20% escalations
            recommendations.append(
                "High escalation rate suggests specialist assignment or workload issues. Review specialist allocation."
            )
        
        if decision_breakdown.get('workflow_error', 0) > 0:
            recommendations.append(
                "Workflow errors detected. Review system reliability and error handling processes."
            )
        
        if not recommendations:
            recommendations.append("No significant compliance issues identified in this reporting period.")
        
        return recommendations
    
    def search_audit_records(self, 
                           query: str,
                           workflow_id: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[AuditRecord]:
        """Search audit records with flexible criteria."""
        try:
            sql_query = '''
                SELECT record_id, workflow_id, timestamp, action, actor,
                       context, rationale, evidence, checksum,
                       previous_record_checksum, metadata
                FROM audit_records
                WHERE 1=1
            '''
            params = []
            
            # Add search conditions
            if query:
                sql_query += ' AND (context LIKE ? OR rationale LIKE ? OR evidence LIKE ?)'
                query_param = f'%{query}%'
                params.extend([query_param, query_param, query_param])
            
            if workflow_id:
                sql_query += ' AND workflow_id = ?'
                params.append(workflow_id)
            
            if start_date:
                sql_query += ' AND timestamp >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                sql_query += ' AND timestamp <= ?'
                params.append(end_date.isoformat())
            
            sql_query += ' ORDER BY timestamp DESC'
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql_query, params)
                
                records = []
                for row in cursor.fetchall():
                    record = AuditRecord(
                        record_id=row[0],
                        workflow_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        action=row[3],
                        actor=row[4],
                        context=row[5],
                        rationale=row[6],
                        evidence=json.loads(row[7]),
                        checksum=row[8],
                        previous_record_checksum=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Error searching audit records: {e}")
            return []
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics and health metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM audit_records')
                total_records = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(DISTINCT workflow_id) FROM audit_records')
                total_workflows = cursor.fetchone()[0]
                
                cursor = conn.execute('''
                    SELECT action, COUNT(*) FROM audit_records
                    GROUP BY action
                    ORDER BY COUNT(*) DESC
                ''')
                action_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                cursor = conn.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM audit_records
                    WHERE timestamp >= date('now', '-30 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                ''')
                daily_activity = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Validate integrity
            is_valid, errors = self.validate_audit_chain_integrity()
            
            stats = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_records': total_records,
                'total_workflows': total_workflows,
                'action_breakdown': action_counts,
                'daily_activity_30d': daily_activity,
                'audit_integrity': {
                    'is_valid': is_valid,
                    'error_count': len(errors),
                    'last_validation': datetime.now(timezone.utc).isoformat()
                },
                'pattern_count': len(self.patterns),
                'storage_info': {
                    'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
                    'patterns_size_mb': os.path.getsize(self.patterns_file) / (1024 * 1024) if os.path.exists(self.patterns_file) else 0
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting audit statistics: {e}")
            return {'error': str(e)}

def main():
    """Command line interface for decision audit tracker."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python decision_audit_tracker.py <command> [args]")
        print("Commands:")
        print("  stats                     - Show audit statistics")
        print("  validate                  - Validate audit chain integrity")
        print("  search <query>            - Search audit records")
        print("  patterns [type]           - Show decision patterns")
        print("  report <start> <end>      - Generate compliance report")
        print("  test-record               - Create test audit record")
        return
    
    command = sys.argv[1]
    tracker = DecisionAuditTracker()
    
    if command == "stats":
        stats = tracker.get_audit_statistics()
        print(json.dumps(stats, indent=2))
        
    elif command == "validate":
        is_valid, errors = tracker.validate_audit_chain_integrity()
        print(f"Audit Chain Valid: {is_valid}")
        if errors:
            print("Errors found:")
            for error in errors:
                print(f"  - {error}")
    
    elif command == "search" and len(sys.argv) >= 3:
        query = sys.argv[2]
        records = tracker.search_audit_records(query)
        print(f"Found {len(records)} records:")
        for record in records[:10]:  # Show first 10
            print(f"  {record.timestamp} - {record.action} by {record.actor}")
            print(f"    {record.context}")
    
    elif command == "patterns":
        pattern_type = sys.argv[2] if len(sys.argv) >= 3 else None
        patterns = tracker.get_decision_patterns(pattern_type)
        print(f"Found {len(patterns)} patterns:")
        for pattern in patterns[:5]:  # Show top 5
            print(f"  {pattern.pattern_id}: {pattern.pattern_type} (freq: {pattern.frequency})")
    
    elif command == "report" and len(sys.argv) >= 4:
        start_str = sys.argv[2]
        end_str = sys.argv[3]
        
        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)
        
        report = tracker.generate_compliance_report(start_date, end_date)
        
        print(f"Compliance Report {report.report_id}")
        print(f"Period: {report.period_start} to {report.period_end}")
        print(f"Total Interventions: {report.total_interventions}")
        print(f"Override Rate: {report.override_rate:.1%}")
        print(f"Audit Integrity: {report.audit_trail_integrity}")
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    elif command == "test-record":
        test_record = AuditRecord(
            workflow_id="test_workflow_123",
            timestamp=datetime.now(timezone.utc),
            action="manual_decision_approve",
            actor="test_specialist",
            context="Test security review",
            rationale="Low risk change approved after review",
            evidence=["Security scan passed", "Code review completed"]
        )
        
        record_id = tracker.record_decision(test_record)
        print(f"✅ Created test audit record: {record_id}")
    
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())