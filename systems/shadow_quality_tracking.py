#!/usr/bin/env python3
"""
Shadow Quality Tracking System - Issue #142
DPIBS Validation Phase 1 Quality Tracking Implementation

Provides comprehensive quality monitoring, adversarial analysis, and evidence consolidation
for continuous parallel execution throughout the validation period (5-7 days).

Key Features:
- Context relevance validation (90% target)
- Benchmarking accuracy validation (85% target)
- Agent improvement metrics tracking
- System performance quality monitoring
- Adversarial evidence consolidation
- Real-time quality decision integration
"""

import os
import sys
import json
import time
import uuid
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import sqlite3
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add RIF to path
sys.path.insert(0, '/Users/cal/DEV/RIF')

from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
from knowledge.database.database_config import DatabaseConfig
from systems.dpibs_benchmarking_enhanced import DPIBSBenchmarkingEngine, create_dpibs_benchmarking_engine


class QualityMetricType(Enum):
    CONTEXT_RELEVANCE = "context_relevance"
    BENCHMARKING_ACCURACY = "benchmarking_accuracy"
    AGENT_IMPROVEMENT = "agent_improvement"
    SYSTEM_PERFORMANCE = "system_performance"
    ADVERSARIAL_FINDINGS = "adversarial_findings"
    QUALITY_DECISION = "quality_decision"


@dataclass
class QualityMetric:
    """Individual quality metric measurement"""
    id: str
    metric_type: QualityMetricType
    timestamp: datetime
    value: float
    target_value: float
    source: str  # Which agent/system generated this metric
    context: Dict[str, Any]
    evidence: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @property
    def target_met(self) -> bool:
        """Check if metric meets target threshold"""
        return self.value >= self.target_value
    
    @property
    def quality_score(self) -> float:
        """Calculate quality score (0.0 to 1.0)"""
        return min(self.value / self.target_value, 1.0) if self.target_value > 0 else 0.0


@dataclass
class QualitySession:
    """Quality tracking session for a specific validation period"""
    session_id: str
    issue_number: int
    start_time: datetime
    end_time: Optional[datetime]
    metrics: List[QualityMetric] = field(default_factory=list)
    adversarial_findings: List[Dict[str, Any]] = field(default_factory=list)
    quality_decisions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_hours(self) -> float:
        """Get session duration in hours"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() / 3600
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active"""
        return self.end_time is None
    
    def get_metrics_by_type(self, metric_type: QualityMetricType) -> List[QualityMetric]:
        """Get all metrics of a specific type"""
        return [m for m in self.metrics if m.metric_type == metric_type]
    
    def get_latest_metric(self, metric_type: QualityMetricType) -> Optional[QualityMetric]:
        """Get the most recent metric of a specific type"""
        type_metrics = self.get_metrics_by_type(metric_type)
        return max(type_metrics, key=lambda m: m.timestamp) if type_metrics else None


class ShadowQualityTracker:
    """
    Shadow Quality Tracking System
    
    Provides continuous quality monitoring with adversarial analysis
    and evidence consolidation throughout the validation period.
    """
    
    def __init__(self, database_path: str = "/Users/cal/DEV/RIF/knowledge/quality_tracking.db"):
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        self.active_sessions: Dict[str, QualitySession] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Quality targets from issue requirements
        self.quality_targets = {
            QualityMetricType.CONTEXT_RELEVANCE: 90.0,  # 90% target
            QualityMetricType.BENCHMARKING_ACCURACY: 85.0,  # 85% target
            QualityMetricType.AGENT_IMPROVEMENT: 75.0,  # Improvement threshold
            QualityMetricType.SYSTEM_PERFORMANCE: 99.9,  # 99.9% availability
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'query_response_ms': 200,  # Sub-200ms query response
            'update_cycle_minutes': 5,  # 5-minute update cycle
            'availability_percent': 99.9,  # 99.9% availability
        }
        
        self._setup_database()
        self._setup_monitoring()
    
    def _setup_database(self):
        """Initialize SQLite database for quality tracking"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        with sqlite3.connect(self.database_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS quality_sessions (
                    session_id TEXT PRIMARY KEY,
                    issue_number INTEGER,
                    start_time TEXT,
                    end_time TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    metric_type TEXT,
                    timestamp TEXT,
                    value REAL,
                    target_value REAL,
                    source TEXT,
                    context TEXT,
                    evidence TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES quality_sessions (session_id)
                );
                
                CREATE TABLE IF NOT EXISTS adversarial_findings (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    finding_type TEXT,
                    severity TEXT,
                    timestamp TEXT,
                    description TEXT,
                    evidence TEXT,
                    impact_assessment TEXT,
                    recommendations TEXT,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES quality_sessions (session_id)
                );
                
                CREATE TABLE IF NOT EXISTS quality_decisions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    decision_type TEXT,
                    timestamp TEXT,
                    decision_data TEXT,
                    rationale TEXT,
                    confidence_score REAL,
                    evidence_quality REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES quality_sessions (session_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_sessions_issue ON quality_sessions (issue_number);
                CREATE INDEX IF NOT EXISTS idx_metrics_session_type ON quality_metrics (session_id, metric_type);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON quality_metrics (timestamp);
                CREATE INDEX IF NOT EXISTS idx_findings_session ON adversarial_findings (session_id);
                CREATE INDEX IF NOT EXISTS idx_decisions_session ON quality_decisions (session_id);
            """)
    
    def _setup_monitoring(self):
        """Setup continuous monitoring infrastructure"""
        self.logger.info("Setting up shadow quality monitoring infrastructure")
        
        # Create monitoring directories
        monitoring_dirs = [
            "/Users/cal/DEV/RIF/knowledge/quality_monitoring",
            "/Users/cal/DEV/RIF/knowledge/quality_monitoring/metrics",
            "/Users/cal/DEV/RIF/knowledge/quality_monitoring/evidence",
            "/Users/cal/DEV/RIF/knowledge/quality_monitoring/dashboards",
            "/Users/cal/DEV/RIF/knowledge/quality_monitoring/alerts",
        ]
        
        for directory in monitoring_dirs:
            os.makedirs(directory, exist_ok=True)
    
    def start_quality_session(self, issue_number: int, session_metadata: Dict[str, Any] = None) -> str:
        """Start a new quality tracking session for an issue"""
        session_id = f"quality-{issue_number}-{int(time.time())}"
        
        session = QualitySession(
            session_id=session_id,
            issue_number=issue_number,
            start_time=datetime.now(),
            end_time=None,
            metadata=session_metadata or {}
        )
        
        # Store in database
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO quality_sessions 
                (session_id, issue_number, start_time, end_time, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                issue_number,
                session.start_time.isoformat(),
                None,
                json.dumps(session.metadata)
            ))
        
        self.active_sessions[session_id] = session
        self.logger.info(f"Started quality tracking session {session_id} for issue #{issue_number}")
        
        # Start continuous monitoring if not already active
        if not self.monitoring_active:
            self.start_monitoring()
        
        return session_id
    
    def end_quality_session(self, session_id: str) -> Dict[str, Any]:
        """End a quality tracking session and generate final report"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or already ended")
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        
        # Update database
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                UPDATE quality_sessions 
                SET end_time = ? 
                WHERE session_id = ?
            """, (session.end_time.isoformat(), session_id))
        
        # Generate final report
        final_report = self._generate_session_report(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        self.logger.info(f"Ended quality tracking session {session_id}")
        return final_report
    
    def record_quality_metric(self, session_id: str, metric_type: QualityMetricType, 
                            value: float, source: str, context: Dict[str, Any] = None,
                            evidence: Dict[str, Any] = None) -> str:
        """Record a quality metric measurement"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or ended")
        
        metric_id = f"metric-{uuid.uuid4().hex[:8]}"
        target_value = self.quality_targets.get(metric_type, 0.0)
        
        metric = QualityMetric(
            id=metric_id,
            metric_type=metric_type,
            timestamp=datetime.now(),
            value=value,
            target_value=target_value,
            source=source,
            context=context or {},
            evidence=evidence or {}
        )
        
        # Add to session
        self.active_sessions[session_id].metrics.append(metric)
        
        # Store in database
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO quality_metrics 
                (id, session_id, metric_type, timestamp, value, target_value, source, context, evidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric_id,
                session_id,
                metric_type.value,
                metric.timestamp.isoformat(),
                value,
                target_value,
                source,
                json.dumps(context or {}),
                json.dumps(evidence or {}),
                json.dumps(metric.metadata)
            ))
        
        # Check if metric triggers alerts
        self._check_quality_alerts(metric)
        
        self.logger.info(f"Recorded {metric_type.value} metric: {value:.2f} (target: {target_value:.2f})")
        return metric_id
    
    def record_adversarial_finding(self, session_id: str, finding_type: str, severity: str,
                                 description: str, evidence: Dict[str, Any],
                                 impact_assessment: str, recommendations: List[str]) -> str:
        """Record an adversarial analysis finding"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or ended")
        
        finding_id = f"finding-{uuid.uuid4().hex[:8]}"
        
        finding = {
            'id': finding_id,
            'finding_type': finding_type,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'evidence': evidence,
            'impact_assessment': impact_assessment,
            'recommendations': recommendations,
            'status': 'open'
        }
        
        # Add to session
        self.active_sessions[session_id].adversarial_findings.append(finding)
        
        # Store in database
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO adversarial_findings 
                (id, session_id, finding_type, severity, timestamp, description, evidence, 
                 impact_assessment, recommendations, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                finding_id,
                session_id,
                finding_type,
                severity,
                finding['timestamp'],
                description,
                json.dumps(evidence),
                impact_assessment,
                json.dumps(recommendations),
                'open'
            ))
        
        self.logger.warning(f"Recorded adversarial finding: {finding_type} ({severity})")
        return finding_id
    
    def record_quality_decision(self, session_id: str, decision_type: str,
                              decision_data: Dict[str, Any], rationale: str,
                              confidence_score: float, evidence_quality: float) -> str:
        """Record a quality-based decision"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or ended")
        
        decision_id = f"decision-{uuid.uuid4().hex[:8]}"
        
        decision = {
            'id': decision_id,
            'decision_type': decision_type,
            'timestamp': datetime.now().isoformat(),
            'decision_data': decision_data,
            'rationale': rationale,
            'confidence_score': confidence_score,
            'evidence_quality': evidence_quality
        }
        
        # Add to session
        self.active_sessions[session_id].quality_decisions.append(decision)
        
        # Store in database
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO quality_decisions 
                (id, session_id, decision_type, timestamp, decision_data, rationale, 
                 confidence_score, evidence_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id,
                session_id,
                decision_type,
                decision['timestamp'],
                json.dumps(decision_data),
                rationale,
                confidence_score,
                evidence_quality
            ))
        
        self.logger.info(f"Recorded quality decision: {decision_type} (confidence: {confidence_score:.2f})")
        return decision_id
    
    def start_monitoring(self):
        """Start continuous monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started continuous quality monitoring")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Stopped continuous quality monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system performance metrics
                self._collect_system_performance()
                
                # Check for quality threshold violations
                self._check_quality_thresholds()
                
                # Generate real-time dashboard data
                self._update_monitoring_dashboard()
                
                # Sleep for monitoring interval
                time.sleep(30)  # 30 second monitoring cycle
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Longer sleep on error
    
    def _collect_system_performance(self):
        """Collect system performance metrics"""
        for session_id, session in self.active_sessions.items():
            try:
                # Measure query response time
                start_time = time.time()
                # Simulate query operation
                self._test_system_responsiveness()
                query_time_ms = (time.time() - start_time) * 1000
                
                # Record query response metric
                self.record_quality_metric(
                    session_id,
                    QualityMetricType.SYSTEM_PERFORMANCE,
                    100.0 if query_time_ms <= self.performance_thresholds['query_response_ms'] else 
                    (self.performance_thresholds['query_response_ms'] / query_time_ms) * 100,
                    'system_monitor',
                    {'query_time_ms': query_time_ms, 'threshold_ms': self.performance_thresholds['query_response_ms']},
                    {'measurement_type': 'query_response', 'timestamp': datetime.now().isoformat()}
                )
                
            except Exception as e:
                self.logger.error(f"Error collecting performance metrics for session {session_id}: {e}")
    
    def _test_system_responsiveness(self):
        """Test system responsiveness"""
        # Simple database query to test responsiveness
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("SELECT COUNT(*) FROM quality_sessions").fetchone()
    
    def _check_quality_alerts(self, metric: QualityMetric):
        """Check if metric triggers quality alerts"""
        if not metric.target_met:
            alert_message = (f"Quality Alert: {metric.metric_type.value} = {metric.value:.2f} "
                           f"(target: {metric.target_value:.2f}) from {metric.source}")
            self.logger.warning(alert_message)
            
            # Write alert to file
            alert_file = f"/Users/cal/DEV/RIF/knowledge/quality_monitoring/alerts/alert_{int(time.time())}.json"
            with open(alert_file, 'w') as f:
                json.dump({
                    'alert_type': 'quality_threshold_violation',
                    'metric': metric.to_dict(),
                    'timestamp': datetime.now().isoformat(),
                    'message': alert_message
                }, f, indent=2)
    
    def _check_quality_thresholds(self):
        """Check all active sessions for quality threshold violations"""
        for session_id, session in self.active_sessions.items():
            # Check recent metrics for each type
            for metric_type in QualityMetricType:
                recent_metrics = [m for m in session.get_metrics_by_type(metric_type) 
                                if (datetime.now() - m.timestamp).seconds < 300]  # Last 5 minutes
                
                if recent_metrics:
                    avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
                    target = self.quality_targets.get(metric_type, 0.0)
                    
                    if avg_value < target * 0.9:  # 10% below target triggers alert
                        self.logger.warning(f"Quality degradation in session {session_id}: "
                                          f"{metric_type.value} = {avg_value:.2f} (target: {target:.2f})")
    
    def _update_monitoring_dashboard(self):
        """Update real-time monitoring dashboard"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(self.active_sessions),
            'session_summary': {}
        }
        
        for session_id, session in self.active_sessions.items():
            session_summary = {
                'issue_number': session.issue_number,
                'duration_hours': session.duration_hours,
                'total_metrics': len(session.metrics),
                'adversarial_findings': len(session.adversarial_findings),
                'quality_decisions': len(session.quality_decisions),
                'latest_metrics': {}
            }
            
            # Get latest metrics by type
            for metric_type in QualityMetricType:
                latest_metric = session.get_latest_metric(metric_type)
                if latest_metric:
                    session_summary['latest_metrics'][metric_type.value] = {
                        'value': latest_metric.value,
                        'target': latest_metric.target_value,
                        'target_met': latest_metric.target_met,
                        'quality_score': latest_metric.quality_score,
                        'timestamp': latest_metric.timestamp.isoformat()
                    }
            
            dashboard_data['session_summary'][session_id] = session_summary
        
        # Write dashboard data
        dashboard_file = "/Users/cal/DEV/RIF/knowledge/quality_monitoring/dashboards/current.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
    
    def _generate_session_report(self, session: QualitySession) -> Dict[str, Any]:
        """Generate comprehensive session report"""
        report = {
            'session_id': session.session_id,
            'issue_number': session.issue_number,
            'duration_hours': session.duration_hours,
            'start_time': session.start_time.isoformat(),
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'quality_summary': {},
            'adversarial_analysis': {},
            'quality_decisions_summary': {},
            'recommendations': [],
            'overall_assessment': {}
        }
        
        # Quality metrics summary
        for metric_type in QualityMetricType:
            type_metrics = session.get_metrics_by_type(metric_type)
            if type_metrics:
                values = [m.value for m in type_metrics]
                target = type_metrics[0].target_value
                
                report['quality_summary'][metric_type.value] = {
                    'total_measurements': len(type_metrics),
                    'average_value': sum(values) / len(values),
                    'min_value': min(values),
                    'max_value': max(values),
                    'target_value': target,
                    'target_achievement_rate': len([v for v in values if v >= target]) / len(values) * 100,
                    'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'stable'
                }
        
        # Adversarial findings summary
        if session.adversarial_findings:
            findings_by_severity = {}
            for finding in session.adversarial_findings:
                severity = finding['severity']
                if severity not in findings_by_severity:
                    findings_by_severity[severity] = []
                findings_by_severity[severity].append(finding)
            
            report['adversarial_analysis'] = {
                'total_findings': len(session.adversarial_findings),
                'findings_by_severity': {k: len(v) for k, v in findings_by_severity.items()},
                'critical_findings': len([f for f in session.adversarial_findings if f['severity'] == 'critical']),
                'high_findings': len([f for f in session.adversarial_findings if f['severity'] == 'high'])
            }
        
        # Quality decisions summary
        if session.quality_decisions:
            avg_confidence = sum(d['confidence_score'] for d in session.quality_decisions) / len(session.quality_decisions)
            avg_evidence_quality = sum(d['evidence_quality'] for d in session.quality_decisions) / len(session.quality_decisions)
            
            report['quality_decisions_summary'] = {
                'total_decisions': len(session.quality_decisions),
                'average_confidence': avg_confidence,
                'average_evidence_quality': avg_evidence_quality,
                'high_confidence_decisions': len([d for d in session.quality_decisions if d['confidence_score'] >= 0.8])
            }
        
        # Overall assessment
        context_relevance = session.get_latest_metric(QualityMetricType.CONTEXT_RELEVANCE)
        benchmarking_accuracy = session.get_latest_metric(QualityMetricType.BENCHMARKING_ACCURACY)
        
        overall_score = 0.0
        targets_met = []
        
        if context_relevance:
            targets_met.append(context_relevance.target_met)
            overall_score += context_relevance.quality_score * 0.4
        
        if benchmarking_accuracy:
            targets_met.append(benchmarking_accuracy.target_met)
            overall_score += benchmarking_accuracy.quality_score * 0.4
        
        # Add performance and improvement scores
        system_performance = session.get_latest_metric(QualityMetricType.SYSTEM_PERFORMANCE)
        if system_performance:
            overall_score += system_performance.quality_score * 0.2
        
        report['overall_assessment'] = {
            'overall_quality_score': overall_score,
            'primary_targets_met': all(targets_met) if targets_met else False,
            'context_relevance_target_met': context_relevance.target_met if context_relevance else False,
            'benchmarking_accuracy_target_met': benchmarking_accuracy.target_met if benchmarking_accuracy else False,
            'recommendation': self._generate_overall_recommendation(overall_score, targets_met, session.adversarial_findings)
        }
        
        return report
    
    def _generate_overall_recommendation(self, overall_score: float, targets_met: List[bool], 
                                       adversarial_findings: List[Dict]) -> str:
        """Generate overall go/no-go recommendation"""
        critical_findings = len([f for f in adversarial_findings if f.get('severity') == 'critical'])
        
        if critical_findings > 0:
            return "NO-GO: Critical adversarial findings must be addressed before proceeding"
        elif overall_score >= 0.85 and all(targets_met):
            return "GO: All quality targets met with high confidence"
        elif overall_score >= 0.75:
            return "CONDITIONAL GO: Quality targets mostly met, monitor closely"
        else:
            return "NO-GO: Quality targets not met, requires remediation"
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a quality tracking session"""
        if session_id not in self.active_sessions:
            # Try to load from database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("""
                    SELECT session_id, issue_number, start_time, end_time, metadata
                    FROM quality_sessions 
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return {'error': f'Session {session_id} not found'}
                
                return {
                    'session_id': row[0],
                    'issue_number': row[1],
                    'start_time': row[2],
                    'end_time': row[3],
                    'status': 'completed' if row[3] else 'unknown',
                    'metadata': json.loads(row[4]) if row[4] else {}
                }
        
        session = self.active_sessions[session_id]
        latest_metrics = {}
        
        for metric_type in QualityMetricType:
            latest_metric = session.get_latest_metric(metric_type)
            if latest_metric:
                latest_metrics[metric_type.value] = {
                    'value': latest_metric.value,
                    'target': latest_metric.target_value,
                    'target_met': latest_metric.target_met,
                    'quality_score': latest_metric.quality_score
                }
        
        return {
            'session_id': session_id,
            'issue_number': session.issue_number,
            'status': 'active',
            'duration_hours': session.duration_hours,
            'total_metrics': len(session.metrics),
            'adversarial_findings': len(session.adversarial_findings),
            'quality_decisions': len(session.quality_decisions),
            'latest_metrics': latest_metrics
        }
    
    def create_monitoring_dashboard(self) -> str:
        """Create HTML monitoring dashboard"""
        dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Shadow Quality Tracking Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .metric.good { border-color: #4CAF50; }
        .metric.warning { border-color: #FF9800; }
        .metric.critical { border-color: #F44336; }
        .session { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
        .refresh { margin: 10px 0; }
    </style>
    <script>
        function refreshDashboard() {
            location.reload();
        }
        setInterval(refreshDashboard, 30000); // Refresh every 30 seconds
    </script>
</head>
<body>
    <h1>Shadow Quality Tracking Dashboard</h1>
    <div class="refresh">
        <button onclick="refreshDashboard()">Refresh Now</button>
        <span>Auto-refresh every 30 seconds</span>
    </div>
    
    <h2>Active Sessions</h2>
"""
        
        for session_id, session in self.active_sessions.items():
            dashboard_html += f"""
    <div class="session">
        <h3>Session: {session_id} (Issue #{session.issue_number})</h3>
        <p>Duration: {session.duration_hours:.1f} hours</p>
        <p>Metrics: {len(session.metrics)} | Findings: {len(session.adversarial_findings)} | Decisions: {len(session.quality_decisions)}</p>
        
        <h4>Latest Quality Metrics</h4>
"""
            
            for metric_type in QualityMetricType:
                latest_metric = session.get_latest_metric(metric_type)
                if latest_metric:
                    status_class = 'good' if latest_metric.target_met else 'critical'
                    dashboard_html += f"""
        <div class="metric {status_class}">
            <strong>{metric_type.value}</strong>: {latest_metric.value:.2f} / {latest_metric.target_value:.2f}
            ({latest_metric.quality_score:.1%} quality score)
        </div>
"""
            
            dashboard_html += "</div>"
        
        dashboard_html += """
    <script>
        // Auto-refresh status
        document.title = "Quality Dashboard - Last Updated: " + new Date().toLocaleTimeString();
    </script>
</body>
</html>
"""
        
        dashboard_path = "/Users/cal/DEV/RIF/knowledge/quality_monitoring/dashboards/dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        return dashboard_path


# Integration functions for RIF system
def create_shadow_quality_tracker() -> ShadowQualityTracker:
    """Create shadow quality tracker instance"""
    return ShadowQualityTracker()


def integrate_with_dpibs(tracker: ShadowQualityTracker, optimizer: DPIBSPerformanceOptimizer) -> DPIBSBenchmarkingEngine:
    """Integrate shadow quality tracker with DPIBS benchmarking system"""
    engine = create_dpibs_benchmarking_engine(optimizer)
    
    # Monkey patch to add quality tracking
    original_benchmark = engine.benchmark_issue
    
    def enhanced_benchmark_issue(issue_number: int, include_evidence: bool = True, session_id: str = None):
        # Get or create quality session
        if not session_id:
            session_id = tracker.start_quality_session(issue_number, {
                'integration_type': 'dpibs_benchmarking',
                'benchmarking_mode': 'enhanced'
            })
        
        # Run original benchmarking
        result = original_benchmark(issue_number, include_evidence)
        
        # Record quality metrics
        tracker.record_quality_metric(
            session_id,
            QualityMetricType.BENCHMARKING_ACCURACY,
            result.nlp_accuracy_score * 100,
            'dpibs_benchmarking',
            {'issue_number': issue_number, 'analysis_duration_ms': result.analysis_duration_ms},
            {'benchmarking_result': result.to_dict()}
        )
        
        # Record context relevance if available
        if hasattr(result, 'knowledge_integration_data'):
            context_relevance = result.knowledge_integration_data.get('patterns_identified', 0) * 20  # Scale to percentage
            tracker.record_quality_metric(
                session_id,
                QualityMetricType.CONTEXT_RELEVANCE,
                min(context_relevance, 100),
                'dpibs_benchmarking',
                {'patterns_found': result.knowledge_integration_data.get('patterns_identified', 0)},
                {'integration_data': result.knowledge_integration_data}
            )
        
        return result
    
    engine.benchmark_issue = enhanced_benchmark_issue
    return engine


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔍 Shadow Quality Tracking System - Issue #142")
    print("=" * 60)
    
    # Create tracker
    tracker = create_shadow_quality_tracker()
    
    # Start quality session for issue #142
    session_id = tracker.start_quality_session(142, {
        'validation_phase': 1,
        'expected_duration_days': 7,
        'quality_focus': 'DPIBS_validation'
    })
    
    print(f"✅ Started quality tracking session: {session_id}")
    
    # Record some sample metrics
    tracker.record_quality_metric(
        session_id,
        QualityMetricType.CONTEXT_RELEVANCE,
        92.5,
        'test_agent',
        {'test_scenario': 'initial_validation'},
        {'measurement_details': 'Simulated context relevance test'}
    )
    
    tracker.record_quality_metric(
        session_id,
        QualityMetricType.BENCHMARKING_ACCURACY,
        87.3,
        'dpibs_benchmarking',
        {'benchmark_type': 'enhanced_nlp'},
        {'accuracy_breakdown': {'nlp': 87.3, 'evidence': 91.2}}
    )
    
    # Record adversarial finding
    tracker.record_adversarial_finding(
        session_id,
        'performance_degradation',
        'medium',
        'Query response time occasionally exceeds 200ms under load',
        {'max_response_time_ms': 234, 'avg_response_time_ms': 186},
        'May impact user experience during peak usage',
        ['Implement query optimization', 'Add caching layer', 'Monitor query performance']
    )
    
    print("📊 Recorded sample quality metrics and adversarial finding")
    
    # Get session status
    status = tracker.get_session_status(session_id)
    print(f"📈 Session Status: {json.dumps(status, indent=2)}")
    
    # Create monitoring dashboard
    dashboard_path = tracker.create_monitoring_dashboard()
    print(f"📋 Created monitoring dashboard: {dashboard_path}")
    
    # Start continuous monitoring for demo
    print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
    try:
        tracker.start_monitoring()
        time.sleep(60)  # Monitor for 1 minute in demo
    except KeyboardInterrupt:
        print("\n⏹️  Stopping monitoring")
    finally:
        tracker.stop_monitoring()
        
        # End session and generate report
        final_report = tracker.end_quality_session(session_id)
        print(f"📊 Final Report: {json.dumps(final_report, indent=2)}")