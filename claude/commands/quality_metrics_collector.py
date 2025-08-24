#!/usr/bin/env python3
"""
Quality Gate Effectiveness Monitoring - Quality Metrics Collector
Issue #94 Phase 1: Core Monitoring Infrastructure

This module implements comprehensive metrics collection for quality gate decisions,
tracking effectiveness, false positive/negative rates, and production outcomes.

Architecture:
- Session-based data collection using Claude Code hooks
- Persistent storage in knowledge/quality_metrics/
- Real-time quality gate decision tracking
- Production defect correlation analysis
- Performance overhead monitoring (<50ms target)
"""

import json
import time
import hashlib
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class QualityDecisionType(Enum):
    """Types of quality gate decisions that can be tracked."""
    PASS = "pass"
    FAIL = "fail" 
    WARNING = "warning"
    SKIP = "skip"
    MANUAL_OVERRIDE = "manual_override"


class QualityGateType(Enum):
    """Types of quality gates in the system."""
    CODE_COVERAGE = "code_coverage"
    SECURITY_SCAN = "security_scan"
    LINTING = "linting"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    EVIDENCE_REQUIREMENTS = "evidence_requirements"
    RISK_ASSESSMENT = "risk_assessment"
    QUALITY_SCORE = "quality_score"


@dataclass
class QualityDecision:
    """Represents a single quality gate decision."""
    decision_id: str
    timestamp: str
    issue_number: int
    gate_type: str
    decision: str
    threshold_value: Optional[float]
    actual_value: Optional[float]
    context: Dict[str, Any]
    processing_time_ms: float
    agent_type: Optional[str]
    confidence_score: Optional[float]
    evidence: List[str]


@dataclass
class ProductionOutcome:
    """Tracks production outcomes for correlation analysis."""
    outcome_id: str
    timestamp: str
    issue_number: int
    change_id: str
    defect_count: int
    defect_severity: List[str]
    production_issues: List[Dict[str, Any]]
    time_to_defect_hours: Optional[float]
    customer_impact: Optional[str]


@dataclass
class EffectivenessMetrics:
    """Calculated effectiveness metrics for quality gates."""
    gate_type: str
    time_period_start: str
    time_period_end: str
    total_decisions: int
    pass_count: int
    fail_count: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy_percent: float
    precision_percent: float
    recall_percent: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    average_processing_time_ms: float


class QualityMetricsCollector:
    """
    Core component for collecting quality gate effectiveness metrics.
    
    Provides session-based metrics collection compatible with Claude Code's
    development workflow while maintaining persistent data between sessions.
    """
    
    def __init__(self, storage_path: str = "knowledge/quality_metrics"):
        """Initialize the quality metrics collector."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.session_id = self._generate_session_id()
        self.setup_logging()
        
        # Initialize storage subdirectories
        for subdir in ['recent', 'archive', 'patterns', 'reports', 'correlations']:
            (self.storage_path / subdir).mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging for quality metrics collection."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - QualityMetricsCollector - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{timestamp}-{time.time()}".encode()).hexdigest()[:12]
    
    def record_quality_decision(
        self, 
        issue_number: int,
        gate_type: QualityGateType,
        decision: QualityDecisionType,
        threshold_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        context: Dict[str, Any] = None,
        agent_type: Optional[str] = None,
        confidence_score: Optional[float] = None,
        evidence: List[str] = None
    ) -> str:
        """
        Record a quality gate decision for effectiveness tracking.
        
        Args:
            issue_number: GitHub issue number
            gate_type: Type of quality gate
            decision: Gate decision (pass/fail/warning/skip)
            threshold_value: Configured threshold for the gate
            actual_value: Actual measured value
            context: Additional context information
            agent_type: Agent that made the decision
            confidence_score: Confidence in the decision
            evidence: Supporting evidence for the decision
            
        Returns:
            Decision ID for tracking purposes
        """
        start_time = time.time()
        
        try:
            # Validate required inputs
            if issue_number is None:
                self.logger.error("Issue number is required for quality decision recording")
                return ""
            
            decision_record = QualityDecision(
                decision_id=self._generate_decision_id(issue_number, gate_type.value),
                timestamp=datetime.now().isoformat(),
                issue_number=issue_number,
                gate_type=gate_type.value,
                decision=decision.value,
                threshold_value=threshold_value,
                actual_value=actual_value,
                context=context or {},
                processing_time_ms=0,  # Will be updated below
                agent_type=agent_type,
                confidence_score=confidence_score,
                evidence=evidence or []
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            decision_record.processing_time_ms = processing_time
            
            # Store the decision
            self._store_decision(decision_record)
            
            # Log performance warning if too slow
            if processing_time > 50:
                self.logger.warning(
                    f"Quality decision recording took {processing_time:.2f}ms "
                    f"(threshold: 50ms) for issue #{issue_number}"
                )
            
            self.logger.info(
                f"Recorded quality decision: {decision.value} for "
                f"{gate_type.value} on issue #{issue_number}"
            )
            
            return decision_record.decision_id
            
        except Exception as e:
            self.logger.error(f"Error recording quality decision: {e}")
            return ""
    
    def track_production_outcome(
        self,
        issue_number: int,
        change_id: str,
        defects: List[Dict[str, Any]],
        time_to_defect_hours: Optional[float] = None,
        customer_impact: Optional[str] = None
    ) -> str:
        """
        Track production outcomes for correlation with quality decisions.
        
        Args:
            issue_number: GitHub issue number
            change_id: Unique identifier for the change
            defects: List of defects found in production
            time_to_defect_hours: Time from deployment to defect discovery
            customer_impact: Description of customer impact
            
        Returns:
            Outcome ID for tracking purposes
        """
        try:
            outcome_record = ProductionOutcome(
                outcome_id=self._generate_outcome_id(issue_number, change_id),
                timestamp=datetime.now().isoformat(),
                issue_number=issue_number,
                change_id=change_id,
                defect_count=len(defects),
                defect_severity=[d.get('severity', 'unknown') for d in defects],
                production_issues=defects,
                time_to_defect_hours=time_to_defect_hours,
                customer_impact=customer_impact
            )
            
            # Store the outcome
            self._store_outcome(outcome_record)
            
            self.logger.info(
                f"Tracked production outcome: {len(defects)} defects for "
                f"issue #{issue_number}, change {change_id}"
            )
            
            return outcome_record.outcome_id
            
        except Exception as e:
            self.logger.error(f"Error tracking production outcome: {e}")
            return ""
    
    def calculate_effectiveness_metrics(
        self,
        gate_type: QualityGateType,
        days_back: int = 30
    ) -> Optional[EffectivenessMetrics]:
        """
        Calculate effectiveness metrics for a specific quality gate type.
        
        Args:
            gate_type: Type of quality gate to analyze
            days_back: Number of days to look back for data
            
        Returns:
            Calculated effectiveness metrics or None if insufficient data
        """
        try:
            # Get time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Load decisions for the gate type and time period
            decisions = self._load_decisions_in_period(gate_type, start_time, end_time)
            if not decisions:
                self.logger.warning(f"No decisions found for {gate_type.value} in last {days_back} days")
                return None
            
            # Load production outcomes for correlation
            outcomes = self._load_outcomes_in_period(start_time, end_time)
            
            # Calculate confusion matrix
            tp, fp, tn, fn = self._calculate_confusion_matrix(decisions, outcomes)
            
            # Calculate derived metrics
            total_decisions = len(decisions)
            pass_count = len([d for d in decisions if d['decision'] == 'pass'])
            fail_count = len([d for d in decisions if d['decision'] == 'fail'])
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
            
            avg_processing_time = sum(d['processing_time_ms'] for d in decisions) / len(decisions)
            
            metrics = EffectivenessMetrics(
                gate_type=gate_type.value,
                time_period_start=start_time.isoformat(),
                time_period_end=end_time.isoformat(),
                total_decisions=total_decisions,
                pass_count=pass_count,
                fail_count=fail_count,
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                accuracy_percent=accuracy,
                precision_percent=precision,
                recall_percent=recall,
                f1_score=f1,
                false_positive_rate=fpr,
                false_negative_rate=fnr,
                average_processing_time_ms=avg_processing_time
            )
            
            # Store metrics
            self._store_metrics(metrics)
            
            self.logger.info(
                f"Calculated effectiveness metrics for {gate_type.value}: "
                f"Accuracy={accuracy:.1f}%, FPR={fpr:.1f}%, FNR={fnr:.1f}%"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating effectiveness metrics: {e}")
            return None
    
    def get_quality_gate_status(self, issue_number: int) -> Dict[str, Any]:
        """Get current quality gate status for an issue."""
        try:
            # Load recent decisions for the issue
            decisions = self._load_decisions_for_issue(issue_number)
            
            if not decisions:
                return {
                    'issue_number': issue_number,
                    'status': 'no_decisions',
                    'gates': {},
                    'overall_score': None
                }
            
            # Aggregate by gate type
            gate_status = {}
            for decision in decisions:
                gate_type = decision['gate_type']
                if gate_type not in gate_status:
                    gate_status[gate_type] = []
                gate_status[gate_type].append(decision)
            
            # Get latest decision for each gate
            latest_gates = {}
            for gate_type, gate_decisions in gate_status.items():
                latest = max(gate_decisions, key=lambda x: x['timestamp'])
                latest_gates[gate_type] = {
                    'decision': latest['decision'],
                    'timestamp': latest['timestamp'],
                    'threshold': latest.get('threshold_value'),
                    'actual': latest.get('actual_value'),
                    'confidence': latest.get('confidence_score')
                }
            
            # Calculate overall score
            pass_count = sum(1 for gate in latest_gates.values() if gate['decision'] == 'pass')
            total_gates = len(latest_gates)
            overall_score = (pass_count / total_gates * 100) if total_gates > 0 else 0
            
            return {
                'issue_number': issue_number,
                'status': 'active',
                'gates': latest_gates,
                'overall_score': overall_score,
                'total_decisions': len(decisions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality gate status: {e}")
            return {
                'issue_number': issue_number,
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_decision_id(self, issue_number: int, gate_type: str) -> str:
        """Generate unique decision ID."""
        timestamp = datetime.now().isoformat()
        data = f"{issue_number}-{gate_type}-{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _generate_outcome_id(self, issue_number: int, change_id: str) -> str:
        """Generate unique outcome ID."""
        timestamp = datetime.now().isoformat()
        data = f"{issue_number}-{change_id}-{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _store_decision(self, decision: QualityDecision) -> None:
        """Store quality decision to persistent storage."""
        # Store in recent directory (hot data)
        recent_file = self.storage_path / "recent" / f"decisions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(recent_file, 'a') as f:
            f.write(json.dumps(asdict(decision)) + '\n')
        
        # Also store session-specific data
        session_file = self.storage_path / "recent" / f"session_{self.session_id}.jsonl"
        with open(session_file, 'a') as f:
            f.write(json.dumps(asdict(decision)) + '\n')
    
    def _store_outcome(self, outcome: ProductionOutcome) -> None:
        """Store production outcome to persistent storage."""
        # Store in correlations directory
        correlation_file = self.storage_path / "correlations" / f"outcomes_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(correlation_file, 'a') as f:
            f.write(json.dumps(asdict(outcome)) + '\n')
    
    def _store_metrics(self, metrics: EffectivenessMetrics) -> None:
        """Store calculated metrics."""
        metrics_file = self.storage_path / "reports" / f"metrics_{metrics.gate_type}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def _load_decisions_in_period(self, gate_type: QualityGateType, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Load decisions for a gate type within a time period."""
        decisions = []
        
        # Check recent directory for files in the time period
        recent_dir = self.storage_path / "recent"
        if not recent_dir.exists():
            return decisions
        
        for decision_file in recent_dir.glob("decisions_*.jsonl"):
            try:
                with open(decision_file, 'r') as f:
                    for line in f:
                        decision = json.loads(line.strip())
                        decision_time = datetime.fromisoformat(decision['timestamp'])
                        
                        if (start_time <= decision_time <= end_time and 
                            decision['gate_type'] == gate_type.value):
                            decisions.append(decision)
            except Exception as e:
                self.logger.warning(f"Error reading decision file {decision_file}: {e}")
                continue
        
        return decisions
    
    def _load_outcomes_in_period(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Load production outcomes within a time period."""
        outcomes = []
        
        correlations_dir = self.storage_path / "correlations"
        if not correlations_dir.exists():
            return outcomes
        
        for outcome_file in correlations_dir.glob("outcomes_*.jsonl"):
            try:
                with open(outcome_file, 'r') as f:
                    for line in f:
                        outcome = json.loads(line.strip())
                        outcome_time = datetime.fromisoformat(outcome['timestamp'])
                        
                        if start_time <= outcome_time <= end_time:
                            outcomes.append(outcome)
            except Exception as e:
                self.logger.warning(f"Error reading outcome file {outcome_file}: {e}")
                continue
        
        return outcomes
    
    def _load_decisions_for_issue(self, issue_number: int) -> List[Dict[str, Any]]:
        """Load all decisions for a specific issue."""
        decisions = []
        
        recent_dir = self.storage_path / "recent"
        if not recent_dir.exists():
            return decisions
        
        for decision_file in recent_dir.glob("decisions_*.jsonl"):
            try:
                with open(decision_file, 'r') as f:
                    for line in f:
                        decision = json.loads(line.strip())
                        if decision['issue_number'] == issue_number:
                            decisions.append(decision)
            except Exception as e:
                self.logger.warning(f"Error reading decision file {decision_file}: {e}")
                continue
        
        return decisions
    
    def _calculate_confusion_matrix(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """
        Calculate confusion matrix by correlating decisions with outcomes.
        
        For quality gates:
        - True Positive (TP): Gate failed and there were production defects (correct rejection)
        - False Positive (FP): Gate failed but no production defects (incorrect rejection)
        - True Negative (TN): Gate passed and no production defects (correct acceptance)
        - False Negative (FN): Gate passed but there were production defects (incorrect acceptance)
        
        Returns:
            Tuple of (true_positives, false_positives, true_negatives, false_negatives)
        """
        tp = fp = tn = fn = 0
        
        # Create mapping of issue numbers to outcomes
        issue_outcomes = {}
        for outcome in outcomes:
            issue_num = outcome['issue_number']
            has_defects = outcome['defect_count'] > 0
            issue_outcomes[issue_num] = has_defects
        
        # Analyze each decision
        for decision in decisions:
            issue_num = decision['issue_number']
            decision_failed = decision['decision'] == 'fail'
            decision_passed = decision['decision'] == 'pass'
            
            # Get actual outcome (if available)
            has_production_defects = issue_outcomes.get(issue_num, None)
            
            if has_production_defects is None:
                # No production data available, skip this decision
                continue
            
            # Calculate confusion matrix values based on gate logic
            if decision_failed and has_production_defects:
                tp += 1  # Correctly rejected bad change
            elif decision_failed and not has_production_defects:
                fp += 1  # Incorrectly rejected good change
            elif decision_passed and not has_production_defects:
                tn += 1  # Correctly accepted good change
            elif decision_passed and has_production_defects:
                fn += 1  # Incorrectly accepted bad change
        
        return tp, fp, tn, fn


def main():
    """Command line interface for quality metrics collector."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_metrics_collector.py <command> [args]")
        print("Commands:")
        print("  record-decision <issue_number> <gate_type> <decision>")
        print("  track-outcome <issue_number> <change_id> <defect_count>")
        print("  calculate-metrics <gate_type> [days_back]")
        print("  get-status <issue_number>")
        return 1
    
    collector = QualityMetricsCollector()
    command = sys.argv[1]
    
    if command == "record-decision" and len(sys.argv) >= 5:
        issue_num = int(sys.argv[2])
        gate_type = QualityGateType(sys.argv[3])
        decision = QualityDecisionType(sys.argv[4])
        
        decision_id = collector.record_quality_decision(issue_num, gate_type, decision)
        print(f"Decision recorded: {decision_id}")
        
    elif command == "track-outcome" and len(sys.argv) >= 5:
        issue_num = int(sys.argv[2])
        change_id = sys.argv[3]
        defect_count = int(sys.argv[4])
        
        # Create mock defects
        defects = [{'severity': 'medium'} for _ in range(defect_count)]
        
        outcome_id = collector.track_production_outcome(issue_num, change_id, defects)
        print(f"Outcome tracked: {outcome_id}")
        
    elif command == "calculate-metrics" and len(sys.argv) >= 3:
        gate_type = QualityGateType(sys.argv[2])
        days_back = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        
        metrics = collector.calculate_effectiveness_metrics(gate_type, days_back)
        if metrics:
            print(json.dumps(asdict(metrics), indent=2))
        else:
            print("No metrics available")
        
    elif command == "get-status" and len(sys.argv) >= 3:
        issue_num = int(sys.argv[2])
        status = collector.get_quality_gate_status(issue_num)
        print(json.dumps(status, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())