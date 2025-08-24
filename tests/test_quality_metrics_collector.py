#!/usr/bin/env python3
"""
Test suite for Quality Metrics Collector
Issue #94: Quality Gate Effectiveness Monitoring
"""

import os
import sys
import json
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Add the claude/commands directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'claude', 'commands'))

from quality_metrics_collector import (
    QualityMetricsCollector,
    QualityDecision,
    ProductionOutcome,
    QualityGateType,
    QualityDecisionType
)

class TestQualityMetricsCollector:
    """Test suite for QualityMetricsCollector class."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = QualityMetricsCollector(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_collector_initialization(self):
        """Test collector initializes correctly."""
        assert self.collector.storage_path.exists()
        
        # Check subdirectories are created
        expected_dirs = ['recent', 'archive', 'patterns', 'reports', 'correlations']
        for subdir in expected_dirs:
            assert (self.collector.storage_path / subdir).exists()
        
        # Check session ID is generated
        assert self.collector.session_id is not None
        assert len(self.collector.session_id) == 12
    
    def test_record_quality_decision(self):
        """Test recording a quality gate decision."""
        # Test basic decision recording
        decision_id = self.collector.record_quality_decision(
            issue_number=123,
            gate_type=QualityGateType.CODE_COVERAGE,
            decision=QualityDecisionType.PASS,
            threshold_value=80.0,
            actual_value=85.0,
            context={'component_type': 'business_logic'},
            agent_type='rif-validator',
            confidence_score=0.95,
            evidence=['Coverage meets minimum requirements', 'All tests passing']
        )
        
        assert decision_id != ""
        assert len(decision_id) == 16
        
        # Check decision file was created
        recent_files = list((self.collector.storage_path / "recent").glob("*.jsonl"))
        assert len(recent_files) > 0
        
        # Check decision was stored correctly
        with open(recent_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
            
            decision_data = json.loads(lines[-1])
            assert decision_data['issue_number'] == 123
            assert decision_data['gate_type'] == 'code_coverage'
            assert decision_data['decision'] == 'pass'
            assert decision_data['threshold_value'] == 80.0
            assert decision_data['actual_value'] == 85.0
            assert decision_data['agent_type'] == 'rif-validator'
            assert decision_data['confidence_score'] == 0.95
            assert len(decision_data['evidence']) == 2
    
    def test_record_multiple_decisions(self):
        """Test recording multiple decisions for metrics calculation."""
        decisions = [
            (QualityGateType.CODE_COVERAGE, QualityDecisionType.PASS, 80.0, 85.0),
            (QualityGateType.SECURITY_SCAN, QualityDecisionType.FAIL, 0, 2),
            (QualityGateType.PERFORMANCE, QualityDecisionType.PASS, 5.0, 2.0),
            (QualityGateType.LINTING, QualityDecisionType.WARNING, 3, 5),
        ]
        
        decision_ids = []
        for gate_type, decision, threshold, actual in decisions:
            decision_id = self.collector.record_quality_decision(
                issue_number=456,
                gate_type=gate_type,
                decision=decision,
                threshold_value=threshold,
                actual_value=actual
            )
            decision_ids.append(decision_id)
        
        # All decisions should be recorded
        assert len(decision_ids) == 4
        assert all(id != "" for id in decision_ids)
        
        # Check all decisions are unique
        assert len(set(decision_ids)) == 4
    
    def test_track_production_outcome(self):
        """Test tracking production outcomes for correlation analysis."""
        # Create mock defects
        defects = [
            {'severity': 'high', 'type': 'regression', 'description': 'API timeout'},
            {'severity': 'medium', 'type': 'security', 'description': 'Input validation bypass'}
        ]
        
        outcome_id = self.collector.track_production_outcome(
            issue_number=789,
            change_id="commit-abc123",
            defects=defects,
            time_to_defect_hours=24.5,
            customer_impact="Service degradation for 2 hours"
        )
        
        assert outcome_id != ""
        assert len(outcome_id) == 16
        
        # Check outcome file was created
        correlation_files = list((self.collector.storage_path / "correlations").glob("*.jsonl"))
        assert len(correlation_files) > 0
        
        # Check outcome was stored correctly
        with open(correlation_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
            
            outcome_data = json.loads(lines[-1])
            assert outcome_data['issue_number'] == 789
            assert outcome_data['change_id'] == "commit-abc123"
            assert outcome_data['defect_count'] == 2
            assert outcome_data['defect_severity'] == ['high', 'medium']
            assert outcome_data['time_to_defect_hours'] == 24.5
            assert len(outcome_data['production_issues']) == 2
    
    def test_get_quality_gate_status(self):
        """Test retrieving quality gate status for an issue."""
        # Record some decisions for an issue
        issue_number = 999
        
        self.collector.record_quality_decision(
            issue_number=issue_number,
            gate_type=QualityGateType.CODE_COVERAGE,
            decision=QualityDecisionType.PASS,
            threshold_value=80.0,
            actual_value=85.0,
            confidence_score=0.9
        )
        
        self.collector.record_quality_decision(
            issue_number=issue_number,
            gate_type=QualityGateType.SECURITY_SCAN,
            decision=QualityDecisionType.FAIL,
            threshold_value=0,
            actual_value=1,
            confidence_score=0.95
        )
        
        # Get status
        status = self.collector.get_quality_gate_status(issue_number)
        
        assert status['issue_number'] == issue_number
        assert status['status'] == 'active'
        assert len(status['gates']) == 2
        assert 'code_coverage' in status['gates']
        assert 'security_scan' in status['gates']
        
        # Check gate details
        coverage_gate = status['gates']['code_coverage']
        assert coverage_gate['decision'] == 'pass'
        assert coverage_gate['threshold'] == 80.0
        assert coverage_gate['actual'] == 85.0
        assert coverage_gate['confidence'] == 0.9
        
        security_gate = status['gates']['security_scan']
        assert security_gate['decision'] == 'fail'
        assert security_gate['threshold'] == 0
        assert security_gate['actual'] == 1
        
        # Check overall score calculation
        assert status['overall_score'] == 50.0  # 1/2 gates passed
        assert status['total_decisions'] == 2
    
    def test_calculate_effectiveness_metrics(self):
        """Test calculation of effectiveness metrics."""
        # Create test data with known outcomes
        issue_numbers = [100, 101, 102, 103, 104]
        gate_type = QualityGateType.CODE_COVERAGE
        
        # Record decisions with varying outcomes
        decisions_data = [
            (QualityDecisionType.PASS, True),   # False negative (pass but has defects)
            (QualityDecisionType.PASS, False),  # True negative (pass and no defects)
            (QualityDecisionType.FAIL, True),   # True positive (fail and has defects)
            (QualityDecisionType.PASS, True),   # False negative (pass but has defects)
            (QualityDecisionType.FAIL, False),  # False positive (fail but no defects)
        ]
        
        for i, (decision, has_defects) in enumerate(decisions_data):
            issue_num = issue_numbers[i]
            
            # Record quality decision
            self.collector.record_quality_decision(
                issue_number=issue_num,
                gate_type=gate_type,
                decision=decision,
                threshold_value=80.0,
                actual_value=85.0 if decision == QualityDecisionType.PASS else 75.0
            )
            
            # Record corresponding production outcome
            defects = [{'severity': 'medium', 'type': 'regression'}] if has_defects else []
            self.collector.track_production_outcome(
                issue_number=issue_num,
                change_id=f"commit-{i}",
                defects=defects
            )
        
        # Calculate metrics
        metrics = self.collector.calculate_effectiveness_metrics(gate_type, days_back=1)
        
        assert metrics is not None
        assert metrics.gate_type == 'code_coverage'
        assert metrics.total_decisions == 5
        assert metrics.pass_count == 3
        assert metrics.fail_count == 2
        
        # Check confusion matrix based on corrected logic:
        # TP=1 (1 fail with defects), FP=1 (1 fail without defects), 
        # TN=1 (1 pass without defects), FN=2 (2 pass with defects)
        assert metrics.true_positives == 1   # Correctly rejected bad changes
        assert metrics.false_positives == 1  # Incorrectly rejected good changes
        assert metrics.true_negatives == 1   # Correctly accepted good changes
        assert metrics.false_negatives == 2  # Incorrectly accepted bad changes
        
        # Check calculated rates
        assert metrics.accuracy_percent > 0
        assert metrics.precision_percent > 0
        assert metrics.false_positive_rate > 0
        assert metrics.false_negative_rate > 0
    
    def test_performance_tracking(self):
        """Test that performance metrics are tracked."""
        # Record a decision and check timing is tracked
        start_time = datetime.now()
        
        decision_id = self.collector.record_quality_decision(
            issue_number=555,
            gate_type=QualityGateType.CODE_COVERAGE,
            decision=QualityDecisionType.PASS,
            threshold_value=80.0,
            actual_value=85.0
        )
        
        # Check decision was recorded with timing
        recent_files = list((self.collector.storage_path / "recent").glob("*.jsonl"))
        with open(recent_files[0], 'r') as f:
            lines = f.readlines()
            decision_data = json.loads(lines[-1])
            
            assert 'processing_time_ms' in decision_data
            assert decision_data['processing_time_ms'] >= 0
            assert decision_data['processing_time_ms'] < 1000  # Should be fast
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with invalid data
        decision_id = self.collector.record_quality_decision(
            issue_number=None,  # Invalid issue number
            gate_type=QualityGateType.CODE_COVERAGE,
            decision=QualityDecisionType.PASS
        )
        
        # Should handle gracefully and return empty string
        assert decision_id == ""
        
        # Test get status for non-existent issue
        status = self.collector.get_quality_gate_status(999999)
        assert status['status'] == 'no_decisions'
        assert status['gates'] == {}
        assert status['overall_score'] is None
    
    def test_session_tracking(self):
        """Test that session-specific data is tracked."""
        # Record a decision
        self.collector.record_quality_decision(
            issue_number=777,
            gate_type=QualityGateType.CODE_COVERAGE,
            decision=QualityDecisionType.PASS
        )
        
        # Check session file was created
        session_files = list((self.collector.storage_path / "recent").glob(f"session_{self.collector.session_id}.jsonl"))
        assert len(session_files) == 1
        
        # Check session file contains decision
        with open(session_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
            
            decision_data = json.loads(lines[-1])
            assert decision_data['issue_number'] == 777


if __name__ == '__main__':
    pytest.main([__file__, '-v'])