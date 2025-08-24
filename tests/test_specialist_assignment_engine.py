#!/usr/bin/env python3
"""
Test suite for Specialist Assignment Engine
Issue #92: Risk-Based Manual Intervention Framework
"""

import os
import sys
import json
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the claude/commands directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'claude', 'commands'))

from specialist_assignment_engine import (
    SpecialistAssignmentEngine,
    SpecialistProfile,
    AssignmentRequest,
    AssignmentResult,
    EvidenceChecklist,
    SpecialistType,
    SpecialistStatus
)

class TestSpecialistAssignmentEngine:
    """Test suite for Specialist Assignment Engine."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config
        self.config_path = Path(self.temp_dir) / "test_risk_config.yaml"
        self._create_test_risk_config()
        
        # Initialize engine
        self.engine = SpecialistAssignmentEngine(config_path=str(self.config_path))
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_risk_config(self):
        """Create test risk assessment configuration."""
        import yaml
        
        config = {
            'specialist_routing': {
                'security': {
                    'triggers': ['security_changes', 'auth_modifications'],
                    'sla_hours': 4,
                    'blocking': True
                },
                'architecture': {
                    'triggers': ['large_changes', 'database_changes'],
                    'sla_hours': 12,
                    'blocking': False
                }
            },
            'evidence_requirements': {
                'security_specialist': {
                    'mandatory': [
                        'Security impact assessment completed',
                        'Threat model reviewed',
                        'Authentication flow validated'
                    ],
                    'recommended': [
                        'Penetration test results',
                        'Security code review'
                    ]
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        assert self.engine.config is not None
        assert len(self.engine.specialists) > 0
        assert SpecialistType.SECURITY in self.engine.specialists
        assert len(self.engine.specialists[SpecialistType.SECURITY]) > 0
    
    def test_security_specialist_assignment(self):
        """Test assignment of security specialist."""
        request = AssignmentRequest(
            issue_number=101,
            risk_score=0.8,
            risk_level="high",
            primary_risk_factors=["security_changes", "authentication_bypass"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["auth/oauth.py", "security/jwt_handler.py"],
            estimated_review_time=6.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should successfully assign a security specialist
        assert result.assigned_specialist is not None
        assert result.assigned_specialist.specialist_type == SpecialistType.SECURITY
        assert result.assignment_confidence > 0.0
        
        # Should have evidence checklist
        assert result.evidence_checklist is not None
        assert len(result.evidence_checklist.mandatory_items) >= 3
        
        # Should have escalation chain
        assert len(result.escalation_chain) > 0
        assert result.escalation_chain[0] == result.assigned_specialist
    
    def test_sla_calculation(self):
        """Test SLA calculation based on urgency."""
        # Critical request should have short SLA
        critical_request = AssignmentRequest(
            issue_number=102,
            risk_score=0.9,
            risk_level="critical",
            primary_risk_factors=["production_outage"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["core/auth.py"],
            estimated_review_time=2.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(critical_request)
        
        # Critical should have SLA ≤ 2 hours
        sla_hours = (result.sla_deadline - datetime.now()).total_seconds() / 3600
        assert sla_hours <= 2.5
        
        # Medium request should have normal SLA
        medium_request = AssignmentRequest(
            issue_number=103,
            risk_score=0.5,
            risk_level="medium",
            primary_risk_factors=["routine_review"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="medium",
            files_changed=["auth/helper.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(medium_request)
        
        # Medium should have normal SLA (4-8 hours)
        sla_hours = (result.sla_deadline - datetime.now()).total_seconds() / 3600
        assert 3 <= sla_hours <= 10
    
    def test_workload_balancing(self):
        """Test that assignments consider current workload."""
        # Get all security specialists
        security_specialists = self.engine.specialists[SpecialistType.SECURITY]
        
        # Record initial workloads
        initial_workloads = {s.specialist_id: s.current_workload for s in security_specialists}
        
        # Make assignment
        request = AssignmentRequest(
            issue_number=104,
            risk_score=0.6,
            risk_level="medium",
            primary_risk_factors=["security_review"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="medium",
            files_changed=["auth/module.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should assign to someone
        assert result.assigned_specialist is not None
        
        # Assigned specialist's workload should increase
        assigned_specialist = result.assigned_specialist
        new_workload = assigned_specialist.current_workload
        initial_workload = initial_workloads[assigned_specialist.specialist_id]
        assert new_workload == initial_workload + 1
    
    def test_expertise_matching(self):
        """Test that specialists are matched by expertise."""
        request = AssignmentRequest(
            issue_number=105,
            risk_score=0.7,
            risk_level="high",
            primary_risk_factors=["cryptography", "encryption", "key_management"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="medium",
            files_changed=["crypto/aes.py", "security/keys.py"],
            estimated_review_time=6.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        assert result.assigned_specialist is not None
        
        # Check if assigned specialist has relevant expertise
        specialist_expertise = result.assigned_specialist.expertise_areas
        has_crypto_expertise = any('crypt' in area.lower() for area in specialist_expertise)
        
        # Should prefer specialist with relevant expertise or have reasonable confidence
        if has_crypto_expertise:
            assert result.assignment_confidence > 0.5
        else:
            # Even without direct expertise match, should still assign someone
            assert result.assignment_confidence > 0.3
    
    def test_evidence_checklist_generation(self):
        """Test evidence checklist generation."""
        request = AssignmentRequest(
            issue_number=106,
            risk_score=0.8,
            risk_level="high",
            primary_risk_factors=["security_vulnerability"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="high",
            files_changed=["auth/login.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should have evidence checklist
        checklist = result.evidence_checklist
        assert checklist is not None
        assert checklist.specialist_type == SpecialistType.SECURITY
        
        # Should have mandatory items from config
        assert len(checklist.mandatory_items) >= 3
        assert any('security' in item.lower() for item in checklist.mandatory_items)
        assert any('threat' in item.lower() for item in checklist.mandatory_items)
        
        # Should have recommended items
        assert len(checklist.recommended_items) > 0
        
        # Should have completion criteria
        assert checklist.completion_criteria != ""
    
    def test_escalation_chain(self):
        """Test escalation chain building."""
        request = AssignmentRequest(
            issue_number=107,
            risk_score=0.8,
            risk_level="high",
            primary_risk_factors=["high_impact_change"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="high",
            files_changed=["security/core.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should have escalation chain
        assert len(result.escalation_chain) >= 1
        
        # First in chain should be assigned specialist
        assert result.escalation_chain[0] == result.assigned_specialist
        
        # Should be ordered by escalation level
        for i in range(len(result.escalation_chain) - 1):
            current_level = result.escalation_chain[i].escalation_level
            next_level = result.escalation_chain[i + 1].escalation_level
            assert current_level <= next_level
    
    @patch('subprocess.run')
    def test_github_issue_creation(self, mock_subprocess):
        """Test GitHub issue creation for specialist review."""
        # Mock successful GitHub issue creation
        mock_result = MagicMock()
        mock_result.stderr = "https://github.com/user/repo/issues/999\n"
        mock_subprocess.return_value = mock_result
        
        request = AssignmentRequest(
            issue_number=108,
            risk_score=0.9,
            risk_level="critical",
            primary_risk_factors=["security_vulnerability"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["security/auth.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should create GitHub issue
        assert result.github_issue_number == 999
        
        # Verify subprocess was called
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        
        assert 'gh' in call_args
        assert 'issue' in call_args
        assert 'create' in call_args
    
    def test_no_available_specialists(self):
        """Test handling when no specialists are available."""
        # Make all specialists busy
        for specialists in self.engine.specialists.values():
            for specialist in specialists:
                specialist.current_workload = specialist.max_concurrent_reviews
                specialist.status = SpecialistStatus.BUSY
        
        request = AssignmentRequest(
            issue_number=109,
            risk_score=0.8,
            risk_level="high",
            primary_risk_factors=["security_issue"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="high",
            files_changed=["auth/core.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should handle gracefully
        assert result.assigned_specialist is None
        assert result.assignment_confidence == 0.0
        
        # Check reasoning (might have emojis)
        reasoning_text = ' '.join(result.assignment_reasoning)
        assert "no available specialists found" in reasoning_text.lower()
        
        # Should still provide evidence checklist
        assert result.evidence_checklist is not None
    
    def test_workload_report(self):
        """Test workload report generation."""
        report = self.engine.get_specialist_workload_report()
        
        # Should have required sections
        assert 'timestamp' in report
        assert 'specialists_by_type' in report
        assert 'workload_summary' in report
        
        # Should have security specialist data
        assert 'security' in report['specialists_by_type']
        security_specialists = report['specialists_by_type']['security']
        
        assert len(security_specialists) > 0
        
        for specialist_data in security_specialists:
            assert 'name' in specialist_data
            assert 'workload' in specialist_data
            assert 'status' in specialist_data
            assert 'utilization' in specialist_data
            
            # Utilization should be valid
            assert 0 <= specialist_data['utilization'] <= 1
    
    def test_assignment_reasoning(self):
        """Test assignment reasoning generation."""
        request = AssignmentRequest(
            issue_number=110,
            risk_score=0.7,
            risk_level="medium",
            primary_risk_factors=["security_review"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="medium",
            files_changed=["auth/token.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should have reasoning
        assert len(result.assignment_reasoning) > 0
        
        # Should include assignment details
        reasoning_text = ' '.join(result.assignment_reasoning)
        assert result.assigned_specialist.name in reasoning_text
        assert 'workload' in reasoning_text.lower()
        
        # Should include confidence score
        assert f"{result.assignment_confidence:.2f}" in reasoning_text
    
    def test_custom_evidence_items(self):
        """Test custom evidence item generation."""
        # Test critical risk level
        critical_request = AssignmentRequest(
            issue_number=111,
            risk_score=0.9,
            risk_level="critical",
            primary_risk_factors=["critical_security_issue"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["auth/core.py"],
            estimated_review_time=2.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(critical_request)
        
        # Should have executive approval for critical
        assert "Executive approval documentation" in result.evidence_checklist.custom_items
        
        # Test database changes
        db_request = AssignmentRequest(
            issue_number=112,
            risk_score=0.6,
            risk_level="medium",
            primary_risk_factors=["database_changes"],
            specialist_type=SpecialistType.ARCHITECTURE,
            urgency_level="medium",
            files_changed=["database/migration_001.sql", "models/user.py"],
            estimated_review_time=6.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(db_request)
        
        # Should have database rollback plan
        assert "Database migration rollback plan" in result.evidence_checklist.custom_items
    
    def test_assignment_history_tracking(self):
        """Test that assignments are tracked in history."""
        initial_count = len(self.engine.assignment_history['assignments'])
        
        request = AssignmentRequest(
            issue_number=113,
            risk_score=0.5,
            risk_level="medium",
            primary_risk_factors=["routine_check"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="medium",
            files_changed=["auth/helper.py"],
            estimated_review_time=3.0,
            special_requirements=[]
        )
        
        result = self.engine.assign_specialist(request)
        
        # Should add to history
        new_count = len(self.engine.assignment_history['assignments'])
        assert new_count == initial_count + 1
        
        # Should update metrics
        metrics = self.engine.assignment_history['metrics']
        assert 'total_assignments' in metrics
        assert metrics['total_assignments'] > 0
        
        # Check latest assignment
        latest = self.engine.assignment_history['assignments'][-1]
        assert latest['issue_number'] == 113
        assert latest['specialist_type'] == 'security'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])