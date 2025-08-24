#!/usr/bin/env python3
"""
Test suite for Risk-Based Manual Intervention System
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
from integrated_risk_assessment import (
    IntegratedRiskAssessment,
    IntegratedRiskProfile
)

class TestRiskBasedIntervention:
    """Test suite for Risk-Based Manual Intervention system."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config
        self.config_path = Path(self.temp_dir) / "test_risk_config.yaml"
        self._create_test_risk_config()
        
        # Initialize components
        self.assignment_engine = SpecialistAssignmentEngine(config_path=str(self.config_path))
        self.risk_assessor = IntegratedRiskAssessment(config_path=str(self.config_path))
    
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
                },
                'compliance': {
                    'triggers': ['privacy_changes', 'audit_modifications'],
                    'sla_hours': 6,
                    'blocking': True
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
                        'Security code review',
                        'Vulnerability scan report'
                    ]
                },
                'architecture_specialist': {
                    'mandatory': [
                        'Architectural design review',
                        'Scalability assessment',
                        'Dependency impact analysis'
                    ],
                    'recommended': [
                        'Performance benchmarks',
                        'Load testing results',
                        'Database migration strategy'
                    ]
                }
            },
            'escalation_triggers': {
                'security_changes': {
                    'patterns': ['auth/**', '**/security/**', '**/payment/**'],
                    'specialist': 'security',
                    'sla_hours': 4,
                    'blocking': True
                },
                'architecture_changes': {
                    'patterns': ['>500 LOC', '>10 files', '**/database/**'],
                    'specialist': 'architecture',
                    'sla_hours': 12,
                    'blocking': False
                },
                'compliance_areas': {
                    'patterns': ['**/audit/**', '**/privacy/**'],
                    'specialist': 'compliance',
                    'sla_hours': 6,
                    'blocking': True
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_security_risk_detection_and_escalation(self):
        """Test detection of security risks and automatic specialist escalation."""
        # Create high-security-risk scenario
        request = AssignmentRequest(
            issue_number=101,
            risk_score=0.8,
            risk_level="high",
            primary_risk_factors=["security_changes", "authentication_bypass", "privilege_escalation"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["auth/oauth.py", "security/jwt_handler.py", "payment/billing.py"],
            estimated_review_time=6.0,
            special_requirements=["executive_approval"]
        )
        
        result = self.assignment_engine.assign_specialist(request)
        
        # Should successfully assign a security specialist
        assert result.assigned_specialist is not None
        assert result.assigned_specialist.specialist_type == SpecialistType.SECURITY
        assert result.assignment_confidence > 0.5
        
        # Should have appropriate evidence checklist for security
        assert len(result.evidence_checklist.mandatory_items) >= 3
        assert any('security' in item.lower() for item in result.evidence_checklist.mandatory_items)
        assert any('threat' in item.lower() for item in result.evidence_checklist.mandatory_items)
        
        # Should have escalation chain
        assert len(result.escalation_chain) > 1
        assert result.escalation_chain[0] == result.assigned_specialist
        
        # Should include custom evidence for critical issues
        assert "Executive approval documentation" in result.evidence_checklist.custom_items
        assert "Security impact assessment" in result.evidence_checklist.custom_items
    
    def test_architecture_risk_detection_conditional_blocking(self):
        """Test architecture risk detection with conditional blocking."""
        request = AssignmentRequest(
            issue_number=102,
            risk_score=0.6,
            risk_level="medium",
            primary_risk_factors=["large_changes", "database_schema_modification", "api_changes"],
            specialist_type=SpecialistType.ARCHITECTURE,
            urgency_level="medium",
            files_changed=[f"database/migration_{i:03d}.sql" for i in range(15)],  # Many files
            estimated_review_time=12.0,
            special_requirements=[]
        )
        
        result = self.assignment_engine.assign_specialist(request)
        
        # Should assign architecture specialist
        assert result.assigned_specialist is not None
        assert result.assigned_specialist.specialist_type == SpecialistType.ARCHITECTURE
        
        # Should have database-specific evidence
        assert "Database migration rollback plan" in result.evidence_checklist.custom_items
        
        # Should have reasonable SLA for architecture review
        sla_hours = (result.sla_deadline - datetime.now()).total_seconds() / 3600
        assert 8 <= sla_hours <= 16  # Should be around 12 hours for medium priority
    
    @patch('subprocess.run')
    def test_github_issue_creation_for_specialist_review(self, mock_subprocess):
        """Test GitHub issue creation for specialist review."""
        # Mock successful GitHub issue creation
        mock_result = MagicMock()
        mock_result.stderr = "https://github.com/user/repo/issues/999\n"
        mock_subprocess.return_value = mock_result
        
        request = AssignmentRequest(
            issue_number=103,
            risk_score=0.9,
            risk_level="critical",
            primary_risk_factors=["security_vulnerability", "data_exposure"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["security/auth.py", "api/user_data.py"],
            estimated_review_time=4.0,
            special_requirements=["immediate_response"]
        )
        
        result = self.assignment_engine.assign_specialist(request)
        
        # Should create GitHub issue
        assert result.github_issue_number == 999
        
        # Verify GitHub CLI was called correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        
        assert 'gh' in call_args
        assert 'issue' in call_args
        assert 'create' in call_args
        assert '--title' in call_args
        assert '--body' in call_args
        assert '--assignee' in call_args
        
        # Should include proper labels
        label_arg_index = call_args.index('--label') + 1
        labels = call_args[label_arg_index]
        assert 'state:blocked' in labels
        assert 'specialist:security' in labels
        assert 'risk:critical' in labels
    
    def test_sla_calculation_and_urgency_adjustment(self):
        """Test SLA calculation with urgency-based adjustments."""
        # Critical urgency should get very short SLA
        critical_request = AssignmentRequest(
            issue_number=104,
            risk_score=0.9,
            risk_level="critical",
            primary_risk_factors=["production_outage"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["core/auth.py"],
            estimated_review_time=2.0,
            special_requirements=[]
        )
        
        result = self.assignment_engine.assign_specialist(critical_request)
        
        # Critical should have SLA ≤ 2 hours
        sla_hours = (result.sla_deadline - datetime.now()).total_seconds() / 3600
        assert sla_hours <= 2.5
        
        # High urgency should get reduced SLA
        high_request = AssignmentRequest(
            issue_number=105,
            risk_score=0.7,
            risk_level="high",
            primary_risk_factors=["security_issue"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="high",
            files_changed=["auth/login.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.assignment_engine.assign_specialist(high_request)
        
        # High should have SLA ≤ 4 hours
        sla_hours = (result.sla_deadline - datetime.now()).total_seconds() / 3600
        assert sla_hours <= 4.5
    
    def test_specialist_workload_balancing(self):
        """Test that specialist assignment considers current workload."""
        # Get initial specialist workloads
        initial_workloads = {}
        for specialist_type, specialists in self.assignment_engine.specialists.items():
            for specialist in specialists:
                initial_workloads[specialist.specialist_id] = specialist.current_workload
        
        # Make multiple assignments to same type
        requests = []
        results = []
        
        for i in range(3):
            request = AssignmentRequest(
                issue_number=200 + i,
                risk_score=0.6,
                risk_level="medium",
                primary_risk_factors=["security_review_needed"],
                specialist_type=SpecialistType.SECURITY,
                urgency_level="medium",
                files_changed=["auth/module.py"],
                estimated_review_time=4.0,
                special_requirements=[]
            )
            
            requests.append(request)
            result = self.assignment_engine.assign_specialist(request)
            results.append(result)
            
            assert result.assigned_specialist is not None
        
        # Check that workload was updated
        assigned_specialists = [r.assigned_specialist for r in results if r.assigned_specialist]
        
        for specialist in assigned_specialists:
            new_workload = specialist.current_workload
            initial_workload = initial_workloads.get(specialist.specialist_id, 0)
            assert new_workload > initial_workload
    
    def test_expertise_matching_in_assignments(self):
        """Test that specialists are matched based on expertise areas."""
        # Request with cryptography-related risk factors
        crypto_request = AssignmentRequest(
            issue_number=301,
            risk_score=0.7,
            risk_level="high",
            primary_risk_factors=["cryptography", "encryption", "key_management"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="medium",
            files_changed=["crypto/aes.py", "security/keys.py"],
            estimated_review_time=6.0,
            special_requirements=[]
        )
        
        result = self.assignment_engine.assign_specialist(crypto_request)
        
        assert result.assigned_specialist is not None
        
        # Should prefer specialist with cryptography expertise
        specialist_expertise = result.assigned_specialist.expertise_areas
        assert any('crypt' in area.lower() for area in specialist_expertise)
        
        # Should have high assignment confidence due to expertise match
        assert result.assignment_confidence > 0.6
        
        # Should include expertise match in reasoning
        reasoning_text = ' '.join(result.assignment_reasoning)
        assert 'expertise' in reasoning_text.lower() or 'match' in reasoning_text.lower()
    
    def test_escalation_chain_building(self):
        """Test building of proper escalation chains."""
        request = AssignmentRequest(
            issue_number=401,
            risk_score=0.8,
            risk_level="high",
            primary_risk_factors=["high_impact_security_change"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="high",
            files_changed=["security/core.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.assignment_engine.assign_specialist(request)
        
        assert result.assigned_specialist is not None
        assert len(result.escalation_chain) >= 2
        
        # Escalation chain should be ordered by escalation level
        for i in range(len(result.escalation_chain) - 1):
            current_level = result.escalation_chain[i].escalation_level
            next_level = result.escalation_chain[i + 1].escalation_level
            assert current_level <= next_level
        
        # Should have backup specialists
        assert len(result.backup_specialists) >= 1
        
        # Backup specialists should be different from assigned specialist
        for backup in result.backup_specialists:
            assert backup.specialist_id != result.assigned_specialist.specialist_id
    
    def test_evidence_checklist_customization(self):
        """Test evidence checklist customization based on request context."""
        # Test different scenarios and their evidence requirements
        scenarios = [
            {
                'risk_factors': ['database_changes', 'migration'],
                'files': ['database/migration_001.sql', 'models/user.py'],
                'expected_custom': 'Database migration rollback plan'
            },
            {
                'risk_factors': ['security_vulnerability', 'critical'],
                'files': ['auth/jwt.py'],
                'risk_level': 'critical',
                'expected_custom': 'Executive approval documentation'
            },
            {
                'risk_factors': ['performance_impact'],
                'files': ['api/core.py'],
                'expected_custom': None  # No specific custom item expected
            }
        ]
        
        for scenario in scenarios:
            request = AssignmentRequest(
                issue_number=500,
                risk_score=0.7,
                risk_level=scenario.get('risk_level', 'medium'),
                primary_risk_factors=scenario['risk_factors'],
                specialist_type=SpecialistType.SECURITY,
                urgency_level="medium",
                files_changed=scenario['files'],
                estimated_review_time=4.0,
                special_requirements=[]
            )
            
            result = self.assignment_engine.assign_specialist(request)
            
            # Check evidence checklist
            assert result.evidence_checklist is not None
            assert len(result.evidence_checklist.mandatory_items) > 0
            
            if scenario.get('expected_custom'):
                assert scenario['expected_custom'] in result.evidence_checklist.custom_items
    
    def test_no_available_specialist_handling(self):
        """Test handling when no specialists are available."""
        # Simulate all specialists being at capacity
        for specialist_type, specialists in self.assignment_engine.specialists.items():
            for specialist in specialists:
                specialist.current_workload = specialist.max_concurrent_reviews
                specialist.status = SpecialistStatus.BUSY
        
        request = AssignmentRequest(
            issue_number=601,
            risk_score=0.8,
            risk_level="high",
            primary_risk_factors=["security_issue"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="high",
            files_changed=["auth/core.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = self.assignment_engine.assign_specialist(request)
        
        # Should handle gracefully
        assert result.assigned_specialist is None
        assert result.assignment_confidence == 0.0
        assert "No available specialists found" in result.assignment_reasoning
        assert "Manual escalation required" in result.assignment_reasoning
        
        # Should still provide evidence checklist for manual review
        assert result.evidence_checklist is not None
        
        # Should have 24-hour fallback SLA
        sla_hours = (result.sla_deadline - datetime.now()).total_seconds() / 3600
        assert 23 <= sla_hours <= 25
    
    def test_assignment_history_tracking(self):
        """Test that assignments are properly tracked in history."""
        request = AssignmentRequest(
            issue_number=701,
            risk_score=0.6,
            risk_level="medium",
            primary_risk_factors=["routine_security_review"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="medium",
            files_changed=["auth/helper.py"],
            estimated_review_time=3.0,
            special_requirements=[]
        )
        
        initial_assignment_count = len(self.assignment_engine.assignment_history['assignments'])
        
        result = self.assignment_engine.assign_specialist(request)
        
        # Should track assignment in history
        new_assignment_count = len(self.assignment_engine.assignment_history['assignments'])
        assert new_assignment_count == initial_assignment_count + 1
        
        # Should update metrics
        total_assignments = self.assignment_engine.assignment_history['metrics'].get('total_assignments', 0)
        assert total_assignments > 0
        
        # Check latest assignment record
        latest_assignment = self.assignment_engine.assignment_history['assignments'][-1]
        assert latest_assignment['issue_number'] == 701
        assert latest_assignment['specialist_type'] == 'security'
        assert latest_assignment['risk_level'] == 'medium'
    
    def test_workload_report_generation(self):
        """Test specialist workload report generation."""
        report = self.assignment_engine.get_specialist_workload_report()
        
        # Should contain key sections
        assert 'timestamp' in report
        assert 'specialists_by_type' in report
        assert 'workload_summary' in report
        
        # Should have data for each specialist type
        for specialist_type in SpecialistType:
            if specialist_type in self.assignment_engine.specialists:
                assert specialist_type.value in report['specialists_by_type']
                assert specialist_type.value in report['workload_summary']
        
        # Should include workload details for specialists
        for type_data in report['specialists_by_type'].values():
            for specialist_data in type_data:
                assert 'name' in specialist_data
                assert 'workload' in specialist_data
                assert 'status' in specialist_data
                assert 'utilization' in specialist_data
                
                # Utilization should be between 0 and 1
                assert 0 <= specialist_data['utilization'] <= 1
    
    def test_integrated_risk_assessment_flow(self):
        """Test integration between risk assessment and specialist assignment."""
        # Simulate integrated flow
        files = ["auth/oauth.py", "security/permissions.py", "payment/stripe.py"]
        
        # First assess risk
        risk_profile = self.risk_assessor.assess_integrated_risk(
            files=files,
            context="critical_algorithms",
            vulnerability_data={'critical': 1, 'high': 2, 'medium': 3},
            performance_data={'regression': 2.5},
            project_context={'security_critical': True}
        )
        
        # Then assign specialist based on risk profile
        if risk_profile.risk_level in ['high', 'critical']:
            specialist_type = SpecialistType.SECURITY
            urgency = "critical" if risk_profile.risk_level == 'critical' else "high"
            
            request = AssignmentRequest(
                issue_number=801,
                risk_score=risk_profile.overall_risk_score,
                risk_level=risk_profile.risk_level,
                primary_risk_factors=risk_profile.compound_risks[:3],
                specialist_type=specialist_type,
                urgency_level=urgency,
                files_changed=files,
                estimated_review_time=risk_profile.overall_risk_score * 8,  # Scale with risk
                special_requirements=[]
            )
            
            result = self.assignment_engine.assign_specialist(request)
            
            # High risk should result in successful assignment
            assert result.assigned_specialist is not None
            
            # Should have appropriate evidence requirements for security
            assert len(result.evidence_checklist.mandatory_items) >= 3
            
            # Should reflect urgency in SLA
            sla_hours = (result.sla_deadline - datetime.now()).total_seconds() / 3600
            if urgency == "critical":
                assert sla_hours <= 3
            else:
                assert sla_hours <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])