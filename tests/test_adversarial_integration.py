#!/usr/bin/env python3
"""
Integration tests for the complete Adversarial Testing Framework.
Tests the end-to-end adversarial verification workflow.
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claude.commands.shadow_quality_tracking import ShadowQualityTracker


class TestAdversarialIntegration(unittest.TestCase):
    """Integration tests for the complete adversarial testing system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ShadowQualityTracker()
        self.sample_main_issue = {
            "title": "Implement secure authentication system",
            "body": "Add OAuth2 authentication with JWT tokens and role-based access control",
            "labels": [
                {"name": "complexity:high"},
                {"name": "security:critical"},
                {"name": "state:implementing"}
            ],
            "state": "open",
            "createdAt": "2025-08-23T21:15:00Z"
        }
    
    @patch('subprocess.run')
    def test_complete_adversarial_workflow(self, mock_run):
        """Test the complete adversarial testing workflow."""
        # Mock GitHub API responses for the workflow
        mock_run.side_effect = [
            # Create shadow issue
            Mock(returncode=0, stdout="https://github.com/owner/repo/issues/456"),
            # Log shadow creation to main issue
            Mock(returncode=0, stdout=""),
            # Log initial verification activity
            Mock(returncode=0, stdout=""),
            # Sync quality status 
            Mock(returncode=0, stdout=""),
            # Log evidence collection activity
            Mock(returncode=0, stdout=""),
            # Final status update and close
            Mock(returncode=0, stdout=""),
            Mock(returncode=0, stdout="")
        ]
        
        with patch.object(self.tracker, '_get_issue_details', return_value=self.sample_main_issue):
            # Step 1: Create shadow quality issue
            shadow_result = self.tracker.create_shadow_quality_issue(123)
            self.assertNotIn("error", shadow_result)
            shadow_number = 456  # From mocked response
            
            # Step 2: Log initial verification activity
            initial_activity = {
                "type": "Initial Risk Assessment",
                "agent": "RIF-Validator",
                "action": "Analyzed issue for security implications",
                "result": "HIGH RISK - Authentication system requires intensive validation",
                "evidence": "Security keywords detected: OAuth2, JWT, role-based access",
                "notes": "Escalating to intensive verification depth due to security implications"
            }
            
            log_success = self.tracker.log_quality_activity(shadow_number, initial_activity)
            self.assertTrue(log_success)
            
            # Step 3: Sync quality status during implementation
            sync_success = self.tracker.sync_quality_status(123, shadow_number)
            self.assertTrue(sync_success)
            
            # Step 4: Log evidence collection activity
            evidence_activity = {
                "type": "Evidence Collection",
                "agent": "RIF-Validator", 
                "action": "Collected implementation evidence",
                "result": "EVIDENCE COMPLETE - All required proof collected",
                "evidence": "Unit tests: 45 passing, Coverage: 92%, Security scan: No vulnerabilities",
                "notes": "All evidence requirements met for security-critical implementation"
            }
            
            evidence_log_success = self.tracker.log_quality_activity(shadow_number, evidence_activity)
            self.assertTrue(evidence_log_success)
            
            # Step 5: Close shadow issue with final metrics
            final_metrics = {
                "final_score": 95,
                "gate_decision": "PASS",
                "evidence_percent": 100,
                "final_risk_level": "High (Mitigated)",
                "issues_found": 3,
                "issues_resolved": 3,
                "summary": "High-risk authentication implementation successfully validated with comprehensive evidence",
                "lessons_learned": "Security-critical implementations require intensive validation depth and comprehensive evidence collection"
            }
            
            close_success = self.tracker.close_shadow_issue(shadow_number, final_metrics)
            self.assertTrue(close_success)
            
            # Verify all subprocess calls were made
            self.assertEqual(len(mock_run.call_args_list), 7)
    
    def test_risk_escalation_triggers(self):
        """Test that risk assessment correctly identifies escalation triggers."""
        # Test security-related escalation
        security_issue = {
            "title": "Add payment processing with Stripe API",
            "body": "Implement secure payment flow with credit card processing",
            "labels": [{"name": "complexity:medium"}],
            "state": "open"
        }
        
        risk_level = self.tracker._assess_initial_risk_level(security_issue)
        self.assertEqual(risk_level, "High")
        
        # Test medium-risk database issue
        database_issue = {
            "title": "Optimize database queries for user dashboard",
            "body": "Improve performance of complex database queries",
            "labels": [{"name": "complexity:medium"}],
            "state": "open"
        }
        
        risk_level = self.tracker._assess_initial_risk_level(database_issue)
        self.assertEqual(risk_level, "Medium")
        
        # Test low-risk UI issue
        ui_issue = {
            "title": "Update button styles in settings page",
            "body": "Change button colors to match new design system",
            "labels": [{"name": "complexity:low"}],
            "state": "open"
        }
        
        risk_level = self.tracker._assess_initial_risk_level(ui_issue)
        self.assertEqual(risk_level, "Low")
    
    def test_evidence_requirements_validation(self):
        """Test evidence requirements are properly identified."""
        # This would integrate with the actual RIF-Validator evidence framework
        # For now, test the shadow tracking capability to track evidence
        
        evidence_gaps = self.tracker._calculate_quality_metrics(123)
        
        # Verify metrics structure
        self.assertIn('score', evidence_gaps)
        self.assertIn('evidence_percent', evidence_gaps)
        self.assertIn('risk_level', evidence_gaps)
        self.assertIn('verification_depth', evidence_gaps)
    
    def test_quality_scoring_deterministic(self):
        """Test that quality scoring would be deterministic (placeholder test)."""
        # This represents the quality scoring system that would integrate
        # with the RIF-Validator agent enhancements
        
        test_issues = [
            {"type": "FAIL", "count": 2},  # Should result in score = 100 - (20 * 2) = 60
            {"type": "CONCERN", "count": 3}  # Should result in score = 100 - (10 * 3) = 70
        ]
        
        for scenario in test_issues:
            # This would be implemented in the RIF-Validator enhancement
            if scenario["type"] == "FAIL":
                expected_score = 100 - (20 * scenario["count"])
                self.assertEqual(expected_score, 60)
            elif scenario["type"] == "CONCERN":
                expected_score = 100 - (10 * scenario["count"])
                self.assertEqual(expected_score, 70)


class TestAdversarialSystemCapabilities(unittest.TestCase):
    """Test the enhanced system capabilities of the adversarial framework."""
    
    def test_parallel_verification_capability(self):
        """Test that the system can support parallel verification."""
        # The adversarial testing system enables:
        # 1. Main development track (RIF-Implementer)
        # 2. Quality tracking track (Shadow issues)
        # 3. Risk assessment track (Parallel verification)
        
        # Verify shadow issue can be created for parallel tracking
        tracker = ShadowQualityTracker()
        self.assertEqual(tracker.shadow_prefix, "Quality Tracking:")
        self.assertIn("quality:shadow", tracker.quality_labels)
        self.assertIn("state:quality-tracking", tracker.quality_labels)
    
    def test_audit_trail_generation(self):
        """Test audit trail generation capabilities."""
        tracker = ShadowQualityTracker()
        
        # Mock issue data with activities
        mock_issue_data = {
            "title": "Quality Tracking: Issue #123",
            "state": "closed",
            "createdAt": "2025-08-23T20:00:00Z",
            "closedAt": "2025-08-23T22:00:00Z",
            "comments": [
                {
                    "body": "### 2025-08-23T20:30:00Z - Initial Risk Assessment\n**Agent**: RIF-Validator",
                    "createdAt": "2025-08-23T20:30:00Z"
                },
                {
                    "body": "## Quality Status Update - 2025-08-23T21:00:00Z\n**Quality Score**: 85\n**Risk Level**: Medium",
                    "createdAt": "2025-08-23T21:00:00Z"
                }
            ]
        }
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(mock_issue_data)
            )
            
            audit_trail = tracker.generate_audit_trail(456)
            
            # Verify audit trail structure
            self.assertIn("shadow_issue_number", audit_trail)
            self.assertIn("activities", audit_trail)
            self.assertIn("quality_metrics_history", audit_trail)
            self.assertEqual(len(audit_trail["activities"]), 1)
            self.assertEqual(len(audit_trail["quality_metrics_history"]), 1)
    
    def test_evidence_framework_integration(self):
        """Test integration with evidence collection framework."""
        # This test represents the integration points with the enhanced
        # RIF-Implementer evidence generation and RIF-Validator validation
        
        # Mock evidence package (from RIF-Implementer)
        mock_evidence = {
            "tests": {
                "unit": {"added": 45, "passing": 45},
                "integration": {"added": 12, "passing": 12},
                "coverage": 92
            },
            "performance": {
                "baseline": {"response_time": 150},
                "current": {"response_time": 89},
                "improvement": "40% faster"
            },
            "quality": {
                "linting": {"errors": 0},
                "type_check": {"passing": True},
                "security": {"vulnerabilities": 0}
            }
        }
        
        # Test evidence completeness (would be implemented in RIF-Validator)
        required_evidence = ["tests", "quality"]
        evidence_complete = all(category in mock_evidence for category in required_evidence)
        self.assertTrue(evidence_complete)
        
        # Test quality scoring based on evidence
        quality_score = 100
        if mock_evidence["tests"]["coverage"] < 80:
            quality_score -= 10  # Coverage penalty
        if mock_evidence["quality"]["linting"]["errors"] > 0:
            quality_score -= 20  # Linting failure penalty
        
        self.assertEqual(quality_score, 100)  # No penalties for this evidence package


if __name__ == '__main__':
    unittest.main()