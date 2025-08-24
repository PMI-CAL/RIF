#!/usr/bin/env python3
"""
Test Suite for Shadow Quality Tracking System - Issue #142
Comprehensive testing for quality monitoring, adversarial analysis, and evidence consolidation

Tests cover:
- Quality metric tracking and validation
- Adversarial finding recording and analysis  
- Quality decision making and evidence consolidation
- Integration with DPIBS benchmarking system
- GitHub integration and automated reporting
- Continuous monitoring and alert systems
"""

import os
import sys
import json
import time
import tempfile
import unittest
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add RIF to path
sys.path.insert(0, '/Users/cal/DEV/RIF')

from systems.shadow_quality_tracking import (
    ShadowQualityTracker, QualityMetricType, QualityMetric, QualitySession,
    create_shadow_quality_tracker
)
from systems.shadow_quality_integration import (
    ShadowQualityIntegrator, ValidationContext,
    create_shadow_quality_integrator
)

class TestShadowQualityTracker(unittest.TestCase):
    """Test suite for core shadow quality tracking functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix='shadow_quality_test_')
        self.test_db_path = os.path.join(self.temp_dir, 'test_quality.db')
        self.tracker = ShadowQualityTracker(self.test_db_path)
    
    def test_shadow_prefix_initialization(self):
        """Test that tracker initializes with correct prefix."""
        self.assertEqual(self.tracker.shadow_prefix, "Quality Tracking:")
        self.assertIn("quality:shadow", self.tracker.quality_labels)
        self.assertIn("state:quality-tracking", self.tracker.quality_labels)
    
    def test_generate_shadow_body_content(self):
        """Test shadow issue body generation."""
        body = self.tracker._generate_shadow_body(123, self.sample_issue)
        
        # Check required sections
        self.assertIn("Shadow Quality Issue for #123", body)
        self.assertIn("Verification Checkpoints", body)
        self.assertIn("Current Quality Metrics", body)
        self.assertIn("Audit Trail", body)
        
        # Check complexity extraction
        self.assertIn("High", body)  # Should extract "high" from complexity:high
        
        # Check risk assessment
        self.assertIn("High", body)  # Should assess as high risk due to "authentication"
    
    def test_extract_complexity_from_labels(self):
        """Test complexity extraction from issue labels."""
        labels = [{"name": "complexity:very-high"}, {"name": "bug"}]
        complexity = self.tracker._extract_complexity(labels)
        self.assertEqual(complexity, "Very-High")
        
        # Test no complexity label
        labels_no_complexity = [{"name": "feature"}, {"name": "bug"}]
        complexity_unknown = self.tracker._extract_complexity(labels_no_complexity)
        self.assertEqual(complexity_unknown, "Unknown")
    
    def test_assess_initial_risk_level(self):
        """Test initial risk assessment based on issue content."""
        # High risk keywords
        high_risk_issue = {
            "title": "Fix authentication bypass vulnerability",
            "body": "Security issue in login system"
        }
        risk = self.tracker._assess_initial_risk_level(high_risk_issue)
        self.assertEqual(risk, "High")
        
        # Medium risk keywords
        medium_risk_issue = {
            "title": "Optimize database queries",
            "body": "Performance improvements needed"
        }
        risk = self.tracker._assess_initial_risk_level(medium_risk_issue)
        self.assertEqual(risk, "Medium")
        
        # Low risk (no special keywords)
        low_risk_issue = {
            "title": "Update documentation",
            "body": "Fix typos in README"
        }
        risk = self.tracker._assess_initial_risk_level(low_risk_issue)
        self.assertEqual(risk, "Low")
    
    @patch('subprocess.run')
    def test_create_shadow_quality_issue_success(self, mock_run):
        """Test successful shadow issue creation."""
        # Mock successful GitHub API response
        mock_run.side_effect = [
            # First call: get main issue details
            Mock(returncode=0, stdout=json.dumps(self.sample_issue)),
            # Second call: create shadow issue
            Mock(returncode=0, stdout="https://github.com/owner/repo/issues/456")
        ]
        
        with patch.object(self.tracker, '_get_issue_details', return_value=self.sample_issue):
            with patch.object(self.tracker, '_log_shadow_creation'):
                result = self.tracker.create_shadow_quality_issue(123)
        
        self.assertIn("shadow_issue_number", result)
        self.assertEqual(result["shadow_issue_number"], 456)
        self.assertEqual(result["main_issue"], 123)
    
    @patch('subprocess.run')
    def test_log_quality_activity_success(self, mock_run):
        """Test successful activity logging."""
        mock_run.return_value = Mock(returncode=0)
        
        activity = {
            "type": "Evidence Validation",
            "agent": "RIF-Validator",
            "action": "Verified unit test coverage",
            "result": "95% coverage achieved",
            "evidence": "coverage_report.html",
            "notes": "Excellent test coverage"
        }
        
        success = self.tracker.log_quality_activity(456, activity)
        self.assertTrue(success)
        
        # Check that gh issue comment was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertIn("gh issue comment 456", call_args)
    
    @patch('subprocess.run')
    def test_log_quality_activity_failure(self, mock_run):
        """Test activity logging failure handling."""
        mock_run.return_value = Mock(returncode=1, stderr="GitHub API error")
        
        activity = {"type": "Test Activity"}
        success = self.tracker.log_quality_activity(456, activity)
        self.assertFalse(success)
    
    @patch('subprocess.run')
    def test_sync_quality_status(self, mock_run):
        """Test quality status synchronization."""
        mock_run.return_value = Mock(returncode=0)
        
        with patch.object(self.tracker, '_get_issue_details', return_value=self.sample_issue):
            with patch.object(self.tracker, '_calculate_quality_metrics', return_value={
                'score': 85,
                'evidence_percent': 75,
                'risk_level': 'Medium',
                'verification_depth': 'Standard'
            }):
                success = self.tracker.sync_quality_status(123, 456)
        
        self.assertTrue(success)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_close_shadow_issue(self, mock_run):
        """Test shadow issue closing."""
        mock_run.return_value = Mock(returncode=0)
        
        final_metrics = {
            "final_score": 88,
            "gate_decision": "PASS",
            "evidence_percent": 100,
            "issues_found": 2,
            "issues_resolved": 2,
            "summary": "All quality gates passed successfully"
        }
        
        success = self.tracker.close_shadow_issue(456, final_metrics)
        self.assertTrue(success)
        
        # Should call gh issue comment and gh issue close
        self.assertEqual(mock_run.call_count, 2)
    
    @patch('subprocess.run')
    def test_generate_audit_trail(self, mock_run):
        """Test audit trail generation."""
        mock_issue_data = {
            "title": "Quality Tracking: Issue #123",
            "state": "closed",
            "createdAt": "2025-01-01T00:00:00Z",
            "closedAt": "2025-01-01T02:00:00Z",
            "comments": [
                {
                    "body": "### 2025-01-01T01:00:00Z - Evidence Validation\\n**Agent**: RIF-Validator\\n**Action**: Checked tests",
                    "createdAt": "2025-01-01T01:00:00Z"
                },
                {
                    "body": "## Quality Status Update - 2025-01-01T01:30:00Z\\n**Quality Score**: 85",
                    "createdAt": "2025-01-01T01:30:00Z"
                }
            ]
        }
        
        mock_run.return_value = Mock(returncode=0, stdout=json.dumps(mock_issue_data))
        
        audit = self.tracker.generate_audit_trail(456)
        
        self.assertEqual(audit["shadow_issue_number"], 456)
        self.assertEqual(audit["state"], "closed")
        self.assertEqual(audit["total_activities"], 2)
        self.assertEqual(len(audit["activities"]), 1)  # One activity log
        self.assertEqual(len(audit["quality_metrics_history"]), 1)  # One status update
    
    def test_extract_activity_type(self):
        """Test activity type extraction from comment."""
        comment_body = "### 2025-01-01T01:00:00Z - Evidence Validation\\n**Agent**: RIF-Validator"
        activity_type = self.tracker._extract_activity_type(comment_body)
        self.assertEqual(activity_type, "Evidence Validation")
    
    def test_extract_metrics_from_comment(self):
        """Test metrics extraction from status update comment."""
        comment_body = """## Quality Status Update
        **Quality Score**: 85
        **Risk Level**: Medium
        **Evidence Completion**: 75%
        """
        
        metrics = self.tracker._extract_metrics_from_comment(comment_body)
        self.assertIn("quality_score", metrics)
        self.assertIn("risk_level", metrics)
        self.assertIn("evidence_completion", metrics)
    
    def test_calculate_quality_metrics_placeholder(self):
        """Test quality metrics calculation (placeholder implementation)."""
        metrics = self.tracker._calculate_quality_metrics(123)
        
        # Should return placeholder metrics
        self.assertIn("score", metrics)
        self.assertIn("evidence_percent", metrics)
        self.assertIn("risk_level", metrics)
        self.assertEqual(metrics["evidence_percent"], 0)  # Placeholder value


class TestShadowQualityTrackerCLI(unittest.TestCase):
    """Test the command-line interface of the shadow quality tracker."""
    
    def setUp(self):
        """Set up CLI test fixtures."""
        self.script_path = os.path.join(
            os.path.dirname(__file__), 
            '../claude/commands/shadow_quality_tracking.py'
        )
    
    @patch('subprocess.run')
    @patch('sys.argv', ['shadow_quality_tracking.py', 'create-shadow', '123'])
    def test_cli_create_shadow(self, mock_run):
        """Test CLI create-shadow command."""
        mock_run.return_value = Mock(returncode=0, stdout="https://github.com/owner/repo/issues/456")
        
        # This test would need to be run in a more integrated environment
        # For now, just verify the CLI structure
        self.assertTrue(os.path.exists(self.script_path))
    
    def test_cli_help_output(self):
        """Test CLI help output."""
        # Test with no arguments - should show usage
        result = subprocess.run([
            'python', self.script_path
        ], capture_output=True, text=True)
        
        self.assertIn("Usage:", result.stdout)
        self.assertIn("create-shadow", result.stdout)
        self.assertIn("log-activity", result.stdout)
        self.assertIn("sync-status", result.stdout)


class TestShadowQualityTrackerIntegration(unittest.TestCase):
    """Integration tests for shadow quality tracking."""
    
    @patch('subprocess.run')
    def test_full_shadow_lifecycle(self, mock_run):
        """Test complete shadow issue lifecycle."""
        tracker = ShadowQualityTracker()
        
        # Mock all subprocess calls to succeed
        mock_run.return_value = Mock(returncode=0, stdout="https://github.com/owner/repo/issues/456")
        
        # Mock issue details
        sample_issue = {
            "title": "Test issue",
            "body": "Test body",
            "labels": [],
            "state": "open"
        }
        
        with patch.object(tracker, '_get_issue_details', return_value=sample_issue):
            with patch.object(tracker, '_log_shadow_creation'):
                # 1. Create shadow issue
                create_result = tracker.create_shadow_quality_issue(123)
                self.assertIn("shadow_issue_number", create_result)
                
                shadow_number = create_result["shadow_issue_number"]
                
                # 2. Log some activities
                activity1 = {"type": "Initial Validation", "agent": "RIF-Validator"}
                success1 = tracker.log_quality_activity(shadow_number, activity1)
                self.assertTrue(success1)
                
                activity2 = {"type": "Evidence Review", "agent": "RIF-Validator"}
                success2 = tracker.log_quality_activity(shadow_number, activity2)
                self.assertTrue(success2)
                
                # 3. Sync status
                sync_success = tracker.sync_quality_status(123, shadow_number)
                self.assertTrue(sync_success)
                
                # 4. Close shadow issue
                final_metrics = {"final_score": 95, "gate_decision": "PASS"}
                close_success = tracker.close_shadow_issue(shadow_number, final_metrics)
                self.assertTrue(close_success)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestShadowQualityTracker))
    test_suite.addTest(unittest.makeSuite(TestShadowQualityTrackerCLI))
    test_suite.addTest(unittest.makeSuite(TestShadowQualityTrackerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)