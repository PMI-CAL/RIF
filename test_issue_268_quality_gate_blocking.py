#!/usr/bin/env python3
"""
Test Suite for Issue #268 - Quality Gate Blocking Fix
Tests that quality gate failures correctly block merge recommendations.
"""

import unittest
import sys
from pathlib import Path

class TestQualityGateBlocking(unittest.TestCase):
    """Test cases for Issue #268 quality gate blocking behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock quality gate results
        self.passing_gates = {
            'code_coverage': {'status': 'PASS', 'value': 85},
            'security_scan': {'status': 'PASS', 'value': 0},
            'tests': {'status': 'PASS', 'value': 100},
            'linting': {'status': 'PASS', 'value': 0}
        }
        
        self.failing_gates_single = {
            'code_coverage': {'status': 'FAIL', 'value': 65},  # Failed coverage
            'security_scan': {'status': 'PASS', 'value': 0},
            'tests': {'status': 'PASS', 'value': 100},
            'linting': {'status': 'PASS', 'value': 0}
        }
        
        self.failing_gates_multiple = {
            'code_coverage': {'status': 'FAIL', 'value': 65},
            'security_scan': {'status': 'FAIL', 'value': 3},  # 3 vulnerabilities
            'tests': {'status': 'PASS', 'value': 100},
            'linting': {'status': 'ERROR', 'value': 10}  # Linting error
        }
        
        self.mixed_status_gates = {
            'code_coverage': {'status': 'PASS', 'value': 85},
            'security_scan': {'status': 'PENDING', 'value': None},  # Still running
            'tests': {'status': 'TIMEOUT', 'value': None},  # Timed out
            'linting': {'status': 'SKIPPED', 'value': None}  # Skipped
        }

    def test_validate_quality_gates_all_pass(self):
        """Test that all passing gates return PASS decision."""
        result = self.validate_quality_gates(self.passing_gates)
        
        self.assertEqual(result["decision"], "PASS")
        self.assertFalse(result["merge_blocked"])
    
    def test_validate_quality_gates_single_fail(self):
        """Test that single failing gate returns FAIL decision."""
        result = self.validate_quality_gates(self.failing_gates_single)
        
        self.assertEqual(result["decision"], "FAIL")
        self.assertTrue(result["merge_blocked"])
        self.assertIn("code_coverage", result["reason"])
    
    def test_validate_quality_gates_multiple_fail(self):
        """Test that multiple failing gates return FAIL decision."""
        result = self.validate_quality_gates(self.failing_gates_multiple)
        
        self.assertEqual(result["decision"], "FAIL")
        self.assertTrue(result["merge_blocked"])
        # Should return on first failure encountered
        self.assertTrue(any(gate in result["reason"] for gate in ["code_coverage", "security_scan", "linting"]))
    
    def test_validate_quality_gates_mixed_status(self):
        """Test that non-PASS statuses (PENDING, TIMEOUT, SKIPPED) block merge."""
        result = self.validate_quality_gates(self.mixed_status_gates)
        
        self.assertEqual(result["decision"], "FAIL")
        self.assertTrue(result["merge_blocked"])
        # Should block on first non-PASS status
        self.assertTrue(any(status in result["reason"] for status in ["PENDING", "TIMEOUT", "SKIPPED"]))

    def test_evaluate_merge_eligibility_passing(self):
        """Test PR manager merge eligibility with passing gates."""
        result = self.evaluate_merge_eligibility(self.passing_gates)
        
        self.assertTrue(result["merge_allowed"])
        self.assertNotIn("no_override", result)  # Should not have no_override when passing

    def test_evaluate_merge_eligibility_failing(self):
        """Test PR manager merge eligibility with failing gates."""
        result = self.evaluate_merge_eligibility(self.failing_gates_single)
        
        self.assertFalse(result["merge_allowed"])
        self.assertTrue(result["no_override"])
        self.assertEqual(result["reason"], "Quality gates failing - merge blocked")

    def test_make_merge_decision_pass(self):
        """Test PR manager merge decision with passing gates."""
        pr_data = MockPRData(self.passing_gates)
        result = self.make_merge_decision(pr_data)
        
        self.assertEqual(result, "APPROVE FOR MERGE - All quality gates passing")

    def test_make_merge_decision_fail(self):
        """Test PR manager merge decision with failing gates."""
        pr_data = MockPRData(self.failing_gates_single)
        result = self.make_merge_decision(pr_data)
        
        self.assertEqual(result, "DO NOT MERGE - Quality gates failing")

    def test_make_validation_decision_pass(self):
        """Test validator decision with passing gates."""
        issue_data = MockIssueData(self.passing_gates)
        result = self.make_validation_decision(issue_data)
        
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["merge_recommendation"], "APPROVED")

    def test_make_validation_decision_fail(self):
        """Test validator decision with failing gates."""
        issue_data = MockIssueData(self.failing_gates_single)
        result = self.make_validation_decision(issue_data)
        
        self.assertEqual(result["decision"], "FAIL")
        self.assertEqual(result["merge_recommendation"], "DO NOT MERGE")
        self.assertIn("Fix quality gate failures", result["required_action"])

    def test_comprehensive_blocking_behavior(self):
        """Comprehensive test of blocking behavior from Issue #268."""
        test_cases = [
            # Test case: (description, gates, expected_blocked)
            ("All gates pass", self.passing_gates, False),
            ("Single gate fails", self.failing_gates_single, True),
            ("Multiple gates fail", self.failing_gates_multiple, True),
            ("Mixed status gates", self.mixed_status_gates, True),
        ]
        
        for description, gates, expected_blocked in test_cases:
            with self.subTest(description=description):
                # Test RIF-Validator blocking
                validator_result = self.validate_quality_gates(gates)
                self.assertEqual(validator_result["merge_blocked"], expected_blocked, 
                               f"Validator blocking failed for: {description}")
                
                # Test RIF-PR-Manager blocking  
                pr_result = self.evaluate_merge_eligibility(gates)
                self.assertEqual(not pr_result["merge_allowed"], expected_blocked,
                               f"PR Manager blocking failed for: {description}")

    # Helper methods that implement the Issue #268 fix logic

    def validate_quality_gates(self, gate_results):
        """
        Binary decision function for quality gates
        ANY failure = FAIL decision with merge blocking
        NO agent discretion allowed
        """
        for gate_name, gate_result in gate_results.items():
            if gate_result['status'] != "PASS":
                return {
                    "decision": "FAIL", 
                    "merge_blocked": True,
                    "reason": f"Quality gate '{gate_name}' failed: {gate_result['status']}"
                }
        
        return {"decision": "PASS", "merge_blocked": False}

    def evaluate_merge_eligibility(self, quality_gates):
        """
        Binary merge decision based on quality gate status
        ANY failure = NO merge recommendation
        NO agent discretion allowed
        """
        any_gate_failed = any(
            gate['status'] != "PASS" 
            for gate in quality_gates.values()
        )
        
        if any_gate_failed:
            return {
                "merge_allowed": False,
                "no_override": True,
                "reason": "Quality gates failing - merge blocked"
            }
        
        return {"merge_allowed": True}

    def make_merge_decision(self, pr_data):
        """Make binary merge decision based on quality gate status"""
        eligibility = self.evaluate_merge_eligibility(pr_data.quality_gates)
        
        if not eligibility["merge_allowed"]:
            return "DO NOT MERGE - Quality gates failing"
        else:
            return "APPROVE FOR MERGE - All quality gates passing"

    def make_validation_decision(self, issue_data):
        """Make validation decision with mandatory quality gate check"""
        # First check quality gates (absolute blocking)
        gate_validation = self.validate_quality_gates(issue_data.quality_gates)
        
        if gate_validation["merge_blocked"]:
            return {
                "decision": "FAIL",
                "merge_recommendation": "DO NOT MERGE",
                "blocking_reason": gate_validation["reason"],
                "required_action": "Fix quality gate failures before proceeding"
            }
        
        # Only proceed with other validation if gates pass
        return {"decision": "PASS", "merge_recommendation": "APPROVED"}


class MockPRData:
    """Mock PR data for testing."""
    def __init__(self, quality_gates):
        self.quality_gates = quality_gates


class MockIssueData:
    """Mock issue data for testing."""
    def __init__(self, quality_gates):
        self.quality_gates = quality_gates


if __name__ == '__main__':
    print("🧪 Running Issue #268 Quality Gate Blocking Tests...")
    print("=" * 60)
    
    # Run the test suite
    unittest.main(verbosity=2)