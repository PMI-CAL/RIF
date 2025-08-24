#!/usr/bin/env python3
"""
Validation Test Suite for Issues #17-23
Tests all adversarial verification system components
"""

import unittest
import os
import sys
import json
import yaml
from datetime import datetime

# Add project path for imports
sys.path.append('/Users/cal/DEV/RIF')

class TestRiskBasedVerification(unittest.TestCase):
    """Test Issue #17 - Risk-Based Verification"""
    
    def test_rif_validator_agent_content(self):
        """Test RIF-Validator agent has required risk-based content"""
        with open('/Users/cal/DEV/RIF/claude/agents/rif-validator.md', 'r') as f:
            content = f.read()
        
        # Check professional identity
        self.assertIn('Test Architect with Quality Advisory Authority', content)
        self.assertIn('Professional skepticism drives verification depth', content)
        
        # Check risk escalation triggers
        self.assertIn('risk_escalation_triggers:', content)
        self.assertIn('security_files_modified: true', content)
        self.assertIn('authentication_changes: true', content)
        self.assertIn('payment_processing: true', content)
        
        # Check verification depth levels
        self.assertIn('Shallow Verification', content)
        self.assertIn('Standard Verification', content) 
        self.assertIn('Deep Verification', content)
        self.assertIn('Intensive Verification', content)
        
        print("✅ Issue #17: Risk-based verification implemented correctly")

class TestEvidenceFramework(unittest.TestCase):
    """Test Issue #18 - Evidence Requirements Framework"""
    
    def test_evidence_requirements_structure(self):
        """Test evidence requirements framework exists"""
        with open('/Users/cal/DEV/RIF/claude/agents/rif-validator.md', 'r') as f:
            content = f.read()
        
        # Check evidence requirements section exists
        self.assertIn('Evidence Requirements Framework', content)
        self.assertIn('evidence_requirements = {', content)
        
        # Check specific evidence types
        self.assertIn('"feature_complete":', content)
        self.assertIn('"bug_fixed":', content)
        self.assertIn('"performance_improved":', content)
        self.assertIn('"security_validated":', content)
        
        # Check evidence validation process
        self.assertIn('Evidence Validation Process', content)
        self.assertIn('Identify Claim Type', content)
        self.assertIn('Check Required Evidence', content)
        
        print("✅ Issue #18: Evidence requirements framework implemented correctly")

class TestQualityScoring(unittest.TestCase):
    """Test Issue #19 - Quality Scoring System"""
    
    def test_quality_scoring_formula(self):
        """Test quality scoring system implementation"""
        with open('/Users/cal/DEV/RIF/claude/agents/rif-validator.md', 'r') as f:
            content = f.read()
        
        # Check scoring formula exists
        self.assertIn('Quality Score Calculation', content)
        self.assertIn('100 - (20 × FAILs) - (10 × CONCERNs)', content)
        
        # Check gate decision criteria
        self.assertIn('Advisory Decision:', content)
        self.assertIn('PASS/CONCERNS/FAIL/WAIVED', content)
        
        # Check objective methodology
        self.assertIn('Quality Score Methodology', content)
        self.assertIn('Objective Quality Score Formula:', content)
        
        print("✅ Issue #19: Quality scoring system implemented correctly")

class TestShadowQualityTracking(unittest.TestCase):
    """Test Issue #20 - Shadow Quality Tracking System"""
    
    def test_shadow_tracking_implementation(self):
        """Test shadow quality tracking system"""
        shadow_file = '/Users/cal/DEV/RIF/claude/commands/shadow_quality_tracking.py'
        self.assertTrue(os.path.exists(shadow_file))
        
        with open(shadow_file, 'r') as f:
            content = f.read()
        
        # Check key functions exist
        self.assertIn('create_shadow_quality_issue', content)
        self.assertIn('log_quality_activity', content)
        self.assertIn('sync_quality_status', content)
        self.assertIn('close_shadow_issue', content)
        
        # Check RIF-Validator integration
        validator_file = '/Users/cal/DEV/RIF/claude/agents/rif-validator.md'
        with open(validator_file, 'r') as f:
            validator_content = f.read()
        
        self.assertIn('Shadow Quality Tracking', validator_content)
        self.assertIn('create_shadow_quality_issue', validator_content)
        
        print("✅ Issue #20: Shadow quality tracking system implemented correctly")

class TestWorkflowConfiguration(unittest.TestCase):
    """Test Issue #21 - Workflow Configuration Updates"""
    
    def test_workflow_yaml_updates(self):
        """Test workflow configuration has adversarial verification states"""
        with open('/Users/cal/DEV/RIF/config/rif-workflow.yaml', 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check new states exist
        states = workflow.get('workflow', {}).get('states', {})
        self.assertIn('skeptical_review', states)
        self.assertIn('evidence_gathering', states)
        self.assertIn('quality_tracking', states)
        
        # Check quality gates
        quality_gates = workflow.get('workflow', {}).get('quality_gates', {})
        self.assertIn('evidence_requirements', quality_gates)
        self.assertIn('quality_score', quality_gates)
        self.assertIn('risk_assessment', quality_gates)
        
        # Check evidence requirements configuration
        evidence_config = workflow.get('workflow', {}).get('evidence_requirements', {})
        self.assertIn('feature_complete', evidence_config)
        self.assertIn('bug_fixed', evidence_config)
        
        # Check shadow quality tracking configuration
        shadow_config = workflow.get('workflow', {}).get('shadow_quality_tracking', {})
        self.assertTrue(shadow_config.get('enabled', False))
        
        print("✅ Issue #21: Workflow configuration updated correctly")

class TestImplementerEvidence(unittest.TestCase):
    """Test Issue #22 - RIF-Implementer Evidence Generation"""
    
    def test_implementer_evidence_generation(self):
        """Test RIF-Implementer has evidence generation capabilities"""
        with open('/Users/cal/DEV/RIF/claude/agents/rif-implementer.md', 'r') as f:
            content = f.read()
        
        # Check evidence generation section
        self.assertIn('Evidence Generation', content)
        self.assertIn('Test Evidence', content)
        self.assertIn('Coverage Reports', content)
        self.assertIn('Performance Baselines', content)
        
        # Check output format includes evidence
        self.assertIn('Evidence Package', content)
        self.assertIn('Pre-Validation Checklist', content)
        
        # Check evidence collection functions
        self.assertIn('collect_implementation_evidence', content)
        
        print("✅ Issue #22: RIF-Implementer evidence generation implemented correctly")

class TestAnalystDecomposition(unittest.TestCase):
    """Test Issue #23 - RIF-Analyst Granular Decomposition"""
    
    def test_analyst_context_window_analysis(self):
        """Test RIF-Analyst has context window analysis"""
        with open('/Users/cal/DEV/RIF/claude/agents/rif-analyst.md', 'r') as f:
            content = f.read()
        
        # Check context window analysis exists
        self.assertIn('Context Window Analysis', content)
        self.assertIn('analyze_context_requirements', content)
        self.assertIn('needs_decomposition', content)
        
        # Check decomposition strategy
        self.assertIn('Proposed Sub-Issues', content)
        self.assertIn('< 500 LOC', content)
        
        # Check evidence requirements analysis
        self.assertIn('Evidence Requirements Analysis', content)
        
        print("✅ Issue #23: RIF-Analyst granular decomposition implemented correctly")

class TestIntegration(unittest.TestCase):
    """Test integration between all components"""
    
    def test_end_to_end_integration(self):
        """Test components work together"""
        # Check that all agent files reference each other appropriately
        
        # RIF-Validator should reference evidence requirements
        with open('/Users/cal/DEV/RIF/claude/agents/rif-validator.md', 'r') as f:
            validator_content = f.read()
        self.assertIn('evidence_requirements', validator_content)
        
        # RIF-Implementer should reference evidence generation
        with open('/Users/cal/DEV/RIF/claude/agents/rif-implementer.md', 'r') as f:
            implementer_content = f.read()
        self.assertIn('Evidence Package', implementer_content)
        
        # RIF-Analyst should reference decomposition
        with open('/Users/cal/DEV/RIF/claude/agents/rif-analyst.md', 'r') as f:
            analyst_content = f.read()
        self.assertIn('Context Window Analysis', analyst_content)
        
        print("✅ Integration: All components properly integrated")

def run_validation_suite():
    """Run the complete validation suite"""
    print("🔍 Running Adversarial Verification System Validation Suite")
    print("=" * 60)
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestRiskBasedVerification,
        TestEvidenceFramework, 
        TestQualityScoring,
        TestShadowQualityTracking,
        TestWorkflowConfiguration,
        TestImplementerEvidence,
        TestAnalystDecomposition,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            lines = traceback.split('\n')
            error_msg = lines[-2] if len(lines) > 1 else "Unknown failure"
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print("\n🔥 ERRORS:")
        for test, traceback in result.errors:
            lines = traceback.split('\n')
            error_msg = lines[-2] if len(lines) > 1 else "Unknown error"
            print(f"  - {test}: {error_msg}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 ALL TESTS PASSED - Adversarial Verification System validated successfully!")
    else:
        print("\n⚠️  VALIDATION ISSUES DETECTED - See details above")
    
    return success, result

if __name__ == "__main__":
    success, result = run_validation_suite()
    sys.exit(0 if success else 1)