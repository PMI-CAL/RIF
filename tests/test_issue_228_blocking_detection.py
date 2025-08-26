#!/usr/bin/env python3
"""
Comprehensive Test Suite for Issue #228 - Enhanced Blocking Detection
RIF-Implementer: Critical orchestration failure prevention tests

This test suite validates the complete enhanced blocking detection system
implemented to resolve Issue #228's critical orchestration failure.

Test Coverage:
1. Enhanced Blocking Detection Engine functionality
2. Orchestration Intelligence Integration
3. Pre-flight validation
4. False positive prevention
5. Comment parsing integration
6. End-to-end orchestration scenarios
"""

import unittest
import json
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Add the commands directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "claude" / "commands"))

try:
    from enhanced_orchestration_intelligence import EnhancedBlockingDetectionEngine
    from orchestration_intelligence_integration import (
        make_intelligent_orchestration_decision,
        validate_orchestration_request,
        validate_orchestration_patterns,
        OrchestrationDecision
    )
    from pre_flight_blocking_validator import PreFlightBlockingValidator
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    IMPORTS_AVAILABLE = False

class TestEnhancedBlockingDetectionEngine(unittest.TestCase):
    """Test the core blocking detection engine"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        self.engine = EnhancedBlockingDetectionEngine()
    
    def test_blocking_phrases_detection(self):
        """Test that blocking phrases are correctly defined"""
        expected_phrases = [
            "this issue blocks all others",
            "this issue blocks all other work", 
            "blocks all other work",
            "blocks all others",
            "stop all work",
            "must complete before all",
            "must complete before all other work",
            "must complete before all others"
        ]
        
        self.assertEqual(self.engine.blocking_phrases, expected_phrases)
    
    @patch('enhanced_orchestration_intelligence.subprocess.run')
    def test_detect_blocking_issues_with_blocking_issue(self, mock_subprocess):
        """Test detection of issues with blocking declarations"""
        # Mock GitHub API response for issue with blocking declaration
        mock_issue_data = {
            'number': 225,
            'title': 'Critical Issue',
            'body': 'This is a critical issue. THIS ISSUE BLOCKS ALL OTHERS until resolved.',
            'state': 'open',
            'labels': [],
            'comments': []
        }
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_issue_data)
        mock_subprocess.return_value = mock_result
        
        # Test blocking detection
        result = self.engine.detect_blocking_issues([225])
        
        self.assertTrue(result['has_blocking_issues'])
        self.assertEqual(result['blocking_issues'], [225])
        self.assertEqual(result['blocking_count'], 1)
        self.assertIn('225', result['blocking_details'])
        
        # Check detailed analysis
        details = result['blocking_details']['225']
        self.assertEqual(len(details['blocking_sources']), 1)
        self.assertEqual(details['blocking_sources'][0]['source'], 'issue_body')
        self.assertIn('this issue blocks all others', details['detected_phrases'])
    
    @patch('enhanced_orchestration_intelligence.subprocess.run')
    def test_detect_blocking_issues_in_comments(self, mock_subprocess):
        """Test detection of blocking declarations in comments"""
        mock_issue_data = {
            'number': 226,
            'title': 'Regular Issue',
            'body': 'This is a regular issue description.',
            'state': 'open',
            'labels': [],
            'comments': [
                {
                    'body': 'Updated: STOP ALL WORK until this is fixed!',
                    'author': {'login': 'maintainer'}
                }
            ]
        }
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_issue_data)
        mock_subprocess.return_value = mock_result
        
        result = self.engine.detect_blocking_issues([226])
        
        self.assertTrue(result['has_blocking_issues'])
        self.assertEqual(result['blocking_issues'], [226])
        
        # Check comment detection
        details = result['blocking_details']['226']
        comment_sources = [s for s in details['blocking_sources'] if s['source'].startswith('comment')]
        self.assertEqual(len(comment_sources), 1)
        self.assertEqual(comment_sources[0]['author'], 'maintainer')
        self.assertIn('stop all work', details['detected_phrases'])
    
    @patch('enhanced_orchestration_intelligence.subprocess.run')
    def test_no_false_positives(self, mock_subprocess):
        """Test that generic urgent terms don't trigger false positives"""
        mock_issue_data = {
            'number': 227,
            'title': 'Urgent Critical Issue',
            'body': 'This is a critical and urgent issue that needs immediate attention. It is blocking some functionality.',
            'state': 'open',
            'labels': [],
            'comments': [
                {
                    'body': 'This is really important and urgent!',
                    'author': {'login': 'user'}
                }
            ]
        }
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_issue_data)
        mock_subprocess.return_value = mock_result
        
        result = self.engine.detect_blocking_issues([227])
        
        # Should NOT detect blocking - these are generic terms
        self.assertFalse(result['has_blocking_issues'])
        self.assertEqual(result['blocking_issues'], [])
        self.assertEqual(result['non_blocking_issues'], [227])
    
    @patch('enhanced_orchestration_intelligence.subprocess.run')
    def test_multiple_issues_mixed_blocking(self, mock_subprocess):
        """Test detection with mix of blocking and non-blocking issues"""
        def mock_subprocess_side_effect(*args, **kwargs):
            issue_num = int(args[0][3])  # Extract issue number from gh command
            
            if issue_num == 225:
                mock_data = {
                    'number': 225,
                    'body': 'BLOCKS ALL OTHERS - critical infrastructure',
                    'comments': []
                }
            elif issue_num == 226:
                mock_data = {
                    'number': 226,
                    'body': 'Regular feature request',
                    'comments': []
                }
            else:  # 227
                mock_data = {
                    'number': 227,
                    'body': 'Another regular issue',
                    'comments': [{'body': 'MUST COMPLETE BEFORE ALL other features', 'author': {'login': 'admin'}}]
                }
            
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(mock_data)
            return mock_result
        
        mock_subprocess.side_effect = mock_subprocess_side_effect
        
        result = self.engine.detect_blocking_issues([225, 226, 227])
        
        self.assertTrue(result['has_blocking_issues'])
        self.assertEqual(set(result['blocking_issues']), {225, 227})
        self.assertEqual(result['non_blocking_issues'], [226])
        self.assertEqual(result['blocking_count'], 2)

class TestOrchestrationIntelligenceIntegration(unittest.TestCase):
    """Test the orchestration intelligence integration module"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_orchestration_decision_structure(self):
        """Test that OrchestrationDecision has required fields"""
        decision = OrchestrationDecision(
            decision_type="test",
            enforcement_action="TEST",
            blocking_issues=[],
            allowed_issues=[],
            blocked_issues=[],
            dependency_rationale="test",
            blocking_analysis={},
            task_launch_codes=[],
            execution_ready=True,
            parallel_execution=False
        )
        
        decision_dict = decision.to_dict()
        required_fields = [
            'decision_type', 'enforcement_action', 'blocking_issues',
            'allowed_issues', 'blocked_issues', 'dependency_rationale',
            'blocking_analysis', 'task_launch_codes', 'execution_ready',
            'parallel_execution', 'timestamp'
        ]
        
        for field in required_fields:
            self.assertIn(field, decision_dict)
    
    @patch('orchestration_intelligence_integration.ENHANCED_INTELLIGENCE_AVAILABLE', True)
    @patch('orchestration_intelligence_integration.EnhancedOrchestrationIntelligence')
    def test_make_intelligent_orchestration_decision_with_blocking(self, mock_intelligence_class):
        """Test orchestration decision when blocking issues are detected"""
        mock_intelligence = MagicMock()
        mock_intelligence.generate_orchestration_plan.return_value = {
            'orchestration_blocked': True,
            'blocking_issues': [225],
            'allowed_issues': [225],
            'blocked_issues': [226, 227],
            'message': 'Orchestration BLOCKED - 1 issues require completion first',
            'blocking_analysis': {
                'has_blocking_issues': True,
                'blocking_count': 1,
                'blocking_details': {
                    '225': {'detected_phrases': ['this issue blocks all others']}
                }
            },
            'task_launch_codes': ['Task(description="Fix blocking issue", ...)'],
            'execution_ready': True,
            'parallel_execution': False
        }
        mock_intelligence_class.return_value = mock_intelligence
        
        decision = make_intelligent_orchestration_decision([225, 226, 227])
        
        self.assertEqual(decision.decision_type, "block_execution")
        self.assertEqual(decision.enforcement_action, "HALT_ALL_ORCHESTRATION")
        self.assertEqual(decision.blocking_issues, [225])
        self.assertEqual(decision.blocked_issues, [226, 227])
        self.assertTrue(decision.execution_ready)
        self.assertFalse(decision.parallel_execution)
    
    @patch('orchestration_intelligence_integration.ENHANCED_INTELLIGENCE_AVAILABLE', True)
    @patch('orchestration_intelligence_integration.EnhancedOrchestrationIntelligence')
    def test_make_intelligent_orchestration_decision_without_blocking(self, mock_intelligence_class):
        """Test orchestration decision when no blocking issues detected"""
        mock_intelligence = MagicMock()
        mock_intelligence.generate_orchestration_plan.return_value = {
            'orchestration_blocked': False,
            'blocking_analysis': {'has_blocking_issues': False, 'blocking_count': 0},
            'task_launch_codes': ['Task(...)', 'Task(...)'],
            'execution_ready': True,
            'parallel_execution': True
        }
        mock_intelligence_class.return_value = mock_intelligence
        
        decision = make_intelligent_orchestration_decision([226, 227])
        
        self.assertEqual(decision.decision_type, "allow_execution")
        self.assertEqual(decision.enforcement_action, "ALLOW")
        self.assertEqual(decision.blocking_issues, [])
        self.assertEqual(decision.allowed_issues, [226, 227])
        self.assertTrue(decision.execution_ready)
        self.assertTrue(decision.parallel_execution)
    
    @patch('orchestration_intelligence_integration.ENHANCED_INTELLIGENCE_AVAILABLE', True)
    @patch('orchestration_intelligence_integration.EnhancedBlockingDetectionEngine')
    def test_validate_orchestration_request(self, mock_engine_class):
        """Test pre-flight orchestration validation"""
        mock_engine = MagicMock()
        mock_engine.detect_blocking_issues.return_value = {
            'has_blocking_issues': True,
            'blocking_issues': [225],
            'non_blocking_issues': [226, 227],
            'blocking_details': {
                '225': {'detected_phrases': ['this issue blocks all others']}
            }
        }
        mock_engine_class.return_value = mock_engine
        
        should_block, message = validate_orchestration_request([225, 226, 227])
        
        self.assertTrue(should_block)
        self.assertIn("ORCHESTRATION BLOCKED", message)
        self.assertIn("Issue #225", message)
        self.assertIn("Complete blocking issues before proceeding", message)

class TestPreFlightBlockingValidator(unittest.TestCase):
    """Test the standalone pre-flight blocking validator"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        self.validator = PreFlightBlockingValidator()
    
    def test_blocking_phrases_match_engine(self):
        """Test that validator uses same blocking phrases as engine"""
        if IMPORTS_AVAILABLE:
            engine = EnhancedBlockingDetectionEngine()
            self.assertEqual(self.validator.blocking_phrases, engine.blocking_phrases)
    
    @patch('pre_flight_blocking_validator.subprocess.run')
    def test_validate_single_issue_blocking(self, mock_subprocess):
        """Test validation of single blocking issue"""
        mock_issue_data = {
            'number': 225,
            'body': 'Emergency fix: THIS ISSUE BLOCKS ALL OTHER WORK',
            'comments': []
        }
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_issue_data)
        mock_subprocess.return_value = mock_result
        
        result = self.validator._validate_single_issue(225)
        
        self.assertTrue(result['is_blocking'])
        self.assertEqual(len(result['blocking_sources']), 1)
        self.assertEqual(result['blocking_sources'][0]['source_type'], 'issue_body')
        self.assertIn('this issue blocks all other work', result['blocking_sources'][0]['detected_phrases'])
        self.assertEqual(result['confidence'], 1.0)
    
    def test_extract_evidence(self):
        """Test evidence extraction from text"""
        text = "This is some text with THIS ISSUE BLOCKS ALL OTHERS in the middle and more text after."
        phrase = "this issue blocks all others"
        
        evidence = self.validator._extract_evidence(text, phrase, context_chars=50)
        
        self.assertIn(phrase.upper(), evidence)
        self.assertTrue(len(evidence) <= 60)  # Should be roughly context_chars + phrase length
    
    @patch('pre_flight_blocking_validator.subprocess.run')
    def test_validate_issues_for_blocking_comprehensive(self, mock_subprocess):
        """Test comprehensive validation of multiple issues"""
        def mock_subprocess_side_effect(*args, **kwargs):
            issue_num = int(args[0][3])  # Extract issue number
            
            if issue_num == 225:
                mock_data = {'number': 225, 'body': 'BLOCKS ALL OTHERS', 'comments': []}
            else:
                mock_data = {'number': issue_num, 'body': 'Regular issue', 'comments': []}
            
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(mock_data)
            return mock_result
        
        mock_subprocess.side_effect = mock_subprocess_side_effect
        
        result = self.validator.validate_issues_for_blocking([225, 226, 227])
        
        self.assertTrue(result['has_blocking_issues'])
        self.assertEqual(result['blocking_issues'], [225])
        self.assertEqual(set(result['non_blocking_issues']), {226, 227})
        self.assertEqual(result['enforcement_recommendation'], 'HALT_ALL_ORCHESTRATION')
        self.assertEqual(result['total_issues_checked'], 3)
        
        # Check detailed analysis
        self.assertIn('225', result['detailed_analysis'])
        self.assertTrue(result['detailed_analysis']['225']['is_blocking'])

class TestOrchestrationPatternValidation(unittest.TestCase):
    """Test orchestration pattern validation to prevent anti-patterns"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_validate_good_orchestration_patterns(self):
        """Test validation passes for good orchestration patterns"""
        good_tasks = [
            {
                'description': 'RIF-Implementer: Implement user authentication',
                'prompt': 'You are RIF-Implementer. Implement user auth for issue #123. Follow all instructions in claude/agents/rif-implementer.md.'
            },
            {
                'description': 'RIF-Validator: Validate API endpoints',
                'prompt': 'You are RIF-Validator. Validate API endpoints for issue #124. Follow all instructions in claude/agents/rif-validator.md.'
            }
        ]
        
        result = validate_orchestration_patterns(good_tasks)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.violations), 0)
        self.assertEqual(len(result.suggestions), 0)
    
    def test_detect_multi_issue_anti_pattern(self):
        """Test detection of multi-issue accelerator anti-pattern"""
        bad_tasks = [
            {
                'description': 'Multi-Issue Accelerator: Handle issues #1, #2, #3',
                'prompt': 'Handle multiple issues in parallel'
            }
        ]
        
        result = validate_orchestration_patterns(bad_tasks)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('multi-issue' in v.lower() for v in result.violations))
        self.assertTrue(any('separate tasks' in s.lower() for s in result.suggestions))
    
    def test_detect_missing_agent_role(self):
        """Test detection of missing agent role specification"""
        bad_tasks = [
            {
                'description': 'Work on issue #123',
                'prompt': 'Please implement this feature for issue #123.'
            }
        ]
        
        result = validate_orchestration_patterns(bad_tasks)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('agent role' in v.lower() for v in result.violations))
        self.assertTrue(any('you are' in s.lower() for s in result.suggestions))
    
    def test_detect_missing_agent_file_reference(self):
        """Test detection of missing agent file reference"""
        bad_tasks = [
            {
                'description': 'RIF-Implementer: Work on issue',
                'prompt': 'You are RIF-Implementer. Implement the feature.'
            }
        ]
        
        result = validate_orchestration_patterns(bad_tasks)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('agent file reference' in v.lower() for v in result.violations))
        self.assertTrue(any('claude/agents/' in s.lower() for s in result.suggestions))

class TestEndToEndOrchestrationScenarios(unittest.TestCase):
    """Test end-to-end orchestration scenarios"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    @patch('orchestration_intelligence_integration.ENHANCED_INTELLIGENCE_AVAILABLE', True)
    @patch('orchestration_intelligence_integration.EnhancedOrchestrationIntelligence')
    def test_issue_225_scenario_resolution(self, mock_intelligence_class):
        """Test the exact Issue #225 scenario that caused the original failure"""
        # This tests the exact scenario: Issue #225 declares "THIS ISSUE BLOCKS ALL OTHERS"
        # but orchestrator ignored it and proceeded with issues #226 and #227
        
        mock_intelligence = MagicMock()
        mock_intelligence.generate_orchestration_plan.return_value = {
            'orchestration_blocked': True,
            'blocking_issues': [225],
            'allowed_issues': [225],
            'blocked_issues': [226, 227],
            'message': 'Orchestration BLOCKED - Issue #225 blocks all others',
            'blocking_analysis': {
                'has_blocking_issues': True,
                'blocking_count': 1,
                'blocking_details': {
                    '225': {
                        'blocking_sources': [{
                            'source': 'comment_0',
                            'author': 'PMI-CAL',
                            'detected_phrases': ['this issue blocks all others', 'blocks all others']
                        }]
                    }
                }
            },
            'task_launch_codes': [
                'Task(description="Resolve BLOCKING issue #225", subagent_type="general-purpose", prompt="You are RIF-Implementer. Resolve BLOCKING issue #225. Follow all instructions in claude/agents/rif-implementer.md.")'
            ],
            'execution_ready': True,
            'parallel_execution': False
        }
        mock_intelligence_class.return_value = mock_intelligence
        
        # Test the fixed behavior
        decision = make_intelligent_orchestration_decision([225, 226, 227])
        
        # Verify the fix prevents the original failure
        self.assertEqual(decision.enforcement_action, "HALT_ALL_ORCHESTRATION")
        self.assertEqual(decision.blocking_issues, [225])
        self.assertEqual(decision.blocked_issues, [226, 227])
        self.assertEqual(len(decision.task_launch_codes), 1)  # Only one task for blocking issue
        self.assertIn("BLOCKING issue #225", decision.task_launch_codes[0])
        
        # Verify that issues #226 and #227 are NOT allowed to proceed
        self.assertNotIn(226, decision.allowed_issues)
        self.assertNotIn(227, decision.allowed_issues)

def run_comprehensive_test_suite():
    """Run all tests and provide summary"""
    print("=" * 80)
    print("🧪 COMPREHENSIVE TEST SUITE: Issue #228 - Enhanced Blocking Detection")
    print("=" * 80)
    
    if not IMPORTS_AVAILABLE:
        print("❌ CRITICAL: Required modules not available for testing")
        print("   Implementation may be incomplete or import paths incorrect")
        return False
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEnhancedBlockingDetectionEngine,
        TestOrchestrationIntelligenceIntegration,
        TestPreFlightBlockingValidator,
        TestOrchestrationPatternValidation,
        TestEndToEndOrchestrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            error_msg = error_lines[-2] if len(error_lines) > 1 else str(traceback)
            print(f"  - {test}: {error_msg}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ ALL TESTS PASSED - Issue #228 implementation validated")
    else:
        print("\n❌ SOME TESTS FAILED - Issue #228 implementation needs fixes")
    
    return success

if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)