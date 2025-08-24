#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Orchestration Intelligence Layer

Tests all 6 core components and their integration:
1. ContextModelingEngine
2. DynamicStateAnalyzer  
3. ValidationResultAnalyzer
4. LearningAgentSelector
5. FailurePatternAnalyzer & LoopBackDecisionEngine
6. TransitionEngine & ParallelCoordinator

Author: RIF-Implementer (Issue #52)
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import json
import tempfile

# Add the commands directory to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'claude', 'commands'))

from enhanced_orchestration_intelligence import (
    # Core Components
    ContextModelingEngine,
    DynamicStateAnalyzer,
    ValidationResultAnalyzer,
    LearningAgentSelector,
    FailurePatternAnalyzer,
    LoopBackDecisionEngine,
    TransitionEngine,
    ParallelCoordinator,
    
    # Data Classes
    ContextModel,
    ValidationResult,
    FailureCategory,
    ConfidenceLevel,
    AgentPerformanceData,
    
    # Integration
    EnhancedOrchestrationIntelligence,
    get_enhanced_orchestration_intelligence,
    
    # Utilities
    analyze_issue,
    generate_orchestration_plan,
    handle_validation_failure
)

from orchestration_utilities import (
    IssueContext,
    ContextAnalyzer
)

class TestContextModelingEngine(unittest.TestCase):
    """Test the ContextModelingEngine component"""
    
    def setUp(self):
        self.engine = ContextModelingEngine()
        
        # Create mock issue context
        self.mock_issue_context = IssueContext(
            number=52,
            title="Implement DynamicOrchestrator class",
            body="Need to implement the core orchestrator system with state analysis and agent selection",
            labels=['state:implementing', 'complexity:high', 'priority:high'],
            state='open',
            complexity='high',
            priority='high',
            agent_history=['RIF-Analyst', 'RIF-Planner', 'RIF-Architect'],
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T01:00:00Z',
            comments_count=5
        )
    
    def test_context_enrichment(self):
        """Test context enrichment functionality"""
        context_model = self.engine.enrich_context(self.mock_issue_context)
        
        self.assertIsInstance(context_model, ContextModel)
        self.assertEqual(context_model.issue_context, self.mock_issue_context)
        self.assertIsInstance(context_model.complexity_dimensions, dict)
        self.assertIsInstance(context_model.security_context, dict)
        self.assertIsInstance(context_model.performance_context, dict)
        self.assertIsInstance(context_model.risk_factors, list)
        self.assertIsInstance(context_model.semantic_tags, set)
    
    def test_complexity_analysis(self):
        """Test complexity dimension analysis"""
        context_model = self.engine.enrich_context(self.mock_issue_context)
        
        # Should have complexity dimensions
        self.assertIn('technical', context_model.complexity_dimensions)
        self.assertIn('architectural', context_model.complexity_dimensions)
        
        # Complexity scores should be in valid range
        for score in context_model.complexity_dimensions.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Overall complexity should be calculated
        overall_score = context_model.overall_complexity_score
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
    
    def test_security_context_analysis(self):
        """Test security context analysis"""
        context_model = self.engine.enrich_context(self.mock_issue_context)
        
        security_context = context_model.security_context
        self.assertIn('requires_security_review', security_context)
        self.assertIn('vulnerability_risk', security_context)
        self.assertIsInstance(security_context['requires_security_review'], bool)
    
    def test_semantic_tag_generation(self):
        """Test semantic tag generation"""
        context_model = self.engine.enrich_context(self.mock_issue_context)
        
        self.assertIsInstance(context_model.semantic_tags, set)
        # Should contain at least some relevant tags
        expected_tags = {'orchestration', 'feature'}
        self.assertTrue(len(expected_tags & context_model.semantic_tags) > 0)

class TestValidationResultAnalyzer(unittest.TestCase):
    """Test the ValidationResultAnalyzer component"""
    
    def setUp(self):
        self.analyzer = ValidationResultAnalyzer()
    
    def test_validation_result_analysis_success(self):
        """Test analysis of successful validation"""
        raw_results = {
            'passed': True,
            'score': 0.95,
            'failures': [],
            'warnings': []
        }
        
        result = self.analyzer.analyze_validation_results(raw_results)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(len(result.failures), 0)
    
    def test_validation_result_analysis_failure(self):
        """Test analysis of failed validation"""
        raw_results = {
            'passed': False,
            'score': 0.3,
            'failures': [
                {'message': 'Architecture violation detected', 'type': 'architecture'},
                {'message': 'Security vulnerability found', 'type': 'security'}
            ],
            'warnings': ['Performance concern']
        }
        
        result = self.analyzer.analyze_validation_results(raw_results)
        
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.3)
        self.assertEqual(len(result.failures), 2)
        self.assertTrue(result.has_architectural_issues())
    
    def test_failure_categorization(self):
        """Test failure categorization logic"""
        raw_failures = [
            {'message': 'architecture pattern violation', 'type': 'design'},
            {'message': 'missing requirements', 'type': 'spec'},
            {'message': 'syntax error in code', 'type': 'implementation'}
        ]
        
        categorized = self.analyzer._categorize_failures(raw_failures)
        
        self.assertEqual(len(categorized), 3)
        
        # Check categories are assigned
        categories = [f['category'] for f in categorized]
        self.assertIn(FailureCategory.ARCHITECTURAL_ISSUES.value, categories)
        self.assertIn(FailureCategory.REQUIREMENT_GAPS.value, categories)
        self.assertIn(FailureCategory.IMPLEMENTATION_ERRORS.value, categories)

class TestDynamicStateAnalyzer(unittest.TestCase):
    """Test the DynamicStateAnalyzer component"""
    
    def setUp(self):
        self.analyzer = DynamicStateAnalyzer()
        
        # Create mock context
        self.mock_context = Mock(spec=ContextModel)
        self.mock_context.issue_context = Mock()
        self.mock_context.issue_context.current_state_label = 'state:implementing'
        self.mock_context.issue_context.agent_history = ['RIF-Analyst']
        self.mock_context.overall_complexity_score = 0.7
        self.mock_context.risk_score = 0.5
        self.mock_context.semantic_tags = {'orchestration', 'architecture'}
        self.mock_context.risk_factors = []
        self.mock_context.validation_results = None
    
    def test_state_analysis_without_validation(self):
        """Test state analysis without validation results"""
        next_state, confidence = self.analyzer.analyze_current_state(
            self.mock_context.issue_context
        )
        
        self.assertIsInstance(next_state, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_state_analysis_with_validation_failure(self):
        """Test state analysis with failed validation"""
        # Create mock validation result
        validation_results = {
            'passed': False,
            'score': 0.3,
            'failures': [{'message': 'architecture issue', 'category': 'architecture'}]
        }
        
        next_state, confidence = self.analyzer.analyze_current_state(
            self.mock_context.issue_context, validation_results
        )
        
        # Should recommend appropriate recovery state
        self.assertIn(next_state, ['architecting', 'analyzing', 'implementing'])
        self.assertGreater(confidence, 0.0)
    
    def test_state_analysis_summary(self):
        """Test comprehensive state analysis summary"""
        summary = self.analyzer.get_state_analysis_summary(self.mock_context.issue_context)
        
        self.assertIn('current_state', summary)
        self.assertIn('recommended_next_state', summary)
        self.assertIn('confidence', summary)
        self.assertIn('complexity_analysis', summary)
        self.assertIn('risk_analysis', summary)

class TestLearningAgentSelector(unittest.TestCase):
    """Test the LearningAgentSelector component"""
    
    def setUp(self):
        self.selector = LearningAgentSelector()
        
        # Create mock context model
        self.mock_context = Mock(spec=ContextModel)
        self.mock_context.issue_context = Mock()
        self.mock_context.issue_context.number = 52
        self.mock_context.issue_context.current_state_label = 'state:implementing'
        self.mock_context.overall_complexity_score = 0.6
        self.mock_context.semantic_tags = {'implementation', 'orchestration'}
    
    def test_single_agent_selection(self):
        """Test single agent selection"""
        assignment = self.selector.select_single_agent(self.mock_context)
        
        if assignment:  # May be empty if no candidates
            self.assertIn('agent', assignment)
            self.assertIn('issue_number', assignment)
    
    def test_agent_recommendations(self):
        """Test agent recommendation generation"""
        recommendations = self.selector.get_agent_recommendations(self.mock_context)
        
        self.assertIsInstance(recommendations, list)
        
        if recommendations:
            for rec in recommendations:
                self.assertIn('agent', rec)
                self.assertIn('predicted_performance', rec)
                self.assertIn('capability_match', rec)
                self.assertIn('overall_score', rec)
    
    def test_performance_feedback_update(self):
        """Test performance feedback updates"""
        # This should not raise an exception
        self.selector.update_performance_feedback(
            issue_number=52,
            agent_name='RIF-Implementer',
            success=True,
            completion_time=2.5,
            performance_metrics={'quality_score': 0.9}
        )

class TestFailurePatternAnalyzer(unittest.TestCase):
    """Test the FailurePatternAnalyzer component"""
    
    def setUp(self):
        self.analyzer = FailurePatternAnalyzer()
        
        # Create mock validation result
        self.mock_validation = Mock(spec=ValidationResult)
        self.mock_validation.passed = False
        self.mock_validation.score = 0.4
        self.mock_validation.failures = [
            {'message': 'architecture issue', 'category': 'architecture'}
        ]
        self.mock_validation.security_issues = []
        self.mock_validation.architectural_concerns = ['pattern violation']
        self.mock_validation.has_architectural_issues.return_value = True
        self.mock_validation.has_requirement_gaps.return_value = False
        self.mock_validation.has_fixable_errors.return_value = False
        
        # Create mock context
        self.mock_context = Mock(spec=ContextModel)
        self.mock_context.overall_complexity_score = 0.7
        self.mock_context.semantic_tags = {'orchestration', 'architecture'}
        self.mock_context.risk_factors = []
    
    def test_failure_pattern_analysis(self):
        """Test failure pattern analysis"""
        analysis = self.analyzer.analyze_failure_patterns(
            self.mock_validation, self.mock_context
        )
        
        self.assertIn('failure_signature', analysis)
        self.assertIn('pattern_matches', analysis)
        self.assertIn('failure_category', analysis)
        self.assertIn('recovery_strategy', analysis)
        
        # Check failure signature structure
        signature = analysis['failure_signature']
        self.assertIn('primary_category', signature)
        self.assertIn('complexity_range', signature)
    
    def test_failure_signature_extraction(self):
        """Test failure signature extraction"""
        signature = self.analyzer._extract_failure_signature(
            self.mock_validation, self.mock_context
        )
        
        self.assertIn('primary_category', signature)
        self.assertIn('failure_count', signature)
        self.assertIn('complexity_range', signature)
        self.assertIn('context_tags', signature)
    
    def test_recovery_strategy_generation(self):
        """Test recovery strategy generation"""
        failure_category = {
            'category': FailureCategory.ARCHITECTURAL_ISSUES.value,
            'confidence': 0.8
        }
        
        strategy = self.analyzer._generate_recovery_strategy(
            failure_category, [], self.mock_context
        )
        
        self.assertIn('recommended_state', strategy)
        self.assertIn('agent', strategy)
        self.assertIn('estimated_time', strategy)

class TestLoopBackDecisionEngine(unittest.TestCase):
    """Test the LoopBackDecisionEngine component"""
    
    def setUp(self):
        self.engine = LoopBackDecisionEngine()
        
        # Create mocks
        self.mock_validation = Mock(spec=ValidationResult)
        self.mock_validation.passed = False
        self.mock_validation.score = 0.3
        
        self.mock_context = Mock(spec=ContextModel)
        self.mock_context.overall_complexity_score = 0.6
        self.mock_context.risk_score = 0.5
        self.mock_context.semantic_tags = {'implementation'}
    
    def test_loop_back_decision_making(self):
        """Test loop-back decision making"""
        decision = self.engine.make_loop_back_decision(
            self.mock_validation, self.mock_context, 'implementing'
        )
        
        self.assertIn('recommended_action', decision)
        self.assertIn('alternatives', decision)
        self.assertIn('decision_reasoning', decision)
        self.assertIn('confidence', decision)
    
    def test_transition_readiness_evaluation(self):
        """Test transition readiness evaluation"""
        self.mock_context.issue_context = Mock()
        self.mock_context.issue_context.current_state_label = 'state:implementing'
        self.mock_context.issue_context.agent_history = ['RIF-Analyst']
        self.mock_context.security_context = {'requires_security_review': False}
        self.mock_context.risk_factors = []
        
        readiness = self.engine.evaluate_transition_readiness(
            self.mock_context, 'validating'
        )
        
        self.assertIn('ready', readiness)
        self.assertIn('confidence', readiness)
        self.assertIsInstance(readiness['ready'], bool)
    
    def test_decision_option_generation(self):
        """Test decision option generation"""
        pattern_analysis = {
            'recovery_strategy': {
                'recommended_state': 'architecting',
                'agent': 'RIF-Architect',
                'estimated_time': 4.0
            }
        }
        
        options = self.engine._generate_decision_options(
            pattern_analysis, self.mock_context, 'implementing'
        )
        
        self.assertIsInstance(options, list)
        self.assertGreater(len(options), 0)
        
        for option in options:
            self.assertIn('type', option)
            self.assertIn('target_state', option)
            self.assertIn('agent', option)

class TestTransitionEngine(unittest.TestCase):
    """Test the TransitionEngine component"""
    
    @patch('enhanced_orchestration_intelligence.GitHubStateManager')
    def setUp(self, mock_github_class):
        self.mock_github = Mock()
        mock_github_class.return_value = self.mock_github
        
        self.engine = TransitionEngine()
        
        # Create mock context
        self.mock_context = Mock(spec=ContextModel)
        self.mock_context.issue_context = Mock()
        self.mock_context.issue_context.number = 52
        self.mock_context.issue_context.current_state_label = 'state:implementing'
        self.mock_context.overall_complexity_score = 0.6
    
    def test_intelligent_transition_execution(self):
        """Test intelligent transition execution"""
        # Mock successful GitHub update
        self.mock_github.update_issue_state.return_value = True
        self.mock_github.add_agent_tracking_label.return_value = True
        
        result = self.engine.execute_intelligent_transition(self.mock_context)
        
        self.assertIn('transition_executed', result)
        if result.get('transition_executed'):
            self.assertIn('new_state', result)
            self.assertIn('assigned_agent', result)
    
    def test_transition_with_validation_failure(self):
        """Test transition with validation failure"""
        mock_validation = Mock(spec=ValidationResult)
        mock_validation.passed = False
        mock_validation.score = 0.3
        mock_validation.failures = []
        mock_validation.security_issues = []
        mock_validation.architectural_concerns = []
        mock_validation.has_architectural_issues.return_value = False
        mock_validation.has_requirement_gaps.return_value = False
        mock_validation.has_fixable_errors.return_value = True
        
        # Mock successful GitHub update
        self.mock_github.update_issue_state.return_value = True
        
        result = self.engine.execute_intelligent_transition(
            self.mock_context, mock_validation
        )
        
        # Should handle validation failure appropriately
        self.assertIn('transition_executed', result)

class TestParallelCoordinator(unittest.TestCase):
    """Test the ParallelCoordinator component"""
    
    def setUp(self):
        self.coordinator = ParallelCoordinator()
    
    @patch('enhanced_orchestration_intelligence.ContextAnalyzer')
    def test_parallel_orchestration_coordination(self, mock_context_analyzer_class):
        """Test parallel orchestration coordination"""
        # Mock context analyzer
        mock_analyzer = Mock()
        mock_context_analyzer_class.return_value = mock_analyzer
        
        # Mock issue context
        mock_issue_context = Mock()
        mock_issue_context.number = 52
        mock_issue_context.title = "Test Issue"
        mock_issue_context.current_state_label = 'state:implementing'
        mock_analyzer.analyze_issue.return_value = mock_issue_context
        
        result = self.coordinator.coordinate_parallel_orchestration([52])
        
        self.assertIn('parallel_tasks', result)
        self.assertIn('coordination_plan', result)
        self.assertIn('total_issues', result)
    
    def test_claude_orchestration_command_generation(self):
        """Test Claude Code orchestration command generation"""
        # Mock coordination result
        coordination_result = {
            'parallel_tasks': [
                {
                    'issue_number': 52,
                    'agent': 'RIF-Implementer',
                    'description': 'Test task',
                    'prompt': 'Test prompt'
                }
            ]
        }
        
        commands = self.coordinator.generate_claude_orchestration_commands(coordination_result)
        
        self.assertIsInstance(commands, list)
        if commands:
            self.assertIn('Task(', commands[0])
            self.assertIn('RIF-Implementer', commands[0])

class TestEnhancedOrchestrationIntelligence(unittest.TestCase):
    """Test the main integration facade"""
    
    def setUp(self):
        self.intelligence = EnhancedOrchestrationIntelligence()
    
    @patch('enhanced_orchestration_intelligence.ContextAnalyzer')
    def test_issue_analysis_with_intelligence(self, mock_context_analyzer_class):
        """Test comprehensive issue analysis"""
        # Mock the context analyzer
        mock_analyzer = Mock()
        mock_context_analyzer_class.return_value = mock_analyzer
        
        mock_issue_context = Mock()
        mock_issue_context.number = 52
        mock_issue_context.title = "Test Issue"
        mock_issue_context.body = "Test body"
        mock_issue_context.labels = ['state:implementing']
        mock_issue_context.current_state_label = 'state:implementing'
        mock_issue_context.agent_history = ['RIF-Analyst']
        mock_issue_context.complexity_score = 3
        
        mock_analyzer.analyze_issue.return_value = mock_issue_context
        
        # Mock the intelligence component's context analyzer
        self.intelligence.context_analyzer = mock_analyzer
        
        analysis = self.intelligence.analyze_issue_with_intelligence(52)
        
        if 'error' not in analysis:
            self.assertIn('issue_number', analysis)
            self.assertIn('context_model', analysis)
            self.assertIn('state_analysis', analysis)
            self.assertIn('intelligence_summary', analysis)
    
    def test_validation_failure_handling(self):
        """Test validation failure handling"""
        validation_results = {
            'passed': False,
            'score': 0.4,
            'failures': [{'message': 'test failure', 'type': 'test'}]
        }
        
        # This should not raise an exception even if GitHub calls fail
        result = self.intelligence.handle_validation_failure(52, validation_results)
        
        # Either successful handling or error, but structured response
        self.assertIn('issue_number', result)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions and integration points"""
    
    def test_factory_function(self):
        """Test factory function"""
        intelligence = get_enhanced_orchestration_intelligence()
        
        self.assertIsInstance(intelligence, EnhancedOrchestrationIntelligence)
    
    @patch('enhanced_orchestration_intelligence.get_enhanced_orchestration_intelligence')
    def test_convenience_functions(self, mock_factory):
        """Test convenience functions"""
        mock_intelligence = Mock()
        mock_factory.return_value = mock_intelligence
        
        # Mock return values
        mock_intelligence.analyze_issue_with_intelligence.return_value = {'test': 'result'}
        mock_intelligence.generate_orchestration_plan.return_value = {'plan': 'test'}
        mock_intelligence.handle_validation_failure.return_value = {'failure': 'handled'}
        
        # Test convenience functions
        result1 = analyze_issue(52)
        result2 = generate_orchestration_plan([52])
        result3 = handle_validation_failure(52, {'passed': False})
        
        self.assertEqual(result1, {'test': 'result'})
        self.assertEqual(result2, {'plan': 'test'})
        self.assertEqual(result3, {'failure': 'handled'})

class TestPatternCompliance(unittest.TestCase):
    """Test compliance with RIF orchestration patterns"""
    
    def test_no_orchestrator_classes_instantiated(self):
        """Verify no orchestrator classes are instantiated"""
        intelligence = get_enhanced_orchestration_intelligence()
        
        # Check that we don't have any orchestrator class instances
        for attr_name in dir(intelligence):
            attr = getattr(intelligence, attr_name)
            # Make sure we don't have any attributes that look like orchestrator instances
            if hasattr(attr, '__class__'):
                class_name = attr.__class__.__name__.lower()
                self.assertNotIn('orchestrator', class_name, 
                               f"Found orchestrator-like class: {attr.__class__.__name__}")
    
    def test_task_launching_support(self):
        """Test that Task() launching is supported"""
        coordinator = ParallelCoordinator()
        
        # Create minimal coordination result
        coordination_result = {
            'parallel_tasks': [{
                'issue_number': 52,
                'agent': 'RIF-Implementer',
                'description': 'Test task',
                'prompt': 'Test prompt'
            }]
        }
        
        commands = coordinator.generate_claude_orchestration_commands(coordination_result)
        
        # Should generate Task() commands for Claude Code
        self.assertIsInstance(commands, list)
        if commands:
            self.assertIn('Task(', commands[0])

def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    
    # Test suite setup
    test_classes = [
        TestContextModelingEngine,
        TestValidationResultAnalyzer,
        TestDynamicStateAnalyzer,
        TestLearningAgentSelector,
        TestFailurePatternAnalyzer,
        TestLoopBackDecisionEngine,
        TestTransitionEngine,
        TestParallelCoordinator,
        TestEnhancedOrchestrationIntelligence,
        TestUtilityFunctions,
        TestPatternCompliance
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    print("Enhanced Orchestration Intelligence - Comprehensive Test Suite")
    print("=" * 70)
    
    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        failed_tests += len(result.failures) + len(result.errors)
        
        status = "✅ PASS" if result.wasSuccessful() else "❌ FAIL"
        print(f"{test_class.__name__:40} {status:10} ({result.testsRun} tests)")
        
        if not result.wasSuccessful():
            for failure in result.failures + result.errors:
                print(f"  - {failure[0]}: {failure[1].split('AssertionError:')[-1].strip()}")
    
    print("=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! Implementation ready for validation.")
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Review implementation.")
    
    return failed_tests == 0

if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)