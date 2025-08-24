#!/usr/bin/env python3
"""
Enhanced Dynamic Orchestrator Test Suite
Issue #52: Comprehensive testing for enhanced DynamicOrchestrator implementation

This module provides comprehensive testing for the enhanced DynamicOrchestrator
including intelligence components, learning capabilities, and loop-back mechanisms.
"""

import pytest
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch

# Add the commands directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'claude' / 'commands'))

try:
    from dynamic_orchestrator import (
        DynamicOrchestrator, 
        ContextModel, 
        ContextModelingEngine,
        ValidationResultAnalyzer,
        EnhancedStateAnalyzer,
        LearningAgentSelector,
        TeamOptimizationEngine,
        PerformanceTrackingSystem,
        FailurePatternAnalyzer,
        LoopBackDecisionEngine,
        RecoveryStrategySelector,
        MockValidationSuccess,
        MockValidationFailure
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


class TestContextModel:
    """Test suite for ContextModel dataclass."""
    
    def test_context_model_creation(self):
        """Test basic context model creation."""
        model = ContextModel(
            github_issues=[1, 2, 3],
            complexity='high',
            priority=2
        )
        
        assert model.github_issues == [1, 2, 3]
        assert model.complexity == 'high'
        assert model.priority == 2
        assert model.retry_count == 0  # Default value
        
    def test_context_model_serialization(self):
        """Test context model serialization to/from dict."""
        original = ContextModel(
            github_issues=[1, 2],
            complexity='medium',
            error_history=['Error 1', 'Error 2'],
            performance_metrics={'speed': 0.8, 'quality': 0.9}
        )
        
        # Test to_dict
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data['github_issues'] == [1, 2]
        assert data['complexity'] == 'medium'
        assert data['error_history'] == ['Error 1', 'Error 2']
        
        # Test from_dict
        restored = ContextModel.from_dict(data)
        assert restored.github_issues == original.github_issues
        assert restored.complexity == original.complexity
        assert restored.error_history == original.error_history
        assert restored.performance_metrics == original.performance_metrics


class TestContextModelingEngine:
    """Test suite for ContextModelingEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ContextModelingEngine()
    
    def test_basic_context_modeling(self):
        """Test basic context modeling functionality."""
        context_data = {
            'github_issues': [1, 2, 3],
            'complexity': 'medium',
            'workflow_type': 'standard',
            'priority': 2
        }
        
        model = self.engine.model_context(context_data)
        
        assert isinstance(model, ContextModel)
        assert model.github_issues == [1, 2, 3]
        assert model.complexity == 'medium'
        assert model.workflow_type == 'standard'
        assert model.priority == 2
        
        # Should have pattern matches and confidence scores
        assert hasattr(model, 'pattern_matches')
        assert hasattr(model, 'confidence_scores')
        
    def test_complexity_analysis(self):
        """Test complexity factor analysis."""
        context = ContextModel(
            github_issues=[1, 2, 3, 4, 5],  # Many issues
            error_history=['Error 1', 'Error 2', 'Error 3'],  # Multiple errors
            retry_count=2,  # Some retries
            workflow_type='complex_system'
        )
        
        analysis = self.engine._analyze_complexity_factors(context, {})
        
        assert 'issue_count' in analysis
        assert 'error_frequency' in analysis
        assert 'retry_rate' in analysis
        assert 'complexity_score' in analysis
        assert 'recommended_complexity' in analysis
        
        # Should recommend higher complexity due to multiple factors
        assert analysis['complexity_score'] > 0.3
        
    def test_security_analysis(self):
        """Test security factor analysis."""
        context = ContextModel()
        raw_data = {
            'security_critical': True,
            'compliance_required': True,
            'data_sensitivity': 'high',
            'external_dependencies': ['dep1', 'dep2', 'dep3', 'dep4', 'dep5', 'dep6']
        }
        
        analysis = self.engine._analyze_security_factors(context, raw_data)
        
        assert 'security_critical' in analysis
        assert 'security_risk_score' in analysis
        assert 'requires_security_agent' in analysis
        
        # High security risk should trigger security agent
        assert analysis['requires_security_agent'] == True
        assert analysis['security_risk_score'] > 0.5
        
    def test_error_pattern_identification(self):
        """Test error pattern identification."""
        error_history = [
            'Connection timeout occurred',
            'Network timeout exceeded',
            'Authentication failed - permission denied',
            'Memory out of bounds error',
            'Validation failed for input data'
        ]
        
        patterns = self.engine._identify_error_patterns(error_history)
        
        assert 'timeout' in patterns
        assert 'authentication' in patterns
        assert 'memory' in patterns
        assert 'validation' in patterns


class TestValidationResultAnalyzer:
    """Test suite for ValidationResultAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ValidationResultAnalyzer()
        self.context = ContextModel(complexity='medium', retry_count=1)
    
    def test_successful_validation_analysis(self):
        """Test analysis of successful validation results."""
        validation_result = MockValidationSuccess()
        
        analysis = self.analyzer.analyze_validation_results(validation_result, self.context)
        
        assert analysis['status'] == 'analyzed'
        assert 'failure_info' in analysis
        assert analysis['failure_info']['success'] == True
        
    def test_failed_validation_analysis(self):
        """Test analysis of failed validation results."""
        validation_result = MockValidationFailure()
        
        analysis = self.analyzer.analyze_validation_results(validation_result, self.context)
        
        assert analysis['status'] == 'analyzed'
        assert 'failure_info' in analysis
        assert analysis['failure_info']['success'] == False
        assert 'failure_categories' in analysis
        assert 'recommended_state' in analysis
        assert 'recovery_strategy' in analysis
        
    def test_failure_categorization(self):
        """Test failure categorization logic."""
        failure_info = {
            'errors': [
                'Architecture design is tightly coupled',
                'Requirements are unclear and ambiguous',
                'Implementation has syntax errors',
                'Performance timeout exceeded'
            ]
        }
        
        categories = self.analyzer._categorize_failures(failure_info)
        
        assert 'architectural' in categories
        assert 'requirements' in categories
        assert 'implementation' in categories
        assert 'performance' in categories
        
    def test_optimal_state_determination(self):
        """Test optimal state determination based on failures."""
        failure_categories = {
            'architectural': ['Design coupling issue'],
            'requirements': ['Unclear specification']
        }
        
        optimal_state = self.analyzer._determine_optimal_state(failure_categories, self.context)
        
        # Should prefer architectural fixes over requirements (priority order)
        assert optimal_state == 'architecting'


class TestEnhancedStateAnalyzer:
    """Test suite for EnhancedStateAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.context_engine = ContextModelingEngine()
        self.validation_analyzer = ValidationResultAnalyzer()
        self.analyzer = EnhancedStateAnalyzer(self.context_engine, self.validation_analyzer)
        
        self.workflow_graph = {
            'states': {
                'analyzing': {
                    'transitions': ['planning', 'implementing'],
                    'agents': ['RIF-Analyst'],
                    'requirements': ['github_issues']
                },
                'planning': {
                    'transitions': ['implementing', 'architecting'],
                    'agents': ['RIF-Planner'],
                    'requirements': ['requirements_analysis']
                },
                'implementing': {
                    'transitions': ['validating', 'analyzing'],
                    'agents': ['RIF-Implementer'],
                    'requirements': ['implementation_plan']
                },
                'validating': {
                    'transitions': ['learning', 'implementing'],
                    'agents': ['RIF-Validator'],
                    'requirements': ['code_complete']
                }
            }
        }
    
    def test_basic_state_analysis(self):
        """Test basic state analysis functionality."""
        context_data = {
            'github_issues': [1, 2],
            'complexity': 'medium',
            'workflow_type': 'standard'
        }
        
        next_state, analysis = self.analyzer.analyze_and_determine_next_state(
            'analyzing', context_data, self.workflow_graph
        )
        
        assert next_state in ['planning', 'implementing']
        assert isinstance(analysis, dict)
        assert 'decision_factors' in analysis
        assert 'confidence' in analysis
        assert 'reasoning' in analysis
        
    def test_state_analysis_with_validation_failure(self):
        """Test state analysis with validation failures."""
        context_data = {
            'github_issues': [1],
            'complexity': 'high',
            'validation_results': MockValidationFailure(),
            'error_history': ['Implementation bug found', 'Test failed']
        }
        
        next_state, analysis = self.analyzer.analyze_and_determine_next_state(
            'validating', context_data, self.workflow_graph
        )
        
        # Should recommend going back to implementation
        assert next_state == 'implementing'
        assert analysis['confidence'] > 0.5
        
    def test_complexity_factor_evaluation(self):
        """Test complexity factor evaluation."""
        context_model = ContextModel(complexity='high', retry_count=0)
        
        complexity_factor = self.analyzer._evaluate_complexity_factor(
            context_model, ['implementing', 'architecting'], self.workflow_graph
        )
        
        assert 'recommendations' in complexity_factor
        # High complexity should prefer architecting
        assert complexity_factor['recommendations']['architecting'] > 0.5
        
    def test_pattern_factor_evaluation(self):
        """Test pattern-based factor evaluation."""
        context_model = ContextModel(
            pattern_matches=[
                {
                    'pattern_name': 'High Complexity Pattern',
                    'confidence': 0.9,
                    'recommendations': ['Add architect agent', 'Increase planning depth']
                }
            ]
        )
        
        pattern_factor = self.analyzer._evaluate_pattern_factor(
            context_model, ['planning', 'architecting']
        )
        
        assert 'recommendations' in pattern_factor
        assert pattern_factor['confidence'] > 0.8


class TestLearningAgentSelector:
    """Test suite for LearningAgentSelector and related components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = LearningAgentSelector()
    
    def test_basic_initialization(self):
        """Test basic initialization of learning components."""
        assert hasattr(self.selector, 'team_optimization_engine')
        assert hasattr(self.selector, 'performance_tracking_system')
        assert hasattr(self.selector, 'learning_weights')
        
    def test_team_optimization_engine(self):
        """Test team optimization engine functionality."""
        engine = TeamOptimizationEngine()
        
        initial_team = ['RIF-Analyst', 'RIF-Implementer', 'RIF-Validator']
        context = {'complexity': 'high', 'max_team_size': 4}
        performance_data = {
            'RIF-Analyst': [0.8, 0.9, 0.7],
            'RIF-Implementer': [0.6, 0.7, 0.8],
            'RIF-Validator': [0.9, 0.8, 0.9]
        }
        
        optimized_team = engine.optimize_team_composition(initial_team, context, performance_data)
        
        assert isinstance(optimized_team, list)
        assert len(optimized_team) <= context['max_team_size']
        # Should prefer agents with better performance (RIF-Validator)
        assert optimized_team[0] in ['RIF-Validator', 'RIF-Analyst']
        
    def test_performance_tracking_system(self):
        """Test performance tracking system."""
        tracker = PerformanceTrackingSystem()
        
        context = {'complexity': 'medium', 'workflow_type': 'standard'}
        performance_data = {'speed': 0.8, 'quality': 0.9, 'reliability': 0.7}
        
        # Record performance
        tracker.record_performance('RIF-Implementer', context, performance_data)
        
        # Check performance prediction
        prediction = tracker.get_performance_prediction('RIF-Implementer', context)
        
        assert 'predicted_score' in prediction
        assert 'confidence' in prediction
        assert 'context_type' in prediction
        assert prediction['predicted_score'] > 0.5  # Should be positive
        
    def test_performance_trends(self):
        """Test performance trend analysis."""
        tracker = PerformanceTrackingSystem()
        
        # Record improving performance over time
        context = {'complexity': 'medium'}
        for i, score in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            tracker.record_performance('RIF-Test-Agent', context, {'overall': score})
        
        trends = tracker.get_performance_trends('RIF-Test-Agent')
        
        assert trends['trend'] == 'improving'
        assert trends['confidence'] > 0.5
        assert trends['data_points'] == 5


class TestFailurePatternAnalyzer:
    """Test suite for FailurePatternAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FailurePatternAnalyzer()
        self.context = {'complexity': 'medium', 'retry_count': 1}
    
    def test_pattern_identification(self):
        """Test failure pattern identification."""
        error_history = [
            'Connection timeout occurred during API call',
            'Module import failed - dependency not found',
            'Architecture coupling issue detected',
            'Requirements specification is unclear'
        ]
        
        analysis = self.analyzer.analyze_failure_patterns(error_history, self.context)
        
        assert 'patterns_found' in analysis
        assert 'pattern_details' in analysis
        assert 'recovery_recommendations' in analysis
        
        # Should identify multiple patterns
        patterns = analysis['patterns_found']
        assert 'timeout_pattern' in patterns
        assert 'dependency_pattern' in patterns
        assert 'architecture_pattern' in patterns
        assert 'requirements_pattern' in patterns
        
    def test_pattern_significance_analysis(self):
        """Test pattern significance analysis."""
        identified_patterns = {
            'timeout_pattern': [0, 2, 4],  # Errors at positions 0, 2, 4
            'dependency_pattern': [1, 3]   # Errors at positions 1, 3
        }
        error_history = ['error1', 'error2', 'error3', 'error4', 'error5']
        
        analysis = self.analyzer._analyze_pattern_significance(identified_patterns, error_history)
        
        assert 'timeout_pattern' in analysis
        assert 'dependency_pattern' in analysis
        
        timeout_analysis = analysis['timeout_pattern']
        assert 'frequency' in timeout_analysis
        assert 'significance' in timeout_analysis
        assert timeout_analysis['frequency'] == 3
        
    def test_recovery_recommendation_generation(self):
        """Test recovery recommendation generation."""
        pattern_analysis = {
            'timeout_pattern': {
                'significance': 0.8,
                'recovery_strategy': 'extend_timeout_and_simplify',
                'recommended_state': 'implementing'
            },
            'dependency_pattern': {
                'significance': 0.6,
                'recovery_strategy': 'resolve_dependencies',
                'recommended_state': 'planning'
            }
        }
        
        recommendations = self.analyzer._generate_recovery_recommendations(pattern_analysis, self.context)
        
        assert 'primary_strategy' in recommendations
        assert 'recommended_state' in recommendations
        assert 'confidence' in recommendations
        
        # Should recommend the most significant pattern's strategy
        assert recommendations['primary_strategy'] == 'extend_timeout_and_simplify'
        assert recommendations['recommended_state'] == 'implementing'


class TestLoopBackDecisionEngine:
    """Test suite for LoopBackDecisionEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.failure_analyzer = FailurePatternAnalyzer()
        self.engine = LoopBackDecisionEngine(self.failure_analyzer)
    
    def test_basic_loop_back_decision(self):
        """Test basic loop-back decision making."""
        context = {
            'error_history': ['Implementation error', 'Test failure'],
            'retry_count': 1,
            'complexity': 'medium'
        }
        possible_transitions = ['implementing', 'analyzing', 'planning']
        
        decision, details = self.engine.make_loop_back_decision('validating', context, possible_transitions)
        
        assert decision in possible_transitions
        assert isinstance(details, dict)
        assert 'decision_factors' in details
        assert 'confidence' in details
        assert 'reasoning' in details
        
    def test_high_retry_count_handling(self):
        """Test handling of high retry counts."""
        context = {
            'error_history': ['Error 1', 'Error 2', 'Error 3', 'Error 4'],
            'retry_count': 4,  # High retry count
            'complexity': 'medium'
        }
        possible_transitions = ['implementing', 'analyzing', 'planning']
        
        decision, details = self.engine.make_loop_back_decision('implementing', context, possible_transitions)
        
        # High retry count should prefer more fundamental states
        assert decision in ['analyzing', 'planning']
        
    def test_transition_outcome_recording(self):
        """Test recording of transition outcomes for learning."""
        # Record some outcomes
        self.engine.record_transition_outcome('validating', 'implementing', True)
        self.engine.record_transition_outcome('validating', 'implementing', True)
        self.engine.record_transition_outcome('validating', 'implementing', False)
        
        # Check that outcomes are recorded
        state_key = 'validating_to_implementing'
        assert state_key in self.engine.state_success_rates
        assert self.engine.state_success_rates[state_key]['successes'] == 2
        assert self.engine.state_success_rates[state_key]['failures'] == 1


class TestRecoveryStrategySelector:
    """Test suite for RecoveryStrategySelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = RecoveryStrategySelector()
    
    def test_strategy_selection_basic(self):
        """Test basic recovery strategy selection."""
        failure_analysis = {
            'recovery_recommendations': {
                'primary_strategy': 'resolve_dependencies',
                'confidence': 0.8
            }
        }
        context = {'retry_count': 1, 'complexity': 'medium', 'priority': 2}
        
        result = self.selector.select_recovery_strategy(failure_analysis, context, 'implementing')
        
        assert 'selected_strategy' in result
        assert 'implementation_plan' in result
        assert 'confidence' in result
        
        assert result['selected_strategy'] in self.selector.recovery_strategies
        
    def test_effort_appropriateness_evaluation(self):
        """Test effort appropriateness evaluation."""
        context_low_retry = {'retry_count': 0, 'complexity': 'low', 'priority': 1}
        context_high_retry = {'retry_count': 4, 'complexity': 'high', 'priority': 2}
        
        # Low effort should be more appropriate for low retry/complexity
        score_low = self.selector._evaluate_effort_appropriateness('low', context_low_retry)
        
        # High effort should be more appropriate for high retry/complexity
        score_high = self.selector._evaluate_effort_appropriateness('high', context_high_retry)
        
        assert score_low > 0.3
        assert score_high > 0.5
        
    def test_strategy_outcome_recording(self):
        """Test recording strategy outcomes for learning."""
        context = {'complexity': 'medium'}
        details = {'duration': 120, 'issues_resolved': 3}
        
        # Record successful outcome
        self.selector.record_strategy_outcome('resolve_dependencies', context, True, details)
        
        # Check that outcome is recorded
        assert 'resolve_dependencies' in self.selector.strategy_history
        history = self.selector.strategy_history['resolve_dependencies']
        assert len(history) == 1
        assert history[0]['success'] == True


class TestDynamicOrchestratorEnhanced:
    """Test suite for enhanced DynamicOrchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = DynamicOrchestrator()
    
    def test_enhanced_initialization(self):
        """Test that enhanced components are properly initialized."""
        assert hasattr(self.orchestrator, 'context_engine')
        assert hasattr(self.orchestrator, 'validation_analyzer')
        assert hasattr(self.orchestrator, 'state_analyzer')
        assert hasattr(self.orchestrator, 'learning_agent_selector')
        assert hasattr(self.orchestrator, 'failure_pattern_analyzer')
        assert hasattr(self.orchestrator, 'loop_back_decision_engine')
        assert hasattr(self.orchestrator, 'recovery_strategy_selector')
        
    def test_enhanced_metrics_tracking(self):
        """Test enhanced metrics tracking."""
        initial_metrics = self.orchestrator.metrics.copy()
        
        # Check that enhanced metrics are initialized
        assert 'intelligent_decisions' in self.orchestrator.metrics
        assert 'pattern_matches' in self.orchestrator.metrics
        assert 'avg_decision_time' in self.orchestrator.metrics
        assert 'decision_confidence_avg' in self.orchestrator.metrics
        
    def test_enhanced_state_analysis(self):
        """Test enhanced state analysis functionality."""
        # Set up context with validation failure
        self.orchestrator.context = {
            'validation_results': MockValidationFailure(),
            'error_history': ['Implementation error'],
            'retry_count': 1,
            'complexity': 'medium'
        }
        self.orchestrator.current_state = 'validating'
        
        next_state = self.orchestrator.analyze_current_state()
        
        # Should use enhanced analysis
        assert next_state in ['implementing', 'analyzing', 'planning']
        assert 'last_analysis' in self.orchestrator.context
        
    def test_enhanced_state_summary(self):
        """Test enhanced state summary functionality."""
        # Add some context for testing
        self.orchestrator.context = {
            'github_issues': [1, 2],
            'complexity': 'high',
            'error_history': ['Error 1']
        }
        
        summary = self.orchestrator.get_state_summary()
        
        # Should have enhanced information
        assert 'intelligence_insights' in summary
        assert 'loop_back_intelligence' in summary
        assert 'agent_performance' in summary
        
        # Check intelligence insights
        insights = summary['intelligence_insights']
        if 'error' not in insights:
            assert 'context_quality' in insights
            assert 'decision_confidence' in insights
        
    def test_performance_recording(self):
        """Test performance recording functionality."""
        agents = ['RIF-Implementer', 'RIF-Validator']
        performance_metrics = {'speed': 0.8, 'quality': 0.9, 'reliability': 0.7}
        
        # Record performance
        self.orchestrator.record_transition_performance(agents, True, performance_metrics)
        
        # Should not raise exceptions and should update internal state
        # (Detailed assertions would require access to internal state)
        
    def test_workflow_with_enhanced_features(self):
        """Test complete workflow with enhanced features."""
        initial_context = {
            'github_issues': [1],
            'complexity': 'medium',
            'workflow_type': 'test_enhanced',
            'priority': 2
        }
        
        # Run a short workflow
        result = self.orchestrator.run_workflow(initial_context, max_iterations=3)
        
        assert isinstance(result, dict)
        assert 'final_state' in result
        assert 'metrics' in result
        assert 'success' in result
        
        # Enhanced metrics should be present
        metrics = result['metrics']
        assert 'intelligent_decisions' in metrics
        assert 'decision_confidence_avg' in metrics


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple enhanced components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = DynamicOrchestrator()
    
    def test_failure_recovery_scenario(self):
        """Test complete failure recovery scenario with enhanced intelligence."""
        # Simulate a validation failure scenario
        self.orchestrator.current_state = 'validating'
        self.orchestrator.context = {
            'validation_results': MockValidationFailure(),
            'error_history': [
                'Architecture coupling issue detected',
                'Implementation has memory leaks',
                'Test timeout exceeded'
            ],
            'retry_count': 2,
            'complexity': 'high',
            'github_issues': [1, 2]
        }
        
        # Analyze state with enhanced intelligence
        next_state = self.orchestrator.analyze_current_state()
        
        # Should make intelligent decision
        assert next_state in ['implementing', 'analyzing', 'architecting', 'planning']
        
        # Should have analysis details stored
        if 'last_analysis' in self.orchestrator.context:
            analysis = self.orchestrator.context['last_analysis']
            assert 'confidence' in analysis
            assert 'reasoning' in analysis
        
    def test_learning_and_optimization_scenario(self):
        """Test learning and optimization over multiple transitions."""
        agents = ['RIF-Implementer', 'RIF-Validator']
        
        # Record multiple performance outcomes
        for i in range(5):
            success = i > 2  # First few fail, later ones succeed
            score = 0.3 + (i * 0.15)  # Improving performance
            
            self.orchestrator.record_transition_performance(
                agents, success, {'reliability': score, 'quality': score}
            )
        
        # Get performance summary
        summary = self.orchestrator.get_state_summary()
        
        if 'agent_performance' in summary and 'error' not in summary['agent_performance']:
            perf_data = summary['agent_performance']
            for agent in agents:
                if agent in perf_data:
                    assert 'trend' in perf_data[agent]
                    # Should detect improvement
                    
    def test_complex_workflow_scenario(self):
        """Test complex workflow scenario with multiple intelligence features."""
        complex_context = {
            'github_issues': [1, 2, 3, 4, 5],  # Many issues
            'complexity': 'very_high',
            'workflow_type': 'complex_system',
            'priority': 4,
            'security_critical': True,
            'required_skills': ['system_design', 'security_analysis', 'performance_optimization'],
            'max_team_size': 5
        }
        
        # Test enhanced team composition
        if hasattr(self.orchestrator, 'learning_agent_selector'):
            team = self.orchestrator.learning_agent_selector.compose_dynamic_team(complex_context)
            
            assert isinstance(team, list)
            assert len(team) <= complex_context['max_team_size']
            # Should include security agent for critical systems
            assert 'RIF-Security' in team


def run_enhanced_orchestrator_tests():
    """Run all enhanced orchestrator tests."""
    print("🧪 Running Enhanced Dynamic Orchestrator Test Suite")
    print("=" * 60)
    
    test_classes = [
        TestContextModel,
        TestContextModelingEngine,
        TestValidationResultAnalyzer,
        TestEnhancedStateAnalyzer,
        TestLearningAgentSelector,
        TestFailurePatternAnalyzer,
        TestLoopBackDecisionEngine,
        TestRecoveryStrategySelector,
        TestDynamicOrchestratorEnhanced,
        TestIntegrationScenarios
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        total_tests += len(test_methods)
        
        # Create instance and run tests
        test_instance = test_class()
        
        for test_method_name in test_methods:
            try:
                # Run setup if available
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test method
                test_method = getattr(test_instance, test_method_name)
                test_method()
                
                print(f"  ✅ {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {test_method_name}: {str(e)}")
                failed_tests += 1
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"ENHANCED ORCHESTRATOR TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")
    
    success = failed_tests == 0
    
    if success:
        print(f"\n🎉 All enhanced orchestrator tests passed!")
        print(f"   Enhanced DynamicOrchestrator implementation is ready for validation")
    else:
        print(f"\n❌ {failed_tests} tests failed")
        print(f"   Review failed tests before proceeding")
    
    return success


if __name__ == "__main__":
    success = run_enhanced_orchestrator_tests()
    sys.exit(0 if success else 1)