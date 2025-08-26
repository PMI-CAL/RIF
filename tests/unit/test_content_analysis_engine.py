#!/usr/bin/env python3
"""
Test suite for ContentAnalysisEngine - Issue #273 Implementation

This test suite validates the ContentAnalysisEngine implementation that replaces
the label dependency system with intelligent content analysis.

Test Coverage:
- State analysis from issue content
- Complexity assessment from requirements
- Dependency extraction from issue text
- Performance benchmarks (sub-100ms requirement)
- Accuracy validation (90%+ accuracy target)
- Edge case handling
"""

import unittest
import time
from unittest.mock import patch, MagicMock

# Import the classes we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'claude', 'commands'))

from content_analysis_engine import (
    ContentAnalysisEngine, 
    StateAnalyzer, 
    ComplexityAnalyzer, 
    DependencyAnalyzer,
    WorkflowState, 
    ComplexityLevel,
    ContentAnalysisResult
)

class TestStateAnalyzer(unittest.TestCase):
    """Test the StateAnalyzer component"""
    
    def setUp(self):
        self.analyzer = StateAnalyzer()
    
    def test_new_state_detection(self):
        """Test detection of new/analysis-needed states"""
        test_cases = [
            ("Need to analyze the user requirements", WorkflowState.NEW),
            ("Investigate how the API works", WorkflowState.NEW),
            ("What is the best approach for this?", WorkflowState.NEW),
            ("New feature request", WorkflowState.NEW),
            ("Initial investigation required", WorkflowState.NEW)
        ]
        
        for text, expected_state in test_cases:
            with self.subTest(text=text):
                state, confidence = self.analyzer.analyze_state(text)
                self.assertEqual(state, expected_state)
                self.assertGreater(confidence, 0.0)
    
    def test_analyzing_state_detection(self):
        """Test detection of analyzing/research states"""
        test_cases = [
            ("Currently analyzing the performance bottleneck", WorkflowState.ANALYZING),
            ("Investigation in progress", WorkflowState.ANALYZING),
            ("Research phase ongoing", WorkflowState.ANALYZING),
            ("Understanding the requirements", WorkflowState.ANALYZING),
            ("Feasibility analysis underway", WorkflowState.ANALYZING)
        ]
        
        for text, expected_state in test_cases:
            with self.subTest(text=text):
                state, confidence = self.analyzer.analyze_state(text)
                self.assertEqual(state, expected_state)
                self.assertGreater(confidence, 0.0)
    
    def test_implementing_state_detection(self):
        """Test detection of implementation states"""
        test_cases = [
            ("Ready to implement the feature", WorkflowState.IMPLEMENTING),
            ("Start implementation of the API", WorkflowState.IMPLEMENTING),
            ("Code the authentication module", WorkflowState.IMPLEMENTING),
            ("Build the user interface", WorkflowState.IMPLEMENTING),
            ("Develop the core functionality", WorkflowState.IMPLEMENTING)
        ]
        
        for text, expected_state in test_cases:
            with self.subTest(text=text):
                state, confidence = self.analyzer.analyze_state(text)
                self.assertEqual(state, expected_state)
                self.assertGreater(confidence, 0.0)
    
    def test_validating_state_detection(self):
        """Test detection of validation/testing states"""
        test_cases = [
            ("Need to test the implementation", WorkflowState.VALIDATING),
            ("Validate the user interface", WorkflowState.VALIDATING),
            ("Quality review required", WorkflowState.VALIDATING),
            ("Check the security aspects", WorkflowState.VALIDATING),
            ("Ready for testing", WorkflowState.VALIDATING)
        ]
        
        for text, expected_state in test_cases:
            with self.subTest(text=text):
                state, confidence = self.analyzer.analyze_state(text)
                self.assertEqual(state, expected_state)
                self.assertGreater(confidence, 0.0)
    
    def test_blocked_state_detection(self):
        """Test detection of blocked states"""
        test_cases = [
            ("Blocked by dependency on issue #123", WorkflowState.BLOCKED),
            ("Cannot proceed without API access", WorkflowState.BLOCKED),
            ("Waiting for prerequisite completion", WorkflowState.BLOCKED),
            ("Stuck on authentication issue", WorkflowState.BLOCKED)
        ]
        
        for text, expected_state in test_cases:
            with self.subTest(text=text):
                state, confidence = self.analyzer.analyze_state(text)
                self.assertEqual(state, expected_state)
                self.assertGreater(confidence, 0.0)

class TestComplexityAnalyzer(unittest.TestCase):
    """Test the ComplexityAnalyzer component"""
    
    def setUp(self):
        self.analyzer = ComplexityAnalyzer()
    
    def test_low_complexity_detection(self):
        """Test detection of low complexity tasks"""
        test_cases = [
            "Fix typo in documentation",
            "Update single configuration variable",
            "Change button text",
            "Simple bug fix in validation function",
            "Quick modification to existing feature"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                complexity, confidence = self.analyzer.analyze_complexity(text)
                self.assertEqual(complexity, ComplexityLevel.LOW)
                self.assertGreater(confidence, 0.0)
    
    def test_medium_complexity_detection(self):
        """Test detection of medium complexity tasks"""
        test_cases = [
            "Implement new user registration feature",
            "Add email notification system",
            "Create dashboard component",
            "Integrate with payment API",
            "Develop search functionality"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                complexity, confidence = self.analyzer.analyze_complexity(text)
                self.assertEqual(complexity, ComplexityLevel.MEDIUM)
                self.assertGreater(confidence, 0.0)
    
    def test_high_complexity_detection(self):
        """Test detection of high complexity tasks"""
        test_cases = [
            "Redesign the entire authentication system",
            "Performance optimization of database queries",
            "Security audit and vulnerability assessment",
            "Migrate to new architecture framework",
            "Refactor legacy codebase structure"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                complexity, confidence = self.analyzer.analyze_complexity(text)
                self.assertEqual(complexity, ComplexityLevel.HIGH)
                self.assertGreater(confidence, 0.0)
    
    def test_very_high_complexity_detection(self):
        """Test detection of very high complexity tasks"""
        test_cases = [
            "Implement distributed microservices architecture",
            "Build real-time machine learning pipeline",
            "Create concurrent data processing system",
            "Design enterprise-grade monitoring solution",
            "Develop AI-powered recommendation algorithm"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                complexity, confidence = self.analyzer.analyze_complexity(text)
                self.assertEqual(complexity, ComplexityLevel.VERY_HIGH)
                self.assertGreater(confidence, 0.0)

class TestDependencyAnalyzer(unittest.TestCase):
    """Test the DependencyAnalyzer component"""
    
    def setUp(self):
        self.analyzer = DependencyAnalyzer()
    
    def test_dependency_extraction(self):
        """Test extraction of issue dependencies"""
        test_cases = [
            ("Depends on issue #123 for API changes", [123], []),
            ("Blocked by issue #456", [456], []),
            ("Requires completion of #789", [789], []),
            ("Waiting for issue #101 and #102", [101, 102], []),
            ("No dependencies mentioned", [], [])
        ]
        
        for text, expected_depends, expected_blocks in test_cases:
            with self.subTest(text=text):
                depends_on, blocks = self.analyzer.analyze_dependencies(text)
                self.assertEqual(set(depends_on), set(expected_depends))
                self.assertEqual(set(blocks), set(expected_blocks))
    
    def test_blocking_relationship_extraction(self):
        """Test extraction of blocking relationships"""
        test_cases = [
            ("This blocks issue #123", [], [123]),
            ("Blocking issue #456 until completion", [], [456]),
            ("This issue blocks all others", [], []),  # General blocking
            ("Must complete before all other work", [], [])  # General blocking
        ]
        
        for text, expected_depends, expected_blocks in test_cases:
            with self.subTest(text=text):
                depends_on, blocks = self.analyzer.analyze_dependencies(text)
                self.assertEqual(set(depends_on), set(expected_depends))
                self.assertEqual(set(blocks), set(expected_blocks))

class TestContentAnalysisEngine(unittest.TestCase):
    """Test the main ContentAnalysisEngine"""
    
    def setUp(self):
        self.engine = ContentAnalysisEngine()
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis of realistic issue content"""
        title = "Implement user authentication system"
        body = """
        We need to implement a secure user authentication system that supports:
        
        1. User registration and login
        2. Password hashing and validation
        3. JWT token generation
        4. Session management
        5. Security audit logging
        
        This is a high-priority security feature that requires careful implementation.
        Performance target: < 200ms response time for authentication requests.
        
        Dependencies:
        - Requires completion of database schema changes in issue #145
        - Blocks the user management feature in issue #200
        """
        
        result = self.engine.analyze_issue_content(title, body)
        
        # Validate result structure
        self.assertIsInstance(result, ContentAnalysisResult)
        self.assertIsInstance(result.state, WorkflowState)
        self.assertIsInstance(result.complexity, ComplexityLevel)
        self.assertIsInstance(result.dependencies, list)
        self.assertIsInstance(result.confidence_score, float)
        self.assertIsInstance(result.analysis_time_ms, float)
        
        # Validate analysis results
        self.assertEqual(result.state, WorkflowState.NEW)  # "implement" indicates new work
        self.assertIn(result.complexity, [ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH])  # Security system is complex
        self.assertIn(145, result.dependencies)  # Should detect dependency on #145
        self.assertGreater(result.confidence_score, 0.5)  # Should have reasonable confidence
        
        # Validate semantic indicators
        self.assertIn('security', result.semantic_indicators.get('domains', []))
        self.assertIn('authentication', result.semantic_indicators.get('domains', []))
        
        # Validate risk factors
        self.assertIn('Security Risk', result.risk_factors)
    
    def test_performance_requirement(self):
        """Test that analysis meets sub-100ms performance requirement"""
        title = "Simple bug fix"
        body = "Fix null pointer exception in user validation method"
        
        # Perform multiple analyses to get average time
        times = []
        for _ in range(10):
            result = self.engine.analyze_issue_content(title, body)
            times.append(result.analysis_time_ms)
        
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 100.0, f"Average analysis time {avg_time:.2f}ms exceeds 100ms requirement")
        
        # Check engine performance stats
        stats = self.engine.get_performance_stats()
        self.assertTrue(stats['performance_target_met'])
    
    def test_empty_content_handling(self):
        """Test handling of empty or minimal content"""
        test_cases = [
            ("", ""),
            ("Title only", ""),
            ("", "Body only"),
            ("Short", "Fix")
        ]
        
        for title, body in test_cases:
            with self.subTest(title=title, body=body):
                result = self.engine.analyze_issue_content(title, body)
                
                # Should still return valid result
                self.assertIsInstance(result, ContentAnalysisResult)
                self.assertIsInstance(result.state, WorkflowState)
                self.assertIsInstance(result.complexity, ComplexityLevel)
                
                # Should have reasonable defaults
                self.assertGreaterEqual(result.confidence_score, 0.0)
                self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_semantic_indicator_extraction(self):
        """Test extraction of semantic indicators"""
        title = "Python API performance optimization"
        body = """
        Optimize the Python REST API for better performance.
        Current response times are too slow for production use.
        Need to implement caching and database query optimization.
        """
        
        result = self.engine.analyze_issue_content(title, body)
        
        # Should detect technology indicators
        self.assertIn('python', result.semantic_indicators.get('technologies', []))
        self.assertIn('api', result.semantic_indicators.get('technologies', []))
        
        # Should detect action indicators
        self.assertIn('optimize', result.semantic_indicators.get('actions', []))
        self.assertIn('implement', result.semantic_indicators.get('actions', []))
        
        # Should detect domain indicators
        self.assertIn('performance', result.semantic_indicators.get('domains', []))
    
    def test_risk_factor_identification(self):
        """Test identification of risk factors"""
        test_cases = [
            ("Security vulnerability fix", ["Security Risk"]),
            ("Database migration script", ["Data Risk"]),
            ("Performance critical optimization", ["Performance Risk"]),
            ("Breaking API changes", ["Breaking Change"]),
            ("Third-party integration", ["Integration Risk"]),
            ("Production deployment", ["Deployment Risk"])
        ]
        
        for text, expected_risks in test_cases:
            with self.subTest(text=text):
                result = self.engine.analyze_issue_content(text, text)
                
                for risk in expected_risks:
                    self.assertIn(risk, result.risk_factors,
                                f"Expected risk '{risk}' not found in: {result.risk_factors}")
    
    def test_confidence_scoring(self):
        """Test confidence scoring accuracy"""
        # Clear, well-defined task should have high confidence
        clear_task = self.engine.analyze_issue_content(
            "Implement user login feature",
            "Create a user login form with username/password validation"
        )
        self.assertGreater(clear_task.confidence_score, 0.7)
        
        # Vague task should have lower confidence
        vague_task = self.engine.analyze_issue_content(
            "Improve system", 
            "Make things work better"
        )
        self.assertLess(vague_task.confidence_score, 0.5)
    
    def test_accuracy_validation(self):
        """Test accuracy against known good examples"""
        # Test cases with expected results for accuracy validation
        test_cases = [
            {
                'title': 'Fix login bug',
                'body': 'Users cannot log in due to validation error',
                'expected_state': WorkflowState.NEW,
                'expected_complexity': ComplexityLevel.LOW
            },
            {
                'title': 'Implement microservices architecture',
                'body': 'Migrate to distributed microservices with service discovery',
                'expected_state': WorkflowState.NEW,
                'expected_complexity': ComplexityLevel.VERY_HIGH
            },
            {
                'title': 'Test user interface',
                'body': 'Validate the new dashboard components work correctly',
                'expected_state': WorkflowState.VALIDATING,
                'expected_complexity': ComplexityLevel.MEDIUM
            }
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases) * 2  # state + complexity
        
        for case in test_cases:
            result = self.engine.analyze_issue_content(case['title'], case['body'])
            
            if result.state == case['expected_state']:
                correct_predictions += 1
            if result.complexity == case['expected_complexity']:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        self.assertGreaterEqual(accuracy, 0.90, f"Accuracy {accuracy:.2%} below 90% requirement")

class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""
    
    def setUp(self):
        self.engine = ContentAnalysisEngine()
    
    def test_issue_273_scenario(self):
        """Test the specific scenario from Issue #273"""
        title = "CRITICAL: Replace Label Dependency with Content Analysis Engine"
        body = """
        ## Problem Statement
        Current orchestration is locked into label dependency via line 668 in enhanced_orchestration_intelligence.py:
        ```python
        current_state = context_model.issue_context.current_state_label  # LABEL DEPENDENT
        ```
        
        This violates the core requirement: "Orchestration needs to stop depending on labels"
        
        ## Required Implementation
        1. ContentAnalysisEngine Class
        2. Content-Based State Determination  
        3. Integration Points
        
        ## Success Criteria
        - Zero references to current_state_label in core orchestration logic
        - 90%+ accuracy in content-derived state determination
        - Sub-100ms content analysis response time
        """
        
        result = self.engine.analyze_issue_content(title, body)
        
        # Should detect as implementing state (ready to code)
        self.assertEqual(result.state, WorkflowState.NEW)
        
        # Should detect as high complexity due to core system changes
        self.assertIn(result.complexity, [ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH])
        
        # Should detect critical priority
        self.assertIn('critical', result.semantic_indicators.get('urgency', []))
        
        # Should be fast analysis
        self.assertLess(result.analysis_time_ms, 100.0)
        
        # Should have high confidence due to clear requirements
        self.assertGreater(result.confidence_score, 0.7)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    def setUp(self):
        self.engine = ContentAnalysisEngine()
    
    def test_bulk_analysis_performance(self):
        """Test performance with bulk analysis"""
        # Generate test cases
        test_cases = [
            ("Bug fix #{}", "Fix validation error in user input"),
            ("Feature #{}", "Implement new dashboard component"),
            ("Security #{}", "Address vulnerability in authentication"),
            ("Performance #{}", "Optimize database query execution"),
            ("Architecture #{}", "Refactor service layer design")
        ] * 10  # 50 total cases
        
        start_time = time.time()
        
        results = []
        for i, (title_template, body) in enumerate(test_cases):
            title = title_template.format(i)
            result = self.engine.analyze_issue_content(title, body)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_analysis = (total_time * 1000) / len(test_cases)  # Convert to ms
        
        # Should maintain sub-100ms average even with bulk processing
        self.assertLess(avg_time_per_analysis, 100.0, 
                       f"Bulk analysis average {avg_time_per_analysis:.2f}ms exceeds 100ms target")
        
        # All analyses should complete successfully
        self.assertEqual(len(results), len(test_cases))
        
        # All results should be valid
        for result in results:
            self.assertIsInstance(result, ContentAnalysisResult)
            self.assertGreaterEqual(result.confidence_score, 0.0)
            self.assertLessEqual(result.confidence_score, 1.0)

if __name__ == '__main__':
    # Set up test environment
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in [
        TestStateAnalyzer,
        TestComplexityAnalyzer, 
        TestDependencyAnalyzer,
        TestContentAnalysisEngine,
        TestIntegrationScenarios,
        TestPerformanceBenchmarks
    ]:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Content Analysis Engine Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, trace in result.errors:
            print(f"  - {test}: {trace.split(chr(10))[-2]}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)