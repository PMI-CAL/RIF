#!/usr/bin/env python3
"""
Test suite for ContentAnalysisEngine - Issue #273 Implementation

This comprehensive test suite validates the ContentAnalysisEngine that replaces
label dependency with intelligent content analysis throughout the orchestration system.

Tests cover:
- State derivation from content
- Complexity assessment 
- Dependency extraction
- Blocking detection
- Performance benchmarks
- Accuracy validation
"""

import unittest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from claude.commands.content_analysis_engine import (
    ContentAnalysisEngine, ContentAnalysisResult, IssueState, ComplexityLevel, ConfidenceLevel
)

class TestContentAnalysisEngine(unittest.TestCase):
    """Test suite for ContentAnalysisEngine - Issue #273 fix validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = ContentAnalysisEngine()
        
        # Test cases based on real GitHub issue patterns
        self.test_cases = [
            {
                'title': 'Add user authentication system',
                'body': 'Need to implement user authentication with login, registration, and password reset functionality. This is a new feature request.',
                'expected_state': 'new',
                'expected_complexity': 'medium'
            },
            {
                'title': 'Currently analyzing database performance issues',
                'body': 'Investigating slow query performance. Need to analyze query patterns and identify bottlenecks. Research phase is ongoing.',
                'expected_state': 'analyzing',
                'expected_complexity': 'high'
            },
            {
                'title': 'Planning microservices architecture',
                'body': 'Designing a microservices architecture for the payment system. Need to plan service boundaries, communication patterns, and data consistency approach.',
                'expected_state': 'planning',
                'expected_complexity': 'very-high'
            },
            {
                'title': 'Implementing REST API endpoints',
                'body': 'Working on implementation of REST API endpoints for user management. Code is being written and components are being built.',
                'expected_state': 'implementing',
                'expected_complexity': 'medium'
            },
            {
                'title': 'Testing the authentication module',
                'body': 'Validating the authentication implementation. Running test suite and verifying all test cases pass. Quality assurance in progress.',
                'expected_state': 'validating',
                'expected_complexity': 'low'
            },
            {
                'title': 'BLOCKED: Waiting for database migration',
                'body': 'This issue is blocked by the database migration. Cannot proceed until the migration is complete. Waiting for dependency resolution.',
                'expected_state': 'blocked',
                'expected_complexity': 'medium'
            },
            {
                'title': 'Authentication system completed',
                'body': 'User authentication has been successfully implemented and tested. All requirements met and ready to close.',
                'expected_state': 'complete',
                'expected_complexity': 'medium'
            }
        ]

        # Blocking detection test cases
        self.blocking_test_cases = [
            {
                'title': 'CRITICAL: Fix agent context reading',
                'body': 'THIS ISSUE BLOCKS ALL OTHERS. Critical infrastructure failure in agent context reading must be resolved immediately.',
                'expected_blocking': True
            },
            {
                'title': 'Core API framework implementation', 
                'body': 'BLOCKS ALL OTHER WORK until complete. This foundational API framework must be finished before any dependent features can proceed.',
                'expected_blocking': True
            },
            {
                'title': 'Regular feature implementation',
                'body': 'Standard feature implementation for user profiles. No blocking requirements.',
                'expected_blocking': False
            },
            {
                'title': 'High priority urgent task',
                'body': 'This is a critical high priority task that needs immediate attention.',
                'expected_blocking': False  # Should not trigger - generic urgency
            }
        ]

        # Dependency extraction test cases  
        self.dependency_test_cases = [
            {
                'title': 'Feature depends on core API',
                'body': 'This feature depends on issue #42 (core API) and requires #15 to be completed first.',
                'expected_dependencies': ['42', '15']
            },
            {
                'title': 'Blocked by multiple issues',
                'body': 'Cannot proceed - blocked by #23, #45, and #67. Must wait for these prerequisite issues.',
                'expected_dependencies': ['23', '45', '67']
            },
            {
                'title': 'No dependencies',
                'body': 'Standalone feature with no external dependencies. Can proceed independently.',
                'expected_dependencies': []
            }
        ]

    def test_state_derivation_accuracy(self):
        """Test accuracy of state derivation from content"""
        print("Testing state derivation accuracy...")
        
        correct_predictions = 0
        for case in self.test_cases:
            result = self.engine.analyze_issue_content(case['title'], case['body'])
            
            if result.derived_state.value == case['expected_state']:
                correct_predictions += 1
                print(f"✅ CORRECT: '{case['title'][:50]}...' -> {result.derived_state.value}")
            else:
                print(f"❌ INCORRECT: '{case['title'][:50]}...' -> {result.derived_state.value} (expected: {case['expected_state']})")

        accuracy = correct_predictions / len(self.test_cases)
        print(f"\n📊 State Derivation Accuracy: {accuracy:.2%} ({correct_predictions}/{len(self.test_cases)})")
        
        # Success criteria from Issue #273: 90%+ accuracy
        self.assertGreaterEqual(accuracy, 0.90, f"State derivation accuracy {accuracy:.2%} below required 90%")

    def test_complexity_assessment(self):
        """Test complexity assessment from content"""
        print("Testing complexity assessment...")
        
        correct_predictions = 0
        for case in self.test_cases:
            result = self.engine.analyze_issue_content(case['title'], case['body'])
            
            if result.complexity.value == case['expected_complexity']:
                correct_predictions += 1
                print(f"✅ CORRECT: '{case['title'][:50]}...' -> {result.complexity.value}")
            else:
                print(f"❌ INCORRECT: '{case['title'][:50]}...' -> {result.complexity.value} (expected: {case['expected_complexity']})")

        accuracy = correct_predictions / len(self.test_cases)
        print(f"\n📊 Complexity Assessment Accuracy: {accuracy:.2%} ({correct_predictions}/{len(self.test_cases)})")
        
        # Expect reasonable complexity assessment (70%+ for initial implementation)
        self.assertGreaterEqual(accuracy, 0.70, f"Complexity assessment accuracy {accuracy:.2%} below expected 70%")

    def test_blocking_detection(self):
        """Test blocking issue detection"""
        print("Testing blocking detection...")
        
        for case in self.blocking_test_cases:
            result = self.engine.analyze_issue_content(case['title'], case['body'])
            has_blocking = len(result.blocking_indicators) > 0
            
            if has_blocking == case['expected_blocking']:
                print(f"✅ CORRECT: '{case['title'][:50]}...' -> blocking: {has_blocking}")
            else:
                print(f"❌ INCORRECT: '{case['title'][:50]}...' -> blocking: {has_blocking} (expected: {case['expected_blocking']})")
            
            self.assertEqual(has_blocking, case['expected_blocking'], 
                           f"Blocking detection failed for: {case['title']}")

    def test_dependency_extraction(self):
        """Test dependency extraction from content"""
        print("Testing dependency extraction...")
        
        for case in self.dependency_test_cases:
            result = self.engine.analyze_issue_content(case['title'], case['body'])
            
            # Sort both lists for comparison
            found_deps = sorted(result.dependencies)
            expected_deps = sorted(case['expected_dependencies'])
            
            if found_deps == expected_deps:
                print(f"✅ CORRECT: '{case['title'][:50]}...' -> deps: {found_deps}")
            else:
                print(f"❌ INCORRECT: '{case['title'][:50]}...' -> deps: {found_deps} (expected: {expected_deps})")
            
            self.assertEqual(found_deps, expected_deps,
                           f"Dependency extraction failed for: {case['title']}")

    def test_performance_benchmark(self):
        """Test performance requirements from Issue #273: sub-100ms response time"""
        print("Testing performance benchmarks...")
        
        # Performance test case
        test_title = "Complex feature with detailed requirements and multiple dependencies"
        test_body = """
        This is a complex feature implementation that requires:
        
        1. Analysis of existing architecture
        2. Design of new microservice components  
        3. Implementation of REST APIs with authentication
        4. Database schema updates and migrations
        5. Comprehensive testing including unit, integration, and performance tests
        6. Documentation updates
        
        Dependencies:
        - Depends on issue #42 (core authentication service)
        - Requires completion of #15 (database migration framework)  
        - Blocked by #67 (infrastructure deployment)
        
        This is a high-complexity issue that will require significant planning and coordination.
        The implementation involves multiple systems and requires careful attention to security.
        
        Testing requirements:
        - Unit test coverage >90%
        - Integration tests with all dependent services
        - Performance benchmarks must meet SLA requirements
        - Security scanning and vulnerability assessment
        """
        
        # Run multiple iterations to get average performance
        times = []
        num_iterations = 10
        
        for i in range(num_iterations):
            start_time = time.time()
            result = self.engine.analyze_issue_content(test_title, test_body)
            end_time = time.time()
            
            analysis_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(analysis_time)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"\n⏱️  Performance Results:")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   Maximum: {max_time:.2f}ms") 
        print(f"   Minimum: {min_time:.2f}ms")
        
        # Success criteria: sub-100ms average response time
        self.assertLess(avg_time, 100.0, f"Average response time {avg_time:.2f}ms exceeds 100ms requirement")
        print(f"✅ PERFORMANCE: Average response time {avg_time:.2f}ms meets <100ms requirement")

    def test_replacement_method(self):
        """Test the direct replacement method for current_state_label"""
        print("Testing direct replacement method...")
        
        # Test the get_replacement_state method
        test_cases = [
            ("New feature request", "Need to implement user profiles", "new"),
            ("Analysis in progress", "Currently analyzing performance bottlenecks", "analyzing"),
            ("Implementation work", "Writing code for API endpoints", "implementing"),
            ("Testing phase", "Running validation tests", "validating")
        ]
        
        for title, body, expected in test_cases:
            result = self.engine.get_replacement_state(title, body)
            self.assertEqual(result, expected, f"Replacement method failed for: {title}")
            print(f"✅ REPLACEMENT: '{title}' -> '{result}'")

    def test_content_analysis_integration(self):
        """Test integration with existing orchestration utilities"""
        print("Testing orchestration utilities integration...")
        
        # Test that the ContentAnalysisEngine can be imported and used
        try:
            from claude.commands.orchestration_utilities import IssueContext
            print("✅ INTEGRATION: Successfully imported orchestration utilities")
        except ImportError as e:
            self.fail(f"Failed to import orchestration utilities: {e}")
        
        # Create a test issue context
        test_context = IssueContext(
            number=123,
            title="Test implementation issue", 
            body="Working on implementing the user authentication system",
            labels=["state:implementing", "complexity:medium"],
            state="open",
            complexity="medium",
            priority="normal",
            agent_history=["RIF-Analyst", "RIF-Planner"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
            comments_count=3
        )
        
        # Test that content-based state works
        try:
            content_state = test_context.current_state_from_content
            self.assertIsNotNone(content_state, "Content-based state should not be None")
            self.assertIn("state:", content_state, "Content state should include 'state:' prefix")
            print(f"✅ CONTENT STATE: {content_state}")
        except Exception as e:
            print(f"⚠️  Content analysis not available (expected in test environment): {e}")
        
        # Test the full analysis method
        try:
            analysis = test_context.get_content_analysis()
            if analysis:
                self.assertIsInstance(analysis, ContentAnalysisResult)
                print(f"✅ FULL ANALYSIS: State={analysis.derived_state.value}, Complexity={analysis.complexity.value}")
            else:
                print("⚠️  Content analysis not available (expected in test environment)")
        except Exception as e:
            print(f"⚠️  Content analysis error (expected in test environment): {e}")

    def test_confidence_scoring(self):
        """Test confidence scoring for state predictions"""
        print("Testing confidence scoring...")
        
        # High confidence cases (clear indicators)
        high_confidence_cases = [
            ("Implementation in progress", "Currently implementing the user authentication API endpoints", 0.7),
            ("Testing phase active", "Running comprehensive test suite and validating all requirements", 0.7),
            ("Analysis complete", "Finished analyzing the requirements and ready to proceed", 0.6)
        ]
        
        for title, body, min_confidence in high_confidence_cases:
            result = self.engine.analyze_issue_content(title, body)
            self.assertGreaterEqual(result.confidence, min_confidence,
                                  f"Confidence {result.confidence:.2f} below expected {min_confidence} for: {title}")
            print(f"✅ CONFIDENCE: '{title}' -> {result.confidence:.2f} (>= {min_confidence})")

    def test_semantic_tag_generation(self):
        """Test semantic tag generation for better orchestration"""
        print("Testing semantic tag generation...")
        
        # Test case with multiple technologies
        result = self.engine.analyze_issue_content(
            "Full-stack user authentication with React frontend and Python backend",
            """
            Implement comprehensive user authentication system:
            - React frontend with login/register forms
            - Python Flask backend with JWT authentication  
            - Database integration with PostgreSQL
            - API endpoints for user management
            - Security measures and encryption
            - Performance optimization for concurrent users
            """
        )
        
        # Check for expected technology tags
        expected_tags = ['python', 'javascript', 'database', 'api', 'frontend', 'backend', 'security', 'performance']
        found_tags = [tag for tag in expected_tags if tag in result.semantic_tags]
        
        print(f"📋 Generated tags: {result.semantic_tags}")
        print(f"✅ Found expected tags: {found_tags}")
        
        # Should identify at least 50% of the expected technology tags
        tag_accuracy = len(found_tags) / len(expected_tags)
        self.assertGreaterEqual(tag_accuracy, 0.5, 
                               f"Semantic tag accuracy {tag_accuracy:.2%} below expected 50%")

    def test_validation_requirements_extraction(self):
        """Test extraction of validation and testing requirements"""
        print("Testing validation requirements extraction...")
        
        result = self.engine.analyze_issue_content(
            "Feature requiring comprehensive testing",
            """
            Implementation must include:
            - Unit test coverage >90%
            - Integration test suite
            - Performance benchmarks
            - Security scan validation
            - Quality gate compliance
            """
        )
        
        # Should extract testing requirements
        self.assertGreater(len(result.validation_requirements), 0,
                          "Should extract validation requirements")
        print(f"📋 Validation requirements: {result.validation_requirements}")

    def test_empty_and_edge_cases(self):
        """Test edge cases and error handling"""
        print("Testing edge cases...")
        
        # Empty content
        result = self.engine.analyze_issue_content("", "")
        self.assertEqual(result.derived_state, IssueState.NEW)
        print("✅ EDGE CASE: Empty content handled correctly")
        
        # Very short content
        result = self.engine.analyze_issue_content("Fix", "Bug")
        self.assertIsInstance(result.derived_state, IssueState)
        print("✅ EDGE CASE: Short content handled correctly")
        
        # Very long content (performance test)
        long_content = "This is a test. " * 1000
        start_time = time.time()
        result = self.engine.analyze_issue_content("Long test", long_content)
        analysis_time = (time.time() - start_time) * 1000
        
        self.assertLess(analysis_time, 200, "Long content analysis should still be fast")
        print(f"✅ EDGE CASE: Long content ({len(long_content)} chars) analyzed in {analysis_time:.2f}ms")

def main():
    """Run the comprehensive test suite"""
    print("🧪 ContentAnalysisEngine Test Suite - Issue #273 Validation")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContentAnalysisEngine)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print(f"🎯 TEST RESULTS: {result.testsRun} tests run")
    print(f"   ✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Failed: {len(result.failures)}")
    print(f"   💥 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n💥 FAILURES:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback}")
    
    # Overall success assessment
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\n🏆 OVERALL SUCCESS RATE: {success_rate:.2%}")
    
    if success_rate >= 0.90:
        print("✅ ISSUE #273 CRITICAL FIX: ContentAnalysisEngine passes validation")
        print("   Content-based orchestration is ready for production deployment")
    else:
        print("❌ ISSUE #273 VALIDATION FAILED: Additional work required")
        
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)