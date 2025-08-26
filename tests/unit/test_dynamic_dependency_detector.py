#!/usr/bin/env python3
"""
Comprehensive Tests for Dynamic Dependency Detection System (Issue #274)

This test suite validates the 95%+ accuracy requirement for the dynamic dependency
detection system that replaces static label-based rules.

Test Coverage:
1. Cross-issue dependency extraction
2. Blocking declaration detection ("THIS ISSUE BLOCKS ALL OTHERS")
3. Dynamic phase detection from content
4. High-confidence dependency identification
5. Phase progression analysis
6. Integration with phase enforcement system
"""

import unittest
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to sys.path to import claude modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from claude.commands.dynamic_dependency_detector import (
    DynamicDependencyDetector,
    DependencyType,
    BlockingLevel,
    PhaseProgressionType,
    get_dynamic_dependency_analysis,
    detect_blocking_issues_dynamic,
    validate_phase_dependencies_dynamic
)

from claude.commands.content_analysis_engine import IssueState, ComplexityLevel


class TestDynamicDependencyDetector(unittest.TestCase):
    """Test suite for DynamicDependencyDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = DynamicDependencyDetector()
        self.test_knowledge_path = Path("/tmp/test_knowledge")
        self.test_knowledge_path.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if self.test_knowledge_path.exists():
            shutil.rmtree(self.test_knowledge_path)
            
    def test_cross_issue_dependency_extraction(self):
        """Test extraction of cross-issue dependencies with high accuracy"""
        
        test_cases = [
            {
                "title": "User Authentication System",
                "body": "Depends on #42 for core API framework. Also requires #15 for database schema.",
                "expected_dependencies": [42, 15],
                "expected_types": [DependencyType.DEPENDS_ON, DependencyType.REQUIRES]
            },
            {
                "title": "Frontend Integration",  
                "body": "This feature is blocked by #23 and cannot proceed until #45 is complete.",
                "expected_dependencies": [23, 45],
                "expected_types": [DependencyType.BLOCKS, DependencyType.REQUIRES]
            },
            {
                "title": "Complex Dependencies",
                "body": "Requires completion of #10, #11, and #12 before proceeding. After #20 (caching) is done, integrate with #25.",
                "expected_dependencies": [10, 11, 12, 20, 25],
                "expected_types": [DependencyType.REQUIRES] * 3 + [DependencyType.AFTER, DependencyType.INTEGRATION]
            }
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                analysis = self.detector.analyze_issue_dependencies(
                    issue_number=100 + i,
                    issue_title=case["title"],
                    issue_body=case["body"]
                )
                
                # Check dependency extraction accuracy
                extracted_deps = [d.target_issue for d in analysis.dependencies]
                self.assertEqual(
                    set(extracted_deps), 
                    set(case["expected_dependencies"]),
                    f"Failed to extract correct dependencies for case {i}"
                )
                
                # Check confidence levels (should be high)
                for dep in analysis.dependencies:
                    self.assertGreaterEqual(
                        dep.confidence, 0.7,
                        f"Low confidence dependency detected: {dep.confidence}"
                    )
                    
    def test_blocking_declaration_detection(self):
        """Test detection of blocking declarations with 95%+ accuracy"""
        
        test_cases = [
            {
                "title": "Critical Infrastructure Fix",
                "body": "THIS ISSUE BLOCKS ALL OTHERS. Must fix core system before any other work.",
                "expected_blocking": True,
                "expected_level": BlockingLevel.CRITICAL,
                "expected_scope": "all_others"
            },
            {
                "title": "Emergency System Halt",
                "body": "HALT ALL ORCHESTRATION. Critical infrastructure failure detected.",
                "expected_blocking": True,
                "expected_level": BlockingLevel.EMERGENCY,
                "expected_scope": "system_wide"
            },
            {
                "title": "Hard Dependency",
                "body": "Critical dependency that must complete before other tasks can proceed.",
                "expected_blocking": True,
                "expected_level": BlockingLevel.HARD,
                "expected_scope": "general"
            },
            {
                "title": "Soft Recommendation",
                "body": "Should complete this before starting other work, but not strictly required.",
                "expected_blocking": True,
                "expected_level": BlockingLevel.SOFT,
                "expected_scope": "general"
            },
            {
                "title": "Regular Task",
                "body": "Implement user authentication with JWT tokens and bcrypt hashing.",
                "expected_blocking": False,
                "expected_level": None,
                "expected_scope": None
            }
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                analysis = self.detector.analyze_issue_dependencies(
                    issue_number=200 + i,
                    issue_title=case["title"],
                    issue_body=case["body"]
                )
                
                # Check blocking detection accuracy
                has_blocking = len(analysis.blocking_declarations) > 0
                self.assertEqual(
                    has_blocking, 
                    case["expected_blocking"],
                    f"Incorrect blocking detection for case {i}: {case['title']}"
                )
                
                if case["expected_blocking"]:
                    blocking = analysis.blocking_declarations[0]
                    
                    # Check blocking level accuracy
                    self.assertEqual(
                        blocking.blocking_level,
                        case["expected_level"],
                        f"Incorrect blocking level for case {i}"
                    )
                    
                    # Check confidence (should be high for explicit patterns)
                    self.assertGreaterEqual(
                        blocking.confidence, 0.85,
                        f"Low confidence blocking detection: {blocking.confidence}"
                    )
                    
    def test_dynamic_phase_detection(self):
        """Test dynamic phase detection from issue content"""
        
        test_cases = [
            {
                "title": "New Feature Request",
                "body": "Need to implement user authentication. Requirements are unclear and need more research.",
                "expected_phase": IssueState.ANALYZING,
                "expected_progression": PhaseProgressionType.RESEARCH_FIRST
            },
            {
                "title": "Architecture Planning",
                "body": "Need to design the microservices architecture. Requirements analysis is complete.",
                "expected_phase": IssueState.PLANNING,
                "expected_progression": PhaseProgressionType.ARCHITECTURE_NEEDED
            },
            {
                "title": "Ready for Implementation", 
                "body": "Design is complete and architecture is finalized. Ready for coding.",
                "expected_phase": IssueState.IMPLEMENTING,
                "expected_progression": PhaseProgressionType.IMPLEMENTATION_READY
            },
            {
                "title": "Code Complete",
                "body": "Implementation is finished. Code is ready for testing and validation.",
                "expected_phase": IssueState.VALIDATING,
                "expected_progression": PhaseProgressionType.VALIDATION_PENDING
            },
            {
                "title": "Post-Implementation Review",
                "body": "Task completed successfully. Need to document lessons learned and update knowledge base.",
                "expected_phase": IssueState.LEARNING,
                "expected_progression": PhaseProgressionType.LEARNING_PHASE
            }
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                analysis = self.detector.analyze_issue_dependencies(
                    issue_number=300 + i,
                    issue_title=case["title"], 
                    issue_body=case["body"]
                )
                
                # Check phase detection accuracy
                detected_phase = analysis.content_analysis.derived_state
                self.assertEqual(
                    detected_phase,
                    case["expected_phase"],
                    f"Incorrect phase detection for case {i}: expected {case['expected_phase']}, got {detected_phase}"
                )
                
                # Check phase progression analysis
                self.assertEqual(
                    analysis.phase_progression.progression_type,
                    case["expected_progression"],
                    f"Incorrect progression type for case {i}"
                )
                
                # Check confidence
                self.assertGreaterEqual(
                    analysis.phase_progression.confidence, 0.5,
                    f"Low phase progression confidence: {analysis.phase_progression.confidence}"
                )
                
    def test_high_confidence_dependency_identification(self):
        """Test identification of high-confidence dependencies (>80%)"""
        
        # Test case with very explicit dependencies
        analysis = self.detector.analyze_issue_dependencies(
            issue_number=400,
            issue_title="Payment Processing Integration",
            issue_body="""
            This payment system depends on issue #42 (API Gateway) and requires issue #15 (Database Schema).
            
            Critical dependencies:
            - Must have API Gateway (#42) complete before integration
            - Database schema (#15) is essential prerequisite  
            - Blocked by security audit (#67) until resolution
            
            Cannot proceed until all dependencies are resolved.
            """
        )
        
        # Should detect 3 high-confidence dependencies
        high_conf_deps = [d for d in analysis.dependencies if d.confidence > 0.8]
        self.assertGreaterEqual(
            len(high_conf_deps), 2,
            f"Expected at least 2 high-confidence dependencies, got {len(high_conf_deps)}"
        )
        
        # Check specific dependency extraction
        dep_issues = [d.target_issue for d in analysis.dependencies]
        expected_deps = [42, 15, 67]
        
        for expected in expected_deps:
            self.assertIn(
                expected, dep_issues,
                f"Missing expected dependency: #{expected}"
            )
            
    def test_phase_progression_blocking_factors(self):
        """Test identification of phase progression blocking factors"""
        
        analysis = self.detector.analyze_issue_dependencies(
            issue_number=500,
            issue_title="Complex Feature Implementation",
            issue_body="""
            This feature implementation is currently blocked by several factors:
            
            - Missing requirements specification
            - Unclear design goals  
            - No system architecture defined
            - Waiting for approval from stakeholders
            - Incomplete analysis of dependencies
            
            Cannot proceed to implementation until these are resolved.
            """
        )
        
        # Should detect multiple blocking factors
        blocking_factors = analysis.phase_progression.blocking_factors
        self.assertGreater(
            len(blocking_factors), 0,
            "No blocking factors detected despite explicit mentions"
        )
        
        # Should not be ready for implementation
        self.assertNotEqual(
            analysis.phase_progression.progression_type,
            PhaseProgressionType.IMPLEMENTATION_READY,
            "Incorrectly marked as implementation ready despite blocking factors"
        )
        
    def test_orchestration_recommendations(self):
        """Test generation of orchestration recommendations"""
        
        analysis = self.detector.analyze_issue_dependencies(
            issue_number=600,
            issue_title="Critical System Fix",
            issue_body="THIS ISSUE BLOCKS ALL OTHER WORK. Critical infrastructure failure. Depends on #42 and #15."
        )
        
        # Should generate orchestration recommendations
        recommendations = analysis.orchestration_recommendations
        self.assertGreater(
            len(recommendations), 0,
            "No orchestration recommendations generated for critical blocking issue"
        )
        
        # Should mention blocking nature
        blocking_mentioned = any(
            "blocks" in rec.lower() or "critical" in rec.lower() 
            for rec in recommendations
        )
        self.assertTrue(
            blocking_mentioned,
            "Orchestration recommendations don't mention blocking nature"
        )
        
    def test_analysis_confidence_calculation(self):
        """Test overall analysis confidence calculation"""
        
        # High-confidence case
        high_conf_analysis = self.detector.analyze_issue_dependencies(
            issue_number=700,
            issue_title="Well-Defined Task",
            issue_body="Clear requirements. Depends on issue #42. Ready for implementation after architecture review."
        )
        
        self.assertGreaterEqual(
            high_conf_analysis.analysis_confidence, 0.8,
            f"Low confidence for well-defined task: {high_conf_analysis.analysis_confidence}"
        )
        
        # Low-confidence case  
        low_conf_analysis = self.detector.analyze_issue_dependencies(
            issue_number=701,
            issue_title="Vague Request",
            issue_body="Maybe fix something. Not sure what exactly. Might need other things."
        )
        
        self.assertLessEqual(
            low_conf_analysis.analysis_confidence, 0.7,
            f"High confidence for vague task: {low_conf_analysis.analysis_confidence}"
        )


class TestIntegrationFunctions(unittest.TestCase):
    """Test suite for integration functions with orchestration system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_issues = [
            {
                "number": 100,
                "title": "Regular Feature",
                "body": "Implement user dashboard with authentication."
            },
            {
                "number": 101,
                "title": "Blocking Issue",
                "body": "THIS ISSUE BLOCKS ALL OTHERS. Fix critical database connection."
            },
            {
                "number": 102, 
                "title": "Dependent Task",
                "body": "Add user preferences. Depends on #100 for authentication system."
            }
        ]
        
        self.sample_tasks = [
            {
                "description": "RIF-Implementer: Feature for issue #100",
                "prompt": "Implement issue #100 authentication feature",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Implementer: Feature for issue #102", 
                "prompt": "Implement issue #102 user preferences",
                "subagent_type": "general-purpose"
            }
        ]
        
    def test_get_dynamic_dependency_analysis(self):
        """Test the main integration function for orchestration"""
        
        analyses = get_dynamic_dependency_analysis(self.sample_issues)
        
        # Should analyze all issues
        self.assertEqual(
            len(analyses), 3,
            f"Expected 3 analyses, got {len(analyses)}"
        )
        
        # Should include all issue numbers
        expected_issues = {100, 101, 102}
        self.assertEqual(
            set(analyses.keys()), expected_issues,
            "Missing issue analyses"
        )
        
        # Each analysis should be comprehensive
        for issue_num, analysis in analyses.items():
            self.assertIsNotNone(analysis.content_analysis)
            self.assertIsNotNone(analysis.phase_progression)
            self.assertGreaterEqual(analysis.analysis_confidence, 0.5)
            
    def test_detect_blocking_issues_dynamic(self):
        """Test dynamic blocking issue detection"""
        
        blocking_issues, blocking_reasons = detect_blocking_issues_dynamic(self.sample_issues)
        
        # Should detect issue #101 as blocking
        self.assertIn(
            101, blocking_issues,
            "Failed to detect blocking issue #101"
        )
        
        # Should provide reasons
        self.assertGreater(
            len(blocking_reasons), 0,
            "No blocking reasons provided"
        )
        
        # Reason should mention the blocking issue
        blocking_reason_found = any("101" in reason for reason in blocking_reasons)
        self.assertTrue(
            blocking_reason_found,
            "Blocking reason doesn't mention issue #101"
        )
        
    def test_validate_phase_dependencies_dynamic(self):
        """Test dynamic phase dependency validation"""
        
        validation_result = validate_phase_dependencies_dynamic(
            self.sample_issues, self.sample_tasks
        )
        
        # Should return validation structure
        expected_keys = [
            'is_execution_allowed', 'violations', 'allowed_tasks', 
            'blocked_tasks', 'analysis_method', 'confidence'
        ]
        
        for key in expected_keys:
            self.assertIn(
                key, validation_result,
                f"Missing key in validation result: {key}"
            )
            
        # Should use dynamic analysis method
        self.assertEqual(
            validation_result['analysis_method'], 
            'dynamic_content_analysis',
            "Not using dynamic content analysis method"
        )
        
        # Should have high confidence
        self.assertGreaterEqual(
            validation_result['confidence'], 0.9,
            f"Low validation confidence: {validation_result['confidence']}"
        )
        
        # Should block execution due to blocking issue #101
        self.assertFalse(
            validation_result['is_execution_allowed'],
            "Execution allowed despite blocking issue"
        )


class TestAccuracyValidation(unittest.TestCase):
    """Test suite for validating 95%+ accuracy requirement"""
    
    def setUp(self):
        """Set up test fixtures with known ground truth"""
        self.detector = DynamicDependencyDetector()
        
        # Ground truth test cases for accuracy validation
        self.accuracy_test_cases = [
            {
                "issue_number": 1001,
                "title": "Authentication System",
                "body": "Implement JWT authentication. Depends on issue #42 (API framework) and requires #15 (database setup).",
                "expected_dependencies": [42, 15],
                "expected_blocking": False,
                "expected_phase": "implementing"
            },
            {
                "issue_number": 1002,
                "title": "Critical Infrastructure",
                "body": "THIS ISSUE BLOCKS ALL OTHER WORK. Critical database failure must be fixed immediately.",
                "expected_dependencies": [],
                "expected_blocking": True,
                "expected_phase": "implementing"
            },
            {
                "issue_number": 1003,
                "title": "Feature Planning",
                "body": "Need to research and analyze requirements for new reporting system. Architecture design needed.",
                "expected_dependencies": [],
                "expected_blocking": False,
                "expected_phase": "analyzing"
            },
            {
                "issue_number": 1004,
                "title": "Integration Task",
                "body": "Connect payment gateway. Blocked by #67 (security audit) and needs #23 (API endpoints) first.",
                "expected_dependencies": [67, 23],
                "expected_blocking": False,
                "expected_phase": "implementing"
            },
            {
                "issue_number": 1005,
                "title": "Testing Phase",
                "body": "Code implementation complete. Ready for testing and validation. Quality gates required.",
                "expected_dependencies": [],
                "expected_blocking": False,
                "expected_phase": "validating"
            },
            {
                "issue_number": 1006,
                "title": "Emergency Stop",
                "body": "HALT ALL ORCHESTRATION. System-wide security breach detected. Stop all work immediately.",
                "expected_dependencies": [],
                "expected_blocking": True,
                "expected_phase": "implementing"
            },
            {
                "issue_number": 1007,
                "title": "Multi-Dependency Task",
                "body": "Complex integration requiring #10, #11, and #12 to be complete. Also depends on #20 (caching layer).",
                "expected_dependencies": [10, 11, 12, 20],
                "expected_blocking": False,
                "expected_phase": "implementing"
            },
            {
                "issue_number": 1008,
                "title": "Research Task",
                "body": "Investigate new frameworks. Need to understand requirements and explore options before planning.",
                "expected_dependencies": [],
                "expected_blocking": False,
                "expected_phase": "analyzing"
            },
            {
                "issue_number": 1009,
                "title": "Soft Blocking",
                "body": "Should complete this before other features but not critical. Recommended to finish first.",
                "expected_dependencies": [],
                "expected_blocking": True,  # Soft blocking is still blocking
                "expected_phase": "implementing"
            },
            {
                "issue_number": 1010,
                "title": "Learning Phase",
                "body": "Feature completed successfully. Document lessons learned and update knowledge base with patterns.",
                "expected_dependencies": [],
                "expected_blocking": False,
                "expected_phase": "learning"
            }
        ]
        
    def test_dependency_detection_accuracy(self):
        """Test dependency detection accuracy against ground truth"""
        
        correct_detections = 0
        total_cases = len(self.accuracy_test_cases)
        
        for case in self.accuracy_test_cases:
            analysis = self.detector.analyze_issue_dependencies(
                case["issue_number"],
                case["title"],
                case["body"]
            )
            
            # Check dependency detection accuracy
            detected_deps = set(d.target_issue for d in analysis.dependencies)
            expected_deps = set(case["expected_dependencies"])
            
            if detected_deps == expected_deps:
                correct_detections += 1
            else:
                print(f"Dependency mismatch for {case['issue_number']}: expected {expected_deps}, got {detected_deps}")
                
        accuracy = correct_detections / total_cases
        print(f"Dependency Detection Accuracy: {accuracy:.1%}")
        
        # Require 95%+ accuracy for dependencies
        self.assertGreaterEqual(
            accuracy, 0.95,
            f"Dependency detection accuracy {accuracy:.1%} below required 95%"
        )
        
    def test_blocking_detection_accuracy(self):
        """Test blocking detection accuracy against ground truth"""
        
        correct_detections = 0
        total_cases = len(self.accuracy_test_cases)
        
        for case in self.accuracy_test_cases:
            analysis = self.detector.analyze_issue_dependencies(
                case["issue_number"],
                case["title"],
                case["body"]
            )
            
            # Check blocking detection accuracy
            has_blocking = len(analysis.blocking_declarations) > 0
            expected_blocking = case["expected_blocking"]
            
            if has_blocking == expected_blocking:
                correct_detections += 1
            else:
                print(f"Blocking mismatch for {case['issue_number']}: expected {expected_blocking}, got {has_blocking}")
                
        accuracy = correct_detections / total_cases
        print(f"Blocking Detection Accuracy: {accuracy:.1%}")
        
        # Require 95%+ accuracy for blocking detection
        self.assertGreaterEqual(
            accuracy, 0.95,
            f"Blocking detection accuracy {accuracy:.1%} below required 95%"
        )
        
    def test_phase_detection_accuracy(self):
        """Test phase detection accuracy against ground truth"""
        
        correct_detections = 0
        total_cases = len(self.accuracy_test_cases)
        
        for case in self.accuracy_test_cases:
            analysis = self.detector.analyze_issue_dependencies(
                case["issue_number"],
                case["title"],
                case["body"]
            )
            
            # Check phase detection accuracy
            detected_phase = analysis.content_analysis.derived_state.value
            expected_phase = case["expected_phase"]
            
            if detected_phase == expected_phase:
                correct_detections += 1
            else:
                print(f"Phase mismatch for {case['issue_number']}: expected {expected_phase}, got {detected_phase}")
                
        accuracy = correct_detections / total_cases  
        print(f"Phase Detection Accuracy: {accuracy:.1%}")
        
        # Require 90%+ accuracy for phase detection (slightly lower threshold due to complexity)
        self.assertGreaterEqual(
            accuracy, 0.90,
            f"Phase detection accuracy {accuracy:.1%} below required 90%"
        )
        
    def test_overall_system_accuracy(self):
        """Test overall system accuracy combining all detection types"""
        
        dependency_correct = 0
        blocking_correct = 0
        phase_correct = 0
        total_cases = len(self.accuracy_test_cases)
        
        for case in self.accuracy_test_cases:
            analysis = self.detector.analyze_issue_dependencies(
                case["issue_number"],
                case["title"],
                case["body"]
            )
            
            # Dependency accuracy
            detected_deps = set(d.target_issue for d in analysis.dependencies)
            expected_deps = set(case["expected_dependencies"])
            if detected_deps == expected_deps:
                dependency_correct += 1
                
            # Blocking accuracy
            has_blocking = len(analysis.blocking_declarations) > 0
            if has_blocking == case["expected_blocking"]:
                blocking_correct += 1
                
            # Phase accuracy
            detected_phase = analysis.content_analysis.derived_state.value
            if detected_phase == case["expected_phase"]:
                phase_correct += 1
                
        # Calculate individual accuracies
        dependency_accuracy = dependency_correct / total_cases
        blocking_accuracy = blocking_correct / total_cases
        phase_accuracy = phase_correct / total_cases
        
        # Calculate overall accuracy (weighted average)
        overall_accuracy = (dependency_accuracy + blocking_accuracy + phase_accuracy) / 3
        
        print(f"\n🎯 ACCURACY TEST RESULTS:")
        print(f"   Dependency Detection: {dependency_accuracy:.1%}")
        print(f"   Blocking Detection: {blocking_accuracy:.1%}")  
        print(f"   Phase Detection: {phase_accuracy:.1%}")
        print(f"   Overall Accuracy: {overall_accuracy:.1%}")
        
        # Require 95%+ overall accuracy
        self.assertGreaterEqual(
            overall_accuracy, 0.95,
            f"Overall system accuracy {overall_accuracy:.1%} below required 95%"
        )


def run_comprehensive_tests():
    """Run all tests and generate accuracy report"""
    
    print("🧪 RUNNING COMPREHENSIVE DYNAMIC DEPENDENCY DETECTION TESTS")
    print("=" * 70)
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDynamicDependencyDetector,
        TestIntegrationFunctions,
        TestAccuracyValidation
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests) * 100
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_tests - failures - errors}")
    print(f"   Failed: {failures}")
    print(f"   Errors: {errors}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("✅ ISSUE #274 REQUIREMENTS MET: 95%+ accuracy achieved")
    else:
        print("❌ ISSUE #274 REQUIREMENTS NOT MET: Below 95% accuracy")
        
    return result


if __name__ == "__main__":
    run_comprehensive_tests()