#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase Dependency Implementation

Tests all components of the phase dependency enforcement system:
- PhaseDependencyValidator
- PhaseDependencyWarningSystem  
- PhaseDependencyOrchestrationIntegration

Issue #223: RIF Orchestration Error: Not Following Phase Dependencies
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the claude commands directory to Python path
sys.path.append(str(Path(__file__).parent / "claude" / "commands"))

try:
    from phase_dependency_validator import (
        PhaseDependencyValidator,
        PhaseType,
        validate_phase_completion,
        enforce_sequential_phases
    )
    from phase_dependency_warning_system import (
        PhaseDependencyWarningSystem,
        AlertLevel,
        detect_phase_violations,
        generate_prevention_report
    )
    from phase_dependency_orchestration_integration import (
        PhaseDependencyOrchestrationIntegration,
        make_enhanced_orchestration_decision_with_phase_validation,
        generate_enhanced_orchestration_template
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all phase dependency components are implemented correctly.")
    sys.exit(1)


class PhaseDependencyTestSuite:
    """Comprehensive test suite for phase dependency system"""
    
    def __init__(self):
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Starting Phase Dependency Implementation Test Suite")
        print("=" * 60)
        
        # Core validation tests
        self.test_phase_completion_validation()
        self.test_sequential_phase_enforcement()
        self.test_foundation_dependency_validation()
        self.test_violation_detection()
        
        # Warning system tests  
        self.test_real_time_violation_detection()
        self.test_actionable_message_generation()
        self.test_auto_redirection()
        self.test_resource_waste_prevention()
        
        # Integration tests
        self.test_orchestration_integration()
        self.test_github_integration()
        self.test_enhanced_decision_making()
        
        # Scenario-based tests
        self.test_critical_violation_scenarios()
        self.test_dpibs_epic_scenario()
        self.test_performance_benchmarks()
        
        # Print final results
        self.print_test_summary()
        
        return self.test_results["failed"] == 0
        
    def test_phase_completion_validation(self):
        """Test Phase 1: Basic phase completion validation"""
        print("\n🔍 Testing Phase Completion Validation...")
        
        # Test case 1: Research phase completion
        research_complete_issue = {
            "number": 1,
            "title": "Research authentication patterns",
            "labels": [
                {"name": "state:planning"},
                {"name": "analysis:complete"}
            ],
            "body": "Research findings: OAuth 2.0 is recommended"
        }
        
        research_incomplete_issue = {
            "number": 2, 
            "title": "Research database patterns",
            "labels": [{"name": "state:analyzing"}],
            "body": "Still researching approaches"
        }
        
        validator = PhaseDependencyValidator()
        
        # Test completed research phase
        self.assert_test(
            validator.validate_phase_completion(research_complete_issue, PhaseType.RESEARCH),
            "Research phase completion detection (complete)"
        )
        
        # Test incomplete research phase  
        self.assert_test(
            not validator.validate_phase_completion(research_incomplete_issue, PhaseType.RESEARCH),
            "Research phase completion detection (incomplete)"
        )
        
        # Test convenience function
        self.assert_test(
            validate_phase_completion(research_complete_issue, "research"),
            "Convenience function: validate_phase_completion"
        )
        
    def test_sequential_phase_enforcement(self):
        """Test Phase 2: Sequential phase enforcement"""
        print("\n📋 Testing Sequential Phase Enforcement...")
        
        issues = [
            {
                "number": 1,
                "title": "Implement authentication",
                "labels": [
                    {"name": "state:implementing"},
                    {"name": "analysis:complete"},
                    {"name": "planning:complete"},
                    {"name": "architecture:complete"}
                ],
                "body": "Ready for implementation"
            },
            {
                "number": 2,
                "title": "Implement database layer", 
                "labels": [{"name": "state:analyzing"}],
                "body": "Still in analysis phase"
            }
        ]
        
        validator = PhaseDependencyValidator()
        
        # Test sequential enforcement for implementation phase
        ready_issues, blocking_reasons = validator.enforce_sequential_phases(
            issues, PhaseType.IMPLEMENTATION
        )
        
        self.assert_test(
            1 in ready_issues,
            "Issue #1 ready for implementation phase"
        )
        
        self.assert_test(
            2 not in ready_issues,
            "Issue #2 blocked from implementation phase"
        )
        
        self.assert_test(
            len(blocking_reasons) > 0,
            "Blocking reasons provided for incomplete issues"
        )
        
        # Test convenience function
        ready_conv, blocking_conv = enforce_sequential_phases(issues, "implementation")
        self.assert_test(
            ready_conv == ready_issues,
            "Convenience function: enforce_sequential_phases"
        )
        
    def test_foundation_dependency_validation(self):
        """Test Phase 3: Foundation dependency validation"""
        print("\n🏗️ Testing Foundation Dependency Validation...")
        
        github_issues = [
            {
                "number": 1,
                "title": "Core API framework",
                "labels": [{"name": "state:implementing"}],
                "body": "Foundation API framework"
            },
            {
                "number": 2,
                "title": "User authentication service",
                "labels": [{"name": "state:implementing"}], 
                "body": "Depends on core API framework"
            }
        ]
        
        proposed_launches = [
            {
                "description": "RIF-Implementer: User authentication implementation",
                "prompt": "Implement user authentication for issue #2",
                "subagent_type": "general-purpose"
            }
        ]
        
        validator = PhaseDependencyValidator()
        result = validator.validate_phase_dependencies(github_issues, proposed_launches)
        
        # Should detect foundation dependency violation
        foundation_violations = [
            v for v in result.violations 
            if v.violation_type == "foundation_dependency_violation"
        ]
        
        self.assert_test(
            len(foundation_violations) > 0,
            "Foundation dependency violation detection"
        )
        
    def test_violation_detection(self):
        """Test Phase 4: Violation detection accuracy"""
        print("\n⚠️ Testing Violation Detection...")
        
        # Critical violation scenario: Implementation without architecture
        issues = [
            {
                "number": 1,
                "title": "Complex payment system",
                "labels": [{"name": "state:analyzing"}],
                "body": "Complex payment processing system"
            }
        ]
        
        launches = [
            {
                "description": "RIF-Implementer: Payment system implementation", 
                "prompt": "Implement payment system for issue #1",
                "subagent_type": "general-purpose"
            }
        ]
        
        validator = PhaseDependencyValidator()
        result = validator.validate_phase_dependencies(issues, launches)
        
        self.assert_test(
            not result.is_valid,
            "Critical violation detected (implementation without prerequisites)"
        )
        
        self.assert_test(
            len(result.violations) > 0,
            "Violations list populated"
        )
        
        # Check violation severity
        critical_violations = [v for v in result.violations if v.severity == "critical"]
        self.assert_test(
            len(critical_violations) > 0,
            "Critical severity assigned correctly"
        )
        
    def test_real_time_violation_detection(self):
        """Test Phase 5: Real-time warning system"""
        print("\n🚨 Testing Real-time Violation Detection...")
        
        issues = [
            {
                "number": 1,
                "title": "Research ML algorithms", 
                "labels": [{"name": "state:new"}],
                "body": "Research machine learning approaches"
            }
        ]
        
        launches = [
            {
                "description": "RIF-Implementer: ML algorithm implementation",
                "prompt": "Implement ML algorithms for issue #1", 
                "subagent_type": "general-purpose"
            }
        ]
        
        warning_system = PhaseDependencyWarningSystem()
        alerts = warning_system.detect_violations_real_time(issues, launches)
        
        self.assert_test(
            len(alerts) > 0,
            "Real-time alerts generated"
        )
        
        # Check alert levels
        critical_alerts = [a for a in alerts if a.alert_level == AlertLevel.CRITICAL]
        self.assert_test(
            len(critical_alerts) > 0,
            "Critical alerts detected"
        )
        
        # Test convenience function
        convenience_alerts = detect_phase_violations(issues, launches)
        self.assert_test(
            len(convenience_alerts) == len(alerts),
            "Convenience function: detect_phase_violations"
        )
        
    def test_actionable_message_generation(self):
        """Test Phase 6: Actionable message generation"""
        print("\n💬 Testing Actionable Message Generation...")
        
        warning_system = PhaseDependencyWarningSystem()
        
        # Create mock alert
        from phase_dependency_warning_system import PhaseWarningAlert, AlertLevel
        
        alert = PhaseWarningAlert(
            alert_id="test_alert_001",
            alert_level=AlertLevel.CRITICAL,
            violation_type="sequential_phase_violation",
            affected_issues=[1],
            attempted_phase="implementation", 
            missing_phases=["analysis", "architecture"],
            warning_message="Implementation attempted without prerequisites",
            actionable_steps=[
                "Launch RIF-Analyst first",
                "Complete architecture phase",
                "Wait for prerequisites"
            ],
            auto_redirect_available=True,
            timestamp="2025-01-01T00:00:00Z"
        )
        
        messages = warning_system.generate_actionable_messages([alert])
        
        self.assert_test(
            len(messages) > 0,
            "Actionable messages generated"
        )
        
        self.assert_test(
            "test_alert_001" in messages,
            "Alert ID mapped correctly"
        )
        
        # Check message contains actionable steps
        message_content = messages["test_alert_001"]
        self.assert_test(
            "Launch RIF-Analyst" in message_content,
            "Actionable steps included in message"
        )
        
    def test_auto_redirection(self):
        """Test Phase 7: Auto-redirection functionality"""
        print("\n🔄 Testing Auto-redirection...")
        
        issues = [
            {
                "number": 1,
                "title": "API implementation", 
                "labels": [{"name": "state:new"}],
                "body": "Implement REST API"
            }
        ]
        
        blocked_launches = [
            {
                "description": "RIF-Implementer: API implementation",
                "prompt": "Implement API for issue #1",
                "subagent_type": "general-purpose"
            }
        ]
        
        warning_system = PhaseDependencyWarningSystem()
        redirections = warning_system.auto_redirect_to_prerequisites(issues, blocked_launches)
        
        self.assert_test(
            len(redirections) > 0,
            "Auto-redirections generated"
        )
        
        if redirections:
            redirection = redirections[0]
            self.assert_test(
                len(redirection.redirected_agents) > 0,
                "Redirected agents provided"
            )
            
            self.assert_test(
                "RIF-Analyst" in str(redirection.redirected_agents),
                "Analysis agent included in redirection"
            )
            
    def test_resource_waste_prevention(self):
        """Test Phase 8: Resource waste prevention metrics"""
        print("\n💰 Testing Resource Waste Prevention...")
        
        validator = PhaseDependencyValidator()
        warning_system = PhaseDependencyWarningSystem()
        
        # Create scenario with multiple critical violations
        issues = [
            {"number": i, "title": f"Feature {i}", "labels": [{"name": "state:new"}], "body": f"Feature {i}"}
            for i in range(1, 4)
        ]
        
        launches = [
            {
                "description": f"RIF-Implementer: Feature {i} implementation",
                "prompt": f"Implement feature {i} for issue #{i}",
                "subagent_type": "general-purpose"
            }
            for i in range(1, 4)
        ]
        
        result = validator.validate_phase_dependencies(issues, launches)
        prevention_metrics = warning_system.prevent_resource_waste(result)
        
        self.assert_test(
            prevention_metrics["blocked_agents"] > 0,
            "Blocked agents counted"
        )
        
        self.assert_test(
            prevention_metrics["saved_agent_hours"] > 0,
            "Saved agent hours calculated"
        )
        
        # Test convenience function
        conv_metrics = generate_prevention_report(result)
        self.assert_test(
            conv_metrics["blocked_agents"] == prevention_metrics["blocked_agents"],
            "Convenience function: generate_prevention_report"
        )
        
    def test_orchestration_integration(self):
        """Test Phase 9: Orchestration framework integration"""
        print("\n🧠 Testing Orchestration Integration...")
        
        issues = [
            {
                "number": 1,
                "title": "Database schema design",
                "labels": [
                    {"name": "state:implementing"},
                    {"name": "analysis:complete"},
                    {"name": "architecture:complete"}
                ],
                "body": "Database schema is designed"
            }
        ]
        
        launches = [
            {
                "description": "RIF-Implementer: Database implementation",
                "prompt": "Implement database schema for issue #1",
                "subagent_type": "general-purpose"
            }
        ]
        
        try:
            decision = make_enhanced_orchestration_decision_with_phase_validation(
                issues, launches
            )
            
            self.assert_test(
                decision is not None,
                "Enhanced orchestration decision created"
            )
            
            self.assert_test(
                hasattr(decision, 'phase_validation_result'),
                "Phase validation result included"
            )
            
            self.assert_test(
                hasattr(decision, 'final_enforcement_action'),
                "Final enforcement action determined"
            )
            
            # Test template generation
            template = generate_enhanced_orchestration_template(issues, launches)
            
            self.assert_test(
                len(template) > 0,
                "Orchestration template generated"
            )
            
            self.assert_test(
                "Phase Validation:" in template,
                "Template includes phase validation status"
            )
            
        except Exception as e:
            self.test_results["errors"].append(f"Orchestration integration test error: {e}")
            self.assert_test(False, "Orchestration integration (error occurred)")
            
    def test_github_integration(self):
        """Test Phase 10: GitHub integration functionality"""  
        print("\n🔗 Testing GitHub Integration...")
        
        integration = PhaseDependencyOrchestrationIntegration()
        
        # Test GitHub state checking (mock)
        issue_numbers = [1, 2, 3]
        states = integration.check_github_state_real_time(issue_numbers)
        
        self.assert_test(
            len(states) == 3,
            "GitHub state checking returns correct count"
        )
        
        self.assert_test(
            all(isinstance(states[i], dict) for i in issue_numbers),
            "GitHub states are properly formatted"
        )
        
        # Test hooks setup
        hooks_config = integration.setup_automated_enforcement_hooks()
        
        self.assert_test(
            isinstance(hooks_config, dict),
            "Hooks configuration returned"
        )
        
        self.assert_test(
            hooks_config.get("pre_agent_launch_validation", False),
            "Pre-launch validation hook enabled"
        )
        
    def test_enhanced_decision_making(self):
        """Test Phase 11: Enhanced decision making"""
        print("\n⚖️ Testing Enhanced Decision Making...")
        
        integration = PhaseDependencyOrchestrationIntegration()
        
        # Test with valid scenario
        valid_issues = [
            {
                "number": 1,
                "title": "Feature implementation",
                "labels": [
                    {"name": "state:implementing"},
                    {"name": "analysis:complete"},
                    {"name": "planning:complete"},
                    {"name": "architecture:complete"}
                ],
                "body": "Ready for implementation"
            }
        ]
        
        valid_launches = [
            {
                "description": "RIF-Implementer: Feature implementation",
                "prompt": "Implement feature for issue #1",
                "subagent_type": "general-purpose"
            }
        ]
        
        decision = integration.make_enhanced_orchestration_decision(
            valid_issues, valid_launches
        )
        
        self.assert_test(
            decision.phase_validation_result.is_valid,
            "Valid scenario passes phase validation"
        )
        
        self.assert_test(
            decision.final_enforcement_action == "allow_execution",
            "Valid scenario allows execution"
        )
        
        # Test with invalid scenario
        invalid_issues = [
            {
                "number": 2,
                "title": "Complex feature",
                "labels": [{"name": "state:new"}],
                "body": "New complex feature"
            }
        ]
        
        invalid_launches = [
            {
                "description": "RIF-Implementer: Complex implementation", 
                "prompt": "Implement complex feature for issue #2",
                "subagent_type": "general-purpose"
            }
        ]
        
        decision = integration.make_enhanced_orchestration_decision(
            invalid_issues, invalid_launches
        )
        
        self.assert_test(
            not decision.phase_validation_result.is_valid,
            "Invalid scenario fails phase validation"
        )
        
        self.assert_test(
            decision.final_enforcement_action.startswith("block"),
            "Invalid scenario blocks execution"
        )
        
    def test_critical_violation_scenarios(self):
        """Test Phase 12: Critical violation scenarios"""
        print("\n🚨 Testing Critical Violation Scenarios...")
        
        # Scenario 1: Implementation without any prerequisites
        scenario1_issues = [
            {
                "number": 1,
                "title": "New complex system",
                "labels": [{"name": "state:new"}],
                "body": "Brand new complex system"
            }
        ]
        
        scenario1_launches = [
            {
                "description": "RIF-Implementer: Complex system implementation",
                "prompt": "Implement complex system for issue #1", 
                "subagent_type": "general-purpose"
            }
        ]
        
        validator = PhaseDependencyValidator()
        result = validator.validate_phase_dependencies(scenario1_issues, scenario1_launches)
        
        self.assert_test(
            not result.is_valid,
            "Scenario 1: Implementation without prerequisites blocked"
        )
        
        critical_violations = [v for v in result.violations if v.severity == "critical"]
        self.assert_test(
            len(critical_violations) > 0,
            "Scenario 1: Critical severity assigned"
        )
        
        # Scenario 2: Multiple dependent issues launched while foundation incomplete  
        scenario2_issues = [
            {
                "number": 1,
                "title": "Core API framework", 
                "labels": [{"name": "state:analyzing"}],
                "body": "Core API - still in analysis"
            },
            {
                "number": 2,
                "title": "User service",
                "labels": [{"name": "state:implementing"}],
                "body": "User management service" 
            },
            {
                "number": 3,
                "title": "Auth service",
                "labels": [{"name": "state:implementing"}], 
                "body": "Authentication service"
            }
        ]
        
        scenario2_launches = [
            {
                "description": "RIF-Implementer: User service",
                "prompt": "Implement user service for issue #2",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Implementer: Auth service",
                "prompt": "Implement auth service for issue #3",
                "subagent_type": "general-purpose"
            }
        ]
        
        result2 = validator.validate_phase_dependencies(scenario2_issues, scenario2_launches)
        
        self.assert_test(
            not result2.is_valid,
            "Scenario 2: Foundation dependency violations detected"
        )
        
        foundation_violations = [
            v for v in result2.violations 
            if v.violation_type == "foundation_dependency_violation"
        ]
        self.assert_test(
            len(foundation_violations) > 0,
            "Scenario 2: Foundation violations identified"
        )
        
    def test_dpibs_epic_scenario(self):
        """Test Phase 13: DPIBS Epic scenario (real-world)"""
        print("\n📊 Testing DPIBS Epic Scenario...")
        
        # Simulate DPIBS Epic with multiple phases
        dpibs_issues = [
            {
                "number": 133,
                "title": "DPIBS Live Context Architecture Research",
                "labels": [{"name": "state:analyzing"}],
                "body": "Research live context architecture for DPIBS"
            },
            {
                "number": 134,  
                "title": "DPIBS Dependency Tracking Framework Research",
                "labels": [{"name": "state:analyzing"}],
                "body": "Research dependency tracking approaches"
            },
            {
                "number": 135,
                "title": "DPIBS Agent Context Delivery", 
                "labels": [{"name": "state:implementing"}],
                "body": "Implement agent context delivery"
            },
            {
                "number": 136,
                "title": "DPIBS Performance Optimization",
                "labels": [{"name": "state:validating"}],
                "body": "Validate DPIBS performance"
            }
        ]
        
        # Problematic launch: trying to implement/validate before research complete
        problematic_launches = [
            {
                "description": "RIF-Implementer: DPIBS context delivery",
                "prompt": "Implement context delivery for issue #135",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Validator: DPIBS performance validation",
                "prompt": "Validate performance for issue #136", 
                "subagent_type": "general-purpose"
            }
        ]
        
        validator = PhaseDependencyValidator()
        result = validator.validate_phase_dependencies(dpibs_issues, problematic_launches)
        
        self.assert_test(
            not result.is_valid,
            "DPIBS scenario: Phase violations detected"
        )
        
        # Should have violations for launching implementation/validation while research incomplete
        research_violations = [
            v for v in result.violations
            if PhaseType.ANALYSIS in v.missing_prerequisite_phases
        ]
        self.assert_test(
            len(research_violations) > 0,
            "DPIBS scenario: Research prerequisite violations found"
        )
        
        # Test auto-redirection for DPIBS
        warning_system = PhaseDependencyWarningSystem()
        redirections = warning_system.auto_redirect_to_prerequisites(dpibs_issues, problematic_launches)
        
        self.assert_test(
            len(redirections) > 0,
            "DPIBS scenario: Auto-redirections generated"
        )
        
        # Should redirect to analysis agents
        if redirections:
            redirection = redirections[0]
            analyst_agents = [
                agent for agent in redirection.redirected_agents 
                if "RIF-Analyst" in agent.get("description", "")
            ]
            self.assert_test(
                len(analyst_agents) > 0,
                "DPIBS scenario: Redirected to analysis agents"
            )
            
    def test_performance_benchmarks(self):
        """Test Phase 14: Performance benchmarks"""
        print("\n⚡ Testing Performance Benchmarks...")
        
        # Create large scenario for performance testing
        large_issues = [
            {
                "number": i,
                "title": f"Feature {i}",
                "labels": [{"name": "state:new"}],
                "body": f"Feature {i} description"
            }
            for i in range(1, 21)  # 20 issues
        ]
        
        large_launches = [
            {
                "description": f"RIF-Implementer: Feature {i}",
                "prompt": f"Implement feature {i} for issue #{i}",
                "subagent_type": "general-purpose"
            }
            for i in range(1, 21)  # 20 launches
        ]
        
        # Benchmark phase validation
        start_time = time.time()
        
        validator = PhaseDependencyValidator()
        result = validator.validate_phase_dependencies(large_issues, large_launches)
        
        validation_time = time.time() - start_time
        
        self.assert_test(
            validation_time < 5.0,  # Should complete within 5 seconds
            f"Performance: Phase validation completed in {validation_time:.2f}s (< 5s)"
        )
        
        # Benchmark enhanced orchestration
        start_time = time.time()
        
        try:
            integration = PhaseDependencyOrchestrationIntegration()
            decision = integration.make_enhanced_orchestration_decision(
                large_issues[:5],  # Limit to 5 for integration test
                large_launches[:5]
            )
            
            integration_time = time.time() - start_time
            
            self.assert_test(
                integration_time < 10.0,  # Should complete within 10 seconds
                f"Performance: Enhanced orchestration completed in {integration_time:.2f}s (< 10s)"
            )
            
        except Exception as e:
            self.test_results["errors"].append(f"Performance benchmark error: {e}")
            self.assert_test(False, "Performance benchmark (error occurred)")
            
    def assert_test(self, condition: bool, description: str):
        """Helper method to track test results"""
        self.test_results["total_tests"] += 1
        
        if condition:
            print(f"  ✅ {description}")
            self.test_results["passed"] += 1
        else:
            print(f"  ❌ {description}")
            self.test_results["failed"] += 1
            
    def print_test_summary(self):
        """Print final test results summary"""
        print("\n" + "=" * 60)
        print("📋 TEST SUITE SUMMARY")
        print("=" * 60)
        
        total = self.test_results["total_tests"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({(passed/total)*100:.1f}%)")
        print(f"Failed: {failed} ({(failed/total)*100:.1f}%)")
        
        if self.test_results["errors"]:
            print(f"\nErrors encountered: {len(self.test_results['errors'])}")
            for i, error in enumerate(self.test_results["errors"], 1):
                print(f"  {i}. {error}")
                
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Phase Dependency Implementation is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please review implementation.")
            
        print("=" * 60)


def run_validation_examples():
    """Run validation examples to demonstrate functionality"""
    print("\n🔍 VALIDATION EXAMPLES")
    print("-" * 40)
    
    # Example 1: Correct phase progression
    print("\n✅ Example 1: Correct Phase Progression")
    correct_issues = [
        {
            "number": 1,
            "title": "User authentication system",
            "labels": [
                {"name": "state:implementing"},
                {"name": "analysis:complete"}, 
                {"name": "planning:complete"},
                {"name": "architecture:complete"}
            ],
            "body": "Authentication system ready for implementation"
        }
    ]
    
    correct_launches = [
        {
            "description": "RIF-Implementer: Authentication implementation",
            "prompt": "Implement authentication system for issue #1",
            "subagent_type": "general-purpose"
        }
    ]
    
    validator = PhaseDependencyValidator()
    result = validator.validate_phase_dependencies(correct_issues, correct_launches)
    
    print(f"  Phase Validation: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Confidence: {result.confidence_score:.2f}")
    
    # Example 2: Phase dependency violation  
    print("\n❌ Example 2: Phase Dependency Violation")
    violation_issues = [
        {
            "number": 2,
            "title": "Payment processing system",
            "labels": [{"name": "state:new"}],
            "body": "New payment system - just created"
        }
    ]
    
    violation_launches = [
        {
            "description": "RIF-Implementer: Payment implementation",
            "prompt": "Implement payment system for issue #2",
            "subagent_type": "general-purpose"
        }
    ]
    
    result = validator.validate_phase_dependencies(violation_issues, violation_launches)
    
    print(f"  Phase Validation: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Confidence: {result.confidence_score:.2f}")
    
    if result.violations:
        violation = result.violations[0]
        print(f"  Violation Type: {violation.violation_type}")
        print(f"  Severity: {violation.severity}")
        print(f"  Missing Phases: {[p.value for p in violation.missing_prerequisite_phases]}")
        print(f"  Remediation: {violation.remediation_actions[0]}")
        
    # Example 3: Enhanced orchestration decision
    print("\n🧠 Example 3: Enhanced Orchestration Decision")
    try:
        decision = make_enhanced_orchestration_decision_with_phase_validation(
            violation_issues, violation_launches
        )
        
        print(f"  Enhanced Decision: {decision.final_enforcement_action}")
        print(f"  Phase Validation: {'✅ PASSED' if decision.phase_validation_result.is_valid else '❌ FAILED'}")
        print(f"  Alerts Generated: {len(decision.phase_warning_alerts)}")
        print(f"  Auto-redirections: {len(decision.auto_redirections)}")
        print(f"  Resource Waste Prevented: {decision.prevented_resource_waste.get('saved_agent_hours', 0)} hours")
        
        if decision.auto_redirections:
            redirection = decision.auto_redirections[0]
            print(f"  Redirection: {redirection.rationale}")
            print(f"  Redirected Agents: {len(redirection.redirected_agents)}")
            
    except Exception as e:
        print(f"  ❌ Error in enhanced orchestration: {e}")


if __name__ == "__main__":
    # Run the complete test suite
    test_suite = PhaseDependencyTestSuite()
    success = test_suite.run_all_tests()
    
    # Run validation examples
    run_validation_examples()
    
    # Create test results file
    results_file = Path(__file__).parent / "test_phase_dependency_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_results": test_suite.test_results,
            "timestamp": time.time(),
            "success": success,
            "issue_reference": 223
        }, f, indent=2)
        
    print(f"\n📄 Test results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)