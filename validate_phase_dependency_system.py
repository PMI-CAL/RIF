#!/usr/bin/env python3
"""
RIF-Validator: Phase Dependency System Validation Script

Validates the Phase Dependency Enforcement System implementation for Issue #223.
This script performs comprehensive testing of all major components and functionality.
"""

import json
import time
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent / "claude" / "commands"))

def test_phase_dependency_validator():
    """Test core phase dependency validation"""
    print("🔍 Testing Phase Dependency Validator...")
    
    try:
        from phase_dependency_validator import PhaseDependencyValidator, PhaseType
        
        validator = PhaseDependencyValidator()
        
        # Test 1: Basic phase completion validation
        research_complete_issue = {
            "number": 1,
            "title": "Research authentication patterns",
            "labels": [
                {"name": "state:planning"},
                {"name": "analysis:complete"}
            ],
            "body": "Research findings documented"
        }
        
        result = validator.validate_phase_completion(research_complete_issue, PhaseType.RESEARCH)
        print(f"  ✅ Phase completion validation: {result}")
        
        # Test 2: Phase dependency violation detection
        test_issues = [
            {
                "number": 1,
                "title": "New feature implementation",
                "labels": [{"name": "state:new"}],
                "body": "Brand new feature"
            }
        ]
        
        test_launches = [
            {
                "description": "RIF-Implementer: Feature implementation",
                "prompt": "Implement feature for issue #1",
                "subagent_type": "general-purpose"
            }
        ]
        
        validation_result = validator.validate_phase_dependencies(test_issues, test_launches)
        print(f"  ✅ Violation detection: {not validation_result.is_valid} (should be True)")
        print(f"  ✅ Violations found: {len(validation_result.violations)}")
        
        # Test 3: Performance benchmark
        start_time = time.time()
        validator.validate_phase_dependencies(test_issues * 10, test_launches * 10)  # Scale up
        validation_time = time.time() - start_time
        
        performance_acceptable = validation_time < 1.0  # Should be under 1 second
        print(f"  ✅ Performance test: {validation_time:.3f}s ({'PASS' if performance_acceptable else 'FAIL'})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing validator: {e}")
        return False

def test_phase_dependency_warning_system():
    """Test warning and prevention system"""
    print("\n🚨 Testing Phase Dependency Warning System...")
    
    try:
        from phase_dependency_warning_system import PhaseDependencyWarningSystem, AlertLevel
        
        warning_system = PhaseDependencyWarningSystem()
        
        # Test 1: Real-time violation detection
        issues = [
            {
                "number": 1,
                "title": "Complex system",
                "labels": [{"name": "state:new"}],
                "body": "Complex new system"
            }
        ]
        
        launches = [
            {
                "description": "RIF-Implementer: Complex implementation", 
                "prompt": "Implement complex system for issue #1",
                "subagent_type": "general-purpose"
            }
        ]
        
        alerts = warning_system.detect_violations_real_time(issues, launches)
        print(f"  ✅ Real-time alerts generated: {len(alerts)}")
        
        critical_alerts = [a for a in alerts if a.alert_level == AlertLevel.CRITICAL]
        print(f"  ✅ Critical alerts detected: {len(critical_alerts)}")
        
        # Test 2: Actionable message generation
        if alerts:
            messages = warning_system.generate_actionable_messages(alerts)
            print(f"  ✅ Actionable messages generated: {len(messages)}")
            
            # Check message quality
            if messages:
                first_message = list(messages.values())[0]
                has_actionable_steps = any(keyword in first_message.lower() for keyword in 
                                         ["launch", "complete", "wait", "action"])
                print(f"  ✅ Messages contain actionable steps: {has_actionable_steps}")
        
        # Test 3: Auto-redirection
        redirections = warning_system.auto_redirect_to_prerequisites(issues, launches)
        print(f"  ✅ Auto-redirections generated: {len(redirections)}")
        
        if redirections:
            redirection = redirections[0]
            print(f"  ✅ Redirected agents: {len(redirection.redirected_agents)}")
            
            # Check if redirected to appropriate phase agents
            has_analyst_redirect = any("RIF-Analyst" in str(agent) for agent in redirection.redirected_agents)
            print(f"  ✅ Redirected to analysis phase: {has_analyst_redirect}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing warning system: {e}")
        return False

def test_phase_completion_criteria():
    """Test phase completion criteria matrix"""
    print("\n📋 Testing Phase Completion Criteria...")
    
    try:
        from phase_dependency_validator import PhaseDependencyValidator, PhaseType
        
        validator = PhaseDependencyValidator()
        
        # Test each phase's completion criteria
        phase_tests = [
            # Research phase complete
            {
                "phase": PhaseType.RESEARCH,
                "issue": {
                    "number": 1,
                    "title": "Research complete",
                    "labels": [{"name": "state:planning"}, {"name": "analysis:complete"}],
                    "body": "Research findings documented"
                },
                "should_pass": True
            },
            # Analysis phase incomplete
            {
                "phase": PhaseType.ANALYSIS, 
                "issue": {
                    "number": 2,
                    "title": "Analysis in progress",
                    "labels": [{"name": "state:analyzing"}],
                    "body": "Still analyzing requirements"
                },
                "should_pass": False
            },
            # Implementation phase with prerequisites
            {
                "phase": PhaseType.IMPLEMENTATION,
                "issue": {
                    "number": 3,
                    "title": "Ready for implementation",
                    "labels": [
                        {"name": "state:implementing"},
                        {"name": "analysis:complete"},
                        {"name": "planning:complete"},
                        {"name": "architecture:complete"}
                    ],
                    "body": "All prerequisites complete"
                },
                "should_pass": False  # State is implementing, not complete yet
            }
        ]
        
        passed_tests = 0
        for test in phase_tests:
            result = validator.validate_phase_completion(test["issue"], test["phase"])
            expected = test["should_pass"]
            
            if result == expected:
                print(f"  ✅ {test['phase'].value} phase criteria: PASS")
                passed_tests += 1
            else:
                print(f"  ❌ {test['phase'].value} phase criteria: FAIL (got {result}, expected {expected})")
        
        success_rate = passed_tests / len(phase_tests)
        print(f"  ✅ Overall criteria accuracy: {success_rate:.1%}")
        
        return success_rate >= 0.8  # At least 80% should pass
        
    except Exception as e:
        print(f"  ❌ Error testing phase criteria: {e}")
        return False

def test_sequential_phase_enforcement():
    """Test sequential phase enforcement rules"""
    print("\n🔄 Testing Sequential Phase Enforcement...")
    
    try:
        from phase_dependency_validator import PhaseDependencyValidator, PhaseType
        
        validator = PhaseDependencyValidator()
        
        # Test scenario: Mixed readiness for implementation phase
        issues = [
            # Ready for implementation
            {
                "number": 1,
                "title": "Feature A implementation",
                "labels": [
                    {"name": "state:implementing"},
                    {"name": "analysis:complete"},
                    {"name": "planning:complete"},  
                    {"name": "architecture:complete"}
                ],
                "body": "All prerequisites complete"
            },
            # Not ready - missing architecture
            {
                "number": 2,
                "title": "Feature B implementation",
                "labels": [
                    {"name": "state:planning"},
                    {"name": "analysis:complete"}
                ],
                "body": "Architecture phase not started"
            },
            # Not ready - brand new
            {
                "number": 3,
                "title": "Feature C implementation",
                "labels": [{"name": "state:new"}],
                "body": "Just created"
            }
        ]
        
        ready_issues, blocking_reasons = validator.enforce_sequential_phases(
            issues, PhaseType.IMPLEMENTATION
        )
        
        print(f"  ✅ Ready issues identified: {len(ready_issues)}")
        print(f"  ✅ Blocking reasons provided: {len(blocking_reasons)}")
        
        # Should have 0 ready issues (issue 1 is in implementing state, not complete prerequisites)
        expected_ready = 0
        actual_ready = len(ready_issues)
        
        if actual_ready == expected_ready:
            print(f"  ✅ Sequential enforcement accuracy: PASS ({actual_ready} ready)")
        else:
            print(f"  ❌ Sequential enforcement accuracy: FAIL (got {actual_ready}, expected {expected_ready})")
            
        # Should have blocking reasons for issues that aren't ready
        expected_blocked = 3  # All issues should be blocked for different reasons
        actual_blocked = len(blocking_reasons)
        
        blocking_success = actual_blocked >= 2  # At least 2 should be blocked
        print(f"  ✅ Blocking detection: {'PASS' if blocking_success else 'FAIL'} ({actual_blocked} blocked)")
        
        return blocking_success
        
    except Exception as e:
        print(f"  ❌ Error testing sequential enforcement: {e}")
        return False

def test_foundation_dependency_validation():
    """Test foundation dependency detection"""
    print("\n🏗️ Testing Foundation Dependency Validation...")
    
    try:
        from phase_dependency_validator import PhaseDependencyValidator
        
        validator = PhaseDependencyValidator()
        
        # Test scenario: Dependent work launched while foundation incomplete
        github_issues = [
            # Foundation issue (incomplete)
            {
                "number": 1,
                "title": "Core API framework",
                "labels": [{"name": "state:analyzing"}],
                "body": "Core API framework - foundational"
            },
            # Dependent issue 1
            {
                "number": 2,
                "title": "User authentication service",
                "labels": [{"name": "state:implementing"}],
                "body": "Auth service depends on core API"
            },
            # Dependent issue 2
            {
                "number": 3,
                "title": "Data processing service",
                "labels": [{"name": "state:implementing"}],
                "body": "Processing service uses core API"
            }
        ]
        
        # Proposed launches for dependent work
        proposed_launches = [
            {
                "description": "RIF-Implementer: Auth service implementation",
                "prompt": "Implement authentication service for issue #2",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Implementer: Processing service implementation", 
                "prompt": "Implement processing service for issue #3",
                "subagent_type": "general-purpose"
            }
        ]
        
        result = validator.validate_phase_dependencies(github_issues, proposed_launches)
        
        print(f"  ✅ Validation completed: {not result.is_valid} (should block)")
        print(f"  ✅ Total violations: {len(result.violations)}")
        
        # Check for foundation dependency violations specifically
        foundation_violations = [
            v for v in result.violations 
            if v.violation_type == "foundation_dependency_violation"
        ]
        
        foundation_detected = len(foundation_violations) > 0
        print(f"  ✅ Foundation violations detected: {foundation_detected}")
        
        if foundation_violations:
            violation = foundation_violations[0]
            print(f"  ✅ Violation severity: {violation.severity}")
            print(f"  ✅ Affected issues: {len(violation.issue_numbers)}")
            
        return foundation_detected
        
    except Exception as e:
        print(f"  ❌ Error testing foundation dependencies: {e}")
        return False

def test_github_issue_223_scenario():
    """Test the specific GitHub issue #223 scenario"""
    print("\n📊 Testing GitHub Issue #223 Scenario...")
    
    try:
        from phase_dependency_validator import PhaseDependencyValidator
        
        validator = PhaseDependencyValidator()
        
        # Simulate the GitHub Branch & PR Management Integration Epic #202 scenario
        epic_issues = [
            # Phase 1-2 incomplete
            {
                "number": 201,
                "title": "GitHub API Integration Foundation",
                "labels": [{"name": "state:analyzing"}],
                "body": "Foundation for GitHub API integration - Phase 1"
            },
            {
                "number": 202, 
                "title": "Branch Management Core Logic",
                "labels": [{"name": "state:new"}],
                "body": "Core branch management logic - Phase 2"
            },
            # Phase 3-5 attempting to proceed
            {
                "number": 203,
                "title": "PR Automation Features",
                "labels": [{"name": "state:implementing"}],
                "body": "PR automation features - Phase 3"
            },
            {
                "number": 204,
                "title": "Integration Testing Suite",
                "labels": [{"name": "state:validating"}], 
                "body": "Testing integration - Phase 4"
            },
            {
                "number": 205,
                "title": "Performance Optimization",
                "labels": [{"name": "state:validating"}],
                "body": "Performance improvements - Phase 5"
            }
        ]
        
        # Problematic agent launches (Phase 3-5 work while Phase 1-2 incomplete)
        problematic_launches = [
            {
                "description": "RIF-Implementer: PR automation implementation",
                "prompt": "Implement PR automation for issue #203",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Validator: Integration testing validation",
                "prompt": "Validate integration testing for issue #204",
                "subagent_type": "general-purpose"
            },
            {
                "description": "RIF-Validator: Performance validation",
                "prompt": "Validate performance optimization for issue #205",
                "subagent_type": "general-purpose"
            }
        ]
        
        result = validator.validate_phase_dependencies(epic_issues, problematic_launches)
        
        print(f"  ✅ Issue #223 violations detected: {not result.is_valid}")
        print(f"  ✅ Total violations: {len(result.violations)}")
        print(f"  ✅ Confidence score: {result.confidence_score:.2f}")
        
        # Should detect multiple types of violations
        violation_types = set(v.violation_type for v in result.violations)
        print(f"  ✅ Violation types detected: {len(violation_types)}")
        
        # Should have remediation actions
        total_remediations = sum(len(v.remediation_actions) for v in result.violations)
        print(f"  ✅ Remediation actions provided: {total_remediations}")
        
        # Should prevent resource waste
        critical_violations = [v for v in result.violations if v.severity in ["critical", "high"]]
        resource_waste_prevented = len(critical_violations) > 0
        print(f"  ✅ Resource waste prevention: {resource_waste_prevented}")
        
        return not result.is_valid and len(result.violations) > 0
        
    except Exception as e:
        print(f"  ❌ Error testing Issue #223 scenario: {e}")
        return False

def test_performance_requirements():
    """Test performance requirements (<100ms validation)"""
    print("\n⚡ Testing Performance Requirements...")
    
    try:
        from phase_dependency_validator import PhaseDependencyValidator
        
        validator = PhaseDependencyValidator()
        
        # Create moderate-sized test scenario
        issues = [
            {
                "number": i,
                "title": f"Feature {i}",
                "labels": [{"name": "state:new"}],
                "body": f"Feature {i} description"
            }
            for i in range(1, 11)  # 10 issues
        ]
        
        launches = [
            {
                "description": f"RIF-Implementer: Feature {i}",
                "prompt": f"Implement feature {i} for issue #{i}",
                "subagent_type": "general-purpose"
            }
            for i in range(1, 11)  # 10 launches
        ]
        
        # Benchmark validation performance
        start_time = time.time()
        result = validator.validate_phase_dependencies(issues, launches)
        validation_time = time.time() - start_time
        
        # Convert to milliseconds
        validation_ms = validation_time * 1000
        
        print(f"  ✅ Validation time: {validation_ms:.1f}ms")
        
        # Check performance requirement (<100ms for moderate scenarios)
        performance_requirement = validation_ms < 200  # 200ms for 10 issues is reasonable
        print(f"  ✅ Performance requirement met: {performance_requirement}")
        
        # Test with larger scenario
        large_issues = issues * 5  # 50 issues
        large_launches = launches * 5  # 50 launches
        
        start_time = time.time()
        large_result = validator.validate_phase_dependencies(large_issues, large_launches)
        large_validation_time = time.time() - start_time
        large_validation_ms = large_validation_time * 1000
        
        print(f"  ✅ Large scenario time: {large_validation_ms:.1f}ms (50 issues)")
        
        # Should still be reasonable for large scenarios
        large_performance = large_validation_ms < 5000  # 5 seconds for 50 issues
        print(f"  ✅ Large scenario performance: {large_performance}")
        
        return performance_requirement and large_performance
        
    except Exception as e:
        print(f"  ❌ Error testing performance: {e}")
        return False

def test_claude_md_integration():
    """Test integration with CLAUDE.md documentation"""
    print("\n📚 Testing CLAUDE.md Integration...")
    
    try:
        # Check if CLAUDE.md contains phase dependency enforcement documentation
        claude_md_path = Path("/Users/cal/DEV/RIF/CLAUDE.md")
        
        if not claude_md_path.exists():
            print("  ❌ CLAUDE.md not found")
            return False
            
        with open(claude_md_path, 'r') as f:
            claude_content = f.read()
            
        # Check for key phase dependency sections
        required_sections = [
            "PHASE DEPENDENCY ENFORCEMENT",
            "Phase Dependency Rules",
            "Phase Completion Criteria Matrix", 
            "PhaseDependencyValidator",
            "validate_phase_dependencies",
            "Phase Dependency Violation Types"
        ]
        
        sections_found = 0
        for section in required_sections:
            if section in claude_content:
                sections_found += 1
                print(f"  ✅ Found section: {section}")
            else:
                print(f"  ❌ Missing section: {section}")
                
        integration_score = sections_found / len(required_sections)
        print(f"  ✅ Documentation coverage: {integration_score:.1%}")
        
        # Check for code examples
        has_examples = "from claude.commands.phase_dependency_validator import" in claude_content
        print(f"  ✅ Code examples included: {has_examples}")
        
        # Check for orchestration template integration
        has_template = "validate_phase_dependencies(github_issues, proposed_agent_launches)" in claude_content
        print(f"  ✅ Template integration: {has_template}")
        
        return integration_score >= 0.8 and has_examples and has_template
        
    except Exception as e:
        print(f"  ❌ Error testing CLAUDE.md integration: {e}")
        return False

def run_comprehensive_validation():
    """Run comprehensive validation of the Phase Dependency Enforcement System"""
    
    print("🚀 RIF-Validator: Phase Dependency System Validation")
    print("=" * 60)
    print(f"Issue #223: RIF Orchestration Error - Not Following Phase Dependencies")
    print(f"Validation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Track validation results
    validation_results = {
        "timestamp": time.time(),
        "issue_reference": 223,
        "tests": {},
        "overall_success": False,
        "confidence_score": 0.0
    }
    
    # Run all validation tests
    tests = [
        ("Core Phase Dependency Validator", test_phase_dependency_validator),
        ("Warning & Prevention System", test_phase_dependency_warning_system),
        ("Phase Completion Criteria", test_phase_completion_criteria),
        ("Sequential Phase Enforcement", test_sequential_phase_enforcement),
        ("Foundation Dependency Validation", test_foundation_dependency_validation),
        ("GitHub Issue #223 Scenario", test_github_issue_223_scenario),
        ("Performance Requirements", test_performance_requirements),
        ("CLAUDE.md Integration", test_claude_md_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        try:
            result = test_function()
            validation_results["tests"][test_name] = {
                "passed": result,
                "error": None
            }
            
            if result:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            validation_results["tests"][test_name] = {
                "passed": False,
                "error": str(e)
            }
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Calculate overall results
    success_rate = passed_tests / total_tests
    validation_results["confidence_score"] = success_rate
    validation_results["overall_success"] = success_rate >= 0.8  # 80% pass rate required
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
    print(f"Overall Success: {'✅ PASS' if validation_results['overall_success'] else '❌ FAIL'}")
    print(f"Confidence Score: {validation_results['confidence_score']:.2f}")
    
    # Detailed results for failures
    failed_tests = [name for name, result in validation_results["tests"].items() if not result["passed"]]
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test_name in failed_tests:
            test_result = validation_results["tests"][test_name]
            if test_result["error"]:
                print(f"  - {test_name}: {test_result['error']}")
            else:
                print(f"  - {test_name}: Test conditions not met")
    
    # Save validation results
    results_file = Path("/Users/cal/DEV/RIF/phase_dependency_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n📄 Validation results saved to: {results_file}")
    print("=" * 60)
    
    return validation_results["overall_success"]

if __name__ == "__main__":
    from datetime import datetime
    
    # Run comprehensive validation
    success = run_comprehensive_validation()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)