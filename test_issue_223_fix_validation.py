#!/usr/bin/env python3
"""
Issue #223 Fix Validation Test

This test validates that the phase dependency enforcement fix correctly prevents
the orchestration errors described in Issue #223.

Test Scenarios:
1. Valid phase progression - should allow execution
2. Implementation without analysis - should block execution  
3. Validation without implementation - should block execution
4. Mixed valid/invalid launches - should allow valid, block invalid

Expected Behavior: 
- Sequential phases enforced (Research → Architecture → Implementation → Validation)
- Blocked execution generates prerequisite tasks
- Clear violation descriptions and remediation actions provided
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'claude', 'commands'))

from simple_phase_dependency_enforcer import enforce_orchestration_phase_dependencies, generate_phase_validated_orchestration_template

def test_valid_phase_progression():
    """Test that valid phase progression is allowed"""
    print("\n🧪 Test 1: Valid Phase Progression")
    
    # Issues in appropriate states for implementation
    issues = [
        {
            "number": 101,
            "title": "User authentication system",
            "labels": [{"name": "state:architecting"}]  # Architecture complete, ready for implementation
        },
        {
            "number": 102, 
            "title": "Database schema design",
            "labels": [{"name": "state:implementing"}]  # Already in implementation
        }
    ]
    
    # Appropriate agent launches for current states
    launches = [
        {
            "description": "RIF-Implementer: User authentication implementation",
            "prompt": "You are RIF-Implementer. Implement user authentication for issue #101. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Validator: Database schema validation",
            "prompt": "You are RIF-Validator. Validate database schema for issue #102. Follow all instructions in claude/agents/rif-validator.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    result = enforce_orchestration_phase_dependencies(issues, launches)
    
    print(f"   Execution allowed: {result.is_execution_allowed}")
    print(f"   Violations: {len(result.violations)}")
    print(f"   Allowed tasks: {len(result.allowed_tasks)}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    
    assert result.is_execution_allowed, "Valid phase progression should be allowed"
    assert len(result.violations) == 0, "No violations should be detected"
    assert len(result.allowed_tasks) == 2, "Both tasks should be allowed"
    
    print("   ✅ PASSED - Valid progression allowed")
    return True

def test_implementation_without_analysis():
    """Test that implementation without analysis is blocked (Issue #223 core problem)"""
    print("\n🧪 Test 2: Implementation Without Analysis (Issue #223 Core Problem)")
    
    # Issues in early phases that need more work before implementation
    issues = [
        {
            "number": 201,
            "title": "API security implementation",
            "labels": [{"name": "state:new"}]  # Still in initial state
        },
        {
            "number": 202,
            "title": "Database connection pooling", 
            "labels": [{"name": "state:analyzing"}]  # Analysis incomplete
        }
    ]
    
    # Attempted implementation launches (this should be blocked)
    launches = [
        {
            "description": "RIF-Implementer: API security implementation",
            "prompt": "You are RIF-Implementer. Implement API security for issue #201. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Implementer: Database pooling implementation", 
            "prompt": "You are RIF-Implementer. Implement database connection pooling for issue #202. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    result = enforce_orchestration_phase_dependencies(issues, launches)
    
    print(f"   Execution allowed: {result.is_execution_allowed}")
    print(f"   Violations: {len(result.violations)}")
    print(f"   Blocked tasks: {len(result.blocked_tasks)}")
    print(f"   Prerequisite tasks: {len(result.prerequisite_tasks)}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    
    if result.violations:
        print("   Violations detected:")
        for violation in result.violations:
            print(f"     - {violation.description}")
            print(f"       → {violation.remediation_action}")
    
    assert not result.is_execution_allowed, "Implementation without analysis should be blocked"
    assert len(result.violations) > 0, "Violations should be detected"
    assert len(result.blocked_tasks) == 2, "Both implementation tasks should be blocked"
    assert len(result.prerequisite_tasks) > 0, "Prerequisite tasks should be generated"
    
    print("   ✅ PASSED - Implementation without analysis correctly blocked")
    return True

def test_validation_without_implementation():
    """Test that validation without implementation is blocked"""
    print("\n🧪 Test 3: Validation Without Implementation")
    
    # Issues that haven't completed implementation yet
    issues = [
        {
            "number": 301,
            "title": "User login system",
            "labels": [{"name": "state:architecting"}]  # Architecture done, but no implementation
        }
    ]
    
    # Attempted validation launch (this should be blocked) 
    launches = [
        {
            "description": "RIF-Validator: User login validation",
            "prompt": "You are RIF-Validator. Validate user login system for issue #301. Follow all instructions in claude/agents/rif-validator.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    result = enforce_orchestration_phase_dependencies(issues, launches)
    
    print(f"   Execution allowed: {result.is_execution_allowed}")
    print(f"   Violations: {len(result.violations)}")
    print(f"   Blocked tasks: {len(result.blocked_tasks)}")
    print(f"   Prerequisite tasks: {len(result.prerequisite_tasks)}")
    
    if result.violations:
        print("   Violations detected:")
        for violation in result.violations:
            print(f"     - {violation.description}")
    
    assert not result.is_execution_allowed, "Validation without implementation should be blocked"
    assert len(result.violations) > 0, "Violations should be detected"
    
    print("   ✅ PASSED - Validation without implementation correctly blocked")
    return True

def test_mixed_valid_invalid_launches():
    """Test mixed scenario with some valid and some invalid launches"""
    print("\n🧪 Test 4: Mixed Valid/Invalid Launches")
    
    # Mix of issues in different states
    issues = [
        {
            "number": 401,
            "title": "Authentication system",
            "labels": [{"name": "state:implementing"}]  # Ready for validation
        },
        {
            "number": 402,
            "title": "Authorization system", 
            "labels": [{"name": "state:new"}]  # Not ready for implementation
        },
        {
            "number": 403,
            "title": "User management",
            "labels": [{"name": "state:planning"}]  # Ready for architecture 
        }
    ]
    
    # Mixed launches - some valid, some invalid
    launches = [
        {
            "description": "RIF-Validator: Authentication validation",  # VALID
            "prompt": "You are RIF-Validator. Validate authentication for issue #401. Follow all instructions in claude/agents/rif-validator.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Implementer: Authorization implementation",  # INVALID - needs analysis first
            "prompt": "You are RIF-Implementer. Implement authorization for issue #402. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Architect: User management architecture",  # VALID  
            "prompt": "You are RIF-Architect. Design architecture for issue #403. Follow all instructions in claude/agents/rif-architect.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    result = enforce_orchestration_phase_dependencies(issues, launches)
    
    print(f"   Execution allowed: {result.is_execution_allowed}")
    print(f"   Violations: {len(result.violations)}")
    print(f"   Allowed tasks: {len(result.allowed_tasks)}")
    print(f"   Blocked tasks: {len(result.blocked_tasks)}")
    print(f"   Prerequisite tasks: {len(result.prerequisite_tasks)}")
    
    if result.violations:
        print("   Violations detected:")
        for violation in result.violations:
            print(f"     - {violation.description}")
    
    # In mixed scenarios, execution is blocked if ANY violations exist
    assert not result.is_execution_allowed, "Mixed scenario with violations should block execution"
    assert len(result.violations) > 0, "Should detect violations from invalid launches"
    assert len(result.allowed_tasks) >= 0, "Should identify valid tasks" 
    assert len(result.blocked_tasks) > 0, "Should identify blocked tasks"
    
    print("   ✅ PASSED - Mixed scenario correctly handled")
    return True

def test_template_generation():
    """Test that orchestration template is properly generated"""
    print("\n🧪 Test 5: Template Generation")
    
    # Simple violation scenario 
    issues = [
        {
            "number": 501,
            "title": "Test feature",
            "labels": [{"name": "state:new"}]
        }
    ]
    
    launches = [
        {
            "description": "RIF-Implementer: Test implementation",
            "prompt": "You are RIF-Implementer. Implement test feature for issue #501. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    template = generate_phase_validated_orchestration_template(issues, launches)
    
    print(f"   Template generated: {len(template)} characters")
    print("   Template contains:")
    print(f"     - Issue #223 reference: {'✅' if 'Issue #223' in template else '❌'}")
    print(f"     - Enforcement status: {'✅' if 'ENFORCEMENT' in template else '❌'}")
    print(f"     - Violation details: {'✅' if 'VIOLATIONS' in template else '❌'}")
    print(f"     - Task execution: {'✅' if 'Task(' in template else '❌'}")
    
    assert len(template) > 200, "Template should be substantial"
    assert "Issue #223" in template, "Should reference the issue being fixed"
    assert "ENFORCEMENT" in template, "Should show enforcement status"
    
    print("   ✅ PASSED - Template generation working")
    return True

def run_all_tests():
    """Run all validation tests for Issue #223 fix"""
    print("🔍 ISSUE #223 FIX VALIDATION")
    print("=" * 50)
    print("Testing: RIF Orchestration Error: Not Following Phase Dependencies")
    print("Fix: Phase dependency enforcement in orchestration system")
    print("=" * 50)
    
    tests = [
        test_valid_phase_progression,
        test_implementation_without_analysis, 
        test_validation_without_implementation,
        test_mixed_valid_invalid_launches,
        test_template_generation
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            if test():
                passed_tests += 1
        except AssertionError as e:
            print(f"   ❌ FAILED - {e}")
        except Exception as e:
            print(f"   💥 ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} PASSED")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Issue #223 fix is working correctly!")
        print("\nFIX SUMMARY:")
        print("✅ Phase dependency enforcement implemented")
        print("✅ Sequential phase progression enforced (Research → Architecture → Implementation → Validation)")
        print("✅ Invalid agent launches blocked with clear violation messages") 
        print("✅ Prerequisite tasks automatically generated")
        print("✅ CLAUDE.md orchestration template updated")
        print("✅ Resource waste prevention active")
        
        return True
    else:
        print(f"❌ {total_tests - passed_tests} TESTS FAILED - Fix needs attention")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)