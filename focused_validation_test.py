#!/usr/bin/env python3
"""
Focused validation test for Phase Dependency Enforcement System
"""

import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent / "claude" / "commands"))

def test_core_functionality():
    """Test core phase dependency functionality"""
    print("🔍 Testing Core Phase Dependency Functionality...")
    
    from phase_dependency_validator import PhaseDependencyValidator, PhaseType
    
    validator = PhaseDependencyValidator()
    
    # Test Issue #223 scenario: Implementation without prerequisites
    github_issues = [
        {
            "number": 203,
            "title": "PR Automation Features",
            "labels": [{"name": "state:new"}],  # No prerequisites completed
            "body": "PR automation features - Phase 3"
        },
        {
            "number": 201,
            "title": "GitHub API Integration Foundation", 
            "labels": [{"name": "state:analyzing"}],  # Foundation incomplete
            "body": "Foundation for GitHub API integration - Phase 1"
        }
    ]
    
    # Problematic agent launch: Implementation without prerequisites
    proposed_launches = [
        {
            "description": "RIF-Implementer: PR automation implementation",
            "prompt": "Implement PR automation for issue #203",
            "subagent_type": "general-purpose"
        }
    ]
    
    # Validate phase dependencies
    result = validator.validate_phase_dependencies(github_issues, proposed_launches)
    
    print(f"  ✅ Validation blocked execution: {not result.is_valid}")
    print(f"  ✅ Violations detected: {len(result.violations)}")
    print(f"  ✅ Warnings generated: {len(result.warnings)}")
    
    if result.violations:
        violation = result.violations[0]
        print(f"  ✅ Violation type: {violation.violation_type}")
        print(f"  ✅ Severity: {violation.severity}")
        print(f"  ✅ Missing phases: {[p.value for p in violation.missing_prerequisite_phases]}")
        print(f"  ✅ Remediation actions: {len(violation.remediation_actions)}")
        
        if violation.remediation_actions:
            print(f"  ✅ First remediation: {violation.remediation_actions[0]}")
    
    print(f"  ✅ Confidence score: {result.confidence_score:.2f}")
    
    return not result.is_valid and len(result.violations) > 0

def test_warning_system_basic():
    """Test basic warning system functionality"""  
    print("\n🚨 Testing Warning System (Basic)...")
    
    try:
        from phase_dependency_warning_system import PhaseDependencyWarningSystem
        
        warning_system = PhaseDependencyWarningSystem()
        
        # Simple test case
        issues = [
            {
                "number": 1,
                "title": "New feature",
                "labels": [{"name": "state:new"}],
                "body": "New feature implementation"
            }
        ]
        
        launches = [
            {
                "description": "RIF-Implementer: Feature implementation",
                "prompt": "Implement feature for issue #1", 
                "subagent_type": "general-purpose"
            }
        ]
        
        # Test basic alert generation
        alerts = warning_system.detect_violations_real_time(issues, launches)
        print(f"  ✅ Alerts generated: {len(alerts)}")
        
        if alerts:
            alert = alerts[0]
            print(f"  ✅ Alert level: {alert.alert_level.value}")
            print(f"  ✅ Affected issues: {alert.affected_issues}")
            print(f"  ✅ Actionable steps: {len(alert.actionable_steps)}")
        
        return len(alerts) > 0
        
    except Exception as e:
        print(f"  ❌ Warning system error: {e}")
        return False

def test_integration_with_claude_md():
    """Test that CLAUDE.md has been properly updated"""
    print("\n📚 Testing CLAUDE.md Integration...")
    
    claude_md_path = Path("/Users/cal/DEV/RIF/CLAUDE.md")
    
    if not claude_md_path.exists():
        print("  ❌ CLAUDE.md not found")
        return False
    
    with open(claude_md_path, 'r') as f:
        content = f.read()
    
    # Check for phase dependency content
    checks = [
        ("Phase Dependency Rules", "Phase Dependency Rules" in content),
        ("Validator Import", "from claude.commands.phase_dependency_validator import" in content),
        ("Validation Function", "validate_phase_dependencies" in content),
        ("Criteria Matrix", "Phase Completion Criteria Matrix" in content),
        ("Violation Types", "Phase Dependency Violation Types" in content),
        ("Enforcement Examples", "❌ PHASE DEPENDENCY VIOLATIONS" in content)
    ]
    
    passed = 0
    for check_name, result in checks:
        if result:
            print(f"  ✅ {check_name}: Found")
            passed += 1
        else:
            print(f"  ❌ {check_name}: Missing")
    
    success_rate = passed / len(checks)
    print(f"  ✅ Integration completeness: {success_rate:.1%}")
    
    return success_rate >= 0.8

def main():
    """Run focused validation tests"""
    print("🚀 Focused Phase Dependency Validation")
    print("=" * 50)
    
    tests = [
        ("Core Functionality", test_core_functionality),
        ("Warning System Basic", test_warning_system_basic), 
        ("CLAUDE.md Integration", test_integration_with_claude_md)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED\n")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}\n")
    
    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All focused tests PASSED!")
    else:
        print(f"⚠️  {len(tests) - passed} test(s) failed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)