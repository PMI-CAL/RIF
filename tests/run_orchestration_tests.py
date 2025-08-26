"""
Test Runner for Orchestration Pattern Validation

Runs comprehensive test suite for Issue #224: RIF Orchestration Error: Incorrect Parallel Task Launching
Includes unit tests, integration tests, performance benchmarks, and regression tests.
"""

import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path


def run_test_suite():
    """Run the complete orchestration validation test suite"""
    print("🧪 Running Orchestration Pattern Validation Test Suite")
    print("=" * 60)
    
    test_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "issue_id": 224,
        "test_suite": "orchestration_pattern_validation",
        "results": {}
    }
    
    # Test files to run
    test_files = [
        ("Unit & Integration Tests", "test_orchestration_pattern_validation.py"),
        ("Performance Benchmarks", "test_orchestration_performance_benchmarks.py")
    ]
    
    all_passed = True
    
    for test_name, test_file in test_files:
        print(f"\n📋 Running {test_name}")
        print("-" * 40)
        
        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--capture=no"
            ], 
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
            )
            
            success = result.returncode == 0
            test_results["results"][test_name] = {
                "passed": success,
                "return_code": result.returncode
            }
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Error running {test_name}: {e}")
            test_results["results"][test_name] = {
                "passed": False,
                "error": str(e)
            }
            all_passed = False
            
    # Run specific regression test for Issue #224
    print(f"\n🔍 Running Issue #224 Regression Tests")
    print("-" * 40)
    
    try:
        # Import and run specific regression tests
        sys.path.append(str(Path(__file__).parent))
        from test_orchestration_pattern_validation import TestRegression
        
        regression_test = TestRegression()
        
        print("Testing Issue #224 Multi-Issue Accelerator anti-pattern...")
        regression_test.test_issue_224_multi_issue_accelerator_regression()
        print("✅ Issue #224 anti-pattern correctly detected")
        
        print("Testing correct fix for Issue #224...")
        regression_test.test_correct_fix_for_issue_224()
        print("✅ Issue #224 correct fix passes validation")
        
        test_results["results"]["Issue #224 Regression"] = {"passed": True}
        
    except Exception as e:
        print(f"❌ Issue #224 regression tests failed: {e}")
        test_results["results"]["Issue #224 Regression"] = {
            "passed": False,
            "error": str(e)
        }
        all_passed = False
        
    # Generate summary
    print(f"\n📊 Test Suite Summary")
    print("=" * 60)
    
    total_tests = len(test_results["results"])
    passed_tests = sum(1 for result in test_results["results"].values() if result["passed"])
    
    print(f"Total test suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    test_results["summary"] = {
        "total_suites": total_tests,
        "passed_suites": passed_tests,
        "failed_suites": total_tests - passed_tests,
        "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "overall_success": all_passed
    }
    
    # Save test results
    results_file = Path(__file__).parent / f"orchestration_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\n💾 Test results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save test results: {e}")
        
    if all_passed:
        print(f"\n🎉 All tests PASSED! Issue #224 orchestration validation is working correctly.")
        return 0
    else:
        print(f"\n❌ Some tests FAILED! Review errors above.")
        return 1


def validate_anti_pattern_examples():
    """Validate the specific anti-patterns mentioned in Issue #224"""
    print("\n🚫 Validating Known Anti-Patterns")
    print("-" * 40)
    
    # Import validation functions
    sys.path.append('/Users/cal/DEV/RIF')
    from claude.commands.orchestration_pattern_validator import validate_task_request
    
    # Test the exact Issue #224 anti-pattern
    issue_224_anti_pattern = [
        {
            "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3",
            "prompt": "You are a Multi-Issue Accelerator agent. Handle these issues in parallel: Issue #1: user auth, Issue #2: database pool, Issue #3: API validation",
            "subagent_type": "general-purpose"
        }
    ]
    
    print("Testing Issue #224 exact anti-pattern...")
    result = validate_task_request(issue_224_anti_pattern)
    
    if not result.is_valid:
        print("✅ Issue #224 anti-pattern correctly BLOCKED")
        print(f"   Violations detected: {len(result.violations)}")
        for violation in result.violations:
            print(f"   - {violation}")
    else:
        print("❌ Issue #224 anti-pattern incorrectly ALLOWED")
        return False
        
    # Test the correct approach
    correct_approach = [
        {
            "description": "RIF-Implementer: User authentication system",
            "prompt": "You are RIF-Implementer. Implement user authentication for issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Implementer: Database connection pooling",
            "prompt": "You are RIF-Implementer. Implement database connection pooling for issue #2. Follow all instructions in claude/agents/rif-implementer.md.",
            "subagent_type": "general-purpose"
        },
        {
            "description": "RIF-Validator: API validation framework",
            "prompt": "You are RIF-Validator. Validate API framework for issue #3. Follow all instructions in claude/agents/rif-validator.md.",
            "subagent_type": "general-purpose"
        }
    ]
    
    print("\nTesting correct parallel approach...")
    result = validate_task_request(correct_approach)
    
    if result.is_valid:
        print("✅ Correct parallel approach correctly ALLOWED")
        print(f"   Pattern type: {result.pattern_type}")
        print(f"   Confidence score: {result.confidence_score}")
    else:
        print("❌ Correct parallel approach incorrectly BLOCKED")
        print(f"   Violations: {result.violations}")
        return False
        
    return True


if __name__ == "__main__":
    # Run anti-pattern validation first
    anti_pattern_valid = validate_anti_pattern_examples()
    
    if not anti_pattern_valid:
        print("❌ Anti-pattern validation failed!")
        sys.exit(1)
        
    # Run full test suite
    exit_code = run_test_suite()
    sys.exit(exit_code)