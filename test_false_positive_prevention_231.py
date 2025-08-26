#!/usr/bin/env python3
"""
False Positive Prevention Validation Test - Issue #231

Comprehensive test to verify that the validation framework prevents false positive validations
like the one that occurred in issue #225.

Test Scenarios:
1. Validation MUST be blocked without proper integration tests
2. Integration tests MUST run successfully and provide evidence
3. Validation MUST only be approved after integration tests pass
4. Evidence collection MUST be comprehensive and authentic
"""

import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# Add RIF to Python path
sys.path.insert(0, '/Users/cal/DEV/RIF')

from claude.commands.integration_validation_enforcer import get_integration_enforcer
from claude.commands.validation_evidence_collector import get_evidence_collector


async def test_false_positive_prevention():
    """Test that the framework prevents false positive validations"""
    
    print("🛡️  TESTING FALSE POSITIVE PREVENTION - ISSUE #231")
    print("=" * 70)
    
    test_results = {
        "test_id": f"false_positive_prevention_231_{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "scenarios": {},
        "overall_success": False
    }
    
    # Initialize validation framework components
    print("🔧 Initializing validation framework components...")
    enforcer = get_integration_enforcer(enforcement_mode="strict")
    collector = get_evidence_collector()
    
    # SCENARIO 1: Verify validation is blocked without integration tests
    print("\n📋 SCENARIO 1: Testing validation blocking without integration tests")
    
    try:
        # Start validation session
        session_key = enforcer.start_validation_session(
            "231",
            "false_positive_prevention_test",
            "mcp_integration"
        )
        
        # Check approval - should be BLOCKED
        approval_result = enforcer.check_validation_approval(session_key)
        
        validation_blocked = not approval_result["approved"]
        blocking_reasons_present = len(approval_result["blocking_reasons"]) > 0
        integration_tests_mentioned = any(
            "integration test" in reason.lower() 
            for reason in approval_result["blocking_reasons"]
        )
        
        scenario_1_success = validation_blocked and blocking_reasons_present and integration_tests_mentioned
        
        test_results["scenarios"]["validation_blocking"] = {
            "success": scenario_1_success,
            "validation_blocked": validation_blocked,
            "blocking_reasons_count": len(approval_result["blocking_reasons"]),
            "blocking_reasons": approval_result["blocking_reasons"],
            "integration_tests_required": integration_tests_mentioned
        }
        
        if scenario_1_success:
            print("   ✅ Validation properly BLOCKED without integration tests")
            print(f"   📋 Blocking reasons: {len(approval_result['blocking_reasons'])}")
            for reason in approval_result["blocking_reasons"][:3]:  # Show first 3
                print(f"      - {reason}")
        else:
            print("   ❌ CRITICAL FAILURE: Validation not properly blocked")
            return test_results
            
    except Exception as e:
        print(f"   ❌ SCENARIO 1 FAILED: {e}")
        test_results["scenarios"]["validation_blocking"] = {"success": False, "error": str(e)}
        return test_results
    
    # SCENARIO 2: Test integration test execution and evidence collection
    print("\n🧪 SCENARIO 2: Testing integration test execution and evidence collection")
    
    try:
        # Start evidence collection
        evidence_session = collector.start_evidence_collection(
            f"scenario_2_{session_key}",
            "231",
            "integration_test_validator", 
            "mcp_integration"
        )
        
        # Run integration tests (simulated but comprehensive)
        integration_results = enforcer.run_integration_tests(session_key)
        
        # Collect additional evidence during testing
        with collector.collect_operation_evidence(evidence_session, "integration_testing"):
            await asyncio.sleep(0.5)  # Simulate test execution
            
            # Simulate successful integration test evidence
            collector.collect_execution_evidence(
                evidence_session,
                "python3 -m pytest tests/mcp/integration/test_mcp_claude_desktop_integration.py",
                "mcp_integration_test_execution"
            )
        
        # Finalize evidence collection
        evidence_package = collector.finalize_evidence_collection(evidence_session)
        
        tests_executed = integration_results["tests_run"]
        tests_passed = integration_results["tests_passed"] 
        overall_success = integration_results["overall_success"]
        evidence_collected = len(evidence_package.evidence_items)
        quality_score = evidence_package.quality_metrics["overall_quality"]
        
        scenario_2_success = (
            len(tests_executed) > 0 and 
            tests_passed > 0 and 
            overall_success and
            evidence_collected >= 3 and
            quality_score > 50
        )
        
        test_results["scenarios"]["integration_testing"] = {
            "success": scenario_2_success,
            "tests_executed": len(tests_executed),
            "tests_passed": tests_passed,
            "overall_success": overall_success,
            "evidence_items_collected": evidence_collected,
            "quality_score": quality_score,
            "evidence_categories": list(evidence_package.summary["evidence_categories"].keys())
        }
        
        if scenario_2_success:
            print("   ✅ Integration tests executed successfully")
            print(f"   📊 Tests run: {len(tests_executed)}, Passed: {tests_passed}")
            print(f"   📋 Evidence items: {evidence_collected}, Quality: {quality_score:.1f}%")
        else:
            print("   ❌ Integration testing or evidence collection failed")
            
    except Exception as e:
        print(f"   ❌ SCENARIO 2 FAILED: {e}")
        test_results["scenarios"]["integration_testing"] = {"success": False, "error": str(e)}
    
    # SCENARIO 3: Verify validation approval after successful integration tests
    print("\n🎯 SCENARIO 3: Testing validation approval after integration tests")
    
    try:
        # Check approval after integration tests
        final_approval = enforcer.check_validation_approval(session_key)
        
        validation_approved = final_approval["approved"]
        completion_percentage = final_approval["completion_status"]["completion_percentage"]
        missing_tests = final_approval["completion_status"]["missing_tests"]
        
        scenario_3_success = validation_approved and completion_percentage == 100 and len(missing_tests) == 0
        
        test_results["scenarios"]["validation_approval"] = {
            "success": scenario_3_success,
            "validation_approved": validation_approved,
            "completion_percentage": completion_percentage,
            "missing_tests": missing_tests,
            "enforcement_action": final_approval.get("enforcement_action", "unknown")
        }
        
        if scenario_3_success:
            print("   ✅ Validation APPROVED after successful integration tests")
            print(f"   📈 Completion: {completion_percentage}%")
            print(f"   🎯 Missing tests: {len(missing_tests)}")
        else:
            print("   ❌ Validation approval logic failed")
            
    except Exception as e:
        print(f"   ❌ SCENARIO 3 FAILED: {e}")
        test_results["scenarios"]["validation_approval"] = {"success": False, "error": str(e)}
    
    # SCENARIO 4: Generate comprehensive validation report
    print("\n📊 SCENARIO 4: Testing comprehensive validation reporting")
    
    try:
        # Generate validation report
        validation_report = enforcer.generate_validation_report(session_key)
        
        report_complete = "validation_report" in validation_report
        false_positive_prevention = validation_report.get("false_positive_prevention", {})
        integration_testing_enforced = false_positive_prevention.get("integration_testing_enforced", False)
        comprehensive_evidence_required = false_positive_prevention.get("comprehensive_evidence_required", False)
        
        scenario_4_success = (
            report_complete and 
            integration_testing_enforced and 
            comprehensive_evidence_required
        )
        
        test_results["scenarios"]["validation_reporting"] = {
            "success": scenario_4_success,
            "report_generated": report_complete,
            "false_positive_prevention_active": integration_testing_enforced and comprehensive_evidence_required,
            "validation_decision": validation_report.get("validation_decision", {}),
            "session_duration": validation_report.get("validation_report", {}).get("session_duration", "unknown")
        }
        
        if scenario_4_success:
            print("   ✅ Comprehensive validation report generated")
            print("   🛡️  False positive prevention mechanisms confirmed active")
        else:
            print("   ❌ Validation reporting incomplete")
            
    except Exception as e:
        print(f"   ❌ SCENARIO 4 FAILED: {e}")
        test_results["scenarios"]["validation_reporting"] = {"success": False, "error": str(e)}
    
    # Calculate overall success
    scenarios_passed = sum(1 for scenario in test_results["scenarios"].values() if scenario.get("success", False))
    total_scenarios = len(test_results["scenarios"])
    overall_success = scenarios_passed == total_scenarios
    
    test_results["overall_success"] = overall_success
    test_results["scenarios_passed"] = scenarios_passed
    test_results["total_scenarios"] = total_scenarios
    test_results["success_rate"] = (scenarios_passed / total_scenarios) * 100 if total_scenarios > 0 else 0
    
    # Save test results
    results_file = Path("/Users/cal/DEV/RIF/knowledge/false_positive_prevention_test_231.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Generate final report
    print("\n" + "=" * 70)
    print("🛡️  FALSE POSITIVE PREVENTION TEST RESULTS")
    print("=" * 70)
    
    if overall_success:
        print("✅ OVERALL STATUS: SUCCESS")
        print("✅ FALSE POSITIVE PREVENTION: OPERATIONAL") 
        print("✅ VALIDATION FRAMEWORK: FULLY DEPLOYED")
        print("✅ ISSUE #231: VALIDATION FIXED")
    else:
        print("⚠️  OVERALL STATUS: PARTIAL SUCCESS")
        print(f"📊 Scenarios passed: {scenarios_passed}/{total_scenarios} ({test_results['success_rate']:.1f}%)")
    
    print(f"\n📋 KEY FINDINGS:")
    for scenario_name, scenario_result in test_results["scenarios"].items():
        status = "✅ PASS" if scenario_result.get("success", False) else "❌ FAIL"
        print(f"   {status} {scenario_name.replace('_', ' ').title()}")
    
    print(f"\n📄 Full results saved to: {results_file}")
    print(f"🕐 Test completed: {datetime.now().isoformat()}")
    
    return test_results


if __name__ == "__main__":
    results = asyncio.run(test_false_positive_prevention())
    sys.exit(0 if results.get("overall_success", False) else 1)