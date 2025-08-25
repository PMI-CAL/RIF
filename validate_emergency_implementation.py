#!/usr/bin/env python3
"""
Validate Emergency Implementation for Issue #232
Test that critical fixes are working correctly
"""

import subprocess
import json
import os
import sys
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=True)
        return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "stdout": e.stdout, "stderr": e.stderr, "error": str(e)}

def test_agent_instructions():
    """Test that agent instructions prevent autonomous closure"""
    print("🔍 Testing Agent Instructions...")
    
    tests = []
    
    # Check RIF-Implementer has user validation requirements
    with open("claude/agents/rif-implementer.md", "r") as f:
        content = f.read()
        if "AGENTS CANNOT CLOSE ISSUES" in content:
            tests.append({"test": "RIF-Implementer closure prevention", "status": "✅ PASS"})
        else:
            tests.append({"test": "RIF-Implementer closure prevention", "status": "❌ FAIL"})
        
        if "awaiting-user-validation" in content:
            tests.append({"test": "RIF-Implementer user validation state", "status": "✅ PASS"})
        else:
            tests.append({"test": "RIF-Implementer user validation state", "status": "❌ FAIL"})
    
    # Check RIF-Validator has user validation requirements
    with open("claude/agents/rif-validator.md", "r") as f:
        content = f.read()
        if "AGENTS CANNOT CLOSE ISSUES" in content:
            tests.append({"test": "RIF-Validator closure prevention", "status": "✅ PASS"})
        else:
            tests.append({"test": "RIF-Validator closure prevention", "status": "❌ FAIL"})
    
    # Check RIF-Learner has user validation requirements  
    with open("claude/agents/rif-learner.md", "r") as f:
        content = f.read()
        if "awaiting-user-validation" in content:
            tests.append({"test": "RIF-Learner user validation state", "status": "✅ PASS"})
        else:
            tests.append({"test": "RIF-Learner user validation state", "status": "❌ FAIL"})
    
    return tests

def test_workflow_configuration():
    """Test that workflow configuration includes user validation state"""
    print("🔍 Testing Workflow Configuration...")
    
    tests = []
    
    with open("config/rif-workflow.yaml", "r") as f:
        content = f.read()
        if "awaiting-user-validation:" in content:
            tests.append({"test": "Workflow has awaiting-user-validation state", "status": "✅ PASS"})
        else:
            tests.append({"test": "Workflow has awaiting-user-validation state", "status": "❌ FAIL"})
        
        if "user_required: true" in content:
            tests.append({"test": "User validation state requires user", "status": "✅ PASS"})
        else:
            tests.append({"test": "User validation state requires user", "status": "❌ FAIL"})
        
        if "to: \"awaiting-user-validation\"" in content:
            tests.append({"test": "Workflow transitions to user validation", "status": "✅ PASS"})
        else:
            tests.append({"test": "Workflow transitions to user validation", "status": "❌ FAIL"})
    
    return tests

def test_branch_management():
    """Test that branch management system is functional"""
    print("🔍 Testing Branch Management...")
    
    tests = []
    
    # Check branch management files exist
    if os.path.exists("claude/commands/branch_manager.py"):
        tests.append({"test": "Branch Manager exists", "status": "✅ PASS"})
    else:
        tests.append({"test": "Branch Manager exists", "status": "❌ FAIL"})
    
    if os.path.exists("claude/commands/branch_workflow_enforcer.py"):
        tests.append({"test": "Branch Workflow Enforcer exists", "status": "✅ PASS"})
    else:
        tests.append({"test": "Branch Workflow Enforcer exists", "status": "❌ FAIL"})
    
    # Test branch validation
    try:
        from claude.commands.branch_workflow_enforcer import validate_current_branch_compliance
        result = validate_current_branch_compliance(232)
        if result.get("compliant"):
            tests.append({"test": "Branch compliance validation works", "status": "✅ PASS"})
        else:
            tests.append({"test": "Branch compliance validation works", "status": "⚠️  WARN - Not on compliant branch"})
    except Exception as e:
        tests.append({"test": "Branch compliance validation works", "status": f"❌ FAIL - {str(e)}"})
    
    return tests

def test_user_validation_system():
    """Test that user validation system works"""
    print("🔍 Testing User Validation System...")
    
    tests = []
    
    # Check if validation request was posted to issue #232
    result = run_command("gh issue view 232 --comments --json comments")
    if result["success"]:
        issue_data = json.loads(result["stdout"])
        comments = issue_data.get("comments", [])
        
        validation_comment_found = False
        for comment in comments:
            if "USER VALIDATION REQUIRED" in comment.get("body", ""):
                validation_comment_found = True
                break
        
        if validation_comment_found:
            tests.append({"test": "User validation request posted", "status": "✅ PASS"})
        else:
            tests.append({"test": "User validation request posted", "status": "❌ FAIL"})
    else:
        tests.append({"test": "User validation request posted", "status": "❌ FAIL - Cannot check comments"})
    
    # Check if issue #225 has validation request
    result = run_command("gh issue view 225 --comments --json comments")
    if result["success"]:
        issue_data = json.loads(result["stdout"])
        comments = issue_data.get("comments", [])
        
        validation_comment_found = False
        for comment in comments:
            if "USER VALIDATION REQUIRED" in comment.get("body", ""):
                validation_comment_found = True
                break
        
        if validation_comment_found:
            tests.append({"test": "Issue #225 validation request posted", "status": "✅ PASS"})
        else:
            tests.append({"test": "Issue #225 validation request posted", "status": "❌ FAIL"})
    else:
        tests.append({"test": "Issue #225 validation request posted", "status": "❌ FAIL - Cannot check comments"})
    
    return tests

def test_issue_state_management():
    """Test that issue states are managed correctly"""
    print("🔍 Testing Issue State Management...")
    
    tests = []
    
    # Check if awaiting-user-validation label exists
    result = run_command("gh label list --json name")
    if result["success"]:
        labels = json.loads(result["stdout"])
        label_names = [label["name"] for label in labels]
        
        if "state:awaiting-user-validation" in label_names:
            tests.append({"test": "User validation label exists", "status": "✅ PASS"})
        else:
            tests.append({"test": "User validation label exists", "status": "❌ FAIL"})
    else:
        tests.append({"test": "User validation label exists", "status": "❌ FAIL - Cannot check labels"})
    
    # Check if issue #232 has appropriate labels
    result = run_command("gh issue view 232 --json labels")
    if result["success"]:
        issue_data = json.loads(result["stdout"])
        label_names = [label["name"] for label in issue_data.get("labels", [])]
        
        if "state:awaiting-user-validation" in label_names:
            tests.append({"test": "Issue #232 has user validation label", "status": "✅ PASS"})
        else:
            tests.append({"test": "Issue #232 has user validation label", "status": "⚠️  WARN - Label not applied"})
    else:
        tests.append({"test": "Issue #232 has user validation label", "status": "❌ FAIL - Cannot check issue labels"})
    
    return tests

def main():
    """Run all tests"""
    print("🚨 EMERGENCY IMPLEMENTATION VALIDATION - Issue #232")
    print("=" * 60)
    
    all_tests = []
    
    # Run all test suites
    all_tests.extend(test_agent_instructions())
    all_tests.extend(test_workflow_configuration())
    all_tests.extend(test_branch_management()) 
    all_tests.extend(test_user_validation_system())
    all_tests.extend(test_issue_state_management())
    
    # Summary
    print("\n📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    pass_count = 0
    fail_count = 0
    warn_count = 0
    
    for test in all_tests:
        print(f"{test['status']} {test['test']}")
        if "✅ PASS" in test['status']:
            pass_count += 1
        elif "❌ FAIL" in test['status']:
            fail_count += 1
        elif "⚠️" in test['status']:
            warn_count += 1
    
    print(f"\n📊 RESULTS: {pass_count} PASS, {fail_count} FAIL, {warn_count} WARN")
    
    if fail_count == 0:
        print("🎉 ALL CRITICAL TESTS PASSED - Emergency implementation successful!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Emergency implementation needs fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())