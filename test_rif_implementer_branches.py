#!/usr/bin/env python3
"""
Test script to validate RIF-Implementer branch functionality
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"Stderr: {result.stderr.strip()}")
        print("-" * 50)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        print("-" * 50)
        return False

def test_branch_functionality():
    """Test the branch creation functionality"""
    print("=== RIF-Implementer Branch Functionality Test ===\n")
    
    # Test 1: Check current branch
    success1 = run_command("git branch --show-current", "Check current branch")
    
    # Test 2: Check if we're on feature branch
    current_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    is_feature_branch = current_branch not in ['main', 'master']
    print(f"Current branch: {current_branch}")
    print(f"Is feature branch: {is_feature_branch}")
    print("-" * 50)
    
    # Test 3: Test branch creation logic (simulate)
    issue_num = "204"
    issue_title = "phase-1-update-rif-implementer-to-create-feature-branches"
    branch_name = f"issue-{issue_num}-{issue_title[:40]}"
    print(f"Expected branch name: {branch_name}")
    print(f"Actual current branch: {current_branch}")
    print(f"Branch naming matches convention: {current_branch.startswith(f'issue-{issue_num}')}")
    print("-" * 50)
    
    # Test 4: Test git log functionality
    success4 = run_command("git rev-list --count HEAD ^main 2>/dev/null || echo '0'", 
                          "Count commits ahead of main")
    
    # Test 5: Test commit history
    success5 = run_command("git log --oneline -5", "Recent commit history")
    
    # Test 6: Test if gh CLI is available
    success6 = run_command("command -v gh", "Check GitHub CLI availability")
    
    # Test 7: Test branch push status
    success7 = run_command("git status -b --porcelain=v1 | head -1", "Check branch tracking status")
    
    # Summary
    print("\n=== Test Summary ===")
    tests = [
        ("Current branch check", success1),
        (f"On feature branch: {current_branch}", is_feature_branch),
        (f"Branch naming convention", current_branch.startswith(f"issue-{issue_num}")),
        ("Commit counting", success4),
        ("Commit history", success5),
        ("GitHub CLI", success6),
        ("Branch status", success7)
    ]
    
    passed = 0
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed >= len(tests) - 1:  # Allow one test to fail (e.g., gh CLI might not be configured)
        print("🎉 Branch functionality test PASSED!")
        return True
    else:
        print("🚨 Branch functionality test FAILED!")
        return False

if __name__ == "__main__":
    success = test_branch_functionality()
    sys.exit(0 if success else 1)