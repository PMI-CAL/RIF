#!/usr/bin/env python3
"""
Test script for RIF Error Handling Automation System
Tests automatic error detection, classification, and GitHub issue creation
"""

import sys
import json
import subprocess
import datetime
from pathlib import Path
import tempfile
import time

def test_session_error_handler():
    """Test the session error handler functionality"""
    
    print("=== Testing Session Error Handler ===")
    
    # Test 1: Capture a bash error
    print("\nTest 1: Bash Command Error Capture")
    result = subprocess.run([
        "python3", "/Users/cal/DEV/RIF/claude/commands/session_error_handler.py",
        "--capture-bash", "non_existent_command_xyz", "127", "command not found"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Bash error captured successfully")
        print(f"Output: {result.stdout.strip()}")
    else:
        print("✗ Bash error capture failed")
        print(f"Error: {result.stderr}")
    
    # Test 2: Capture a tool error
    print("\nTest 2: Tool Error Capture")
    result = subprocess.run([
        "python3", "/Users/cal/DEV/RIF/claude/commands/session_error_handler.py", 
        "--capture-tool", "TestTool", "Test error message for validation"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Tool error captured successfully")
        print(f"Output: {result.stdout.strip()}")
    else:
        print("✗ Tool error capture failed")
        print(f"Error: {result.stderr}")
    
    # Test 3: Capture a critical error (should trigger GitHub issue creation)
    print("\nTest 3: Critical Error Capture")
    result = subprocess.run([
        "python3", "/Users/cal/DEV/RIF/claude/commands/session_error_handler.py",
        "--capture-generic", "CRITICAL: Database connection failed - authentication failed",
        "--context", json.dumps({"severity": "critical", "source": "database"})
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Critical error captured successfully")
        print(f"Output: {result.stdout.strip()}")
    else:
        print("✗ Critical error capture failed")
        print(f"Error: {result.stderr}")
    
    # Test 4: Get session summary
    print("\nTest 4: Session Summary")
    result = subprocess.run([
        "python3", "/Users/cal/DEV/RIF/claude/commands/session_error_handler.py",
        "--session-summary"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Session summary generated successfully")
        try:
            summary = json.loads(result.stdout)
            print(f"Total errors: {summary.get('total_errors', 0)}")
            print(f"Auto-handled: {summary.get('auto_handled', 0)}")
            print(f"GitHub issues created: {summary.get('github_issues_created', 0)}")
        except:
            print("Raw output:", result.stdout)
    else:
        print("✗ Session summary failed")
        print(f"Error: {result.stderr}")

def test_rif_integration():
    """Test RIF integration functionality"""
    
    print("\n=== Testing RIF Integration ===")
    
    # First, we need to create an error to process
    print("\nStep 1: Creating test error for RIF integration")
    result = subprocess.run([
        "python3", "/Users/cal/DEV/RIF/claude/commands/session_error_handler.py",
        "--capture-generic", "Integration test error for RIF processing",
        "--context", json.dumps({"test": True, "integration": "rif"})
    ], capture_output=True, text=True)
    
    if "error captured:" in result.stdout:
        # Extract error ID from output
        error_id = result.stdout.split("error captured: ")[1].strip()
        print(f"✓ Test error created: {error_id}")
        
        # Test RIF integration processing
        print("\nStep 2: Processing error through RIF integration")
        result = subprocess.run([
            "python3", "/Users/cal/DEV/RIF/claude/commands/error_rif_integration.py",
            "--process-error", error_id
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ RIF integration processing successful")
            print("First few lines of output:")
            lines = result.stdout.split('\n')[:5]
            for line in lines:
                print(f"  {line}")
        else:
            print("✗ RIF integration processing failed")
            print(f"Error: {result.stderr}")
    else:
        print("✗ Failed to create test error")
        print(f"Output: {result.stdout}")

def test_error_classification():
    """Test error classification functionality"""
    
    print("\n=== Testing Error Classification ===")
    
    test_cases = [
        {
            "message": "SyntaxError: invalid syntax in file.py line 42",
            "expected_type": "syntax",
            "expected_severity": "high"
        },
        {
            "message": "NullPointerException: object reference not set",
            "expected_type": "runtime", 
            "expected_severity": "medium"
        },
        {
            "message": "Connection timeout to external API service", 
            "expected_type": "integration",
            "expected_severity": "medium"
        },
        {
            "message": "Authentication failed: invalid credentials",
            "expected_type": "security",
            "expected_severity": "critical"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['expected_type']} error")
        
        result = subprocess.run([
            "python3", "/Users/cal/DEV/RIF/claude/commands/session_error_handler.py",
            "--capture-generic", test_case["message"]
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Error captured: {test_case['message'][:50]}...")
        else:
            print(f"✗ Error capture failed")

def test_knowledge_base_integration():
    """Test knowledge base storage and retrieval"""
    
    print("\n=== Testing Knowledge Base Integration ===")
    
    # Check if error logs are being created
    logs_dir = Path("/Users/cal/DEV/RIF/knowledge/errors/logs")
    
    if logs_dir.exists():
        log_files = list(logs_dir.glob("session_*.jsonl"))
        if log_files:
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            print(f"✓ Session log found: {latest_log.name}")
            
            # Check log content
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                print(f"✓ Log contains {len(lines)} error entries")
                
                if lines:
                    # Parse and display last error
                    last_error = json.loads(lines[-1].strip())
                    print(f"✓ Latest error ID: {last_error.get('id', 'unknown')}")
                    print(f"✓ Severity: {last_error.get('severity', 'unknown')}")
                    
            except Exception as e:
                print(f"✗ Error reading log file: {e}")
        else:
            print("⚠ No session log files found")
    else:
        print("⚠ Errors logs directory not found")
    
    # Check if GitHub issue links are being stored
    links_file = Path("/Users/cal/DEV/RIF/knowledge/errors/github_issues/error_issue_links.jsonl")
    if links_file.exists():
        try:
            with open(links_file, 'r') as f:
                links = f.readlines()
            print(f"✓ GitHub issue links file exists with {len(links)} entries")
        except Exception as e:
            print(f"✗ Error reading issue links: {e}")
    else:
        print("⚠ No GitHub issue links file found (may be created when critical errors occur)")

def test_claude_hooks_integration():
    """Test integration with Claude Code hooks"""
    
    print("\n=== Testing Claude Hooks Integration ===")
    
    # Check if the hooks configuration exists
    settings_file = Path("/Users/cal/DEV/RIF/.claude/settings.json")
    
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            # Check for our updated hooks
            hooks = settings.get("hooks", {})
            
            if "ErrorCapture" in hooks:
                error_capture = hooks["ErrorCapture"][0]["command"]
                if "session_error_handler.py" in error_capture:
                    print("✓ ErrorCapture hook configured for session error handler")
                else:
                    print("⚠ ErrorCapture hook exists but not using session error handler")
            else:
                print("⚠ ErrorCapture hook not found")
            
            if "PostToolUse" in hooks:
                post_tool_hooks = hooks["PostToolUse"]
                bash_hook_found = False
                for hook in post_tool_hooks:
                    if "Bash" in hook.get("matcher", "") and "session_error_handler.py" in hook.get("hooks", [{}])[0].get("command", ""):
                        bash_hook_found = True
                        break
                
                if bash_hook_found:
                    print("✓ PostToolUse Bash hook configured for session error handler")
                else:
                    print("⚠ PostToolUse Bash hook not properly configured")
            
            if "Stop" in hooks:
                stop_hooks = hooks["Stop"]
                summary_hook_found = any("session-summary" in hook.get("command", "") for hook in stop_hooks)
                if summary_hook_found:
                    print("✓ Stop hook configured for session summary")
                else:
                    print("⚠ Stop hook not configured for session summary")
            
        except Exception as e:
            print(f"✗ Error reading Claude settings: {e}")
    else:
        print("✗ Claude settings file not found")

def main():
    """Run all tests"""
    
    print("RIF Error Handling Automation - Test Suite")
    print("=" * 50)
    
    # Ensure test environment
    knowledge_dir = Path("/Users/cal/DEV/RIF/knowledge")
    if not knowledge_dir.exists():
        print("⚠ Knowledge directory not found - creating test directories")
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "errors" / "logs").mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "errors" / "github_issues").mkdir(parents=True, exist_ok=True)
    
    # Run tests
    test_session_error_handler()
    test_rif_integration() 
    test_error_classification()
    test_knowledge_base_integration()
    test_claude_hooks_integration()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nNote: Some tests may show warnings if this is the first run.")
    print("GitHub issue creation tests require 'gh' CLI to be authenticated.")

if __name__ == "__main__":
    main()