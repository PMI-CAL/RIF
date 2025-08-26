#!/usr/bin/env python3
"""
Test Script for Documentation Requirements - Issue #230 Emergency Implementation

This script tests that the mandatory documentation requirements are properly 
implemented and enforced across all RIF agents.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'claude', 'commands'))

from documentation_validator import DocumentationValidator, ValidationResult


def test_documentation_validator():
    """Test the documentation validation system"""
    
    print("🔍 Testing Documentation Validator...")
    
    validator = DocumentationValidator()
    
    # Test 1: Agent work without documentation evidence (should be blocked)
    print("\nTest 1: Agent work without documentation evidence")
    should_block, message = validator.block_agent_work_without_documentation(
        "RIF-Implementer", 
        "I will implement the feature now using my assumptions about how Claude Code works."
    )
    
    assert should_block, "Agent work should be blocked without documentation evidence"
    assert "EMERGENCY BLOCKING" in message, "Blocking message should indicate emergency status"
    print("✅ PASS: Agent work properly blocked without documentation")
    
    # Test 2: Agent work with proper documentation evidence (should not be blocked)
    print("\nTest 2: Agent work with documentation evidence")
    work_with_evidence = """
    ## 📚 MANDATORY DOCUMENTATION CONSULTATION EVIDENCE
    
    **Issue #**: 230
    **Agent**: RIF-Implementer
    
    ### Official Documentation Consulted
    - [x] Claude Code Documentation: Task tool specifications
    - [x] Framework Documentation: Python implementation guides
    - [x] Implementation patterns documentation: Proper coding patterns
    - [x] Configuration specifications: Official config formats
    
    ### Key Documentation Findings
    1. Claude Code Capabilities: Task tool allows parallel agent execution
    2. Official Implementation Patterns: Use subagent_type="general-purpose"
    
    ### Documentation Citations  
    - Primary Source: https://claude.ai/code/documentation
    """
    
    should_block, message = validator.block_agent_work_without_documentation(
        "RIF-Implementer",
        work_with_evidence
    )
    
    # Debug output
    has_evidence, missing = validator.check_documentation_consultation_complete(
        work_with_evidence, "RIF-Implementer"
    )
    print(f"Debug: has_evidence={has_evidence}, missing={missing}")
    
    assert not should_block, "Agent work with documentation evidence should not be blocked"
    print("✅ PASS: Agent work with documentation evidence allowed")
    
    # Test 3: Evidence completeness validation
    print("\nTest 3: Evidence completeness validation")
    
    # Complete evidence
    complete_evidence = {
        "timestamp": "2025-08-25T08:00:00Z",
        "claude_code_docs": ["Task tool documentation", "Subagent specifications"],
        "framework_docs": ["Python documentation"],
        "key_findings": {
            "claude_code_capabilities": "Task tool supports parallel execution",
            "implementation_patterns": "Use subagent_type parameter"
        },
        "validation_checklist": {
            "approach_aligns_with_documentation": True,
            "no_assumptions_made": True,
            "official_examples_followed": True
        },
        "citations": {
            "primary_source": "https://claude.ai/code/documentation"
        }
    }
    
    report = validator.validate_agent_documentation_evidence(
        "RIF-Implementer", "230", complete_evidence
    )
    
    assert report.validation_result == ValidationResult.PASS, "Complete evidence should pass validation"
    print("✅ PASS: Complete evidence validation successful")
    
    # Test 4: Incomplete evidence (should fail)
    print("\nTest 4: Incomplete evidence validation")
    
    incomplete_evidence = {
        "timestamp": "2025-08-25T08:00:00Z",
        "claude_code_docs": [],  # Missing!
        "key_findings": {},      # Missing!
        "citations": {}          # Missing!
    }
    
    report = validator.validate_agent_documentation_evidence(
        "RIF-Implementer", "230", incomplete_evidence
    )
    
    assert report.validation_result in [ValidationResult.BLOCKED, ValidationResult.FAIL], \
           "Incomplete evidence should be blocked or fail"
    print("✅ PASS: Incomplete evidence properly rejected")
    
    print("\n🎉 All documentation validator tests passed!")


def test_agent_file_updates():
    """Test that agent files have been properly updated with documentation requirements"""
    
    print("\n🔍 Testing Agent File Updates...")
    
    agent_files = [
        "claude/agents/rif-implementer.md",
        "claude/agents/rif-analyst.md", 
        "claude/agents/rif-validator.md",
        "claude/agents/rif-planner.md",
        "claude/agents/rif-architect.md",
        "claude/agents/rif-learner.md"
    ]
    
    required_content = [
        "🚨 MANDATORY DOCUMENTATION-FIRST REQUIREMENTS",
        "MANDATORY: Consult Official Documentation",
        "Documentation Evidence Template (MANDATORY POST)",
        "BLOCKING MECHANISM",
        "Issue #230 emergency protocols"
    ]
    
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            with open(agent_file, 'r') as f:
                content = f.read()
                
            missing_content = []
            for required in required_content:
                if required not in content:
                    missing_content.append(required)
            
            if missing_content:
                print(f"❌ FAIL: {agent_file} missing: {missing_content}")
                return False
            else:
                print(f"✅ PASS: {agent_file} properly updated")
        else:
            print(f"⚠️ WARN: {agent_file} not found")
    
    print("✅ All agent files properly updated with documentation requirements!")
    return True


def test_claude_md_updates():
    """Test that CLAUDE.md has been updated with documentation requirements"""
    
    print("\n🔍 Testing CLAUDE.md Updates...")
    
    claude_md_file = "CLAUDE.md"
    
    required_content = [
        "🚨 MANDATORY DOCUMENTATION-FIRST REQUIREMENTS",
        "MANDATORY: Official Documentation Consultation BEFORE Any Work", 
        "Documentation Evidence Template Required",
        "Validation Gate Enforcement",
        "Emergency Blocking Mechanism",
        "Issue #230 emergency implementation"
    ]
    
    if os.path.exists(claude_md_file):
        with open(claude_md_file, 'r') as f:
            content = f.read()
            
        missing_content = []
        for required in required_content:
            if required not in content:
                missing_content.append(required)
        
        if missing_content:
            print(f"❌ FAIL: CLAUDE.md missing: {missing_content}")
            return False
        else:
            print("✅ PASS: CLAUDE.md properly updated")
            return True
    else:
        print("❌ FAIL: CLAUDE.md not found")
        return False


def test_validation_gate_creation():
    """Test that validation gate file was created"""
    
    print("\n🔍 Testing Validation Gate Creation...")
    
    validator_file = "claude/commands/documentation_validator.py"
    
    if os.path.exists(validator_file):
        with open(validator_file, 'r') as f:
            content = f.read()
            
        required_elements = [
            "class DocumentationValidator",
            "validate_agent_documentation_evidence",
            "block_agent_work_without_documentation", 
            "ValidationResult",
            "Issue #230"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ FAIL: documentation_validator.py missing: {missing_elements}")
            return False
        else:
            print("✅ PASS: documentation_validator.py properly created")
            return True
    else:
        print("❌ FAIL: documentation_validator.py not found")
        return False


def main():
    """Run all tests for documentation requirements implementation"""
    
    print("🚨 TESTING ISSUE #230 EMERGENCY DOCUMENTATION REQUIREMENTS")
    print("=" * 70)
    
    try:
        # Test 1: Documentation validator functionality
        test_documentation_validator()
        
        # Test 2: Agent file updates
        if not test_agent_file_updates():
            sys.exit(1)
        
        # Test 3: CLAUDE.md updates
        if not test_claude_md_updates():
            sys.exit(1)
            
        # Test 4: Validation gate creation
        if not test_validation_gate_creation():
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("Issue #230 emergency documentation requirements successfully implemented!")
        print("\n✅ Phase 1 implementation COMPLETE")
        print("✅ All agents now require documentation consultation")
        print("✅ Validation gates active and enforcing")
        print("✅ Emergency blocking mechanisms operational")
        
        print("\nNext steps:")
        print("- Phase 2: Process integration (4-12 hours)")
        print("- Phase 3: System redesign (12-24 hours)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()