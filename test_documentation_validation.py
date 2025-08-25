#!/usr/bin/env python3
"""
Test Suite for Documentation-First Enforcement
Adversarial audit verification for Issue #229

This test verifies that the documentation validation system is working
as implemented for Issue #230 emergency response.
"""

import json
import sys
import os
from pathlib import Path

# Add claude commands to path
sys.path.insert(0, str(Path(__file__).parent / "claude" / "commands"))

try:
    from documentation_validator import DocumentationValidator, ValidationResult
    VALIDATOR_AVAILABLE = True
except ImportError as e:
    VALIDATOR_AVAILABLE = False
    IMPORT_ERROR = str(e)


def test_documentation_validator_exists():
    """Test 1: Documentation validator exists and imports"""
    print("🧪 TEST 1: Documentation Validator Availability")
    
    if VALIDATOR_AVAILABLE:
        print("✅ PASS: DocumentationValidator imported successfully")
        return True
    else:
        print(f"❌ FAIL: DocumentationValidator import failed: {IMPORT_ERROR}")
        return False


def test_agent_blocking_mechanism():
    """Test 2: Agent work blocking without documentation"""
    print("\n🧪 TEST 2: Agent Work Blocking Mechanism")
    
    if not VALIDATOR_AVAILABLE:
        print("❌ SKIP: Validator not available")
        return False
    
    validator = DocumentationValidator()
    
    # Test case: Agent work without documentation evidence
    test_agent_work = "I will implement MCP server configuration"
    should_block, message = validator.block_agent_work_without_documentation(
        "RIF-Implementer", test_agent_work
    )
    
    if should_block:
        print("✅ PASS: Agent work correctly blocked without documentation")
        print(f"  Blocking message contains emergency protocols: {'EMERGENCY BLOCKING' in message}")
        return True
    else:
        print("❌ FAIL: Agent work should be blocked but was allowed")
        return False


def test_documentation_evidence_validation():
    """Test 3: Documentation evidence validation"""
    print("\n🧪 TEST 3: Documentation Evidence Validation")
    
    if not VALIDATOR_AVAILABLE:
        print("❌ SKIP: Validator not available")
        return False
    
    validator = DocumentationValidator()
    
    # Test case: Complete evidence
    complete_evidence = {
        "timestamp": "2025-08-25T12:00:00Z",
        "claude_code_docs": ["Tool usage documentation", "API reference"],
        "framework_docs": ["MCP server specification"],
        "key_findings": {"claude_code": "Has built-in tool system", "mcp": "Requires specific config format"},
        "validation_checklist": {
            "approach_aligns_with_documentation": True,
            "no_assumptions_made": True,
            "official_examples_followed": True
        },
        "citations": {"primary_source": "https://docs.anthropic.com/claude/"}
    }
    
    report = validator.validate_agent_documentation_evidence(
        "RIF-Implementer", "229", complete_evidence
    )
    
    if report.validation_result == ValidationResult.PASS:
        print("✅ PASS: Complete documentation evidence validates successfully")
        return True
    else:
        print(f"❌ FAIL: Complete evidence failed validation: {report.violations}")
        return False


def test_incomplete_evidence_rejection():
    """Test 4: Incomplete evidence rejection"""
    print("\n🧪 TEST 4: Incomplete Evidence Rejection")
    
    if not VALIDATOR_AVAILABLE:
        print("❌ SKIP: Validator not available")
        return False
    
    validator = DocumentationValidator()
    
    # Test case: Incomplete evidence (missing key components)
    incomplete_evidence = {
        "timestamp": "2025-08-25T12:00:00Z",
        # Missing claude_code_docs
        "key_findings": {},  # Empty findings
        "validation_checklist": {},  # Empty checklist
        "citations": {}  # Empty citations
    }
    
    report = validator.validate_agent_documentation_evidence(
        "RIF-Implementer", "229", incomplete_evidence
    )
    
    if report.validation_result in [ValidationResult.FAIL, ValidationResult.BLOCKED]:
        print("✅ PASS: Incomplete evidence correctly rejected")
        print(f"  Violations detected: {len(report.violations)}")
        return True
    else:
        print(f"❌ FAIL: Incomplete evidence should be rejected but passed: {report.validation_result}")
        return False


def test_claude_md_documentation_first():
    """Test 5: CLAUDE.md contains documentation-first requirements"""
    print("\n🧪 TEST 5: CLAUDE.md Documentation-First Requirements")
    
    claude_md_path = Path(__file__).parent / "CLAUDE.md"
    
    if not claude_md_path.exists():
        print("❌ FAIL: CLAUDE.md not found")
        return False
    
    with open(claude_md_path, 'r') as f:
        content = f.read()
    
    required_sections = [
        "MANDATORY DOCUMENTATION-FIRST REQUIREMENTS",
        "MANDATORY: Official Documentation Consultation BEFORE Any Work",
        "NO AGENTS CAN BE LAUNCHED WITHOUT DOCUMENTATION EVIDENCE",
        "EMERGENCY ENFORCEMENT: Issue #230 protocols"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if not missing_sections:
        print("✅ PASS: All required documentation-first sections found in CLAUDE.md")
        return True
    else:
        print(f"❌ FAIL: Missing sections in CLAUDE.md: {missing_sections}")
        return False


def test_agent_instructions_updated():
    """Test 6: Agent instructions contain documentation requirements"""
    print("\n🧪 TEST 6: Agent Instructions Documentation Requirements")
    
    agents_dir = Path(__file__).parent / "claude" / "agents"
    core_agents = ["rif-implementer.md", "rif-analyst.md", "rif-validator.md"]
    
    results = {}
    for agent_file in core_agents:
        agent_path = agents_dir / agent_file
        
        if not agent_path.exists():
            results[agent_file] = "FILE_NOT_FOUND"
            continue
        
        with open(agent_path, 'r') as f:
            content = f.read()
        
        required_patterns = [
            "MANDATORY DOCUMENTATION-FIRST REQUIREMENTS",
            "official documentation",
            "documentation consultation"
        ]
        
        found_patterns = [p for p in required_patterns if p.lower() in content.lower()]
        
        if len(found_patterns) >= 2:  # At least 2 out of 3 patterns
            results[agent_file] = "PASS"
        else:
            results[agent_file] = "FAIL"
    
    passes = sum(1 for result in results.values() if result == "PASS")
    total = len(core_agents)
    
    if passes >= 2:  # At least 2 out of 3 agents updated
        print(f"✅ PASS: {passes}/{total} core agents have documentation requirements")
        for agent, result in results.items():
            print(f"  {agent}: {result}")
        return True
    else:
        print(f"❌ FAIL: Only {passes}/{total} core agents have documentation requirements")
        for agent, result in results.items():
            print(f"  {agent}: {result}")
        return False


def test_knowledge_base_learning():
    """Test 7: Knowledge base contains learning from Issue #229/#230"""
    print("\n🧪 TEST 7: Knowledge Base Learning Capture")
    
    knowledge_dir = Path(__file__).parent / "knowledge"
    pattern_files = list(knowledge_dir.glob("patterns/*documentation*.json"))
    
    if not pattern_files:
        print("❌ FAIL: No documentation-related patterns found in knowledge base")
        return False
    
    # Check if emergency documentation pattern exists
    emergency_pattern_path = knowledge_dir / "patterns" / "emergency-documentation-first-enforcement.json"
    
    if emergency_pattern_path.exists():
        with open(emergency_pattern_path, 'r') as f:
            pattern = json.load(f)
        
        required_fields = ["pattern_id", "problem", "solution", "lessons_learned"]
        if all(field in pattern for field in required_fields):
            print("✅ PASS: Emergency documentation pattern captured with complete learning")
            print(f"  Pattern ID: {pattern.get('pattern_id')}")
            print(f"  Issue Reference: {pattern.get('issue_reference')}")
            return True
        else:
            print("❌ FAIL: Emergency documentation pattern incomplete")
            return False
    else:
        print(f"❌ FAIL: Emergency documentation pattern not found")
        print(f"  Available patterns: {[f.name for f in pattern_files]}")
        return False


def run_adversarial_audit():
    """Run complete adversarial audit test suite"""
    print("=" * 70)
    print("🔍 ADVERSARIAL AUDIT: Issue #229 Implementation Verification")
    print("=" * 70)
    
    tests = [
        test_documentation_validator_exists,
        test_agent_blocking_mechanism,
        test_documentation_evidence_validation,
        test_incomplete_evidence_rejection,
        test_claude_md_documentation_first,
        test_agent_instructions_updated,
        test_knowledge_base_learning
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 70)
    print("📊 AUDIT SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 AUDIT RESULT: FULLY IMPLEMENTED - Issue #229 resolved")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print("✅ AUDIT RESULT: SUBSTANTIALLY IMPLEMENTED - Minor gaps remain")
        return True
    else:
        print("❌ AUDIT RESULT: INSUFFICIENT IMPLEMENTATION - Major work needed")
        return False


if __name__ == "__main__":
    success = run_adversarial_audit()
    sys.exit(0 if success else 1)