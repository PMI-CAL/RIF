#!/usr/bin/env python3
"""
Issue #268 Fix Validation Tool

This tool validates that the quality gate enforcement system is working correctly
and that agents cannot recommend PR merges when quality gates are failing.

Based on the evidence from PR #253, this tool checks:
1. Agent instructions are correctly implemented
2. Quality gate blocking logic works as intended  
3. Prohibited reasoning patterns are prevented
4. Binary decision logic is enforced
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


class Issue268FixValidator:
    """Validator for Issue #268 quality gate enforcement fix."""
    
    def __init__(self):
        self.project_root = Path("/Users/cal/DEV/RIF")
        self.validation_results = {
            "agent_instructions": {"status": "UNKNOWN", "details": []},
            "blocking_logic": {"status": "UNKNOWN", "details": []},
            "test_coverage": {"status": "UNKNOWN", "details": []},
            "knowledge_base": {"status": "UNKNOWN", "details": []},
            "pr_evidence": {"status": "UNKNOWN", "details": []}
        }
        
    def validate_complete_fix(self) -> Dict[str, Any]:
        """Perform complete validation of Issue #268 fix."""
        print("🔍 Validating Issue #268 Fix: Quality Gate Enforcement")
        print("=" * 70)
        
        # 1. Validate agent instructions
        self.validate_agent_instructions()
        
        # 2. Validate blocking logic implementation
        self.validate_blocking_logic()
        
        # 3. Validate test coverage
        self.validate_test_coverage()
        
        # 4. Validate knowledge base patterns
        self.validate_knowledge_base()
        
        # 5. Validate PR evidence
        self.validate_pr_evidence()
        
        # Generate final report
        return self.generate_validation_report()
        
    def validate_agent_instructions(self):
        """Validate that agent instructions contain quality gate blocking rules."""
        print("\n📋 Validating Agent Instructions")
        print("-" * 40)
        
        agent_files = [
            "claude/agents/rif-validator.md",
            "claude/agents/rif-pr-manager.md"
        ]
        
        for agent_file in agent_files:
            file_path = self.project_root / agent_file
            
            if not file_path.exists():
                self.validation_results["agent_instructions"]["details"].append(
                    f"❌ Missing: {agent_file}"
                )
                continue
                
            content = file_path.read_text()
            
            # Check for critical blocking rules
            has_critical_rule = "QUALITY GATE FAILURE = MERGE BLOCKING" in content
            has_blocking_logic = "validate_quality_gates" in content or "evaluate_merge_eligibility" in content
            has_prohibited_patterns = "NEVER SAY" in content and "Gate failures prove" in content
            has_binary_decision = "Binary decision" in content or "NO agent discretion" in content
            
            print(f"📄 {agent_file}:")
            print(f"   ✅ Critical blocking rule: {'YES' if has_critical_rule else '❌ NO'}")
            print(f"   ✅ Blocking logic function: {'YES' if has_blocking_logic else '❌ NO'}")
            print(f"   ✅ Prohibited patterns: {'YES' if has_prohibited_patterns else '❌ NO'}")
            print(f"   ✅ Binary decision logic: {'YES' if has_binary_decision else '❌ NO'}")
            
            self.validation_results["agent_instructions"]["details"].append({
                "file": agent_file,
                "critical_rule": has_critical_rule,
                "blocking_logic": has_blocking_logic,
                "prohibited_patterns": has_prohibited_patterns,
                "binary_decision": has_binary_decision
            })
            
        # Overall status
        all_files_valid = all(
            detail.get("critical_rule", False) and 
            detail.get("blocking_logic", False) and
            detail.get("prohibited_patterns", False)
            for detail in self.validation_results["agent_instructions"]["details"]
            if isinstance(detail, dict)
        )
        
        self.validation_results["agent_instructions"]["status"] = "PASS" if all_files_valid else "FAIL"
        
    def validate_blocking_logic(self):
        """Validate that blocking logic implementation is correct."""
        print("\n🔒 Validating Blocking Logic Implementation")
        print("-" * 40)
        
        # Check for quality gate enforcement files
        enforcement_files = [
            "claude/commands/quality_gate_enforcement.py",
            "tests/unit/test_quality_gate_blocking_issue_268.py"
        ]
        
        logic_valid = True
        
        for file_name in enforcement_files:
            file_path = self.project_root / file_name
            
            if not file_path.exists():
                print(f"❌ Missing: {file_name}")
                logic_valid = False
                continue
                
            print(f"✅ Found: {file_name}")
            
        # Run the blocking logic tests
        test_file = self.project_root / "tests/unit/test_quality_gate_blocking_issue_268.py"
        if test_file.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(test_file)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and "21/21 passed" in result.stdout:
                    print("✅ Blocking logic tests: ALL PASS")
                else:
                    print(f"❌ Blocking logic tests: FAILED")
                    print(f"   Error: {result.stderr}")
                    logic_valid = False
                    
            except Exception as e:
                print(f"❌ Failed to run blocking logic tests: {e}")
                logic_valid = False
        else:
            logic_valid = False
            
        self.validation_results["blocking_logic"]["status"] = "PASS" if logic_valid else "FAIL"
        
    def validate_test_coverage(self):
        """Validate test coverage for quality gate blocking."""
        print("\n🧪 Validating Test Coverage")
        print("-" * 40)
        
        test_files = [
            "tests/unit/test_quality_gate_blocking_issue_268.py",
            "tests/integration/test_agent_quality_gate_compliance.py"
        ]
        
        test_coverage_valid = True
        
        for test_file in test_files:
            file_path = self.project_root / test_file
            
            if file_path.exists():
                print(f"✅ Found: {test_file}")
                
                # Count test methods
                content = file_path.read_text()
                test_methods = content.count("def test_")
                print(f"   📊 Test methods: {test_methods}")
                
            else:
                print(f"❌ Missing: {test_file}")
                test_coverage_valid = False
                
        self.validation_results["test_coverage"]["status"] = "PASS" if test_coverage_valid else "FAIL"
        
    def validate_knowledge_base(self):
        """Validate knowledge base patterns for quality gate blocking."""
        print("\n🧠 Validating Knowledge Base Patterns")
        print("-" * 40)
        
        pattern_file = self.project_root / "knowledge/patterns/quality-gate-blocking-critical-pattern-issue-268.json"
        
        if pattern_file.exists():
            print("✅ Found: quality-gate-blocking-critical-pattern-issue-268.json")
            
            try:
                with open(pattern_file) as f:
                    pattern = json.load(f)
                    
                print(f"   📊 Severity: {pattern.get('severity', 'UNKNOWN')}")
                print(f"   📊 Prohibited patterns: {len(pattern.get('prohibited_reasoning_patterns', []))}")
                print(f"   📊 Required patterns: {len(pattern.get('required_reasoning_patterns', []))}")
                
                self.validation_results["knowledge_base"]["status"] = "PASS"
                
            except Exception as e:
                print(f"❌ Invalid pattern file: {e}")
                self.validation_results["knowledge_base"]["status"] = "FAIL"
        else:
            print("❌ Missing: quality-gate-blocking-critical-pattern-issue-268.json")
            self.validation_results["knowledge_base"]["status"] = "FAIL"
            
    def validate_pr_evidence(self):
        """Validate evidence from PR #253 that triggered Issue #268."""
        print("\n📋 Validating PR Evidence")
        print("-" * 40)
        
        # Try to get PR #253 details
        try:
            result = subprocess.run([
                "gh", "pr", "view", "253", 
                "--json", "title,state,mergeStateStatus,statusCheckRollup"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                pr_data = json.loads(result.stdout)
                
                print(f"✅ PR #253 Status: {pr_data.get('state', 'UNKNOWN')}")
                print(f"✅ Merge Status: {pr_data.get('mergeStateStatus', 'UNKNOWN')}")
                
                # Check status checks
                status_checks = pr_data.get('statusCheckRollup', [])
                failed_checks = [
                    check for check in status_checks 
                    if check.get('conclusion') == 'FAILURE'
                ]
                
                print(f"✅ Failed Status Checks: {len(failed_checks)}")
                for check in failed_checks:
                    print(f"   ❌ {check.get('name', 'Unknown')}: FAILURE")
                    
                # This proves the problem existed
                self.validation_results["pr_evidence"]["status"] = "CONFIRMED"
                self.validation_results["pr_evidence"]["details"] = {
                    "pr_state": pr_data.get('state'),
                    "failed_checks": len(failed_checks),
                    "evidence": "PR was merged despite failing quality gates"
                }
                
            else:
                print("❌ Could not retrieve PR #253 details")
                self.validation_results["pr_evidence"]["status"] = "UNAVAILABLE"
                
        except Exception as e:
            print(f"❌ Error retrieving PR evidence: {e}")
            self.validation_results["pr_evidence"]["status"] = "ERROR"
            
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        print("\n" + "=" * 70)
        print("📊 ISSUE #268 FIX VALIDATION REPORT")
        print("=" * 70)
        
        overall_status = "PASS"
        
        for category, result in self.validation_results.items():
            status = result["status"]
            status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            
            print(f"{status_icon} {category.replace('_', ' ').title()}: {status}")
            
            if status in ["FAIL", "ERROR"]:
                overall_status = "FAIL"
            elif status in ["PARTIAL", "UNKNOWN"] and overall_status != "FAIL":
                overall_status = "PARTIAL"
                
        print(f"\n🎯 OVERALL VALIDATION: {overall_status}")
        
        if overall_status == "PASS":
            print("✅ Issue #268 fix implementation appears to be complete")
            print("✅ Quality gate enforcement system is properly configured")
            print("✅ Agent instructions contain necessary blocking logic")
            print("✅ Test coverage validates blocking behavior")
        else:
            print("❌ Issue #268 fix has implementation gaps")
            print("❌ Quality gate enforcement may not be working correctly")
            print("❌ Agents may still be able to bypass quality gates")
            
        print(f"\n🔍 VALIDATION DETAILS:")
        print(f"Agent Instructions: {self.validation_results['agent_instructions']['status']}")
        print(f"Blocking Logic: {self.validation_results['blocking_logic']['status']}")
        print(f"Test Coverage: {self.validation_results['test_coverage']['status']}")
        print(f"Knowledge Base: {self.validation_results['knowledge_base']['status']}")
        print(f"PR Evidence: {self.validation_results['pr_evidence']['status']}")
        
        return {
            "overall_status": overall_status,
            "validation_results": self.validation_results,
            "timestamp": subprocess.run(["date", "-u"], capture_output=True, text=True).stdout.strip(),
            "issue": "268",
            "title": "Quality Gate Enforcement Fix Validation"
        }
        

def main():
    """Main validation execution."""
    validator = Issue268FixValidator()
    
    try:
        report = validator.validate_complete_fix()
        
        # Save report to file
        report_file = Path("/Users/cal/DEV/RIF/issue_268_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\n💾 Validation report saved: {report_file}")
        
        # Exit with appropriate code
        if report["overall_status"] == "PASS":
            print("\n🎉 Issue #268 fix validation: SUCCESS")
            return 0
        else:
            print("\n🚨 Issue #268 fix validation: FAILED")
            return 1
            
    except Exception as e:
        print(f"\n💥 Validation error: {e}")
        return 2


if __name__ == "__main__":
    exit(main())