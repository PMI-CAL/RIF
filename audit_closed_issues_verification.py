#!/usr/bin/env python3
"""
Comprehensive Audit Verification for Closed Issues
RIF-Auditor adversarial testing suite for issue #234

This script performs adversarial auditing of closed issues to verify:
1. Claims of completion match reality
2. Implementation evidence exists
3. Verification tests pass
4. User validation occurred where required
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class ClosedIssueAuditor:
    """
    Adversarial auditor for closed issues.
    Assumes nothing, verifies everything.
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.audit_results = {}
        
    def audit_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Perform comprehensive adversarial audit of a closed issue.
        
        Returns:
            Dict with detailed audit findings
        """
        print(f"\n🔍 AUDITING ISSUE #{issue_number}")
        print("=" * 50)
        
        # Get issue details
        issue_details = self._get_issue_details(issue_number)
        if not issue_details:
            return {
                "issue_number": issue_number,
                "status": "AUDIT_FAILED",
                "reason": "Could not retrieve issue details",
                "should_reopen": True
            }
        
        audit_result = {
            "issue_number": issue_number,
            "title": issue_details.get("title", "Unknown"),
            "closed_at": issue_details.get("closedAt", "Unknown"),
            "state": issue_details.get("state", "Unknown"),
            "audit_timestamp": datetime.now().isoformat(),
            "classification": self._classify_issue(issue_details),
            "tests": {},
            "evidence_found": {},
            "violations": [],
            "should_reopen": False,
            "audit_decision": "PENDING"
        }
        
        print(f"📋 Issue: {audit_result['title']}")
        print(f"📅 Closed: {audit_result['closed_at']}")
        print(f"🏷️  Classification: {audit_result['classification']}")
        
        # Run specific audit tests based on classification
        if audit_result['classification'] == "CURRENT_FEATURE":
            audit_result = self._audit_feature_implementation(audit_result, issue_details)
        elif audit_result['classification'] == "PROCESS_ISSUE":
            audit_result = self._audit_process_fix(audit_result, issue_details)
        elif audit_result['classification'] == "META_ISSUE":
            audit_result = self._audit_meta_issue(audit_result, issue_details)
        else:
            audit_result['violations'].append("Unknown issue classification")
        
        # Final audit decision
        audit_result = self._make_audit_decision(audit_result)
        
        return audit_result
    
    def _get_issue_details(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Get detailed issue information from GitHub."""
        try:
            result = subprocess.run([
                'gh', 'issue', 'view', str(issue_number),
                '--json', 'title,body,state,closedAt,labels,comments'
            ], capture_output=True, text=True, check=True)
            
            return json.loads(result.stdout)
        except:
            return None
    
    def _classify_issue(self, issue_details: Dict[str, Any]) -> str:
        """
        Classify the issue type for appropriate audit approach.
        
        Returns:
            Classification string
        """
        title = issue_details.get("title", "").lower()
        body = issue_details.get("body", "").lower()
        
        # Check for process/meta issues
        meta_keywords = ["critical", "process failure", "root cause", "meta", "audit", "emergency"]
        if any(keyword in title for keyword in meta_keywords):
            return "PROCESS_ISSUE"
        
        # Check for feature implementation
        feature_keywords = ["implement", "add", "create", "build", "develop"]
        if any(keyword in title for keyword in feature_keywords):
            return "CURRENT_FEATURE"
        
        # Check for questions/discussions
        if title.startswith("what is") or "?" in title:
            return "META_ISSUE"
        
        return "UNKNOWN"
    
    def _audit_feature_implementation(self, audit_result: Dict[str, Any], issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit a feature implementation issue.
        """
        issue_number = audit_result["issue_number"]
        
        # Test 1: Branch Evidence
        audit_result["tests"]["branch_created"] = self._test_branch_exists(issue_number)
        
        # Test 2: Code Implementation Evidence
        audit_result["tests"]["implementation_found"] = self._test_implementation_exists(issue_number, issue_details)
        
        # Test 3: Configuration Changes
        audit_result["tests"]["config_updated"] = self._test_configuration_changes(issue_number, issue_details)
        
        # Test 4: Test Coverage
        audit_result["tests"]["tests_exist"] = self._test_verification_tests_exist(issue_number)
        
        # Test 5: User Validation
        audit_result["tests"]["user_validated"] = self._test_user_validation(issue_details)
        
        return audit_result
    
    def _audit_process_fix(self, audit_result: Dict[str, Any], issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit a process/critical failure fix.
        """
        issue_number = audit_result["issue_number"]
        
        # Test 1: Documentation Updates
        audit_result["tests"]["documentation_updated"] = self._test_documentation_evidence(issue_details)
        
        # Test 2: System Changes
        audit_result["tests"]["system_modified"] = self._test_system_changes(issue_number, issue_details)
        
        # Test 3: Prevention Measures
        audit_result["tests"]["prevention_implemented"] = self._test_prevention_measures(issue_number, issue_details)
        
        # Test 4: Validation Framework
        audit_result["tests"]["validation_working"] = self._test_validation_system(issue_number)
        
        return audit_result
    
    def _audit_meta_issue(self, audit_result: Dict[str, Any], issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit a meta/discussion issue.
        """
        # Test 1: Question Answered
        audit_result["tests"]["question_answered"] = self._test_question_answered(issue_details)
        
        # Test 2: Action Items Completed
        audit_result["tests"]["actions_completed"] = self._test_action_items(issue_details)
        
        return audit_result
    
    def _test_branch_exists(self, issue_number: int) -> Dict[str, Any]:
        """Test if issue branch was created."""
        try:
            result = subprocess.run([
                'git', 'branch', '-a'
            ], capture_output=True, text=True, check=True)
            
            issue_branches = [
                line.strip() for line in result.stdout.split('\n')
                if f'issue-{issue_number}' in line.lower()
            ]
            
            return {
                "passed": len(issue_branches) > 0,
                "evidence": issue_branches,
                "message": f"Found {len(issue_branches)} branches for issue #{issue_number}"
            }
        except Exception as e:
            return {
                "passed": False,
                "evidence": [],
                "message": f"Error checking branches: {e}"
            }
    
    def _test_implementation_exists(self, issue_number: int, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if implementation code exists."""
        try:
            # Search for files modified related to this issue
            title_words = issue_details.get("title", "").lower().split()
            search_terms = [word for word in title_words if len(word) > 3]
            
            evidence_files = []
            for term in search_terms[:3]:  # Limit to avoid too many searches
                try:
                    result = subprocess.run([
                        'find', '.', '-name', '*.py', '-exec', 'grep', '-l', term, '{}', ';'
                    ], capture_output=True, text=True, timeout=10)
                    
                    files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                    evidence_files.extend(files)
                except:
                    continue
            
            # Remove duplicates
            evidence_files = list(set(evidence_files))
            
            return {
                "passed": len(evidence_files) > 0,
                "evidence": evidence_files[:10],  # Limit display
                "message": f"Found {len(evidence_files)} potentially related files"
            }
        except Exception as e:
            return {
                "passed": False,
                "evidence": [],
                "message": f"Error searching for implementation: {e}"
            }
    
    def _test_configuration_changes(self, issue_number: int, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if configuration files were updated."""
        config_files = [
            "CLAUDE.md", "config/rif-workflow.yaml", ".claude/settings.json",
            "config/multi-agent.yaml", "config/framework-variables.yaml"
        ]
        
        updated_configs = []
        for config_file in config_files:
            config_path = self.repo_path / config_file
            if config_path.exists():
                # Check if file was modified recently (rough heuristic)
                try:
                    result = subprocess.run([
                        'git', 'log', '--oneline', '--since=1 day ago', '--', config_file
                    ], capture_output=True, text=True)
                    
                    if result.stdout.strip():
                        updated_configs.append(config_file)
                except:
                    pass
        
        return {
            "passed": len(updated_configs) > 0,
            "evidence": updated_configs,
            "message": f"Found {len(updated_configs)} recently updated config files"
        }
    
    def _test_verification_tests_exist(self, issue_number: int) -> Dict[str, Any]:
        """Test if verification tests exist for the issue."""
        test_files = []
        
        # Look for test files mentioning this issue
        try:
            result = subprocess.run([
                'find', '.', '-name', '*.py', '-path', '*/test*', '-exec', 'grep', '-l', f'issue.*{issue_number}', '{}', ';'
            ], capture_output=True, text=True, timeout=10)
            
            test_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except:
            pass
        
        # Also check for generic test files that might test the functionality
        try:
            result = subprocess.run([
                'find', '.', '-name', f'test*{issue_number}*.py'
            ], capture_output=True, text=True, timeout=10)
            
            specific_tests = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            test_files.extend(specific_tests)
        except:
            pass
        
        test_files = list(set(test_files))
        
        return {
            "passed": len(test_files) > 0,
            "evidence": test_files,
            "message": f"Found {len(test_files)} test files for issue #{issue_number}"
        }
    
    def _test_user_validation(self, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if user validated the fix."""
        comments = issue_details.get("comments", [])
        
        user_validation_found = False
        last_comment_author = None
        
        if comments:
            # Check if last comment was from user (not a bot/system)
            last_comment = comments[-1]
            last_comment_author = last_comment.get("author", {}).get("login", "")
            
            # Simple heuristic: if last comment is from non-bot user and contains positive words
            positive_words = ["works", "fixed", "good", "correct", "validated", "confirmed"]
            comment_body = last_comment.get("body", "").lower()
            
            if not any(bot_indicator in last_comment_author.lower() for bot_indicator in ["bot", "rif", "system"]):
                user_validation_found = any(word in comment_body for word in positive_words)
        
        return {
            "passed": user_validation_found,
            "evidence": {
                "last_comment_author": last_comment_author,
                "total_comments": len(comments)
            },
            "message": f"User validation: {'Found' if user_validation_found else 'NOT FOUND'}"
        }
    
    def _test_documentation_evidence(self, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if documentation was updated for process issues."""
        # Check if CLAUDE.md was updated
        claude_md_path = self.repo_path / "CLAUDE.md"
        
        documentation_evidence = []
        
        if claude_md_path.exists():
            try:
                result = subprocess.run([
                    'git', 'log', '--oneline', '--since=1 day ago', '--', "CLAUDE.md"
                ], capture_output=True, text=True)
                
                if result.stdout.strip():
                    documentation_evidence.append("CLAUDE.md updated recently")
            except:
                pass
        
        # Check for new documentation files
        try:
            result = subprocess.run([
                'find', '.', '-name', '*.md', '-newer', str(claude_md_path)
            ], capture_output=True, text=True, timeout=10)
            
            new_docs = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            documentation_evidence.extend(new_docs)
        except:
            pass
        
        return {
            "passed": len(documentation_evidence) > 0,
            "evidence": documentation_evidence,
            "message": f"Found {len(documentation_evidence)} documentation updates"
        }
    
    def _test_system_changes(self, issue_number: int, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if system files were modified."""
        system_paths = [
            "claude/agents/", "claude/commands/", "config/", "systems/"
        ]
        
        modified_files = []
        for path in system_paths:
            try:
                result = subprocess.run([
                    'git', 'log', '--oneline', '--since=1 day ago', '--', path
                ], capture_output=True, text=True)
                
                if result.stdout.strip():
                    modified_files.append(path)
            except:
                pass
        
        return {
            "passed": len(modified_files) > 0,
            "evidence": modified_files,
            "message": f"Found {len(modified_files)} modified system paths"
        }
    
    def _test_prevention_measures(self, issue_number: int, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if prevention measures were implemented."""
        # Look for new validation/prevention code
        prevention_indicators = ["validation", "prevention", "enforce", "block", "check"]
        
        prevention_files = []
        for indicator in prevention_indicators:
            try:
                result = subprocess.run([
                    'find', '.', '-name', '*.py', '-exec', 'grep', '-l', indicator, '{}', ';'
                ], capture_output=True, text=True, timeout=10)
                
                files = [f.strip() for f in result.stdout.split('\n') if f.strip() and 'claude/commands' in f]
                prevention_files.extend(files)
            except:
                continue
        
        prevention_files = list(set(prevention_files))
        
        return {
            "passed": len(prevention_files) > 0,
            "evidence": prevention_files[:5],  # Limit display
            "message": f"Found {len(prevention_files)} prevention-related files"
        }
    
    def _test_validation_system(self, issue_number: int) -> Dict[str, Any]:
        """Test if validation system is working."""
        validation_files = [
            "test_documentation_validation.py",
            "claude/commands/documentation_validator.py", 
            "claude/commands/branch_workflow_enforcer.py"
        ]
        
        working_validators = []
        for validator_file in validation_files:
            validator_path = self.repo_path / validator_file
            if validator_path.exists():
                try:
                    # Try to import or run basic syntax check
                    result = subprocess.run([
                        'python3', '-m', 'py_compile', str(validator_path)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        working_validators.append(validator_file)
                except:
                    pass
        
        return {
            "passed": len(working_validators) > 0,
            "evidence": working_validators,
            "message": f"Found {len(working_validators)} working validation systems"
        }
    
    def _test_question_answered(self, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if question in meta issue was answered."""
        comments = issue_details.get("comments", [])
        body = issue_details.get("body", "")
        
        # Check if there are substantive comments
        substantive_comments = [
            comment for comment in comments
            if len(comment.get("body", "")) > 100  # Longer than just "yes" or "no"
        ]
        
        return {
            "passed": len(substantive_comments) > 0,
            "evidence": {
                "total_comments": len(comments),
                "substantive_comments": len(substantive_comments)
            },
            "message": f"Found {len(substantive_comments)} substantive responses"
        }
    
    def _test_action_items(self, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Test if action items were completed."""
        body = issue_details.get("body", "")
        comments = issue_details.get("comments", [])
        
        # Look for checkbox patterns
        import re
        checkboxes = re.findall(r'- \[([ x])\]', body)
        
        checked_boxes = len([box for box in checkboxes if box == 'x'])
        total_boxes = len(checkboxes)
        
        return {
            "passed": total_boxes > 0 and checked_boxes >= total_boxes * 0.8,  # 80% completion
            "evidence": {
                "total_checkboxes": total_boxes,
                "checked_checkboxes": checked_boxes,
                "completion_rate": checked_boxes / max(total_boxes, 1)
            },
            "message": f"Action items: {checked_boxes}/{total_boxes} completed"
        }
    
    def _make_audit_decision(self, audit_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final audit decision based on test results.
        """
        tests = audit_result.get("tests", {})
        
        # Count passed tests
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test.get("passed", False))
        
        if total_tests == 0:
            audit_result["audit_decision"] = "INSUFFICIENT_TESTING"
            audit_result["should_reopen"] = True
            audit_result["violations"].append("No verification tests were performed")
        elif passed_tests == 0:
            audit_result["audit_decision"] = "IMPLEMENTATION_NOT_FOUND"
            audit_result["should_reopen"] = True
            audit_result["violations"].append("No evidence of implementation found")
        elif passed_tests < total_tests * 0.6:  # Less than 60% pass rate
            audit_result["audit_decision"] = "INSUFFICIENT_IMPLEMENTATION"
            audit_result["should_reopen"] = True
            audit_result["violations"].append(f"Only {passed_tests}/{total_tests} verification tests passed")
        elif passed_tests < total_tests:
            audit_result["audit_decision"] = "PARTIAL_IMPLEMENTATION"
            audit_result["should_reopen"] = False  # Don't reopen, but note issues
            audit_result["violations"].append(f"Some implementation gaps: {passed_tests}/{total_tests} tests passed")
        else:
            audit_result["audit_decision"] = "VERIFIED_COMPLETE"
            audit_result["should_reopen"] = False
        
        # Special case: Critical issues must have user validation
        if "critical" in audit_result.get("title", "").lower():
            user_validation_test = tests.get("user_validated", {})
            if not user_validation_test.get("passed", False):
                audit_result["audit_decision"] = "MISSING_USER_VALIDATION"
                audit_result["should_reopen"] = True
                audit_result["violations"].append("Critical issue closed without user validation")
        
        audit_result["pass_rate"] = passed_tests / max(total_tests, 1)
        audit_result["total_tests"] = total_tests
        audit_result["passed_tests"] = passed_tests
        
        return audit_result
    
    def audit_batch(self, issue_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Audit a batch of closed issues.
        """
        results = []
        
        for issue_number in issue_numbers:
            try:
                result = self.audit_issue(issue_number)
                results.append(result)
                
                # Brief summary
                decision = result.get("audit_decision", "UNKNOWN")
                should_reopen = result.get("should_reopen", False)
                print(f"🔍 Issue #{issue_number}: {decision} {'(SHOULD REOPEN)' if should_reopen else ''}")
                
            except Exception as e:
                print(f"❌ Error auditing issue #{issue_number}: {e}")
                results.append({
                    "issue_number": issue_number,
                    "status": "AUDIT_ERROR",
                    "error": str(e),
                    "should_reopen": True
                })
        
        return results


def main():
    """Run the audit for specified issues."""
    if len(sys.argv) < 2:
        print("Usage: python audit_closed_issues_verification.py <issue_number> [issue_number ...]")
        return 1
    
    issue_numbers = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
    
    if not issue_numbers:
        print("No valid issue numbers provided")
        return 1
    
    auditor = ClosedIssueAuditor()
    results = auditor.audit_batch(issue_numbers)
    
    # Summary
    print("\n" + "=" * 70)
    print("🔍 COMPREHENSIVE AUDIT SUMMARY")
    print("=" * 70)
    
    total_audited = len(results)
    should_reopen = sum(1 for r in results if r.get("should_reopen", False))
    verified_complete = sum(1 for r in results if r.get("audit_decision") == "VERIFIED_COMPLETE")
    
    print(f"📊 Issues Audited: {total_audited}")
    print(f"✅ Verified Complete: {verified_complete}")
    print(f"⚠️  Should Reopen: {should_reopen}")
    print(f"📈 Verification Rate: {(verified_complete/total_audited)*100:.1f}%")
    
    # Save detailed results
    output_file = f"knowledge/audits/closed_issues_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("knowledge/audits", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "audit_timestamp": datetime.now().isoformat(),
            "auditor": "RIF-Auditor",
            "issues_audited": issue_numbers,
            "summary": {
                "total_audited": total_audited,
                "verified_complete": verified_complete,
                "should_reopen": should_reopen,
                "verification_rate": (verified_complete/total_audited)*100
            },
            "detailed_results": results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())