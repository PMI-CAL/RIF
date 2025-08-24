#!/usr/bin/env python3
"""
Emergency Context Compliance Audit System
Addresses Issue #145: Agents are still missing context in issues

Audits all issues for context compliance failures including:
- Missing literature review requirements
- Incomplete research methodology
- Validation without proper evidence
- Context consumption verification
"""

import json
import os
import subprocess
import re
from typing import Dict, List, Any
from datetime import datetime

class EmergencyContextAuditor:
    def __init__(self):
        self.audit_results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "audit_trigger": "Issue #145 - Agents are still missing context in issues",
            "issues_audited": [],
            "compliance_failures": [],
            "critical_findings": [],
            "recommendations": []
        }
    
    def audit_all_issues(self) -> Dict[str, Any]:
        """Audit all open and recently completed issues for context compliance"""
        
        # Get issues from various states
        states = ["open", "closed"]
        
        for state in states:
            issues = self._get_issues_by_state(state)
            for issue in issues:
                self._audit_single_issue(issue)
        
        self._generate_recommendations()
        self._save_audit_results()
        
        return self.audit_results
    
    def _get_issues_by_state(self, state: str) -> List[Dict[str, Any]]:
        """Get issues by state using GitHub CLI"""
        try:
            cmd = [
                "gh", "issue", "list", 
                "--state", state,
                "--limit", "50",
                "--json", "number,title,body,labels,createdAt,updatedAt"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        
        except subprocess.CalledProcessError as e:
            print(f"Error getting {state} issues: {e}")
            return []
    
    def _audit_single_issue(self, issue: Dict[str, Any]) -> None:
        """Audit a single issue for context compliance"""
        
        issue_number = issue["number"]
        issue_title = issue["title"] 
        issue_body = issue.get("body", "")
        labels = issue.get("labels", [])
        
        audit_item = {
            "issue_number": issue_number,
            "title": issue_title,
            "audit_findings": [],
            "compliance_status": "compliant",
            "evidence_status": "sufficient"
        }
        
        # Check for research requirements
        if self._has_research_requirements(issue_body):
            literature_evidence = self._check_literature_review_evidence(issue_number)
            if not literature_evidence["has_evidence"]:
                audit_item["audit_findings"].append({
                    "type": "missing_literature_review",
                    "severity": "critical",
                    "description": "Issue requires literature review but no evidence found",
                    "required_by": "Issue body research methodology section",
                    "evidence_checked": literature_evidence["locations_checked"]
                })
                audit_item["compliance_status"] = "non_compliant"
                audit_item["evidence_status"] = "insufficient"
        
        # Check for validation without evidence
        if self._is_completed_issue(labels):
            evidence_check = self._verify_completion_evidence(issue_number, issue_body)
            if not evidence_check["sufficient"]:
                audit_item["audit_findings"].append({
                    "type": "insufficient_validation_evidence", 
                    "severity": "high",
                    "description": "Issue marked complete without sufficient evidence",
                    "missing_evidence": evidence_check["missing"],
                    "evidence_found": evidence_check["found"]
                })
                audit_item["compliance_status"] = "non_compliant"
        
        # Check for context reading failures
        context_check = self._check_context_consumption(issue_number, issue_body)
        if not context_check["context_properly_consumed"]:
            audit_item["audit_findings"].append({
                "type": "context_consumption_failure",
                "severity": "medium", 
                "description": "Evidence suggests agent did not fully consume issue context",
                "indicators": context_check["failure_indicators"]
            })
        
        self.audit_results["issues_audited"].append(audit_item)
        
        # Track compliance failures
        if audit_item["compliance_status"] == "non_compliant":
            self.audit_results["compliance_failures"].append(issue_number)
            
            # Critical findings for blocking issues
            if any(f["severity"] == "critical" for f in audit_item["audit_findings"]):
                self.audit_results["critical_findings"].append({
                    "issue_number": issue_number,
                    "title": issue_title,
                    "critical_failures": [f for f in audit_item["audit_findings"] if f["severity"] == "critical"]
                })
    
    def _has_research_requirements(self, body: str) -> bool:
        """Check if issue body contains research requirements"""
        research_indicators = [
            "literature review",
            "academic papers",
            "research methodology",
            "industry best practices",
            "research deliverable",
            "academic research",
            "research objective"
        ]
        
        body_lower = body.lower()
        return any(indicator in body_lower for indicator in research_indicators)
    
    def _check_literature_review_evidence(self, issue_number: int) -> Dict[str, Any]:
        """Check for evidence of completed literature review"""
        
        evidence_locations = [
            f"/Users/cal/DEV/RIF/knowledge/checkpoints/issue-{issue_number}-*",
            f"/Users/cal/DEV/RIF/knowledge/learning/issue-{issue_number}-*",
            f"/Users/cal/DEV/RIF/knowledge/research/*{issue_number}*"
        ]
        
        has_evidence = False
        found_evidence = []
        
        for pattern in evidence_locations:
            try:
                cmd = f"find /Users/cal/DEV/RIF -path '{pattern}' -type f 2>/dev/null"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.stdout.strip():
                    files = result.stdout.strip().split('\n')
                    for file_path in files:
                        if self._file_contains_literature_evidence(file_path):
                            has_evidence = True
                            found_evidence.append(file_path)
                            
            except Exception as e:
                print(f"Error checking evidence pattern {pattern}: {e}")
        
        return {
            "has_evidence": has_evidence,
            "evidence_files": found_evidence,
            "locations_checked": evidence_locations
        }
    
    def _file_contains_literature_evidence(self, file_path: str) -> bool:
        """Check if file contains actual literature review evidence"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
            # Look for actual academic evidence
            literature_indicators = [
                "academic sources",
                "papers analyzed", 
                "citations",
                "bibliography",
                "academic literature",
                "systematic review",
                "peer-reviewed",
                "conference paper",
                "journal article"
            ]
            
            return any(indicator in content for indicator in literature_indicators)
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False
    
    def _is_completed_issue(self, labels: List[Dict[str, Any]]) -> bool:
        """Check if issue is marked as completed"""
        for label in labels:
            if label.get("name", "").startswith("state:complete"):
                return True
        return False
    
    def _verify_completion_evidence(self, issue_number: int, body: str) -> Dict[str, Any]:
        """Verify completion evidence for completed issues"""
        
        # Check for required deliverables in body
        required_evidence = []
        if "deliverable" in body.lower():
            # Extract deliverables from issue body
            deliverable_lines = [line for line in body.split('\n') if 'deliverable' in line.lower()]
            required_evidence.extend(deliverable_lines)
        
        # Check for success criteria
        if "success criteria" in body.lower() or "acceptance criteria" in body.lower():
            criteria_section = self._extract_criteria_section(body)
            required_evidence.extend(criteria_section)
        
        # Check for actual evidence
        evidence_found = []
        evidence_files = self._find_completion_artifacts(issue_number)
        
        for file_path in evidence_files:
            if os.path.exists(file_path):
                evidence_found.append(file_path)
        
        sufficient = len(evidence_found) > 0 and len(required_evidence) > 0
        
        return {
            "sufficient": sufficient,
            "required": required_evidence,
            "found": evidence_found,
            "missing": [req for req in required_evidence if not any(req.lower() in found.lower() for found in evidence_found)]
        }
    
    def _extract_criteria_section(self, body: str) -> List[str]:
        """Extract success/acceptance criteria from issue body"""
        lines = body.split('\n')
        criteria = []
        in_criteria_section = False
        
        for line in lines:
            if any(term in line.lower() for term in ['success criteria', 'acceptance criteria']):
                in_criteria_section = True
                continue
            elif line.startswith('#') and in_criteria_section:
                break
            elif in_criteria_section and line.strip():
                criteria.append(line.strip())
        
        return criteria
    
    def _find_completion_artifacts(self, issue_number: int) -> List[str]:
        """Find completion artifacts for an issue"""
        patterns = [
            f"/Users/cal/DEV/RIF/knowledge/checkpoints/issue-{issue_number}-*-complete.json",
            f"/Users/cal/DEV/RIF/knowledge/learning/issue-{issue_number}-*-complete.json", 
            f"/Users/cal/DEV/RIF/*{issue_number}*COMPLETE*.md",
            f"/Users/cal/DEV/RIF/systems/*{issue_number}*"
        ]
        
        found_files = []
        for pattern in patterns:
            try:
                cmd = f"find /Users/cal/DEV/RIF -path '{pattern}' -type f 2>/dev/null"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.stdout.strip():
                    found_files.extend(result.stdout.strip().split('\n'))
                    
            except Exception as e:
                print(f"Error finding artifacts for pattern {pattern}: {e}")
        
        return found_files
    
    def _check_context_consumption(self, issue_number: int, body: str) -> Dict[str, Any]:
        """Check for evidence of proper context consumption"""
        
        failure_indicators = []
        
        # Check if implementation missed obvious requirements
        if "requirement" in body.lower() and len(body) > 1000:
            # Large issue body suggests complex requirements
            implementation_files = self._find_implementation_files(issue_number)
            if len(implementation_files) < 2:
                failure_indicators.append("Complex issue with minimal implementation artifacts")
        
        # Check for validation shortcuts
        validation_files = self._find_validation_artifacts(issue_number)
        if not validation_files:
            failure_indicators.append("No validation artifacts found for completed issue")
        
        return {
            "context_properly_consumed": len(failure_indicators) == 0,
            "failure_indicators": failure_indicators
        }
    
    def _find_implementation_files(self, issue_number: int) -> List[str]:
        """Find implementation files for an issue"""
        cmd = f"find /Users/cal/DEV/RIF -name '*{issue_number}*' -type f | grep -E '\\.(py|md|json)$'"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except:
            return []
    
    def _find_validation_artifacts(self, issue_number: int) -> List[str]:
        """Find validation artifacts for an issue"""
        patterns = [
            f"validation*{issue_number}*",
            f"test*{issue_number}*",
            f"*{issue_number}*validation*"
        ]
        
        found_files = []
        for pattern in patterns:
            cmd = f"find /Users/cal/DEV/RIF -name '{pattern}' -type f 2>/dev/null"
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.stdout.strip():
                    found_files.extend(result.stdout.strip().split('\n'))
            except:
                continue
        
        return found_files
    
    def _generate_recommendations(self) -> None:
        """Generate emergency recommendations based on audit findings"""
        
        critical_count = len(self.audit_results["critical_findings"])
        failure_count = len(self.audit_results["compliance_failures"])
        
        self.audit_results["recommendations"] = [
            {
                "priority": "immediate",
                "action": "Suspend validation for all incomplete issues",
                "rationale": f"{failure_count} issues found with compliance failures",
                "implementation": "Activate emergency validation suspension protocol"
            },
            {
                "priority": "immediate", 
                "action": "Re-open and remediate critical issues",
                "rationale": f"{critical_count} issues with critical compliance failures identified",
                "issues": [f["issue_number"] for f in self.audit_results["critical_findings"]]
            },
            {
                "priority": "high",
                "action": "Implement enhanced context verification",
                "rationale": "Systematic context consumption failures detected",
                "implementation": "Deploy context consumption verification system"
            },
            {
                "priority": "high",
                "action": "Activate knowledge consultation enforcer",
                "rationale": "Evidence of agents bypassing required research steps", 
                "implementation": "Enforce mandatory knowledge consultation for all research issues"
            }
        ]
    
    def _save_audit_results(self) -> None:
        """Save audit results to knowledge base"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audit_file = f"/Users/cal/DEV/RIF/knowledge/context_compliance_audit_{timestamp}.json"
        
        try:
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(self.audit_results, f, indent=2)
            
            print(f"Audit results saved to: {audit_file}")
            
        except Exception as e:
            print(f"Error saving audit results: {e}")


def main():
    """Run emergency context compliance audit"""
    print("🚨 EMERGENCY CONTEXT COMPLIANCE AUDIT 🚨")
    print("Addressing Issue #145: Agents are still missing context in issues")
    print()
    
    auditor = EmergencyContextAuditor()
    results = auditor.audit_all_issues()
    
    print(f"Issues audited: {len(results['issues_audited'])}")
    print(f"Compliance failures: {len(results['compliance_failures'])}")
    print(f"Critical findings: {len(results['critical_findings'])}")
    print()
    
    if results["critical_findings"]:
        print("CRITICAL ISSUES FOUND:")
        for finding in results["critical_findings"]:
            print(f"  Issue #{finding['issue_number']}: {finding['title']}")
            for failure in finding["critical_failures"]:
                print(f"    - {failure['type']}: {failure['description']}")
        print()
    
    print("EMERGENCY RECOMMENDATIONS:")
    for rec in results["recommendations"]:
        print(f"  [{rec['priority'].upper()}] {rec['action']}")
        print(f"    Rationale: {rec['rationale']}")
        print()
    
    return results


if __name__ == "__main__":
    main()