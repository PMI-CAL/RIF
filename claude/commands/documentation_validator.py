#!/usr/bin/env python3
"""
Documentation Validation Gate - Issue #230 Emergency Implementation

CRITICAL: This module implements mandatory documentation consultation validation
to prevent assumption-based development failures.

Created as emergency response to Issue #230: "Critical Process Failure: 
Not Reading Official Documentation Before Implementation"
"""

import json
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationResult(Enum):
    """Documentation validation results"""
    PASS = "PASS"
    FAIL = "FAIL" 
    BLOCKED = "BLOCKED"
    MISSING = "MISSING"


@dataclass
class DocumentationEvidence:
    """Structure for documentation consultation evidence"""
    agent_name: str
    issue_number: str
    timestamp: str
    claude_code_docs: List[str]
    framework_docs: List[str] 
    api_docs: List[str]
    integration_docs: List[str]
    key_findings: Dict[str, str]
    validation_checklist: Dict[str, bool]
    citations: Dict[str, str]
    
    
@dataclass
class ValidationReport:
    """Documentation validation report"""
    agent_name: str
    issue_number: str
    validation_result: ValidationResult
    evidence_provided: bool
    required_evidence: List[str]
    missing_evidence: List[str]
    violations: List[str]
    recommendations: List[str]
    blocking_reasons: List[str]
    

class DocumentationValidator:
    """
    Emergency Documentation Validation Gate
    
    Implements mandatory documentation consultation requirements
    per Issue #230 emergency protocols.
    """
    
    def __init__(self):
        self.emergency_protocols_active = True
        self.issue_230_enforcement = True
        
    def validate_agent_documentation_evidence(
        self, 
        agent_name: str, 
        issue_number: str, 
        evidence_data: Dict
    ) -> ValidationReport:
        """
        Validates that an agent has provided proper documentation evidence
        before proceeding with work.
        
        BLOCKING MECHANISM: Returns BLOCKED if evidence is insufficient.
        """
        
        # Extract evidence from agent submission
        try:
            evidence = self._parse_evidence_data(agent_name, issue_number, evidence_data)
        except Exception as e:
            return ValidationReport(
                agent_name=agent_name,
                issue_number=issue_number,
                validation_result=ValidationResult.BLOCKED,
                evidence_provided=False,
                required_evidence=self._get_required_evidence(agent_name),
                missing_evidence=self._get_required_evidence(agent_name),
                violations=[f"Evidence parsing failed: {str(e)}"],
                recommendations=["Resubmit evidence using proper template format"],
                blocking_reasons=["Invalid evidence format prevents validation"]
            )
        
        # Validate evidence completeness
        validation_result = self._validate_evidence_completeness(evidence)
        
        # Check for critical violations
        critical_violations = self._check_critical_violations(evidence)
        
        # Generate final validation report
        report = ValidationReport(
            agent_name=agent_name,
            issue_number=issue_number,
            validation_result=validation_result,
            evidence_provided=bool(evidence.claude_code_docs or evidence.framework_docs),
            required_evidence=self._get_required_evidence(agent_name),
            missing_evidence=self._identify_missing_evidence(evidence, agent_name),
            violations=critical_violations,
            recommendations=self._generate_recommendations(evidence, agent_name),
            blocking_reasons=self._get_blocking_reasons(evidence, critical_violations)
        )
        
        return report
    
    def check_documentation_consultation_complete(
        self, 
        agent_comment: str,
        agent_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Checks if an agent comment contains proper documentation consultation evidence.
        
        Returns (is_complete, missing_requirements)
        """
        
        # Required evidence markers
        required_markers = [
            "MANDATORY DOCUMENTATION CONSULTATION EVIDENCE",
            "Official Documentation Consulted",
            "Key Documentation Findings", 
            "Documentation Citations"
        ]
        
        missing_requirements = []
        
        for marker in required_markers:
            if marker not in agent_comment:
                missing_requirements.append(marker)
        
        # Check for specific agent requirements
        agent_specific = self._get_agent_specific_requirements(agent_name)
        for requirement in agent_specific:
            if requirement not in agent_comment:
                missing_requirements.append(f"Agent-specific: {requirement}")
        
        is_complete = len(missing_requirements) == 0
        
        return is_complete, missing_requirements
    
    def block_agent_work_without_documentation(
        self, 
        agent_name: str, 
        proposed_work: str
    ) -> Tuple[bool, str]:
        """
        Blocks agent work if documentation consultation evidence is not found.
        
        Returns (should_block, blocking_message)
        """
        
        if not self.emergency_protocols_active:
            return False, ""
        
        # Check if work contains documentation evidence
        has_evidence, missing = self.check_documentation_consultation_complete(
            proposed_work, agent_name
        )
        
        if has_evidence:
            return False, ""
        
        # Generate blocking message
        blocking_message = f"""
🚨 EMERGENCY BLOCKING - ISSUE #230 PROTOCOLS ACTIVE

Agent: {agent_name}
Violation: Attempted work without mandatory documentation consultation

REQUIRED EVIDENCE MISSING:
{chr(10).join(f"- {req}" for req in missing)}

EMERGENCY ACTION REQUIRED:
1. Stop all current work immediately
2. Consult official documentation first
3. Post documentation evidence using template
4. Wait for validation approval before proceeding

NO WORK CAN PROCEED UNTIL DOCUMENTATION CONSULTATION IS EVIDENCED.

This is an emergency enforcement per Issue #230: "Critical Process Failure: 
Not Reading Official Documentation Before Implementation"
"""
        
        return True, blocking_message.strip()
    
    def _parse_evidence_data(
        self, 
        agent_name: str, 
        issue_number: str, 
        evidence_data: Dict
    ) -> DocumentationEvidence:
        """Parse evidence data from agent submission"""
        
        return DocumentationEvidence(
            agent_name=agent_name,
            issue_number=issue_number,
            timestamp=evidence_data.get("timestamp", datetime.now().isoformat()),
            claude_code_docs=evidence_data.get("claude_code_docs", []),
            framework_docs=evidence_data.get("framework_docs", []),
            api_docs=evidence_data.get("api_docs", []),
            integration_docs=evidence_data.get("integration_docs", []),
            key_findings=evidence_data.get("key_findings", {}),
            validation_checklist=evidence_data.get("validation_checklist", {}),
            citations=evidence_data.get("citations", {})
        )
    
    def _validate_evidence_completeness(self, evidence: DocumentationEvidence) -> ValidationResult:
        """Validate that evidence meets minimum requirements"""
        
        # Critical requirements
        if not evidence.claude_code_docs:
            return ValidationResult.BLOCKED
            
        if not evidence.citations.get("primary_source"):
            return ValidationResult.BLOCKED
            
        if not evidence.key_findings:
            return ValidationResult.BLOCKED
        
        # Check validation checklist completion
        required_checks = [
            "approach_aligns_with_documentation",
            "no_assumptions_made", 
            "official_examples_followed"
        ]
        
        for check in required_checks:
            if not evidence.validation_checklist.get(check, False):
                return ValidationResult.FAIL
        
        return ValidationResult.PASS
    
    def _check_critical_violations(self, evidence: DocumentationEvidence) -> List[str]:
        """Check for critical documentation violations"""
        
        violations = []
        
        # No Claude Code documentation consulted
        if not evidence.claude_code_docs:
            violations.append("CRITICAL: No Claude Code documentation consulted")
        
        # No citations provided
        if not evidence.citations:
            violations.append("CRITICAL: No documentation citations provided")
        
        # Key findings empty (indicates no actual reading)
        if not evidence.key_findings:
            violations.append("CRITICAL: No documented findings from reading")
        
        # Validation checklist not completed
        if not any(evidence.validation_checklist.values()):
            violations.append("CRITICAL: Validation checklist not completed")
        
        return violations
    
    def _get_required_evidence(self, agent_name: str) -> List[str]:
        """Get required evidence types for specific agent"""
        
        base_requirements = [
            "Claude Code Documentation consultation",
            "Key documentation findings",
            "Official citations",
            "Validation checklist completion"
        ]
        
        agent_specific = {
            "RIF-Implementer": [
                "Implementation pattern documentation",
                "Configuration specification docs", 
                "Integration protocol documentation"
            ],
            "RIF-Analyst": [
                "Analysis methodology documentation",
                "Requirements specification docs",
                "Pattern recognition documentation"
            ],
            "RIF-Validator": [
                "Testing framework documentation",
                "Quality gate specification docs",
                "Validation procedure documentation"
            ],
            "RIF-Architect": [
                "Architecture pattern documentation",
                "System design specification docs",
                "Integration architecture documentation"
            ],
            "RIF-Planner": [
                "Planning methodology documentation",
                "Workflow specification docs",
                "Resource planning documentation"
            ],
            "RIF-Learner": [
                "Knowledge management documentation",
                "Learning pattern specification docs",
                "Storage procedure documentation"
            ]
        }
        
        return base_requirements + agent_specific.get(agent_name, [])
    
    def _identify_missing_evidence(
        self, 
        evidence: DocumentationEvidence, 
        agent_name: str
    ) -> List[str]:
        """Identify what evidence is missing"""
        
        missing = []
        
        if not evidence.claude_code_docs:
            missing.append("Claude Code documentation sections")
            
        if not evidence.framework_docs:
            missing.append("Framework documentation review")
            
        if not evidence.key_findings:
            missing.append("Key documentation findings")
            
        if not evidence.citations.get("primary_source"):
            missing.append("Primary documentation citations")
        
        # Check agent-specific requirements
        required_checks = self._get_agent_specific_checks(agent_name)
        for check_key, check_desc in required_checks.items():
            if not evidence.validation_checklist.get(check_key, False):
                missing.append(check_desc)
        
        return missing
    
    def _generate_recommendations(
        self, 
        evidence: DocumentationEvidence, 
        agent_name: str
    ) -> List[str]:
        """Generate recommendations for improving evidence"""
        
        recommendations = []
        
        if not evidence.claude_code_docs:
            recommendations.append(
                "Read official Claude Code documentation sections relevant to your work"
            )
        
        if not evidence.key_findings:
            recommendations.append(
                "Document specific findings from official documentation that inform your approach"
            )
            
        if not evidence.citations.get("primary_source"):
            recommendations.append(
                "Provide complete citations to official documentation sources"
            )
        
        if len(evidence.validation_checklist) < 3:
            recommendations.append(
                "Complete all validation checklist items to confirm approach alignment"
            )
            
        recommendations.append(
            "Use the official documentation evidence template for consistent formatting"
        )
        
        return recommendations
    
    def _get_blocking_reasons(
        self, 
        evidence: DocumentationEvidence, 
        violations: List[str]
    ) -> List[str]:
        """Get reasons why work should be blocked"""
        
        blocking_reasons = []
        
        if violations:
            blocking_reasons.extend([f"Critical violation: {v}" for v in violations])
            
        if not evidence.claude_code_docs:
            blocking_reasons.append(
                "No Claude Code documentation consultation prevents verification of approach"
            )
            
        if not evidence.citations:
            blocking_reasons.append(
                "No citations provided prevents validation of documentation sources"
            )
            
        if not blocking_reasons:
            blocking_reasons = ["No blocking issues - evidence sufficient"]
        
        return blocking_reasons
    
    def _get_agent_specific_requirements(self, agent_name: str) -> List[str]:
        """Get agent-specific documentation requirements"""
        
        requirements = {
            "RIF-Implementer": [
                "Implementation patterns documentation",
                "Configuration specifications"
            ],
            "RIF-Analyst": [
                "Analysis methodology documentation",
                "Requirements patterns"
            ],
            "RIF-Validator": [
                "Testing framework documentation", 
                "Quality standards"
            ],
            "RIF-Architect": [
                "Architecture patterns documentation",
                "System design specifications"
            ],
            "RIF-Planner": [
                "Planning methodology documentation",
                "Workflow specifications"
            ],
            "RIF-Learner": [
                "Knowledge management documentation",
                "Learning patterns"
            ]
        }
        
        return requirements.get(agent_name, [])
    
    def _get_agent_specific_checks(self, agent_name: str) -> Dict[str, str]:
        """Get agent-specific validation checks"""
        
        return {
            "approach_aligns_with_documentation": f"{agent_name} approach follows official documentation",
            "no_assumptions_made": f"{agent_name} made no undocumented assumptions", 
            "official_examples_followed": f"{agent_name} referenced official examples",
            "specifications_verified": f"{agent_name} verified against official specifications"
        }


def main():
    """Command-line interface for documentation validation"""
    
    if len(sys.argv) < 4:
        print("Usage: python documentation_validator.py <agent_name> <issue_number> <evidence_file>")
        print("\nEmergency validation for Issue #230 documentation requirements")
        sys.exit(1)
    
    agent_name = sys.argv[1]
    issue_number = sys.argv[2] 
    evidence_file = sys.argv[3]
    
    # Load evidence data
    try:
        with open(evidence_file, 'r') as f:
            evidence_data = json.load(f)
    except Exception as e:
        print(f"Error loading evidence file: {e}")
        sys.exit(1)
    
    # Validate documentation evidence
    validator = DocumentationValidator()
    report = validator.validate_agent_documentation_evidence(
        agent_name, issue_number, evidence_data
    )
    
    # Output validation report
    print(f"Documentation Validation Report - Issue #{issue_number}")
    print(f"Agent: {agent_name}")
    print(f"Result: {report.validation_result.value}")
    print(f"Evidence Provided: {report.evidence_provided}")
    
    if report.violations:
        print("\nVIOLATIONS:")
        for violation in report.violations:
            print(f"  - {violation}")
    
    if report.missing_evidence:
        print("\nMISSING EVIDENCE:")
        for missing in report.missing_evidence:
            print(f"  - {missing}")
    
    if report.recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    if report.validation_result == ValidationResult.BLOCKED:
        print("\n🚨 WORK BLOCKED - DOCUMENTATION EVIDENCE REQUIRED")
        print("Agent work cannot proceed until proper evidence is provided.")
        sys.exit(2)
    elif report.validation_result == ValidationResult.FAIL:
        print("\n⚠️ VALIDATION FAILED - IMPROVEMENTS NEEDED")
        sys.exit(1)
    else:
        print("\n✅ VALIDATION PASSED - WORK MAY PROCEED")
        sys.exit(0)


if __name__ == "__main__":
    main()