#!/usr/bin/env python3
"""
RIF Context Validation Enforcer
Issue #115 & #145: Automatic validation that prevents context failures

This system automatically validates that agents have properly consumed context
before allowing them to proceed with work, preventing issues like #145.

CRITICAL VALIDATIONS:
1. Full issue text has been read and understood
2. All requirements have been extracted and documented
3. Research methodology requirements are identified and planned
4. Validation requirements are understood
5. Evidence of understanding is provided
"""

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class ValidationResult:
    """Result of context validation"""
    passed: bool
    compliance_score: float
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class ContextValidationEnforcer:
    """
    Enforces context consumption validation before agents can proceed with work
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.knowledge_base_path = knowledge_base_path
        self.validation_log_file = os.path.join(knowledge_base_path, "context_validation_log.json")
        
        # Validation thresholds
        self.minimum_compliance_score = 0.8
        self.critical_requirements = [
            "full_issue_read",
            "requirements_extracted", 
            "research_methodology_identified",
            "validation_requirements_understood"
        ]
        
        # Create log directory
        os.makedirs(os.path.dirname(self.validation_log_file), exist_ok=True)
    
    def validate_agent_context_consumption(self, agent_type: str, issue_number: int, 
                                         agent_response: str) -> ValidationResult:
        """
        Validate that agent has properly consumed context before proceeding
        """
        print(f"Validating context consumption for {agent_type} on issue #{issue_number}...")
        
        violations = []
        warnings = []
        recommendations = []
        
        # Get issue context for validation
        issue_context = self._get_issue_context(issue_number)
        
        # Validation checks
        full_issue_read = self._validate_full_issue_reading(agent_response, issue_context)
        requirements_extracted = self._validate_requirements_extraction(agent_response, issue_context)
        research_methodology = self._validate_research_methodology_compliance(agent_response, issue_context)
        validation_requirements = self._validate_validation_requirements_understanding(agent_response, issue_context)
        evidence_provided = self._validate_evidence_of_understanding(agent_response)
        
        # Calculate compliance score
        validations = {
            "full_issue_read": full_issue_read if full_issue_read is not None else False,
            "requirements_extracted": requirements_extracted if requirements_extracted is not None else False,
            "research_methodology_identified": research_methodology if research_methodology is not None else False,
            "validation_requirements_understood": validation_requirements if validation_requirements is not None else False,
            "evidence_provided": evidence_provided if evidence_provided is not None else False
        }
        
        # Convert to numeric values for calculation
        numeric_validations = {k: 1 if v else 0 for k, v in validations.items()}
        compliance_score = sum(numeric_validations.values()) / len(numeric_validations)
        
        # Check for violations
        if not full_issue_read:
            violations.append("Agent has not demonstrated reading the complete issue context")
            
        if not requirements_extracted:
            violations.append("Agent has not extracted and documented all requirements")
            
        if not research_methodology:
            violations.append("Agent has not identified research methodology requirements (critical for preventing issue #115 failures)")
            
        if not validation_requirements:
            violations.append("Agent has not understood validation requirements")
            
        if not evidence_provided:
            violations.append("Agent has not provided evidence of understanding")
        
        # Generate recommendations
        if compliance_score < self.minimum_compliance_score:
            recommendations.append("Agent must re-read issue context and provide comprehensive understanding evidence")
            
        if not research_methodology:
            recommendations.append("CRITICAL: Agent must identify and plan for research methodology requirements to prevent issue #115 type failures")
            
        # Determine pass/fail
        passed = compliance_score >= self.minimum_compliance_score and len(violations) == 0
        
        result = ValidationResult(
            passed=passed,
            compliance_score=compliance_score,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        self._log_validation_result(agent_type, issue_number, result)
        
        return result
    
    def _get_issue_context(self, issue_number: int) -> Dict[str, Any]:
        """Get issue context for validation"""
        try:
            result = subprocess.run([
                "gh", "issue", "view", str(issue_number), "--json",
                "number,title,body,comments"
            ], capture_output=True, text=True, check=True)
            
            return json.loads(result.stdout)
        except Exception as e:
            return {"error": str(e), "number": issue_number}
    
    def _validate_full_issue_reading(self, agent_response: str, issue_context: Dict[str, Any]) -> bool:
        """Validate that agent has read the full issue"""
        issue_body = issue_context.get("body", "")
        issue_title = issue_context.get("title", "")
        
        # Check if agent response references key elements from issue
        title_referenced = issue_title.lower() in agent_response.lower()
        
        # Extract key phrases from issue body
        key_phrases = self._extract_key_phrases(issue_body)
        phrases_referenced = sum(1 for phrase in key_phrases if phrase.lower() in agent_response.lower())
        phrase_coverage = phrases_referenced / max(len(key_phrases), 1)
        
        # Check if comments are referenced
        comments_referenced = "comment" in agent_response.lower() or "discussion" in agent_response.lower()
        
        return title_referenced and phrase_coverage > 0.3
    
    def _validate_requirements_extraction(self, agent_response: str, issue_context: Dict[str, Any]) -> bool:
        """Validate that agent has extracted requirements"""
        issue_body = issue_context.get("body", "")
        
        # Look for requirement extraction indicators in agent response
        requirement_indicators = [
            "requirement", "must", "should", "shall", "deliverable",
            "acceptance criteria", "success criteria", "objective"
        ]
        
        indicators_found = sum(1 for indicator in requirement_indicators 
                             if indicator in agent_response.lower())
        
        # Check if agent lists or documents requirements
        has_requirement_list = (
            "requirements:" in agent_response.lower() or
            "criteria:" in agent_response.lower() or
            "deliverables:" in agent_response.lower() or
            re.search(r"[-*]\s+.*requirement", agent_response.lower())
        )
        
        return indicators_found >= 3 or has_requirement_list
    
    def _validate_research_methodology_compliance(self, agent_response: str, issue_context: Dict[str, Any]) -> bool:
        """Validate research methodology compliance (critical for issue #115 prevention)"""
        issue_body = issue_context.get("body", "")
        
        # Check if issue contains research methodology requirements
        research_indicators = [
            "literature review", "academic papers", "research into", "investigation of",
            "analysis of", "study of", "comparative", "benchmark", "survey"
        ]
        
        has_research_requirements = any(indicator in issue_body.lower() for indicator in research_indicators)
        
        if not has_research_requirements:
            return True  # No research methodology required
        
        # If research is required, check if agent acknowledges it
        research_acknowledged = any(indicator in agent_response.lower() for indicator in research_indicators)
        
        # Specific check for issue #115 type requirements
        if "algorithm" in issue_body.lower() and "research" in issue_body.lower():
            algorithm_research_acknowledged = (
                "algorithm" in agent_response.lower() and
                ("literature" in agent_response.lower() or "research" in agent_response.lower())
            )
            return research_acknowledged and algorithm_research_acknowledged
            
        return research_acknowledged
    
    def _validate_validation_requirements_understanding(self, agent_response: str, issue_context: Dict[str, Any]) -> bool:
        """Validate that validation requirements are understood"""
        issue_body = issue_context.get("body", "")
        
        # Check if issue contains validation requirements
        validation_indicators = [
            "validation", "testing", "performance", "success criteria",
            "acceptance criteria", "benchmark", "metric"
        ]
        
        has_validation_requirements = any(indicator in issue_body.lower() for indicator in validation_indicators)
        
        if not has_validation_requirements:
            return True  # No validation requirements
            
        # Check if agent acknowledges validation requirements
        validation_acknowledged = any(indicator in agent_response.lower() for indicator in validation_indicators)
        
        return validation_acknowledged
    
    def _validate_evidence_of_understanding(self, agent_response: str) -> bool:
        """Validate that agent provides evidence of understanding"""
        evidence_indicators = [
            "I understand", "I will", "I have analyzed", "I have identified",
            "requirements include", "approach will be", "plan is to",
            "analysis shows", "identified the following"
        ]
        
        evidence_count = sum(1 for indicator in evidence_indicators 
                           if indicator.lower() in agent_response.lower())
        
        # Check for structured responses
        has_structure = (
            re.search(r"##?\s+\w+", agent_response) or  # Headers
            re.search(r"[-*]\s+", agent_response) or   # Bullet points
            ":" in agent_response  # Structured content
        )
        
        return evidence_count >= 2 or has_structure
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for validation"""
        # Simple key phrase extraction
        sentences = re.split(r'[.!?]\s+', text)
        key_phrases = []
        
        for sentence in sentences[:10]:  # First 10 sentences
            words = sentence.split()
            if len(words) >= 3:
                # Extract noun phrases and important terms
                important_words = [w for w in words if len(w) > 4 and w.isalpha()]
                if important_words:
                    key_phrases.extend(important_words[:2])  # First 2 important words
        
        return list(set(key_phrases))  # Remove duplicates
    
    def _log_validation_result(self, agent_type: str, issue_number: int, result: ValidationResult):
        """Log validation result for audit trail"""
        try:
            # Load existing log
            if os.path.exists(self.validation_log_file):
                with open(self.validation_log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {"validation_results": []}
            
            # Add new result
            log_entry = {
                "agent_type": agent_type,
                "issue_number": issue_number,
                "validation_result": result.to_dict()
            }
            
            log_data["validation_results"].append(log_entry)
            
            # Save updated log
            with open(self.validation_log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error logging validation result: {e}")
    
    def generate_validation_report(self, agent_type: str, issue_number: int, result: ValidationResult) -> str:
        """Generate validation report for agent"""
        report_parts = []
        
        report_parts.append("# CONTEXT VALIDATION REPORT")
        report_parts.append(f"**Agent**: {agent_type}")
        report_parts.append(f"**Issue**: #{issue_number}")
        report_parts.append(f"**Validation Status**: {'✅ PASSED' if result.passed else '❌ FAILED'}")
        report_parts.append(f"**Compliance Score**: {result.compliance_score:.1%}")
        report_parts.append("")
        
        if result.violations:
            report_parts.append("## ❌ CRITICAL VIOLATIONS")
            for violation in result.violations:
                report_parts.append(f"- {violation}")
            report_parts.append("")
        
        if result.warnings:
            report_parts.append("## ⚠️ WARNINGS")
            for warning in result.warnings:
                report_parts.append(f"- {warning}")
            report_parts.append("")
        
        if result.recommendations:
            report_parts.append("## 📋 RECOMMENDATIONS")
            for rec in result.recommendations:
                report_parts.append(f"- {rec}")
            report_parts.append("")
        
        if not result.passed:
            report_parts.append("## ⛔ ACTION REQUIRED")
            report_parts.append("Agent must address violations before proceeding with work.")
            report_parts.append("Please provide evidence of:")
            report_parts.append("1. Complete issue context reading and understanding")
            report_parts.append("2. Comprehensive requirements extraction")
            report_parts.append("3. Research methodology compliance (if applicable)")
            report_parts.append("4. Validation requirements understanding")
            report_parts.append("")
        
        return "\n".join(report_parts)
    
    def create_emergency_prevention_checkpoint(self, issue_number: int) -> Dict[str, Any]:
        """Create checkpoint to prevent emergency context failures like issue #145"""
        checkpoint = {
            "checkpoint_type": "emergency_prevention",
            "issue_number": issue_number,
            "timestamp": datetime.now().isoformat(),
            "prevention_measures": {
                "full_issue_context_required": True,
                "research_methodology_validation": True,
                "validation_requirements_verification": True,
                "evidence_documentation_required": True
            },
            "reference_emergency": "issue_145_context_failure_prevention",
            "critical_validations": self.critical_requirements,
            "minimum_compliance_score": self.minimum_compliance_score
        }
        
        # Save checkpoint
        checkpoint_file = os.path.join(self.knowledge_base_path, "checkpoints", f"emergency_prevention_issue_{issue_number}.json")
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
        return checkpoint

def validate_agent_work(agent_type: str, issue_number: int, agent_response: str) -> Tuple[bool, str]:
    """
    Main validation function for agent work
    Returns (passed, report) tuple
    """
    enforcer = ContextValidationEnforcer()
    result = enforcer.validate_agent_context_consumption(agent_type, issue_number, agent_response)
    report = enforcer.generate_validation_report(agent_type, issue_number, result)
    
    # Create emergency prevention checkpoint if needed
    if not result.passed:
        enforcer.create_emergency_prevention_checkpoint(issue_number)
    
    return result.passed, report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Context Validation Enforcer")
    parser.add_argument("--agent", type=str, help="Agent type")
    parser.add_argument("--issue", type=int, help="Issue number")
    parser.add_argument("--response", type=str, help="Agent response file to validate")
    parser.add_argument("--test", action="store_true", help="Run validation test")
    
    args = parser.parse_args()
    
    if args.test:
        # Test validation with sample data
        sample_response = """
        I understand the issue requires implementing a context optimization algorithm.
        Requirements include: multi-factor relevance scoring, agent-specific filtering,
        performance targets under 200ms, and integration with existing systems.
        I have analyzed the research methodology requirements including literature review
        and comparative analysis. The validation requirements include performance testing
        and benchmarking against academic alternatives.
        """
        
        passed, report = validate_agent_work("rif-implementer", 115, sample_response)
        print(report)
        print(f"\nValidation Result: {'PASSED' if passed else 'FAILED'}")
        
    elif args.agent and args.issue and args.response:
        # Validate specific agent response
        with open(args.response, 'r') as f:
            response_text = f.read()
            
        passed, report = validate_agent_work(args.agent, args.issue, response_text)
        print(report)
        
    else:
        parser.print_help()