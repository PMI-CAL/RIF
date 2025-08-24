#!/usr/bin/env python3
"""
Success Criteria Validation Component
Part of Enhanced Context Verification System

Validates that all success criteria from issues are properly
addressed and validated before issue completion.
"""

import json
import re
import subprocess
from typing import Dict, List, Any

class SuccessCriteriaValidator:
    def __init__(self):
        self.criteria_patterns = [
            r"\[\s*[\]x]\s*",  # Checkbox patterns
            r"success\s+criteria:",
            r"acceptance\s+criteria:",
            r"must\s+(?:achieve|complete|implement|provide)",
            r"target:\s*",
            r"threshold:\s*",
            r"requirement:\s*"
        ]
    
    def validate_success_criteria(self, issue_number: int) -> Dict[str, Any]:
        """Validate success criteria completion for issue"""
        
        # Get issue data
        issue_data = self._get_issue_data(issue_number)
        
        # Extract success criteria
        criteria = self._extract_success_criteria(issue_data["body"])
        
        # Check validation evidence
        validation_evidence = self._find_validation_evidence(issue_number, criteria)
        
        # Assess criteria completion
        completion_assessment = self._assess_criteria_completion(criteria, validation_evidence)
        
        return {
            "issue_number": issue_number,
            "criteria_extracted": criteria,
            "validation_evidence": validation_evidence,
            "completion_assessment": completion_assessment,
            "status": "compliant" if completion_assessment["all_validated"] else "non_compliant"
        }
    
    def _get_issue_data(self, issue_number: int) -> Dict[str, Any]:
        """Get issue data from GitHub"""
        try:
            cmd = ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception:
            return {"title": "", "body": "", "labels": []}
    
    def _extract_success_criteria(self, body: str) -> List[Dict[str, Any]]:
        """Extract success criteria from issue body"""
        
        criteria = []
        lines = body.split('\n')
        in_criteria_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for criteria section headers
            if re.search(r"success\s+criteria|acceptance\s+criteria", line, re.IGNORECASE):
                in_criteria_section = True
                continue
            elif line.startswith('#') and in_criteria_section:
                in_criteria_section = False
                continue
            
            # Extract criteria items
            if in_criteria_section or any(re.search(pattern, line, re.IGNORECASE) for pattern in self.criteria_patterns):
                if line and not line.startswith('#'):
                    criterion = {
                        "text": line,
                        "line_number": i + 1,
                        "type": self._classify_criterion(line),
                        "measurable": self._is_measurable(line),
                        "checkbox": self._has_checkbox(line)
                    }
                    criteria.append(criterion)
        
        return criteria
    
    def _classify_criterion(self, text: str) -> str:
        """Classify the type of success criterion"""
        
        text_lower = text.lower()
        
        if "test" in text_lower or "coverage" in text_lower:
            return "testing"
        elif "performance" in text_lower or "speed" in text_lower or "latency" in text_lower:
            return "performance"
        elif "documentation" in text_lower or "document" in text_lower:
            return "documentation"
        elif "integration" in text_lower or "integrate" in text_lower:
            return "integration"
        elif "deliverable" in text_lower or "artifact" in text_lower:
            return "deliverable"
        else:
            return "functional"
    
    def _is_measurable(self, text: str) -> bool:
        """Check if criterion contains measurable targets"""
        
        measurable_patterns = [
            r"\d+\s*%",  # Percentages
            r"\d+\s*(?:seconds?|minutes?|hours?)",  # Time
            r"\d+\s*(?:mb|gb|kb)",  # Size
            r"[<>]=?\s*\d+",  # Comparison operators
            r"at\s+least\s+\d+",
            r"more\s+than\s+\d+",
            r"\d+\+",  # Number plus
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in measurable_patterns)
    
    def _has_checkbox(self, text: str) -> bool:
        """Check if criterion has checkbox format"""
        return bool(re.search(r"\[\s*[\]x]\s*", text))
    
    def _find_validation_evidence(self, issue_number: int, criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find evidence that criteria have been validated"""
        
        evidence = {
            "validation_files": [],
            "test_results": [],
            "performance_metrics": [],
            "documentation_artifacts": [],
            "evidence_by_criterion": {}
        }
        
        # Find validation artifacts
        validation_files = self._find_validation_artifacts(issue_number)
        evidence["validation_files"] = validation_files
        
        # Analyze evidence for each criterion
        for criterion in criteria:
            criterion_evidence = self._analyze_criterion_evidence(criterion, validation_files)
            evidence["evidence_by_criterion"][criterion["text"]] = criterion_evidence
        
        return evidence
    
    def _find_validation_artifacts(self, issue_number: int) -> List[str]:
        """Find validation artifacts for issue"""
        
        search_patterns = [
            f"validation*{issue_number}*",
            f"test*{issue_number}*", 
            f"*{issue_number}*validation*",
            f"*{issue_number}*test*",
            f"*{issue_number}*results*"
        ]
        
        artifacts = []
        for pattern in search_patterns:
            try:
                cmd = f"find /Users/cal/DEV/RIF -name '{pattern}' -type f 2>/dev/null"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.stdout.strip():
                    artifacts.extend(result.stdout.strip().split('\n'))
                    
            except Exception:
                continue
        
        return artifacts
    
    def _analyze_criterion_evidence(self, criterion: Dict[str, Any], validation_files: List[str]) -> Dict[str, Any]:
        """Analyze evidence for specific criterion"""
        
        evidence = {
            "evidence_found": False,
            "evidence_files": [],
            "evidence_strength": 0.0,
            "validation_type": criterion["type"]
        }
        
        # Extract key terms from criterion
        key_terms = self._extract_key_terms_from_criterion(criterion["text"])
        
        for file_path in validation_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                term_matches = sum(1 for term in key_terms if term.lower() in content)
                match_strength = term_matches / len(key_terms) if key_terms else 0
                
                if match_strength > 0.3:  # Minimum 30% term overlap
                    evidence["evidence_found"] = True
                    evidence["evidence_files"].append(file_path)
                    evidence["evidence_strength"] = max(evidence["evidence_strength"], match_strength)
                    
            except Exception:
                continue
        
        return evidence
    
    def _extract_key_terms_from_criterion(self, text: str) -> List[str]:
        """Extract key terms from criterion text"""
        
        # Remove checkbox and common words
        text = re.sub(r"\[\s*[\]x]\s*", "", text)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'must', 'should', 'be', 'have'}
        
        words = re.findall(r'\b\w{3,}\b', text.lower())
        key_terms = [word for word in words if word not in common_words]
        
        return key_terms[:8]  # Limit to top 8 terms
    
    def _assess_criteria_completion(self, criteria: List[Dict[str, Any]], validation_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Assess completion of all criteria"""
        
        assessment = {
            "total_criteria": len(criteria),
            "validated_criteria": 0,
            "partially_validated": 0,
            "unvalidated_criteria": [],
            "validation_percentage": 0.0,
            "all_validated": False,
            "quality_score": 0
        }
        
        for criterion in criteria:
            evidence = validation_evidence["evidence_by_criterion"].get(criterion["text"], {})
            
            if evidence.get("evidence_found", False) and evidence.get("evidence_strength", 0) >= 0.6:
                assessment["validated_criteria"] += 1
                assessment["quality_score"] += 2
            elif evidence.get("evidence_found", False) and evidence.get("evidence_strength", 0) >= 0.3:
                assessment["partially_validated"] += 1
                assessment["quality_score"] += 1
            else:
                assessment["unvalidated_criteria"].append(criterion["text"])
        
        # Calculate percentages
        if assessment["total_criteria"] > 0:
            assessment["validation_percentage"] = (assessment["validated_criteria"] / assessment["total_criteria"]) * 100
            assessment["all_validated"] = assessment["validation_percentage"] >= 80  # 80% threshold
        
        return assessment


def validate_issue_success_criteria(issue_number: int) -> Dict[str, Any]:
    """Validate success criteria for specific issue"""
    validator = SuccessCriteriaValidator()
    return validator.validate_success_criteria(issue_number)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        issue_num = int(sys.argv[1])
        result = validate_issue_success_criteria(issue_num)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python success_criteria_validator.py <issue_number>")
