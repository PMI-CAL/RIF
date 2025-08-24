#!/usr/bin/env python3
"""
Requirement Extraction Verification Component
Part of Enhanced Context Verification System

Verifies that agents properly extract and address all requirements
from issue descriptions before implementation.
"""

import json
import re
import subprocess
from typing import Dict, List, Any, Set

class RequirementExtractionVerifier:
    def __init__(self):
        self.requirement_patterns = [
            r"must\s+(?:be|have|include|support|provide|implement)",
            r"shall\s+(?:be|have|include|support|provide|implement)",
            r"should\s+(?:be|have|include|support|provide|implement)",
            r"required?\s+to\s+(?:be|have|include|support|provide|implement)",
            r"needs?\s+to\s+(?:be|have|include|support|provide|implement)",
            r"deliverable:",
            r"success\s+criteria:",
            r"acceptance\s+criteria:",
            r"objective:",
            r"goal:",
            r"requirement:",
            r"constraint:",
            r"specification:"
        ]
        
        self.deliverable_patterns = [
            r"deliverable.*?:",
            r"output.*?:",
            r"artifact.*?:",
            r"document.*?:",
            r"framework.*?:",
            r"system.*?:",
            r"analysis.*?:",
            r"report.*?:"
        ]
    
    def verify_requirement_extraction(self, issue_number: int) -> Dict[str, Any]:
        """Verify comprehensive requirement extraction for issue"""
        
        # Get issue data
        issue_data = self._get_issue_data(issue_number)
        
        # Extract requirements from issue body
        extracted_requirements = self._extract_requirements(issue_data["body"])
        
        # Check implementation artifacts for requirement coverage
        coverage_analysis = self._analyze_requirement_coverage(issue_number, extracted_requirements)
        
        # Validate completeness
        completeness_check = self._validate_requirement_completeness(extracted_requirements, coverage_analysis)
        
        return {
            "issue_number": issue_number,
            "requirements_extracted": extracted_requirements,
            "coverage_analysis": coverage_analysis,
            "completeness_check": completeness_check,
            "status": "compliant" if completeness_check["complete"] else "non_compliant"
        }
    
    def _get_issue_data(self, issue_number: int) -> Dict[str, Any]:
        """Get issue data from GitHub"""
        try:
            cmd = ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception:
            return {"title": "", "body": "", "labels": []}
    
    def _extract_requirements(self, body: str) -> Dict[str, List[str]]:
        """Extract all requirements from issue body"""
        
        requirements = {
            "functional_requirements": [],
            "deliverables": [],
            "success_criteria": [],
            "constraints": [],
            "objectives": []
        }
        
        lines = body.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Identify section headers
            if re.search(r"success\s+criteria", line, re.IGNORECASE):
                current_section = "success_criteria"
                continue
            elif re.search(r"acceptance\s+criteria", line, re.IGNORECASE):
                current_section = "success_criteria"
                continue
            elif re.search(r"deliverable", line, re.IGNORECASE):
                current_section = "deliverables"
                continue
            elif re.search(r"objective|goal", line, re.IGNORECASE):
                current_section = "objectives"
                continue
            elif re.search(r"constraint|requirement", line, re.IGNORECASE):
                current_section = "constraints"
                continue
            elif line.startswith('#'):
                current_section = None
            
            # Extract requirements based on patterns
            for pattern in self.requirement_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if current_section:
                        requirements[current_section].append(line)
                    else:
                        requirements["functional_requirements"].append(line)
        
        # Extract deliverables separately
        for pattern in self.deliverable_patterns:
            matches = re.findall(f"{pattern}.*", body, re.IGNORECASE | re.MULTILINE)
            requirements["deliverables"].extend(matches)
        
        return requirements
    
    def _analyze_requirement_coverage(self, issue_number: int, requirements: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze how well implementation covers extracted requirements"""
        
        # Find implementation artifacts
        artifact_files = self._find_implementation_artifacts(issue_number)
        
        coverage = {
            "artifacts_found": len(artifact_files),
            "requirement_coverage": {},
            "uncovered_requirements": [],
            "coverage_percentage": 0
        }
        
        total_requirements = sum(len(reqs) for reqs in requirements.values())
        covered_count = 0
        
        # Analyze coverage for each requirement category
        for category, req_list in requirements.items():
            coverage["requirement_coverage"][category] = {
                "total": len(req_list),
                "covered": 0,
                "partially_covered": 0,
                "uncovered": []
            }
            
            for requirement in req_list:
                coverage_score = self._check_requirement_in_artifacts(requirement, artifact_files)
                
                if coverage_score >= 0.8:
                    coverage["requirement_coverage"][category]["covered"] += 1
                    covered_count += 1
                elif coverage_score >= 0.3:
                    coverage["requirement_coverage"][category]["partially_covered"] += 1
                    covered_count += 0.5
                else:
                    coverage["requirement_coverage"][category]["uncovered"].append(requirement)
                    coverage["uncovered_requirements"].append(requirement)
        
        coverage["coverage_percentage"] = (covered_count / total_requirements * 100) if total_requirements > 0 else 0
        
        return coverage
    
    def _find_implementation_artifacts(self, issue_number: int) -> List[str]:
        """Find implementation artifacts for issue"""
        
        search_patterns = [
            f"*{issue_number}*",
            f"issue-{issue_number}-*",
            f"issue_{issue_number}_*"
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
    
    def _check_requirement_in_artifacts(self, requirement: str, artifacts: List[str]) -> float:
        """Check how well a requirement is covered in artifacts"""
        
        # Extract key terms from requirement
        key_terms = self._extract_key_terms(requirement)
        
        if not key_terms:
            return 0.0
        
        total_score = 0.0
        artifact_count = 0
        
        for artifact_file in artifacts:
            try:
                with open(artifact_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                artifact_count += 1
                term_matches = 0
                
                for term in key_terms:
                    if term.lower() in content:
                        term_matches += 1
                
                artifact_score = term_matches / len(key_terms)
                total_score += artifact_score
                
            except Exception:
                continue
        
        return (total_score / artifact_count) if artifact_count > 0 else 0.0
    
    def _extract_key_terms(self, requirement: str) -> List[str]:
        """Extract key terms from requirement text"""
        
        # Remove common words and extract meaningful terms
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'must', 'should', 'shall', 'be', 'have', 'include', 'support', 'provide', 'implement'}
        
        words = re.findall(r'\b\w{3,}\b', requirement.lower())
        key_terms = [word for word in words if word not in common_words]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    def _validate_requirement_completeness(self, requirements: Dict[str, List[str]], coverage: Dict[str, Any]) -> Dict[str, Any]:
        """Validate completeness of requirement coverage"""
        
        completeness = {
            "complete": False,
            "score": 0,
            "thresholds": {
                "minimum_coverage": 80,
                "minimum_artifacts": 2,
                "minimum_deliverables": 1
            },
            "assessment": {
                "coverage_sufficient": False,
                "artifacts_sufficient": False,
                "deliverables_addressed": False
            },
            "recommendations": []
        }
        
        # Check coverage threshold
        if coverage["coverage_percentage"] >= completeness["thresholds"]["minimum_coverage"]:
            completeness["assessment"]["coverage_sufficient"] = True
            completeness["score"] += 4
        else:
            completeness["recommendations"].append(f"Increase requirement coverage from {coverage['coverage_percentage']:.1f}% to 80%")
        
        # Check artifact count
        if coverage["artifacts_found"] >= completeness["thresholds"]["minimum_artifacts"]:
            completeness["assessment"]["artifacts_sufficient"] = True
            completeness["score"] += 3
        else:
            completeness["recommendations"].append(f"Create more implementation artifacts (found {coverage['artifacts_found']}, need {completeness['thresholds']['minimum_artifacts']})")
        
        # Check deliverables
        deliverable_coverage = coverage["requirement_coverage"].get("deliverables", {})
        if deliverable_coverage.get("covered", 0) >= completeness["thresholds"]["minimum_deliverables"]:
            completeness["assessment"]["deliverables_addressed"] = True
            completeness["score"] += 3
        else:
            completeness["recommendations"].append("Address required deliverables in implementation")
        
        # Overall completeness (score >= 8 out of 10)
        completeness["complete"] = completeness["score"] >= 8
        
        return completeness


def verify_issue_requirements(issue_number: int) -> Dict[str, Any]:
    """Verify requirement extraction for specific issue"""
    verifier = RequirementExtractionVerifier()
    return verifier.verify_requirement_extraction(issue_number)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        issue_num = int(sys.argv[1])
        result = verify_issue_requirements(issue_num)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python requirement_extraction_verifier.py <issue_number>")
