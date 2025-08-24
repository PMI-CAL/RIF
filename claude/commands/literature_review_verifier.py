#!/usr/bin/env python3
"""
Literature Review Verification Component
Part of Enhanced Context Verification System

Verifies that research issues requiring literature review
have actual academic evidence before being marked complete.
"""

import json
import os
import re
import subprocess
from typing import Dict, List, Any, Optional

class LiteratureReviewVerifier:
    def __init__(self):
        self.academic_evidence_patterns = [
            r"academic\s+sources?",
            r"papers?\s+analyzed",
            r"systematic\s+review",
            r"literature\s+review",
            r"peer[-\s]?reviewed",
            r"academic\s+research",
            r"scholarly\s+articles?",
            r"research\s+papers?",
            r"citations?",
            r"bibliography",
            r"doi:",
            r"arxiv:",
            r"ieee\s+xplore",
            r"acm\s+digital"
        ]
        
        self.minimum_evidence_threshold = {
            "academic_sources": 5,
            "industry_tools": 3,
            "evidence_indicators": 3
        }
    
    def verify_literature_review_completion(self, issue_number: int) -> Dict[str, Any]:
        """Verify literature review completion for research issues"""
        
        # Check if issue requires literature review
        issue_data = self._get_issue_data(issue_number)
        if not self._requires_literature_review(issue_data["body"]):
            return {"required": False, "status": "not_applicable"}
        
        # Search for evidence
        evidence = self._find_literature_evidence(issue_number)
        
        # Validate evidence quality
        validation = self._validate_evidence_quality(evidence)
        
        return {
            "issue_number": issue_number,
            "required": True,
            "evidence_found": evidence,
            "validation_result": validation,
            "status": "compliant" if validation["sufficient"] else "non_compliant",
            "missing_requirements": validation.get("missing", [])
        }
    
    def _get_issue_data(self, issue_number: int) -> Dict[str, Any]:
        """Get issue data from GitHub"""
        try:
            cmd = ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            return {"title": "", "body": "", "labels": []}
    
    def _requires_literature_review(self, body: str) -> bool:
        """Check if issue body indicates literature review requirement"""
        indicators = [
            "literature review",
            "academic papers",
            "research methodology",
            "academic research",
            "scholarly sources",
            "research deliverable"
        ]
        
        body_lower = body.lower()
        return any(indicator in body_lower for indicator in indicators)
    
    def _find_literature_evidence(self, issue_number: int) -> Dict[str, Any]:
        """Find literature review evidence in knowledge base"""
        
        evidence_locations = [
            f"/Users/cal/DEV/RIF/knowledge/checkpoints/issue-{issue_number}-*",
            f"/Users/cal/DEV/RIF/knowledge/learning/issue-{issue_number}-*",
            f"/Users/cal/DEV/RIF/knowledge/research/*{issue_number}*",
            f"/Users/cal/DEV/RIF/*{issue_number}*RESEARCH*",
            f"/Users/cal/DEV/RIF/*{issue_number}*LITERATURE*"
        ]
        
        found_evidence = {
            "files_with_evidence": [],
            "academic_indicators": 0,
            "source_count_estimates": 0,
            "evidence_quality": "insufficient"
        }
        
        for pattern in evidence_locations:
            try:
                cmd = f"find /Users/cal/DEV/RIF -path '{pattern}' -type f 2>/dev/null"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.stdout.strip():
                    files = result.stdout.strip().split('\n')
                    for file_path in files:
                        if os.path.exists(file_path):
                            evidence_data = self._analyze_file_for_literature_evidence(file_path)
                            if evidence_data["has_evidence"]:
                                found_evidence["files_with_evidence"].append({
                                    "file": file_path,
                                    "evidence": evidence_data
                                })
                                found_evidence["academic_indicators"] += evidence_data["indicator_count"]
                                found_evidence["source_count_estimates"] += evidence_data["estimated_sources"]
                                
            except Exception as e:
                continue
        
        # Determine evidence quality
        if found_evidence["academic_indicators"] >= 5 and found_evidence["source_count_estimates"] >= 10:
            found_evidence["evidence_quality"] = "sufficient"
        elif found_evidence["academic_indicators"] >= 3 and found_evidence["source_count_estimates"] >= 5:
            found_evidence["evidence_quality"] = "partial"
        else:
            found_evidence["evidence_quality"] = "insufficient"
        
        return found_evidence
    
    def _analyze_file_for_literature_evidence(self, file_path: str) -> Dict[str, Any]:
        """Analyze file for literature review evidence"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content_lower = content.lower()
            
            evidence_data = {
                "has_evidence": False,
                "indicator_count": 0,
                "estimated_sources": 0,
                "evidence_types": []
            }
            
            # Check for academic patterns
            for pattern in self.academic_evidence_patterns:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                if matches > 0:
                    evidence_data["has_evidence"] = True
                    evidence_data["indicator_count"] += matches
                    evidence_data["evidence_types"].append(pattern)
            
            # Estimate source count
            source_indicators = [
                r"\d+\+?\s+(?:academic\s+)?(?:sources?|papers?)",
                r"\d+\+?\s+(?:tools?\s+)?analyzed",
                r"\d+\+?\s+documents?"
            ]
            
            for pattern in source_indicators:
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    # Extract numbers
                    numbers = re.findall(r'\d+', match)
                    if numbers:
                        evidence_data["estimated_sources"] += int(numbers[0])
            
            return evidence_data
            
        except Exception as e:
            return {"has_evidence": False, "indicator_count": 0, "estimated_sources": 0, "evidence_types": []}
    
    def _validate_evidence_quality(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality of literature review evidence"""
        
        validation = {
            "sufficient": False,
            "score": 0,
            "criteria_met": [],
            "criteria_failed": [],
            "missing": []
        }
        
        # Check academic indicators threshold
        if evidence["academic_indicators"] >= self.minimum_evidence_threshold["academic_sources"]:
            validation["criteria_met"].append("sufficient_academic_indicators")
            validation["score"] += 3
        else:
            validation["criteria_failed"].append("insufficient_academic_indicators")
            validation["missing"].append(f"Need {self.minimum_evidence_threshold['academic_sources']} academic indicators, found {evidence['academic_indicators']}")
        
        # Check source count threshold
        if evidence["source_count_estimates"] >= 10:
            validation["criteria_met"].append("sufficient_source_count")
            validation["score"] += 3
        else:
            validation["criteria_failed"].append("insufficient_source_count")
            validation["missing"].append(f"Need 10+ sources, estimated {evidence['source_count_estimates']}")
        
        # Check evidence file presence
        if len(evidence["files_with_evidence"]) >= 2:
            validation["criteria_met"].append("multiple_evidence_files")
            validation["score"] += 2
        else:
            validation["criteria_failed"].append("insufficient_evidence_files")
            validation["missing"].append(f"Need 2+ evidence files, found {len(evidence['files_with_evidence'])}")
        
        # Overall sufficiency (score >= 6 out of 8)
        validation["sufficient"] = validation["score"] >= 6
        
        return validation


def verify_issue_literature_review(issue_number: int) -> Dict[str, Any]:
    """Verify literature review for specific issue"""
    verifier = LiteratureReviewVerifier()
    return verifier.verify_literature_review_completion(issue_number)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        issue_num = int(sys.argv[1])
        result = verify_issue_literature_review(issue_num)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python literature_review_verifier.py <issue_number>")
