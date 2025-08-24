#!/usr/bin/env python3
"""
Enhanced Context Consumption Verification System
Addresses Issue #145: Systematic agent context failures

Implements comprehensive verification that agents are fully consuming
and properly utilizing issue context before proceeding with work.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class EnhancedContextVerification:
    def __init__(self):
        self.verification_config = {
            "system_name": "Enhanced Context Consumption Verification System",
            "version": "1.0.0",
            "activation_timestamp": datetime.utcnow().isoformat() + "Z",
            "trigger_issue": 145,
            "verification_levels": ["basic", "comprehensive", "critical"],
            "requirements": {
                "literature_review_verification": True,
                "requirement_extraction_verification": True,
                "success_criteria_validation": True,
                "deliverable_tracking": True,
                "evidence_requirement_enforcement": True
            }
        }
    
    def deploy_verification_system(self) -> Dict[str, Any]:
        """Deploy complete enhanced context verification system"""
        
        print("🔍 DEPLOYING ENHANCED CONTEXT VERIFICATION SYSTEM 🔍")
        print()
        
        deployment_results = {
            "deployment_timestamp": datetime.utcnow().isoformat() + "Z",
            "components_deployed": [],
            "verification_rules_active": [],
            "integration_status": {}
        }
        
        # Component 1: Literature Review Verifier
        self._deploy_literature_review_verifier(deployment_results)
        
        # Component 2: Requirement Extraction Verifier
        self._deploy_requirement_extraction_verifier(deployment_results)
        
        # Component 3: Success Criteria Validator
        self._deploy_success_criteria_validator(deployment_results)
        
        # Component 4: Evidence Requirement Enforcer
        self._deploy_evidence_requirement_enforcer(deployment_results)
        
        # Component 5: Context Consumption Monitor
        self._deploy_context_consumption_monitor(deployment_results)
        
        # Component 6: Integration with RIF Workflow
        self._integrate_with_rif_workflow(deployment_results)
        
        # Save deployment configuration
        self._save_deployment_config(deployment_results)
        
        return deployment_results
    
    def _deploy_literature_review_verifier(self, results: Dict[str, Any]) -> None:
        """Deploy literature review verification component"""
        
        print("Step 1: Deploying Literature Review Verifier...")
        
        verifier_code = '''#!/usr/bin/env python3
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
            r"academic\\s+sources?",
            r"papers?\\s+analyzed",
            r"systematic\\s+review",
            r"literature\\s+review",
            r"peer[-\\s]?reviewed",
            r"academic\\s+research",
            r"scholarly\\s+articles?",
            r"research\\s+papers?",
            r"citations?",
            r"bibliography",
            r"doi:",
            r"arxiv:",
            r"ieee\\s+xplore",
            r"acm\\s+digital"
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
                    files = result.stdout.strip().split('\\n')
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
                r"\\d+\\+?\\s+(?:academic\\s+)?(?:sources?|papers?)",
                r"\\d+\\+?\\s+(?:tools?\\s+)?analyzed",
                r"\\d+\\+?\\s+documents?"
            ]
            
            for pattern in source_indicators:
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    # Extract numbers
                    numbers = re.findall(r'\\d+', match)
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
'''
        
        verifier_file = "/Users/cal/DEV/RIF/claude/commands/literature_review_verifier.py"
        
        try:
            with open(verifier_file, 'w', encoding='utf-8') as f:
                f.write(verifier_code)
            
            os.chmod(verifier_file, 0o755)
            
            print(f"  ✅ Literature Review Verifier deployed: {verifier_file}")
            
            results["components_deployed"].append({
                "component": "literature_review_verifier",
                "file": verifier_file,
                "status": "deployed"
            })
            
            results["verification_rules_active"].append("literature_review_mandatory")
            
        except Exception as e:
            print(f"  ❌ Failed to deploy Literature Review Verifier: {e}")
        
        print()
    
    def _deploy_requirement_extraction_verifier(self, results: Dict[str, Any]) -> None:
        """Deploy requirement extraction verification component"""
        
        print("Step 2: Deploying Requirement Extraction Verifier...")
        
        verifier_code = '''#!/usr/bin/env python3
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
            r"must\\s+(?:be|have|include|support|provide|implement)",
            r"shall\\s+(?:be|have|include|support|provide|implement)",
            r"should\\s+(?:be|have|include|support|provide|implement)",
            r"required?\\s+to\\s+(?:be|have|include|support|provide|implement)",
            r"needs?\\s+to\\s+(?:be|have|include|support|provide|implement)",
            r"deliverable:",
            r"success\\s+criteria:",
            r"acceptance\\s+criteria:",
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
        
        lines = body.split('\\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Identify section headers
            if re.search(r"success\\s+criteria", line, re.IGNORECASE):
                current_section = "success_criteria"
                continue
            elif re.search(r"acceptance\\s+criteria", line, re.IGNORECASE):
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
                    artifacts.extend(result.stdout.strip().split('\\n'))
                    
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
        
        words = re.findall(r'\\b\\w{3,}\\b', requirement.lower())
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
'''
        
        verifier_file = "/Users/cal/DEV/RIF/claude/commands/requirement_extraction_verifier.py"
        
        try:
            with open(verifier_file, 'w', encoding='utf-8') as f:
                f.write(verifier_code)
            
            os.chmod(verifier_file, 0o755)
            
            print(f"  ✅ Requirement Extraction Verifier deployed: {verifier_file}")
            
            results["components_deployed"].append({
                "component": "requirement_extraction_verifier",
                "file": verifier_file,
                "status": "deployed"
            })
            
            results["verification_rules_active"].append("requirement_extraction_mandatory")
            
        except Exception as e:
            print(f"  ❌ Failed to deploy Requirement Extraction Verifier: {e}")
        
        print()
    
    def _deploy_success_criteria_validator(self, results: Dict[str, Any]) -> None:
        """Deploy success criteria validation component"""
        
        print("Step 3: Deploying Success Criteria Validator...")
        
        validator_code = '''#!/usr/bin/env python3
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
            r"\\[\\s*[\\]x]\\s*",  # Checkbox patterns
            r"success\\s+criteria:",
            r"acceptance\\s+criteria:",
            r"must\\s+(?:achieve|complete|implement|provide)",
            r"target:\\s*",
            r"threshold:\\s*",
            r"requirement:\\s*"
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
        lines = body.split('\\n')
        in_criteria_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for criteria section headers
            if re.search(r"success\\s+criteria|acceptance\\s+criteria", line, re.IGNORECASE):
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
            r"\\d+\\s*%",  # Percentages
            r"\\d+\\s*(?:seconds?|minutes?|hours?)",  # Time
            r"\\d+\\s*(?:mb|gb|kb)",  # Size
            r"[<>]=?\\s*\\d+",  # Comparison operators
            r"at\\s+least\\s+\\d+",
            r"more\\s+than\\s+\\d+",
            r"\\d+\\+",  # Number plus
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in measurable_patterns)
    
    def _has_checkbox(self, text: str) -> bool:
        """Check if criterion has checkbox format"""
        return bool(re.search(r"\\[\\s*[\\]x]\\s*", text))
    
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
                    artifacts.extend(result.stdout.strip().split('\\n'))
                    
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
        text = re.sub(r"\\[\\s*[\\]x]\\s*", "", text)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'must', 'should', 'be', 'have'}
        
        words = re.findall(r'\\b\\w{3,}\\b', text.lower())
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
'''
        
        validator_file = "/Users/cal/DEV/RIF/claude/commands/success_criteria_validator.py"
        
        try:
            with open(validator_file, 'w', encoding='utf-8') as f:
                f.write(validator_code)
            
            os.chmod(validator_file, 0o755)
            
            print(f"  ✅ Success Criteria Validator deployed: {validator_file}")
            
            results["components_deployed"].append({
                "component": "success_criteria_validator",
                "file": validator_file,
                "status": "deployed"
            })
            
            results["verification_rules_active"].append("success_criteria_validation_mandatory")
            
        except Exception as e:
            print(f"  ❌ Failed to deploy Success Criteria Validator: {e}")
        
        print()
    
    def _deploy_evidence_requirement_enforcer(self, results: Dict[str, Any]) -> None:
        """Deploy evidence requirement enforcement component"""
        
        print("Step 4: Deploying Evidence Requirement Enforcer...")
        
        enforcer_code = '''#!/usr/bin/env python3
"""
Evidence Requirement Enforcer Component
Part of Enhanced Context Verification System

Enforces that adequate evidence exists before allowing
issue state transitions to complete.
"""

import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

class EvidenceRequirementEnforcer:
    def __init__(self):
        self.evidence_requirements = {
            "research_issues": {
                "minimum_artifacts": 3,
                "required_types": ["research_findings", "methodology", "validation"],
                "literature_review_mandatory": True
            },
            "implementation_issues": {
                "minimum_artifacts": 2,
                "required_types": ["implementation", "testing"],
                "code_evidence_mandatory": True
            },
            "validation_issues": {
                "minimum_artifacts": 2,
                "required_types": ["test_results", "validation_report"],
                "quantitative_results_mandatory": True
            }
        }
    
    def enforce_evidence_requirements(self, issue_number: int, target_state: str) -> Dict[str, Any]:
        """Enforce evidence requirements for state transition"""
        
        # Get issue data
        issue_data = self._get_issue_data(issue_number)
        
        # Classify issue type
        issue_type = self._classify_issue(issue_data)
        
        # Find existing evidence
        evidence_inventory = self._inventory_evidence(issue_number)
        
        # Check requirements
        requirements_check = self._check_evidence_requirements(issue_type, evidence_inventory, target_state)
        
        # Enforcement decision
        enforcement_result = self._make_enforcement_decision(requirements_check, target_state)
        
        return {
            "issue_number": issue_number,
            "issue_type": issue_type,
            "target_state": target_state,
            "evidence_inventory": evidence_inventory,
            "requirements_check": requirements_check,
            "enforcement_result": enforcement_result,
            "transition_allowed": enforcement_result["allowed"]
        }
    
    def _get_issue_data(self, issue_number: int) -> Dict[str, Any]:
        """Get issue data from GitHub"""
        try:
            cmd = ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception:
            return {"title": "", "body": "", "labels": []}
    
    def _classify_issue(self, issue_data: Dict[str, Any]) -> str:
        """Classify issue type based on content"""
        
        body = issue_data.get("body", "").lower()
        title = issue_data.get("title", "").lower()
        content = f"{title} {body}"
        
        # Research issue indicators
        research_indicators = [
            "research", "literature review", "analysis", "methodology",
            "academic", "study", "investigation", "survey"
        ]
        
        # Implementation issue indicators  
        implementation_indicators = [
            "implement", "develop", "create", "build", "code",
            "development", "programming", "system", "feature"
        ]
        
        # Validation issue indicators
        validation_indicators = [
            "validation", "testing", "verify", "test", "quality",
            "benchmark", "performance", "compliance"
        ]
        
        research_score = sum(1 for indicator in research_indicators if indicator in content)
        implementation_score = sum(1 for indicator in implementation_indicators if indicator in content)
        validation_score = sum(1 for indicator in validation_indicators if indicator in content)
        
        if research_score >= implementation_score and research_score >= validation_score:
            return "research_issues"
        elif validation_score >= implementation_score:
            return "validation_issues"
        else:
            return "implementation_issues"
    
    def _inventory_evidence(self, issue_number: int) -> Dict[str, Any]:
        """Create inventory of existing evidence for issue"""
        
        inventory = {
            "total_artifacts": 0,
            "artifact_types": {},
            "evidence_files": [],
            "evidence_quality": {}
        }
        
        # Search for evidence files
        search_patterns = [
            f"*{issue_number}*",
            f"issue-{issue_number}-*",
            f"issue_{issue_number}_*"
        ]
        
        for pattern in search_patterns:
            try:
                cmd = f"find /Users/cal/DEV/RIF -name '{pattern}' -type f 2>/dev/null"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.stdout.strip():
                    files = result.stdout.strip().split('\\n')
                    for file_path in files:
                        if os.path.exists(file_path):
                            artifact_info = self._analyze_artifact(file_path)
                            inventory["evidence_files"].append({
                                "path": file_path,
                                "type": artifact_info["type"],
                                "quality": artifact_info["quality"],
                                "size": artifact_info["size"]
                            })
                            
                            artifact_type = artifact_info["type"]
                            inventory["artifact_types"][artifact_type] = inventory["artifact_types"].get(artifact_type, 0) + 1
                            
            except Exception:
                continue
        
        inventory["total_artifacts"] = len(inventory["evidence_files"])
        
        return inventory
    
    def _analyze_artifact(self, file_path: str) -> Dict[str, Any]:
        """Analyze individual artifact for type and quality"""
        
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path).lower()
            
            # Determine artifact type
            if "research" in file_name or "literature" in file_name or "analysis" in file_name:
                artifact_type = "research_findings"
            elif "test" in file_name or "validation" in file_name or "results" in file_name:
                artifact_type = "test_results"
            elif "implementation" in file_name or "complete" in file_name:
                artifact_type = "implementation"
            elif "methodology" in file_name or "method" in file_name:
                artifact_type = "methodology"
            elif file_name.endswith('.py'):
                artifact_type = "code"
            elif file_name.endswith('.md'):
                artifact_type = "documentation"
            elif file_name.endswith('.json'):
                artifact_type = "data"
            else:
                artifact_type = "unknown"
            
            # Assess quality based on size and content (basic heuristic)
            if file_size < 100:
                quality = "low"
            elif file_size < 5000:
                quality = "medium"
            else:
                quality = "high"
            
            # Try to read content for better quality assessment
            if file_path.endswith(('.json', '.md', '.py', '.txt')):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if len(content.split('\\n')) > 50 or len(content) > 2000:
                        quality = "high"
                    elif len(content.split('\\n')) > 20 or len(content) > 1000:
                        quality = "medium"
                        
                except Exception:
                    pass  # Keep existing quality assessment
            
            return {
                "type": artifact_type,
                "quality": quality,
                "size": file_size
            }
            
        except Exception:
            return {
                "type": "unknown",
                "quality": "low", 
                "size": 0
            }
    
    def _check_evidence_requirements(self, issue_type: str, evidence_inventory: Dict[str, Any], target_state: str) -> Dict[str, Any]:
        """Check if evidence meets requirements for issue type"""
        
        requirements = self.evidence_requirements.get(issue_type, self.evidence_requirements["implementation_issues"])
        
        check_result = {
            "issue_type": issue_type,
            "target_state": target_state,
            "requirements_met": {},
            "overall_compliance": False,
            "missing_requirements": [],
            "recommendations": []
        }
        
        # Check minimum artifacts
        artifact_count = evidence_inventory["total_artifacts"]
        min_artifacts = requirements["minimum_artifacts"]
        
        check_result["requirements_met"]["minimum_artifacts"] = artifact_count >= min_artifacts
        if not check_result["requirements_met"]["minimum_artifacts"]:
            check_result["missing_requirements"].append(f"Need {min_artifacts} artifacts, found {artifact_count}")
            check_result["recommendations"].append(f"Create {min_artifacts - artifact_count} additional evidence artifacts")
        
        # Check required types
        artifact_types = set(evidence_inventory["artifact_types"].keys())
        required_types = set(requirements["required_types"])
        missing_types = required_types - artifact_types
        
        check_result["requirements_met"]["required_types"] = len(missing_types) == 0
        if missing_types:
            check_result["missing_requirements"].append(f"Missing artifact types: {list(missing_types)}")
            check_result["recommendations"].append(f"Create artifacts of types: {list(missing_types)}")
        
        # Check special requirements
        if requirements.get("literature_review_mandatory", False) and issue_type == "research_issues":
            has_literature = any("research" in f["type"] for f in evidence_inventory["evidence_files"])
            check_result["requirements_met"]["literature_review"] = has_literature
            if not has_literature:
                check_result["missing_requirements"].append("Literature review evidence required")
                check_result["recommendations"].append("Complete and document literature review")
        
        # Overall compliance
        compliance_score = sum(1 for met in check_result["requirements_met"].values() if met)
        total_requirements = len(check_result["requirements_met"])
        check_result["overall_compliance"] = (compliance_score / total_requirements) >= 0.8 if total_requirements > 0 else False
        
        return check_result
    
    def _make_enforcement_decision(self, requirements_check: Dict[str, Any], target_state: str) -> Dict[str, Any]:
        """Make enforcement decision based on requirements check"""
        
        decision = {
            "allowed": False,
            "reason": "",
            "required_actions": [],
            "severity": "blocking"
        }
        
        if requirements_check["overall_compliance"]:
            decision["allowed"] = True
            decision["reason"] = "All evidence requirements met"
            decision["severity"] = "none"
        else:
            decision["reason"] = f"Evidence requirements not met: {len(requirements_check['missing_requirements'])} deficiencies"
            decision["required_actions"] = requirements_check["recommendations"]
            
            # Determine severity based on target state
            if target_state in ["state:complete", "state:validating"]:
                decision["severity"] = "blocking"
            elif target_state in ["state:implementing"]:
                decision["severity"] = "warning"
            else:
                decision["severity"] = "advisory"
        
        return decision
    
    def enforce_for_issue_transition(self, issue_number: int, from_state: str, to_state: str) -> bool:
        """Enforce evidence requirements for specific state transition"""
        
        enforcement_result = self.enforce_evidence_requirements(issue_number, to_state)
        
        if not enforcement_result["transition_allowed"]:
            # Block transition and add comment
            self._block_transition_with_comment(issue_number, enforcement_result)
            return False
        
        return True
    
    def _block_transition_with_comment(self, issue_number: int, enforcement_result: Dict[str, Any]) -> None:
        """Block transition and add explanatory comment"""
        
        try:
            comment = f"""🚫 **EVIDENCE REQUIREMENT ENFORCER** 🚫

**Transition Blocked**: State transition to `{enforcement_result["target_state"]}` blocked due to insufficient evidence

**Issue Type**: {enforcement_result["issue_type"]}
**Evidence Found**: {enforcement_result["evidence_inventory"]["total_artifacts"]} artifacts

**Missing Requirements**:
{chr(10).join(f"  - {req}" for req in enforcement_result["requirements_check"]["missing_requirements"])}

**Required Actions**:
{chr(10).join(f"  - {action}" for action in enforcement_result["enforcement_result"]["required_actions"])}

**Status**: Issue will remain in current state until evidence requirements met
**Emergency Protocol**: Enhanced Context Verification System Active"""
            
            cmd = ["gh", "issue", "comment", str(issue_number), "--body", comment]
            subprocess.run(cmd, check=True)
            
        except Exception as e:
            print(f"Error adding enforcement comment to issue #{issue_number}: {e}")


def enforce_evidence_for_issue(issue_number: int, target_state: str) -> Dict[str, Any]:
    """Enforce evidence requirements for specific issue"""
    enforcer = EvidenceRequirementEnforcer()
    return enforcer.enforce_evidence_requirements(issue_number, target_state)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        issue_num = int(sys.argv[1])
        target = sys.argv[2]
        result = enforce_evidence_for_issue(issue_num, target)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python evidence_requirement_enforcer.py <issue_number> <target_state>")
'''
        
        enforcer_file = "/Users/cal/DEV/RIF/claude/commands/evidence_requirement_enforcer.py"
        
        try:
            with open(enforcer_file, 'w', encoding='utf-8') as f:
                f.write(enforcer_code)
            
            os.chmod(enforcer_file, 0o755)
            
            print(f"  ✅ Evidence Requirement Enforcer deployed: {enforcer_file}")
            
            results["components_deployed"].append({
                "component": "evidence_requirement_enforcer",
                "file": enforcer_file,
                "status": "deployed"
            })
            
            results["verification_rules_active"].append("evidence_requirements_enforced")
            
        except Exception as e:
            print(f"  ❌ Failed to deploy Evidence Requirement Enforcer: {e}")
        
        print()
    
    def _deploy_context_consumption_monitor(self, results: Dict[str, Any]) -> None:
        """Deploy context consumption monitoring component"""
        
        print("Step 5: Deploying Context Consumption Monitor...")
        
        monitor_code = '''#!/usr/bin/env python3
"""
Context Consumption Monitor Component  
Part of Enhanced Context Verification System

Monitors and verifies that agents are properly consuming
all available context before making decisions.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

class ContextConsumptionMonitor:
    def __init__(self):
        self.consumption_metrics = {
            "context_utilization_threshold": 0.7,  # 70% of context should be utilized
            "requirement_coverage_threshold": 0.8,  # 80% of requirements should be addressed
            "decision_justification_required": True,
            "evidence_traceability_required": True
        }
    
    def monitor_context_consumption(self, issue_number: int, agent_type: str = "unknown") -> Dict[str, Any]:
        """Monitor context consumption for specific issue and agent"""
        
        # Get issue context
        issue_context = self._get_issue_context(issue_number)
        
        # Find agent artifacts
        agent_artifacts = self._find_agent_artifacts(issue_number, agent_type)
        
        # Analyze context utilization
        utilization_analysis = self._analyze_context_utilization(issue_context, agent_artifacts)
        
        # Check decision traceability
        traceability_check = self._check_decision_traceability(issue_context, agent_artifacts)
        
        # Generate consumption report
        consumption_report = self._generate_consumption_report(utilization_analysis, traceability_check)
        
        return {
            "issue_number": issue_number,
            "agent_type": agent_type,
            "monitoring_timestamp": datetime.utcnow().isoformat() + "Z",
            "issue_context": issue_context,
            "utilization_analysis": utilization_analysis,
            "traceability_check": traceability_check,
            "consumption_report": consumption_report,
            "context_properly_consumed": consumption_report["compliant"]
        }
    
    def _get_issue_context(self, issue_number: int) -> Dict[str, Any]:
        """Extract comprehensive context from issue"""
        
        try:
            cmd = ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels,comments"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            issue_data = json.loads(result.stdout)
            
            context = {
                "title": issue_data.get("title", ""),
                "body": issue_data.get("body", ""),
                "labels": issue_data.get("labels", []),
                "comments": issue_data.get("comments", []),
                "context_elements": []
            }
            
            # Extract context elements
            context["context_elements"] = self._extract_context_elements(context)
            
            return context
            
        except Exception as e:
            return {
                "title": "",
                "body": "",
                "labels": [],
                "comments": [],
                "context_elements": [],
                "error": str(e)
            }
    
    def _extract_context_elements(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key context elements from issue data"""
        
        elements = []
        body = context["body"]
        
        # Extract objectives
        objectives = self._extract_section(body, r"objective|goal")
        for obj in objectives:
            elements.append({"type": "objective", "content": obj, "importance": "high"})
        
        # Extract requirements
        requirements = self._extract_section(body, r"requirement|must|shall|should")
        for req in requirements:
            elements.append({"type": "requirement", "content": req, "importance": "high"})
        
        # Extract success criteria
        criteria = self._extract_section(body, r"success\\s+criteria|acceptance\\s+criteria")
        for criterion in criteria:
            elements.append({"type": "success_criterion", "content": criterion, "importance": "critical"})
        
        # Extract deliverables
        deliverables = self._extract_section(body, r"deliverable|output|artifact")
        for deliverable in deliverables:
            elements.append({"type": "deliverable", "content": deliverable, "importance": "high"})
        
        # Extract constraints
        constraints = self._extract_section(body, r"constraint|limitation|restriction")
        for constraint in constraints:
            elements.append({"type": "constraint", "content": constraint, "importance": "medium"})
        
        return elements
    
    def _extract_section(self, text: str, pattern: str) -> List[str]:
        """Extract sections matching pattern from text"""
        
        lines = text.split('\\n')
        extracted = []
        in_section = False
        
        for line in lines:
            line = line.strip()
            
            if re.search(pattern, line, re.IGNORECASE):
                in_section = True
                if line:
                    extracted.append(line)
            elif line.startswith('#') and in_section:
                in_section = False
            elif in_section and line and not line.startswith('#'):
                extracted.append(line)
        
        return extracted
    
    def _find_agent_artifacts(self, issue_number: int, agent_type: str) -> List[Dict[str, Any]]:
        """Find artifacts created by agents for this issue"""
        
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
                    files = result.stdout.strip().split('\\n')
                    for file_path in files:
                        if os.path.exists(file_path):
                            artifact_data = self._analyze_artifact_content(file_path)
                            artifacts.append({
                                "path": file_path,
                                "type": artifact_data["type"],
                                "content_summary": artifact_data["summary"],
                                "key_elements": artifact_data["key_elements"]
                            })
                            
            except Exception:
                continue
        
        return artifacts
    
    def _analyze_artifact_content(self, file_path: str) -> Dict[str, Any]:
        """Analyze artifact content for context utilization"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine artifact type
            if file_path.endswith('.json'):
                artifact_type = "data"
            elif file_path.endswith('.md'):
                artifact_type = "documentation"
            elif file_path.endswith('.py'):
                artifact_type = "code"
            else:
                artifact_type = "unknown"
            
            # Extract key elements
            key_elements = []
            
            # Look for decision justifications
            decision_patterns = [
                r"decision:.*",
                r"rationale:.*",
                r"justification:.*",
                r"reason:.*"
            ]
            
            for pattern in decision_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                key_elements.extend(matches)
            
            # Look for requirement references
            requirement_patterns = [
                r"requirement.*(?:met|addressed|implemented)",
                r"criteria.*(?:satisfied|achieved|completed)",
                r"objective.*(?:accomplished|reached|fulfilled)"
            ]
            
            for pattern in requirement_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                key_elements.extend(matches)
            
            return {
                "type": artifact_type,
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "key_elements": key_elements,
                "content_length": len(content)
            }
            
        except Exception:
            return {
                "type": "unknown",
                "summary": "Could not read content",
                "key_elements": [],
                "content_length": 0
            }
    
    def _analyze_context_utilization(self, issue_context: Dict[str, Any], agent_artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well agents utilized available context"""
        
        context_elements = issue_context["context_elements"]
        total_elements = len(context_elements)
        
        utilization = {
            "total_context_elements": total_elements,
            "addressed_elements": 0,
            "partially_addressed_elements": 0,
            "ignored_elements": [],
            "utilization_score": 0.0,
            "element_analysis": {}
        }
        
        # Analyze each context element
        for element in context_elements:
            element_key = f"{element['type']}_{hash(element['content']) % 1000}"
            
            addressing_score = self._check_element_addressing(element, agent_artifacts)
            
            utilization["element_analysis"][element_key] = {
                "element": element,
                "addressing_score": addressing_score,
                "status": "addressed" if addressing_score >= 0.7 else "partially_addressed" if addressing_score >= 0.3 else "ignored"
            }
            
            if addressing_score >= 0.7:
                utilization["addressed_elements"] += 1
            elif addressing_score >= 0.3:
                utilization["partially_addressed_elements"] += 1
            else:
                utilization["ignored_elements"].append(element)
        
        # Calculate overall utilization score
        if total_elements > 0:
            utilization["utilization_score"] = (utilization["addressed_elements"] + 0.5 * utilization["partially_addressed_elements"]) / total_elements
        
        return utilization
    
    def _check_element_addressing(self, element: Dict[str, Any], artifacts: List[Dict[str, Any]]) -> float:
        """Check how well a context element is addressed in artifacts"""
        
        element_content = element["content"].lower()
        element_terms = self._extract_key_terms(element_content)
        
        if not element_terms:
            return 0.0
        
        best_score = 0.0
        
        for artifact in artifacts:
            # Check in content summary
            summary = artifact["content_summary"].lower()
            
            # Check in key elements
            key_elements_text = " ".join(artifact["key_elements"]).lower()
            
            combined_text = f"{summary} {key_elements_text}"
            
            # Calculate term coverage
            term_matches = sum(1 for term in element_terms if term in combined_text)
            coverage_score = term_matches / len(element_terms) if element_terms else 0
            
            best_score = max(best_score, coverage_score)
        
        return best_score
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'must', 'should', 'be', 'have', 'include', 'support', 'provide', 'implement'}
        
        words = re.findall(r'\\b\\w{3,}\\b', text.lower())
        key_terms = [word for word in words if word not in common_words]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    def _check_decision_traceability(self, issue_context: Dict[str, Any], agent_artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if decisions are properly traced to context"""
        
        traceability = {
            "decisions_found": 0,
            "traceable_decisions": 0,
            "untraceable_decisions": [],
            "traceability_score": 0.0,
            "decision_analysis": []
        }
        
        # Find decisions in artifacts
        decision_patterns = [
            r"decision:.*",
            r"chose\\s+to.*",
            r"decided\\s+to.*",
            r"selected.*because.*",
            r"implemented.*approach.*"
        ]
        
        for artifact in agent_artifacts:
            content = f"{artifact['content_summary']} {' '.join(artifact['key_elements'])}"
            
            for pattern in decision_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                
                for decision in matches:
                    traceability["decisions_found"] += 1
                    
                    # Check if decision references context
                    traced = self._check_decision_context_reference(decision, issue_context)
                    
                    decision_analysis = {
                        "decision": decision,
                        "traced": traced,
                        "artifact": artifact["path"]
                    }
                    
                    traceability["decision_analysis"].append(decision_analysis)
                    
                    if traced:
                        traceability["traceable_decisions"] += 1
                    else:
                        traceability["untraceable_decisions"].append(decision)
        
        # Calculate traceability score
        if traceability["decisions_found"] > 0:
            traceability["traceability_score"] = traceability["traceable_decisions"] / traceability["decisions_found"]
        
        return traceability
    
    def _check_decision_context_reference(self, decision: str, issue_context: Dict[str, Any]) -> bool:
        """Check if decision references issue context"""
        
        decision_lower = decision.lower()
        
        # Check references to issue elements
        context_terms = []
        
        # Extract terms from title and body
        title_terms = self._extract_key_terms(issue_context["title"])
        body_terms = self._extract_key_terms(issue_context["body"][:500])  # First 500 chars
        
        context_terms.extend(title_terms[:5])  # Top 5 from title
        context_terms.extend(body_terms[:10])  # Top 10 from body
        
        # Check if decision mentions context terms
        context_references = sum(1 for term in context_terms if term in decision_lower)
        
        # Also check for explicit references to requirements, criteria, etc.
        explicit_references = [
            "requirement", "criteria", "objective", "goal", 
            "issue", "specification", "constraint"
        ]
        
        explicit_refs = sum(1 for ref in explicit_references if ref in decision_lower)
        
        return (context_references >= 2) or (explicit_refs >= 1)
    
    def _generate_consumption_report(self, utilization_analysis: Dict[str, Any], traceability_check: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall context consumption report"""
        
        report = {
            "compliant": False,
            "overall_score": 0.0,
            "compliance_factors": {},
            "deficiencies": [],
            "recommendations": []
        }
        
        # Check utilization threshold
        utilization_score = utilization_analysis["utilization_score"]
        utilization_compliant = utilization_score >= self.consumption_metrics["context_utilization_threshold"]
        
        report["compliance_factors"]["context_utilization"] = {
            "score": utilization_score,
            "threshold": self.consumption_metrics["context_utilization_threshold"],
            "compliant": utilization_compliant
        }
        
        if not utilization_compliant:
            report["deficiencies"].append(f"Context utilization {utilization_score:.1%} below threshold {self.consumption_metrics['context_utilization_threshold']:.1%}")
            report["recommendations"].append("Address more context elements from issue description")
        
        # Check decision traceability
        traceability_score = traceability_check["traceability_score"]
        traceability_compliant = traceability_score >= 0.5  # 50% of decisions should be traceable
        
        report["compliance_factors"]["decision_traceability"] = {
            "score": traceability_score,
            "threshold": 0.5,
            "compliant": traceability_compliant
        }
        
        if not traceability_compliant:
            report["deficiencies"].append(f"Decision traceability {traceability_score:.1%} below threshold 50%")
            report["recommendations"].append("Provide clear justification linking decisions to issue context")
        
        # Calculate overall score and compliance
        report["overall_score"] = (utilization_score + traceability_score) / 2
        report["compliant"] = utilization_compliant and traceability_compliant
        
        return report


def monitor_issue_context_consumption(issue_number: int, agent_type: str = "unknown") -> Dict[str, Any]:
    """Monitor context consumption for specific issue"""
    monitor = ContextConsumptionMonitor()
    return monitor.monitor_context_consumption(issue_number, agent_type)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        issue_num = int(sys.argv[1])
        agent = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        result = monitor_issue_context_consumption(issue_num, agent)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python context_consumption_monitor.py <issue_number> [agent_type]")
'''
        
        monitor_file = "/Users/cal/DEV/RIF/claude/commands/context_consumption_monitor.py"
        
        try:
            with open(monitor_file, 'w', encoding='utf-8') as f:
                f.write(monitor_code)
            
            os.chmod(monitor_file, 0o755)
            
            print(f"  ✅ Context Consumption Monitor deployed: {monitor_file}")
            
            results["components_deployed"].append({
                "component": "context_consumption_monitor",
                "file": monitor_file,
                "status": "deployed"
            })
            
            results["verification_rules_active"].append("context_consumption_monitoring")
            
        except Exception as e:
            print(f"  ❌ Failed to deploy Context Consumption Monitor: {e}")
        
        print()
    
    def _integrate_with_rif_workflow(self, results: Dict[str, Any]) -> None:
        """Integrate verification system with RIF workflow"""
        
        print("Step 6: Integrating with RIF workflow...")
        
        # Create workflow integration hook
        hook_config = {
            "verification_system_active": True,
            "activation_timestamp": datetime.utcnow().isoformat() + "Z",
            "integration_points": [
                {
                    "trigger": "state_transition_to_implementing",
                    "verification": "requirement_extraction_verifier",
                    "enforcement_level": "warning"
                },
                {
                    "trigger": "state_transition_to_validating", 
                    "verification": "evidence_requirement_enforcer",
                    "enforcement_level": "blocking"
                },
                {
                    "trigger": "state_transition_to_complete",
                    "verification": ["literature_review_verifier", "success_criteria_validator", "evidence_requirement_enforcer"],
                    "enforcement_level": "blocking"
                },
                {
                    "trigger": "agent_activation",
                    "verification": "context_consumption_monitor",
                    "enforcement_level": "advisory"
                }
            ],
            "escalation_procedures": {
                "blocking_violations": "create_github_comment_and_prevent_transition",
                "warning_violations": "create_github_comment_and_log",
                "advisory_violations": "log_only"
            }
        }
        
        hook_file = "/Users/cal/DEV/RIF/config/context_verification_hooks.json"
        
        try:
            with open(hook_file, 'w', encoding='utf-8') as f:
                json.dump(hook_config, f, indent=2)
            
            print(f"  ✅ RIF workflow integration configured: {hook_file}")
            
            results["integration_status"]["rif_workflow"] = "integrated"
            
        except Exception as e:
            print(f"  ❌ Failed to integrate with RIF workflow: {e}")
            results["integration_status"]["rif_workflow"] = f"failed: {e}"
        
        print()
    
    def _save_deployment_config(self, results: Dict[str, Any]) -> None:
        """Save deployment configuration"""
        
        print("Step 7: Saving deployment configuration...")
        
        config = {
            "system_config": self.verification_config,
            "deployment_results": results
        }
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        config_file = f"/Users/cal/DEV/RIF/config/enhanced_context_verification_deployment_{timestamp}.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            print(f"  ✅ Deployment configuration saved: {config_file}")
            
        except Exception as e:
            print(f"  ❌ Failed to save deployment configuration: {e}")
        
        print()


def main():
    """Deploy enhanced context verification system"""
    
    system = EnhancedContextVerification()
    results = system.deploy_verification_system()
    
    print("🔍 ENHANCED CONTEXT VERIFICATION SYSTEM DEPLOYMENT COMPLETE 🔍")
    print()
    print("DEPLOYMENT SUMMARY:")
    print(f"  Components deployed: {len(results['components_deployed'])}")
    print(f"  Verification rules active: {len(results['verification_rules_active'])}")
    print()
    print("ACTIVE VERIFICATION RULES:")
    for rule in results['verification_rules_active']:
        print(f"  ✅ {rule}")
    print()
    print("INTEGRATION STATUS:")
    for integration, status in results['integration_status'].items():
        print(f"  {integration}: {status}")
    print()
    
    return results


if __name__ == "__main__":
    main()