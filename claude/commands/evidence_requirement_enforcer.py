#!/usr/bin/env python3
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
                    files = result.stdout.strip().split('\n')
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
                        
                    if len(content.split('\n')) > 50 or len(content) > 2000:
                        quality = "high"
                    elif len(content.split('\n')) > 20 or len(content) > 1000:
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
