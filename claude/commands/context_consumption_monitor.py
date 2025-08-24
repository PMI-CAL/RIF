#!/usr/bin/env python3
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
        criteria = self._extract_section(body, r"success\s+criteria|acceptance\s+criteria")
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
        
        lines = text.split('\n')
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
                    files = result.stdout.strip().split('\n')
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
        
        words = re.findall(r'\b\w{3,}\b', text.lower())
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
            r"chose\s+to.*",
            r"decided\s+to.*",
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
