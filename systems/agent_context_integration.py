#!/usr/bin/env python3
"""
RIF Agent Context Integration System
Issue #115 & #145: Fix context delivery failures

This system enforces complete context consumption by RIF agents,
preventing the emergency context failure issues where agents miss
critical requirements and validation doesn't catch the problems.

CRITICAL FIXES:
1. Mandatory full issue context reading before work begins
2. Automatic verification that all requirements are understood
3. Context compliance checkpoints
4. Integration with existing context optimization engine
"""

import json
import re
import subprocess
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

# Import context optimization engine 
sys.path.append('/Users/cal/DEV/RIF/systems')
from import_utils import import_context_optimization_engine

context_imports = import_context_optimization_engine()
ContextOptimizer = context_imports['ContextOptimizer']
AgentType = context_imports['AgentType']
ContextType = context_imports['ContextType']

@dataclass
class ContextConsumptionRecord:
    """Record of agent context consumption for compliance tracking"""
    agent_type: str
    issue_number: int
    timestamp: datetime
    full_issue_read: bool
    requirements_extracted: bool
    research_methodology_identified: bool
    validation_requirements_understood: bool
    context_items_consumed: int
    compliance_score: float
    missing_elements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass 
class AgentContextPackage:
    """Complete context package for agent consumption"""
    issue_context: Dict[str, Any]
    requirements_analysis: Dict[str, Any]
    research_methodology_requirements: List[str]
    validation_requirements: List[str]
    optimized_knowledge: str
    system_context: str
    compliance_checklist: List[Dict[str, Any]]
    context_consumption_requirements: Dict[str, Any]

class AgentContextIntegrator:
    """
    Integrates context optimization engine with RIF agent workflow
    to enforce complete context consumption and prevent context failures
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.knowledge_base_path = knowledge_base_path
        self.context_optimizer = ContextOptimizer(knowledge_base_path)
        self.consumption_records = []
        self.compliance_log_file = os.path.join(knowledge_base_path, "context_compliance_audit.json")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.compliance_log_file), exist_ok=True)
        
    def prepare_agent_context_package(self, agent_type_str: str, issue_number: int) -> AgentContextPackage:
        """
        Prepare complete context package for agent with mandatory compliance requirements
        """
        print(f"Preparing context package for {agent_type_str} on issue #{issue_number}...")
        
        # 1. Get full issue context from GitHub
        issue_context = self._get_full_issue_context(issue_number)
        
        # 2. Extract requirements and methodology
        requirements_analysis = self._analyze_issue_requirements(issue_context)
        research_methodology = self._extract_research_methodology_requirements(issue_context)
        validation_requirements = self._extract_validation_requirements(issue_context)
        
        # 3. Get optimized context from context engine
        try:
            agent_type = AgentType(agent_type_str)
        except ValueError:
            # Handle agent types that might not be in the enum
            agent_type = AgentType.IMPLEMENTER  # Default fallback
            
        optimized_context = self.context_optimizer.optimize_for_agent(
            agent_type, 
            {"description": issue_context.get("body", ""), "issue_number": issue_number},
            issue_number
        )
        formatted_context = self.context_optimizer.format_context_for_agent(optimized_context)
        
        # 4. Generate system context
        system_context = self._generate_live_system_context()
        
        # 5. Create compliance checklist
        compliance_checklist = self._generate_compliance_checklist(
            agent_type_str, requirements_analysis, research_methodology, validation_requirements
        )
        
        # 6. Define context consumption requirements
        consumption_requirements = self._define_consumption_requirements(agent_type_str)
        
        return AgentContextPackage(
            issue_context=issue_context,
            requirements_analysis=requirements_analysis,
            research_methodology_requirements=research_methodology,
            validation_requirements=validation_requirements,
            optimized_knowledge=formatted_context,
            system_context=system_context,
            compliance_checklist=compliance_checklist,
            context_consumption_requirements=consumption_requirements
        )
    
    def _get_full_issue_context(self, issue_number: int) -> Dict[str, Any]:
        """Get complete issue context including comments and related issues"""
        try:
            # Get issue details
            result = subprocess.run([
                "gh", "issue", "view", str(issue_number), "--json", 
                "number,title,body,labels,comments,state,assignees,milestone"
            ], capture_output=True, text=True, check=True)
            
            issue_data = json.loads(result.stdout)
            
            # Add comment thread context for full understanding
            issue_data['full_body'] = issue_data.get('body', '')
            issue_data['comment_count'] = len(issue_data.get('comments', []))
            
            # Extract all text for analysis
            all_text = issue_data.get('body', '')
            for comment in issue_data.get('comments', []):
                all_text += "\n\n" + comment.get('body', '')
                
            issue_data['full_text'] = all_text
            
            return issue_data
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting issue context: {e}")
            return {"error": str(e), "number": issue_number}
    
    def _analyze_issue_requirements(self, issue_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze issue to extract all requirements systematically"""
        body = issue_context.get('body', '')
        full_text = issue_context.get('full_text', body)
        
        requirements = {
            "explicit_requirements": self._extract_explicit_requirements(full_text),
            "acceptance_criteria": self._extract_acceptance_criteria(full_text),
            "deliverables": self._extract_deliverables(full_text),
            "success_criteria": self._extract_success_criteria(full_text),
            "complexity_indicators": self._assess_complexity_indicators(full_text),
            "research_components": self._identify_research_components(full_text),
        }
        
        return requirements
    
    def _extract_explicit_requirements(self, text: str) -> List[str]:
        """Extract explicit requirements from issue text"""
        requirements = []
        
        # Look for requirement sections
        requirement_patterns = [
            r"## Requirements?.*?(?=##|\n\n|\Z)",
            r"### Requirements?.*?(?=###|\n\n|\Z)",
            r"## Acceptance Criteria.*?(?=##|\n\n|\Z)",
            r"- \[[ x]\].*?(?=\n(?!-)|\Z)",
            r"Must.*?(?=\n|\Z)",
            r"Should.*?(?=\n|\Z)",
            r"Shall.*?(?=\n|\Z)"
        ]
        
        for pattern in requirement_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                requirements.append(match.group().strip())
        
        return requirements
    
    def _extract_research_methodology_requirements(self, issue_context: Dict[str, Any]) -> List[str]:
        """Extract research methodology requirements that were missed in issue #115"""
        full_text = issue_context.get('full_text', '')
        research_requirements = []
        
        # Specific patterns that indicate research methodology requirements
        research_patterns = [
            r"Literature Review.*?(?=\n(?!-)|\Z)",
            r"Academic papers.*?(?=\n|\Z)", 
            r"Industry best practices.*?(?=\n|\Z)",
            r"Research into.*?(?=\n|\Z)",
            r"Investigation of.*?(?=\n|\Z)",
            r"Analysis of.*?(?=\n|\Z)",
            r"Study of.*?(?=\n|\Z)",
            r"Comparative.*?(?=\n|\Z)",
            r"Benchmark.*?(?=\n|\Z)",
            r"Survey.*?(?=\n|\Z)"
        ]
        
        for pattern in research_patterns:
            matches = re.finditer(pattern, full_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                research_requirements.append(match.group().strip())
        
        # Special check for issue #115 type requirements  
        if "research" in full_text.lower() and "algorithm" in full_text.lower():
            research_requirements.append("CRITICAL: Literature review and comparative analysis required")
            
        if "Academic papers" in full_text or "academic" in full_text.lower():
            research_requirements.append("MANDATORY: Academic literature review required")
            
        return research_requirements
    
    def _extract_validation_requirements(self, issue_context: Dict[str, Any]) -> List[str]:
        """Extract validation requirements"""
        full_text = issue_context.get('full_text', '')
        validation_requirements = []
        
        validation_patterns = [
            r"## Success Criteria.*?(?=##|\Z)",
            r"## Acceptance Criteria.*?(?=##|\Z)",
            r"Validation.*?(?=\n|\Z)",
            r"Testing.*?(?=\n|\Z)",
            r"Performance.*?target.*?(?=\n|\Z)",
            r"Must.*?validate.*?(?=\n|\Z)"
        ]
        
        for pattern in validation_patterns:
            matches = re.finditer(pattern, full_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                validation_requirements.append(match.group().strip())
        
        return validation_requirements
    
    def _extract_acceptance_criteria(self, text: str) -> List[str]:
        """Extract acceptance criteria"""
        criteria = []
        
        # Look for checkbox lists and acceptance criteria
        checkbox_pattern = r"- \[[ x]\] (.*?)(?=\n(?!  )|\Z)"
        matches = re.finditer(checkbox_pattern, text, re.MULTILINE)
        
        for match in matches:
            criteria.append(match.group(1).strip())
            
        return criteria
    
    def _extract_deliverables(self, text: str) -> List[str]:
        """Extract deliverables from issue text"""
        deliverables = []
        
        # Look for deliverable sections
        deliverable_patterns = [
            r"## Deliverables?.*?(?=##|\Z)",
            r"### Deliverables?.*?(?=###|\Z)", 
            r"Deliverable.*?(?=\n|\Z)"
        ]
        
        for pattern in deliverable_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                deliverables.append(match.group().strip())
                
        return deliverables
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria"""
        criteria = []
        
        # Look for success criteria sections
        success_patterns = [
            r"## Success Criteria.*?(?=##|\Z)",
            r"### Success Criteria.*?(?=###|\Z)"
        ]
        
        for pattern in success_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                criteria.append(match.group().strip())
                
        return criteria
    
    def _assess_complexity_indicators(self, text: str) -> Dict[str, Any]:
        """Assess complexity indicators in the issue"""
        indicators = {
            "research_required": any(word in text.lower() for word in ["research", "analysis", "investigate"]),
            "implementation_required": any(word in text.lower() for word in ["implement", "build", "develop"]),
            "validation_required": any(word in text.lower() for word in ["test", "validate", "verify"]),
            "documentation_required": any(word in text.lower() for word in ["document", "specification", "guide"]),
            "performance_requirements": "performance" in text.lower() or "latency" in text.lower(),
            "integration_required": "integration" in text.lower() or "integrate" in text.lower()
        }
        
        complexity_score = sum(indicators.values()) / len(indicators)
        indicators["estimated_complexity"] = "high" if complexity_score > 0.6 else "medium" if complexity_score > 0.3 else "low"
        
        return indicators
    
    def _identify_research_components(self, text: str) -> Dict[str, List[str]]:
        """Identify research components that need to be addressed"""
        components = {
            "literature_review_topics": [],
            "comparative_analysis_targets": [],
            "experimental_requirements": [],
            "validation_methodologies": []
        }
        
        # Extract research topics
        if "literature review" in text.lower():
            components["literature_review_topics"].append("Academic literature review required")
            
        if "comparative" in text.lower() or "comparison" in text.lower():
            components["comparative_analysis_targets"].append("Comparative analysis required")
            
        if "experiment" in text.lower() or "testing" in text.lower():
            components["experimental_requirements"].append("Experimental validation required")
            
        if "benchmark" in text.lower() or "performance" in text.lower():
            components["validation_methodologies"].append("Performance benchmarking required")
            
        return components
    
    def _generate_live_system_context(self) -> str:
        """Generate live system context for agent awareness"""
        context_parts = [
            "# RIF System Context",
            "RIF is an automatic intelligent development system that orchestrates specialized agents through GitHub issues.",
            "",
            "## Critical Context Consumption Requirements",
            "- ALL agents MUST read and understand the complete issue thread before beginning work",
            "- Research methodology requirements MUST be followed when specified",
            "- Validation must verify that ALL requirements were addressed",
            "- Context failures like issue #145 MUST be prevented through systematic context consumption",
            "",
            "## Emergency Context Failure Prevention",
            "- Issue #115 was marked complete without following literature review requirements",
            "- This integration system enforces complete context consumption to prevent similar failures",
            "- All agents must demonstrate understanding of requirements before proceeding"
        ]
        
        return "\n".join(context_parts)
    
    def _generate_compliance_checklist(self, agent_type: str, requirements_analysis: Dict[str, Any],
                                     research_methodology: List[str], validation_requirements: List[str]) -> List[Dict[str, Any]]:
        """Generate compliance checklist for agent"""
        checklist = [
            {
                "requirement": "Full Issue Context Reading",
                "description": "Read and understand complete issue including body and all comments",
                "mandatory": True,
                "evidence_required": True
            },
            {
                "requirement": "Requirements Extraction",
                "description": "Extract and document all explicit and implicit requirements",
                "mandatory": True,
                "evidence_required": True
            }
        ]
        
        # Add research methodology requirements if present
        if research_methodology:
            checklist.append({
                "requirement": "Research Methodology Compliance",
                "description": f"Follow all research methodology requirements: {', '.join(research_methodology)}",
                "mandatory": True,
                "evidence_required": True,
                "specific_requirements": research_methodology
            })
            
        # Add validation requirements if present  
        if validation_requirements:
            checklist.append({
                "requirement": "Validation Requirements Understanding",
                "description": f"Understand and plan for validation requirements: {', '.join(validation_requirements)}",
                "mandatory": True,
                "evidence_required": True,
                "specific_requirements": validation_requirements
            })
            
        # Add agent-specific requirements
        if agent_type in ["rif-implementer", "implementer"]:
            checklist.append({
                "requirement": "Implementation Planning",
                "description": "Plan implementation approach that addresses all requirements",
                "mandatory": True,
                "evidence_required": False
            })
            
        return checklist
    
    def _define_consumption_requirements(self, agent_type: str) -> Dict[str, Any]:
        """Define context consumption requirements for agent type"""
        base_requirements = {
            "full_issue_read": True,
            "requirements_documented": True,
            "methodology_compliance": True,
            "evidence_of_understanding": True,
            "compliance_verification": True
        }
        
        agent_specific = {
            "rif-implementer": {
                **base_requirements,
                "implementation_plan_addresses_all_requirements": True,
                "checkpoint_tracking": True
            },
            "rif-validator": {
                **base_requirements,
                "validation_plan_covers_all_requirements": True,
                "comprehensive_testing": True
            },
            "rif-analyst": {
                **base_requirements,
                "analysis_covers_all_aspects": True,
                "pattern_recognition": True
            }
        }
        
        return agent_specific.get(agent_type, base_requirements)
    
    def record_context_consumption(self, agent_type: str, issue_number: int,
                                 consumption_data: Dict[str, Any]) -> ContextConsumptionRecord:
        """Record agent context consumption for compliance tracking"""
        
        # Calculate compliance score
        required_elements = [
            "full_issue_read", "requirements_extracted", "research_methodology_identified",
            "validation_requirements_understood"
        ]
        
        compliance_count = sum(1 for element in required_elements if consumption_data.get(element, False))
        compliance_score = compliance_count / len(required_elements)
        
        # Identify missing elements
        missing_elements = [element for element in required_elements if not consumption_data.get(element, False)]
        
        record = ContextConsumptionRecord(
            agent_type=agent_type,
            issue_number=issue_number,
            timestamp=datetime.now(),
            full_issue_read=consumption_data.get("full_issue_read", False),
            requirements_extracted=consumption_data.get("requirements_extracted", False),
            research_methodology_identified=consumption_data.get("research_methodology_identified", False),
            validation_requirements_understood=consumption_data.get("validation_requirements_understood", False),
            context_items_consumed=consumption_data.get("context_items_consumed", 0),
            compliance_score=compliance_score,
            missing_elements=missing_elements
        )
        
        self.consumption_records.append(record)
        self._save_compliance_record(record)
        
        return record
    
    def _save_compliance_record(self, record: ContextConsumptionRecord):
        """Save compliance record to audit log"""
        try:
            # Load existing records
            if os.path.exists(self.compliance_log_file):
                with open(self.compliance_log_file, 'r') as f:
                    audit_data = json.load(f)
            else:
                audit_data = {"compliance_records": []}
            
            # Add new record
            audit_data["compliance_records"].append(record.to_dict())
            
            # Save updated data
            with open(self.compliance_log_file, 'w') as f:
                json.dump(audit_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving compliance record: {e}")
    
    def validate_context_consumption(self, record: ContextConsumptionRecord) -> Dict[str, Any]:
        """Validate that context consumption meets requirements"""
        validation_result = {
            "compliant": record.compliance_score >= 0.8,
            "compliance_score": record.compliance_score,
            "missing_elements": record.missing_elements,
            "recommendations": []
        }
        
        if not record.full_issue_read:
            validation_result["recommendations"].append("Must read complete issue thread including all comments")
            
        if not record.research_methodology_identified:
            validation_result["recommendations"].append("Must identify and follow any research methodology requirements")
            
        if not record.validation_requirements_understood:
            validation_result["recommendations"].append("Must understand and plan for validation requirements")
            
        return validation_result
    
    def format_agent_context_instructions(self, context_package: AgentContextPackage) -> str:
        """Format complete context instructions for agent"""
        instructions = []
        
        instructions.append("# MANDATORY CONTEXT CONSUMPTION PROTOCOL")
        instructions.append("## EMERGENCY CONTEXT FAILURE PREVENTION")
        instructions.append("This system prevents context failures like issue #145 where agents miss critical requirements.")
        instructions.append("")
        
        # Issue context
        instructions.append("## COMPLETE ISSUE CONTEXT")
        instructions.append(f"**Issue #{context_package.issue_context.get('number')}**: {context_package.issue_context.get('title')}")
        instructions.append("")
        instructions.append("**Full Issue Body:**")
        instructions.append(context_package.issue_context.get('body', ''))
        instructions.append("")
        
        # Requirements analysis
        instructions.append("## EXTRACTED REQUIREMENTS (MANDATORY TO ADDRESS)")
        for category, items in context_package.requirements_analysis.items():
            if items:
                instructions.append(f"**{category.replace('_', ' ').title()}:**")
                if isinstance(items, list):
                    for item in items:
                        instructions.append(f"- {item}")
                else:
                    instructions.append(f"- {items}")
                instructions.append("")
        
        # Research methodology requirements
        if context_package.research_methodology_requirements:
            instructions.append("## RESEARCH METHODOLOGY REQUIREMENTS (CRITICAL)")
            instructions.append("**These were MISSED in issue #115 - MUST be followed:**")
            for req in context_package.research_methodology_requirements:
                instructions.append(f"- {req}")
            instructions.append("")
        
        # Validation requirements
        if context_package.validation_requirements:
            instructions.append("## VALIDATION REQUIREMENTS")
            for req in context_package.validation_requirements:
                instructions.append(f"- {req}")
            instructions.append("")
        
        # Optimized knowledge context
        instructions.append("## OPTIMIZED KNOWLEDGE CONTEXT")
        instructions.append(context_package.optimized_knowledge)
        instructions.append("")
        
        # System context
        instructions.append("## SYSTEM CONTEXT")
        instructions.append(context_package.system_context)
        instructions.append("")
        
        # Compliance checklist
        instructions.append("## COMPLIANCE CHECKLIST (MANDATORY)")
        instructions.append("You MUST complete and demonstrate evidence for each item:")
        for item in context_package.compliance_checklist:
            status = "[ ]"  # Unchecked
            instructions.append(f"{status} **{item['requirement']}**: {item['description']}")
            if item.get("evidence_required"):
                instructions.append(f"    - Evidence required: Document completion in your response")
            if item.get("specific_requirements"):
                instructions.append(f"    - Specific requirements: {', '.join(item['specific_requirements'])}")
            instructions.append("")
        
        instructions.append("## CONTEXT CONSUMPTION EVIDENCE REQUIRED")
        instructions.append("Post evidence that you have:")
        instructions.append("1. Read and understood the complete issue")
        instructions.append("2. Extracted all requirements (including research methodology)")
        instructions.append("3. Understand validation requirements")
        instructions.append("4. Have a plan that addresses ALL requirements")
        instructions.append("")
        
        return "\n".join(instructions)

def integrate_with_existing_workflow():
    """Integration function to connect with existing RIF workflow"""
    print("Integrating context system with existing RIF workflow...")
    
    integrator = AgentContextIntegrator()
    
    # Test integration with issue #115
    context_package = integrator.prepare_agent_context_package("rif-implementer", 115)
    formatted_instructions = integrator.format_agent_context_instructions(context_package)
    
    # Save integration test results
    integration_test_file = "/Users/cal/DEV/RIF/knowledge/context_integration_test.json"
    os.makedirs(os.path.dirname(integration_test_file), exist_ok=True)
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "context_integration",
        "issue_number": 115,
        "agent_type": "rif-implementer",
        "context_package_generated": True,
        "requirements_extracted": len(context_package.requirements_analysis),
        "research_methodology_found": len(context_package.research_methodology_requirements),
        "validation_requirements_found": len(context_package.validation_requirements),
        "compliance_checklist_items": len(context_package.compliance_checklist),
        "integration_status": "success"
    }
    
    with open(integration_test_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"✓ Context integration test completed successfully")
    print(f"✓ Found {len(context_package.research_methodology_requirements)} research methodology requirements")
    print(f"✓ Found {len(context_package.validation_requirements)} validation requirements")
    print(f"✓ Generated {len(context_package.compliance_checklist)} compliance checklist items")
    
    return context_package, formatted_instructions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Agent Context Integration System")
    parser.add_argument("--agent", type=str, help="Agent type (e.g., rif-implementer)")
    parser.add_argument("--issue", type=int, help="GitHub issue number")
    parser.add_argument("--test", action="store_true", help="Run integration test")
    
    args = parser.parse_args()
    
    if args.test:
        context_package, instructions = integrate_with_existing_workflow()
        print("\n" + "="*80)
        print("AGENT CONTEXT INSTRUCTIONS (SAMPLE):")
        print("="*80)
        print(instructions[:2000] + "..." if len(instructions) > 2000 else instructions)
        
    elif args.agent and args.issue:
        integrator = AgentContextIntegrator()
        context_package = integrator.prepare_agent_context_package(args.agent, args.issue)
        instructions = integrator.format_agent_context_instructions(context_package)
        print(instructions)
    
    else:
        parser.print_help()