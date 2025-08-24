"""
RIF Agent Integration Module for LightRAG
Provides specialized RAG capabilities for each RIF agent type.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add parent directory to path for LightRAG core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lightrag_core import LightRAGCore, get_lightrag_instance


class RIFAgentRAG:
    """
    Base class for RIF agent RAG integration.
    Provides common functionality for all agents.
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize agent RAG integration.
        
        Args:
            agent_name: Name of the RIF agent (e.g., 'analyst', 'architect')
        """
        self.agent_name = agent_name
        self.rag = get_lightrag_instance()
        self.logger = logging.getLogger(f"rif.agent.{agent_name}")
        
    def capture_knowledge(self, content: str, knowledge_type: str, metadata: Dict[str, Any]) -> str:
        """
        Capture knowledge discovered during agent execution.
        
        Args:
            content: Knowledge content to store
            knowledge_type: Type of knowledge (pattern, decision, etc.)
            metadata: Additional metadata
            
        Returns:
            Document ID of stored knowledge
        """
        # Add agent context to metadata
        metadata = metadata.copy()
        metadata.update({
            "source_agent": self.agent_name,
            "captured_at": datetime.utcnow().isoformat(),
            "session_id": getattr(self, 'session_id', 'unknown')
        })
        
        # Map knowledge types to collections
        collection_mapping = {
            "pattern": "patterns",
            "decision": "decisions", 
            "code": "code_snippets",
            "resolution": "issue_resolutions"
        }
        
        collection_name = collection_mapping.get(knowledge_type, "patterns")
        
        return self.rag.store_knowledge(collection_name, content, metadata)
    
    def query_similar_work(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query for similar past work across all collections.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of relevant past work items
        """
        return self.rag.retrieve_knowledge(query, n_results=limit)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about this agent's knowledge contributions."""
        stats = self.rag.get_collection_stats()
        
        # Filter for this agent's contributions
        agent_stats = {}
        for collection, data in stats.items():
            if isinstance(data, dict) and 'count' in data:
                # Would need to query with filters in real implementation
                agent_stats[collection] = {
                    "total_items": data['count'],
                    "agent_contributions": "unknown"  # Placeholder
                }
        
        return agent_stats


class RIFAnalystRAG(RIFAgentRAG):
    """
    LightRAG integration for RIF-Analyst agent.
    Specializes in pattern recognition and requirements analysis.
    """
    
    def __init__(self):
        super().__init__("analyst")
    
    def find_similar_issues(self, issue_description: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar issues from past resolutions.
        
        Args:
            issue_description: Description of current issue
            limit: Number of similar issues to return
            
        Returns:
            List of similar past issues with resolutions
        """
        return self.rag.retrieve_knowledge(
            query=issue_description,
            collection_name="issue_resolutions",
            n_results=limit
        )
    
    def find_relevant_patterns(self, requirements: str, complexity: str = None) -> List[Dict[str, Any]]:
        """
        Find patterns relevant to given requirements.
        
        Args:
            requirements: Requirements description
            complexity: Complexity level filter (low, medium, high, very-high)
            
        Returns:
            List of relevant patterns
        """
        filters = {}
        if complexity:
            filters["complexity"] = complexity
            
        return self.rag.retrieve_knowledge(
            query=requirements,
            collection_name="patterns",
            n_results=5,
            filters=filters if filters else None
        )
    
    def analyze_with_history(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis using historical knowledge.
        
        Args:
            issue_data: Current issue information
            
        Returns:
            Analysis results with historical context
        """
        title = issue_data.get('title', '')
        description = issue_data.get('description', '')
        query = f"{title} {description}"
        
        # Find similar issues
        similar_issues = self.find_similar_issues(query)
        
        # Find relevant patterns
        patterns = self.find_relevant_patterns(query)
        
        # Build analysis context
        analysis = {
            "similar_issues_found": len(similar_issues),
            "patterns_found": len(patterns),
            "recommendations": [],
            "complexity_indicators": [],
            "similar_issues": similar_issues,
            "relevant_patterns": patterns
        }
        
        # Extract insights from similar issues
        for issue in similar_issues:
            if issue.get('metadata', {}).get('resolution_success'):
                analysis["recommendations"].append({
                    "source": "similar_issue",
                    "recommendation": f"Consider approach from issue {issue.get('id', 'unknown')}",
                    "confidence": 1.0 - (issue.get('distance', 0.5))
                })
        
        # Extract insights from patterns
        for pattern in patterns:
            complexity = pattern.get('metadata', {}).get('complexity', 'medium')
            analysis["complexity_indicators"].append(complexity)
        
        return analysis
    
    def store_analysis_results(self, issue_id: str, analysis_results: Dict[str, Any]) -> str:
        """Store analysis results for future reference."""
        content = json.dumps(analysis_results, indent=2)
        metadata = {
            "type": "analysis",
            "issue_id": issue_id,
            "complexity": analysis_results.get('complexity', 'medium'),
            "patterns_used": len(analysis_results.get('relevant_patterns', [])),
            "tags": ["analysis", "requirements"]
        }
        
        return self.capture_knowledge(content, "pattern", metadata)


class RIFArchitectRAG(RIFAgentRAG):
    """
    LightRAG integration for RIF-Architect agent.
    Specializes in architectural decisions and design patterns.
    """
    
    def __init__(self):
        super().__init__("architect")
    
    def find_architectural_decisions(self, context: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find relevant architectural decisions.
        
        Args:
            context: Architecture context or problem description
            limit: Number of decisions to return
            
        Returns:
            List of relevant architectural decisions
        """
        return self.rag.retrieve_knowledge(
            query=context,
            collection_name="decisions",
            n_results=limit
        )
    
    def find_design_patterns(self, requirements: str, technology: str = None) -> List[Dict[str, Any]]:
        """
        Find applicable design patterns.
        
        Args:
            requirements: System requirements
            technology: Technology stack filter
            
        Returns:
            List of applicable design patterns
        """
        filters = {}
        if technology:
            filters["technology"] = technology
            
        return self.rag.retrieve_knowledge(
            query=requirements,
            collection_name="patterns", 
            n_results=5,
            filters=filters if filters else None
        )
    
    def design_with_knowledge(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create architecture design using knowledge base.
        
        Args:
            requirements: System requirements and constraints
            
        Returns:
            Architecture design with knowledge-based recommendations
        """
        system_desc = requirements.get('description', '')
        technology = requirements.get('technology', '')
        
        # Find relevant decisions and patterns
        decisions = self.find_architectural_decisions(system_desc)
        patterns = self.find_design_patterns(system_desc, technology)
        
        design = {
            "based_on_decisions": len(decisions),
            "applicable_patterns": len(patterns),
            "recommendations": [],
            "design_decisions": [],
            "risk_factors": [],
            "reference_decisions": decisions,
            "suggested_patterns": patterns
        }
        
        # Process decisions for recommendations
        for decision in decisions:
            metadata = decision.get('metadata', {})
            if metadata.get('status') == 'active':
                design["recommendations"].append({
                    "source": "architectural_decision",
                    "recommendation": f"Apply decision pattern from {decision.get('id')}",
                    "rationale": metadata.get('rationale', 'Proven successful'),
                    "confidence": 1.0 - (decision.get('distance', 0.5))
                })
        
        # Process patterns for design decisions
        for pattern in patterns:
            metadata = pattern.get('metadata', {})
            design["design_decisions"].append({
                "pattern": metadata.get('pattern_name', 'unknown'),
                "applicability": 1.0 - (pattern.get('distance', 0.5)),
                "complexity": metadata.get('complexity', 'medium')
            })
        
        return design
    
    def store_design_decision(self, decision_data: Dict[str, Any]) -> str:
        """Store architectural decision for future reference."""
        content = json.dumps(decision_data, indent=2)
        metadata = {
            "type": "decision",
            "status": decision_data.get('status', 'active'),
            "impact": decision_data.get('impact', 'medium'),
            "technology": decision_data.get('technology', 'general'),
            "tags": ["architecture", "design"]
        }
        
        return self.capture_knowledge(content, "decision", metadata)


class RIFImplementerRAG(RIFAgentRAG):
    """
    LightRAG integration for RIF-Implementer agent.
    Specializes in code patterns and implementation solutions.
    """
    
    def __init__(self):
        super().__init__("implementer")
    
    def find_code_examples(self, functionality: str, language: str = None) -> List[Dict[str, Any]]:
        """
        Find relevant code examples and snippets.
        
        Args:
            functionality: Description of functionality to implement
            language: Programming language filter
            
        Returns:
            List of relevant code examples
        """
        filters = {}
        if language:
            filters["language"] = language
            
        return self.rag.retrieve_knowledge(
            query=functionality,
            collection_name="code_snippets",
            n_results=5,
            filters=filters if filters else None
        )
    
    def find_implementation_patterns(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Find implementation patterns for a given task.
        
        Args:
            task_description: Description of implementation task
            
        Returns:
            List of relevant implementation patterns
        """
        return self.rag.retrieve_knowledge(
            query=task_description,
            collection_name="patterns",
            n_results=3,
            filters={"type": "implementation"}
        )
    
    def implement_with_knowledge(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan implementation using knowledge base.
        
        Args:
            task_data: Implementation task information
            
        Returns:
            Implementation plan with knowledge-based suggestions
        """
        description = task_data.get('description', '')
        language = task_data.get('language', '')
        
        # Find relevant code and patterns
        code_examples = self.find_code_examples(description, language)
        patterns = self.find_implementation_patterns(description)
        
        implementation = {
            "code_examples_found": len(code_examples),
            "patterns_found": len(patterns),
            "implementation_suggestions": [],
            "reusable_components": [],
            "best_practices": [],
            "reference_code": code_examples,
            "applicable_patterns": patterns
        }
        
        # Process code examples
        for example in code_examples:
            metadata = example.get('metadata', {})
            implementation["reusable_components"].append({
                "component": metadata.get('component_name', 'unknown'),
                "language": metadata.get('language', 'unknown'),
                "complexity": metadata.get('complexity', 'medium'),
                "relevance": 1.0 - (example.get('distance', 0.5))
            })
        
        # Process patterns
        for pattern in patterns:
            metadata = pattern.get('metadata', {})
            implementation["implementation_suggestions"].append({
                "pattern": metadata.get('pattern_name', 'unknown'),
                "approach": metadata.get('approach', 'standard'),
                "confidence": 1.0 - (pattern.get('distance', 0.5))
            })
        
        return implementation
    
    def store_code_snippet(self, code_data: Dict[str, Any]) -> str:
        """Store code snippet for future reuse."""
        content = json.dumps(code_data, indent=2)
        metadata = {
            "type": "code",
            "language": code_data.get('language', 'unknown'),
            "component_name": code_data.get('name', 'unnamed'),
            "complexity": code_data.get('complexity', 'medium'),
            "tags": ["code", "implementation"]
        }
        
        return self.capture_knowledge(content, "code", metadata)


class RIFValidatorRAG(RIFAgentRAG):
    """
    LightRAG integration for RIF-Validator agent.
    Specializes in test patterns and quality validation.
    """
    
    def __init__(self):
        super().__init__("validator")
    
    def find_test_patterns(self, functionality: str, test_type: str = None) -> List[Dict[str, Any]]:
        """
        Find relevant test patterns and strategies.
        
        Args:
            functionality: Functionality being tested
            test_type: Type of test (unit, integration, e2e)
            
        Returns:
            List of relevant test patterns
        """
        filters = {"type": "test"}
        if test_type:
            filters["test_type"] = test_type
            
        return self.rag.retrieve_knowledge(
            query=functionality,
            collection_name="patterns",
            n_results=5,
            filters=filters
        )
    
    def find_quality_gates(self, project_type: str) -> List[Dict[str, Any]]:
        """
        Find applicable quality gates for project type.
        
        Args:
            project_type: Type of project (web, api, mobile, etc.)
            
        Returns:
            List of relevant quality gate configurations
        """
        return self.rag.retrieve_knowledge(
            query=f"{project_type} quality gates",
            collection_name="patterns",
            n_results=3,
            filters={"type": "quality_gate"}
        )
    
    def validate_with_knowledge(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan validation using knowledge base.
        
        Args:
            validation_data: Validation requirements and context
            
        Returns:
            Validation plan with knowledge-based test strategies
        """
        functionality = validation_data.get('functionality', '')
        project_type = validation_data.get('project_type', 'general')
        
        # Find relevant test patterns and quality gates
        test_patterns = self.find_test_patterns(functionality)
        quality_gates = self.find_quality_gates(project_type)
        
        validation = {
            "test_patterns_found": len(test_patterns),
            "quality_gates_found": len(quality_gates),
            "test_strategies": [],
            "quality_requirements": [],
            "coverage_recommendations": [],
            "reference_patterns": test_patterns,
            "applicable_gates": quality_gates
        }
        
        # Process test patterns
        for pattern in test_patterns:
            metadata = pattern.get('metadata', {})
            validation["test_strategies"].append({
                "strategy": metadata.get('strategy_name', 'unknown'),
                "test_type": metadata.get('test_type', 'unit'),
                "coverage_target": metadata.get('coverage_target', 80),
                "confidence": 1.0 - (pattern.get('distance', 0.5))
            })
        
        # Process quality gates
        for gate in quality_gates:
            metadata = gate.get('metadata', {})
            validation["quality_requirements"].append({
                "gate": metadata.get('gate_name', 'unknown'),
                "threshold": metadata.get('threshold', 'standard'),
                "blocking": metadata.get('blocking', True)
            })
        
        return validation
    
    def store_test_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Store test pattern for future reference."""
        content = json.dumps(pattern_data, indent=2)
        metadata = {
            "type": "test",
            "test_type": pattern_data.get('test_type', 'unit'),
            "strategy_name": pattern_data.get('name', 'unnamed'),
            "coverage_target": pattern_data.get('coverage_target', 80),
            "tags": ["testing", "quality"]
        }
        
        return self.capture_knowledge(content, "pattern", metadata)


class RIFPlannerRAG(RIFAgentRAG):
    """
    LightRAG integration for RIF-Planner agent.
    Specializes in project planning templates and workflow patterns.
    """
    
    def __init__(self):
        super().__init__("planner")
    
    def find_planning_templates(self, complexity: str, project_type: str = None) -> List[Dict[str, Any]]:
        """
        Find planning templates for given complexity and project type.
        
        Args:
            complexity: Project complexity (low, medium, high, very-high)
            project_type: Type of project
            
        Returns:
            List of relevant planning templates
        """
        filters = {"type": "planning", "complexity": complexity}
        if project_type:
            filters["project_type"] = project_type
            
        return self.rag.retrieve_knowledge(
            query=f"{complexity} complexity planning",
            collection_name="patterns",
            n_results=3,
            filters=filters
        )
    
    def find_workflow_patterns(self, requirements: str) -> List[Dict[str, Any]]:
        """
        Find workflow patterns for given requirements.
        
        Args:
            requirements: Project requirements description
            
        Returns:
            List of relevant workflow patterns
        """
        return self.rag.retrieve_knowledge(
            query=requirements,
            collection_name="patterns",
            n_results=5,
            filters={"type": "workflow"}
        )
    
    def plan_with_knowledge(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create project plan using knowledge base.
        
        Args:
            planning_data: Planning requirements and constraints
            
        Returns:
            Project plan with knowledge-based recommendations
        """
        complexity = planning_data.get('complexity', 'medium')
        description = planning_data.get('description', '')
        project_type = planning_data.get('project_type', 'general')
        
        # Find relevant templates and workflows
        templates = self.find_planning_templates(complexity, project_type)
        workflows = self.find_workflow_patterns(description)
        
        plan = {
            "templates_found": len(templates),
            "workflows_found": len(workflows),
            "planning_recommendations": [],
            "workflow_suggestions": [],
            "timeline_estimates": [],
            "reference_templates": templates,
            "applicable_workflows": workflows
        }
        
        # Process templates
        for template in templates:
            metadata = template.get('metadata', {})
            plan["planning_recommendations"].append({
                "template": metadata.get('template_name', 'unknown'),
                "estimated_duration": metadata.get('estimated_duration', 'unknown'),
                "success_rate": metadata.get('success_rate', 0.8),
                "relevance": 1.0 - (template.get('distance', 0.5))
            })
        
        # Process workflows
        for workflow in workflows:
            metadata = workflow.get('metadata', {})
            plan["workflow_suggestions"].append({
                "workflow": metadata.get('workflow_name', 'unknown'),
                "parallel_capable": metadata.get('parallel_capable', False),
                "checkpoint_frequency": metadata.get('checkpoint_frequency', 'standard')
            })
        
        return plan
    
    def store_planning_template(self, template_data: Dict[str, Any]) -> str:
        """Store planning template for future use."""
        content = json.dumps(template_data, indent=2)
        metadata = {
            "type": "planning",
            "template_name": template_data.get('name', 'unnamed'),
            "complexity": template_data.get('complexity', 'medium'),
            "estimated_duration": template_data.get('duration', 'unknown'),
            "tags": ["planning", "template"]
        }
        
        return self.capture_knowledge(content, "pattern", metadata)


# Factory function for creating agent RAG instances
def create_agent_rag(agent_type: str) -> RIFAgentRAG:
    """
    Factory function to create appropriate agent RAG instance.
    
    Args:
        agent_type: Type of RIF agent (analyst, architect, implementer, validator, planner)
        
    Returns:
        Appropriate agent RAG instance
        
    Raises:
        ValueError: If agent type is not recognized
    """
    agent_classes = {
        "analyst": RIFAnalystRAG,
        "architect": RIFArchitectRAG,
        "implementer": RIFImplementerRAG,
        "validator": RIFValidatorRAG,
        "planner": RIFPlannerRAG
    }
    
    agent_class = agent_classes.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_class()


# Convenience functions for direct agent RAG usage
def get_analyst_rag() -> RIFAnalystRAG:
    """Get RIF-Analyst RAG instance."""
    return RIFAnalystRAG()

def get_architect_rag() -> RIFArchitectRAG:
    """Get RIF-Architect RAG instance."""
    return RIFArchitectRAG()

def get_implementer_rag() -> RIFImplementerRAG:
    """Get RIF-Implementer RAG instance."""
    return RIFImplementerRAG()

def get_validator_rag() -> RIFValidatorRAG:
    """Get RIF-Validator RAG instance."""
    return RIFValidatorRAG()

def get_planner_rag() -> RIFPlannerRAG:
    """Get RIF-Planner RAG instance."""
    return RIFPlannerRAG()