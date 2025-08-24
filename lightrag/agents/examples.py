"""
Example usage patterns for RIF Agent LightRAG integration.
Demonstrates how each agent type can leverage the knowledge base.
"""

import json
from datetime import datetime
from typing import Dict, Any

from agent_integration import (
    get_analyst_rag, get_architect_rag, get_implementer_rag, 
    get_validator_rag, get_planner_rag
)
from utils import agent_knowledge_session, KnowledgePatternMatcher


def example_analyst_workflow():
    """
    Example of how RIF-Analyst uses LightRAG for issue analysis.
    """
    print("=== RIF-Analyst LightRAG Example ===")
    
    # Simulate GitHub issue data
    issue_data = {
        "title": "Implement user authentication system",
        "description": "Need to add JWT-based authentication with role-based access control",
        "labels": ["enhancement", "security"],
        "complexity": "high"
    }
    
    with agent_knowledge_session('analyst', 'demo_analyst_001') as session:
        analyst = session.agent_rag
        
        # Step 1: Analyze with historical knowledge
        print("Analyzing issue with historical context...")
        analysis = analyst.analyze_with_history(issue_data)
        
        print(f"Found {analysis['similar_issues_found']} similar issues")
        print(f"Found {analysis['patterns_found']} relevant patterns")
        
        # Step 2: Extract insights from similar issues
        if analysis['similar_issues']:
            print("\nSimilar issues found:")
            for issue in analysis['similar_issues'][:2]:
                print(f"- {issue.get('id', 'unknown')}: {issue.get('content', '')[:100]}...")
        
        # Step 3: Store analysis results
        analysis_content = {
            "issue_id": "gh_issue_123",
            "complexity": "high",
            "estimated_effort": "2-3 weeks",
            "dependencies": ["user management", "session handling"],
            "success_factors": analysis['recommendations']
        }
        
        doc_id = analyst.store_analysis_results("gh_issue_123", analysis_content)
        print(f"\nStored analysis results: {doc_id}")
        
        # Show session summary
        summary = session.get_session_summary()
        print(f"\nSession summary: {summary['queries_performed']} queries, "
              f"{summary['knowledge_captured']} items captured")


def example_architect_workflow():
    """
    Example of how RIF-Architect uses LightRAG for system design.
    """
    print("\n=== RIF-Architect LightRAG Example ===")
    
    # Simulate design requirements
    requirements = {
        "description": "Microservices architecture for e-commerce platform",
        "technology": "Python",
        "scale": "high",
        "constraints": ["low latency", "high availability"]
    }
    
    with agent_knowledge_session('architect', 'demo_architect_001') as session:
        architect = session.agent_rag
        
        # Step 1: Design with knowledge base
        print("Creating architecture design with knowledge base...")
        design = architect.design_with_knowledge(requirements)
        
        print(f"Based on {design['based_on_decisions']} past decisions")
        print(f"Found {design['applicable_patterns']} applicable patterns")
        
        # Step 2: Show recommendations
        if design['recommendations']:
            print("\nRecommendations from past decisions:")
            for rec in design['recommendations'][:2]:
                print(f"- {rec['recommendation']} (confidence: {rec['confidence']:.2f})")
        
        # Step 3: Store design decision
        decision_data = {
            "decision_id": "arch_decision_456",
            "context": "E-commerce microservices architecture",
            "decision": "Use event-driven architecture with message queues",
            "rationale": "Proven pattern for scalable e-commerce systems",
            "status": "active",
            "impact": "high"
        }
        
        doc_id = architect.store_design_decision(decision_data)
        print(f"\nStored design decision: {doc_id}")


def example_implementer_workflow():
    """
    Example of how RIF-Implementer uses LightRAG for code implementation.
    """
    print("\n=== RIF-Implementer LightRAG Example ===")
    
    # Simulate implementation task
    task_data = {
        "description": "Implement JWT token validation middleware",
        "language": "Python",
        "framework": "FastAPI",
        "requirements": ["token validation", "error handling", "logging"]
    }
    
    with agent_knowledge_session('implementer', 'demo_implementer_001') as session:
        implementer = session.agent_rag
        
        # Step 1: Plan implementation with knowledge
        print("Planning implementation with knowledge base...")
        implementation = implementer.implement_with_knowledge(task_data)
        
        print(f"Found {implementation['code_examples_found']} code examples")
        print(f"Found {implementation['patterns_found']} implementation patterns")
        
        # Step 2: Show reusable components
        if implementation['reusable_components']:
            print("\nReusable components found:")
            for comp in implementation['reusable_components'][:2]:
                print(f"- {comp['component']} ({comp['language']}) - "
                      f"relevance: {comp['relevance']:.2f}")
        
        # Step 3: Store code snippet
        code_data = {
            "name": "jwt_validation_middleware",
            "language": "Python",
            "framework": "FastAPI",
            "code": """
from fastapi import HTTPException, Depends
from jose import JWTError, jwt

def validate_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
""",
            "complexity": "medium",
            "tags": ["authentication", "middleware", "FastAPI"]
        }
        
        doc_id = implementer.store_code_snippet(code_data)
        print(f"\nStored code snippet: {doc_id}")


def example_validator_workflow():
    """
    Example of how RIF-Validator uses LightRAG for test planning.
    """
    print("\n=== RIF-Validator LightRAG Example ===")
    
    # Simulate validation requirements
    validation_data = {
        "functionality": "JWT authentication middleware",
        "project_type": "web_api",
        "requirements": ["unit tests", "integration tests", "security tests"]
    }
    
    with agent_knowledge_session('validator', 'demo_validator_001') as session:
        validator = session.agent_rag
        
        # Step 1: Plan validation with knowledge
        print("Planning validation with knowledge base...")
        validation = validator.validate_with_knowledge(validation_data)
        
        print(f"Found {validation['test_patterns_found']} test patterns")
        print(f"Found {validation['quality_gates_found']} quality gates")
        
        # Step 2: Show test strategies
        if validation['test_strategies']:
            print("\nTest strategies found:")
            for strategy in validation['test_strategies'][:2]:
                print(f"- {strategy['strategy']} ({strategy['test_type']}) - "
                      f"target coverage: {strategy['coverage_target']}%")
        
        # Step 3: Store test pattern
        pattern_data = {
            "name": "jwt_middleware_test_pattern",
            "test_type": "unit",
            "strategy": "mock-based testing",
            "coverage_target": 95,
            "test_cases": [
                "valid token acceptance",
                "invalid token rejection", 
                "expired token handling",
                "malformed token handling"
            ]
        }
        
        doc_id = validator.store_test_pattern(pattern_data)
        print(f"\nStored test pattern: {doc_id}")


def example_planner_workflow():
    """
    Example of how RIF-Planner uses LightRAG for project planning.
    """
    print("\n=== RIF-Planner LightRAG Example ===")
    
    # Simulate planning requirements
    planning_data = {
        "description": "Multi-service authentication system implementation",
        "complexity": "high", 
        "project_type": "microservices",
        "constraints": ["2 week deadline", "3 developers", "security critical"]
    }
    
    with agent_knowledge_session('planner', 'demo_planner_001') as session:
        planner = session.agent_rag
        
        # Step 1: Create plan with knowledge
        print("Creating project plan with knowledge base...")
        plan = planner.plan_with_knowledge(planning_data)
        
        print(f"Found {plan['templates_found']} planning templates")
        print(f"Found {plan['workflows_found']} workflow patterns")
        
        # Step 2: Show planning recommendations
        if plan['planning_recommendations']:
            print("\nPlanning recommendations:")
            for rec in plan['planning_recommendations'][:2]:
                print(f"- {rec['template']} (duration: {rec['estimated_duration']}, "
                      f"success rate: {rec['success_rate']:.1%})")
        
        # Step 3: Store planning template
        template_data = {
            "name": "auth_system_implementation_template",
            "complexity": "high",
            "duration": "2-3 weeks",
            "phases": [
                "Analysis and design (3 days)",
                "Core implementation (7 days)", 
                "Testing and validation (3 days)",
                "Documentation and deployment (2 days)"
            ],
            "success_factors": ["early security review", "incremental testing"]
        }
        
        doc_id = planner.store_planning_template(template_data)
        print(f"\nStored planning template: {doc_id}")


def example_pattern_analysis():
    """
    Example of advanced pattern analysis across agents.
    """
    print("\n=== Advanced Pattern Analysis Example ===")
    
    # Create pattern matcher with analyst agent
    analyst_rag = get_analyst_rag()
    pattern_matcher = KnowledgePatternMatcher(analyst_rag)
    
    # Find evolution patterns
    print("Finding evolution patterns for 'authentication'...")
    evolution = pattern_matcher.find_evolution_patterns("authentication", 30)
    print(f"Found {len(evolution)} evolution points")
    
    # Find success patterns
    print("\nFinding success patterns for 'microservices'...")
    success_patterns = pattern_matcher.find_success_patterns("microservices")
    print(f"Found {len(success_patterns)} success patterns")
    
    # Find anti-patterns
    print("\nFinding anti-patterns for 'authentication'...")
    anti_patterns = pattern_matcher.find_anti_patterns("authentication")
    print(f"Found {len(anti_patterns)} anti-patterns to avoid")


def example_multi_agent_session():
    """
    Example of coordinated multi-agent knowledge usage.
    """
    print("\n=== Multi-Agent Coordination Example ===")
    
    # Simulate workflow where analysts findings inform architect
    issue_context = "Implement real-time notification system"
    
    # Step 1: Analyst phase
    print("Phase 1: Analysis...")
    with agent_knowledge_session('analyst') as analyst_session:
        analyst = analyst_session.agent_rag
        
        # Query for similar notification systems
        similar_systems = analyst.query_similar_work("real-time notifications")
        print(f"Analyst found {len(similar_systems)} similar systems")
        
        # Store analysis insight
        analysis_insight = {
            "finding": "Real-time notifications commonly use WebSocket or Server-Sent Events",
            "complexity": "medium",
            "technologies": ["WebSocket", "SSE", "message queues"]
        }
        
        analyst.capture_knowledge(
            json.dumps(analysis_insight), 
            "pattern", 
            {"type": "analysis_insight", "domain": "notifications"}
        )
    
    # Step 2: Architect phase - uses analyst's findings
    print("\nPhase 2: Architecture...")
    with agent_knowledge_session('architect') as architect_session:
        architect = architect_session.agent_rag
        
        # Query for recent analysis insights
        recent_insights = architect.query_similar_work("real-time notifications analysis")
        print(f"Architect found {len(recent_insights)} recent insights")
        
        # Create architecture based on insights
        arch_decision = {
            "decision": "Use WebSocket with Redis pub/sub for real-time notifications",
            "based_on": "Analysis findings and proven patterns",
            "trade_offs": "Higher complexity but better scalability"
        }
        
        architect.capture_knowledge(
            json.dumps(arch_decision),
            "decision",
            {"type": "architecture_decision", "domain": "notifications"}
        )
    
    print("Multi-agent coordination completed successfully!")


def run_all_examples():
    """Run all example workflows."""
    print("Running RIF Agent LightRAG Integration Examples")
    print("=" * 60)
    
    try:
        example_analyst_workflow()
        example_architect_workflow() 
        example_implementer_workflow()
        example_validator_workflow()
        example_planner_workflow()
        example_pattern_analysis()
        example_multi_agent_session()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This is expected if LightRAG database is not properly initialized")


if __name__ == "__main__":
    run_all_examples()