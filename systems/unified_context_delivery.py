#!/usr/bin/env python3
"""
RIF Unified Context Delivery System

Provides high-level interface for agents to access unified context from the
Context Integration Hub. This is the primary interface agents should use
to get relevant context information.

Part of Issues #112/#113 implementation - Agent Context Delivery System.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import the context integration hub
try:
    from context_integration_hub import (
        ContextIntegrationHub, ContextRequest, UnifiedContextResponse,
        AgentType, ContextScope, RelevanceWeight
    )
except ImportError:
    # Fallback for development
    print("Warning: Could not import ContextIntegrationHub - using fallback")
    class ContextIntegrationHub:
        def __init__(self, *args, **kwargs): pass
        def get_unified_context(self, *args, **kwargs): return None
        def format_unified_context_for_agent(self, *args, **kwargs): return "Context unavailable"
        def cleanup(self): pass
    
    class ContextRequest:
        def __init__(self, *args, **kwargs): pass
    
    class AgentType(Enum):
        IMPLEMENTER = "rif-implementer"
        ANALYST = "rif-analyst"

class ContextPriority(Enum):
    """Priority levels for context requests"""
    URGENT = "urgent"           # Critical for agent operation
    HIGH = "high"              # Important for quality
    NORMAL = "normal"          # Standard priority  
    LOW = "low"               # Nice to have
    BACKGROUND = "background"  # Can be delayed

@dataclass
class AgentContextProfile:
    """Context profile for different agent types"""
    agent_type: str
    preferred_scope: str
    max_context_tokens: int
    priority_keywords: List[str]
    requires_dependencies: bool
    requires_benchmarking: bool
    typical_tasks: List[str]

class UnifiedContextDelivery:
    """Main interface for agents to request unified context"""
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF"):
        self.repo_path = repo_path
        self.hub = ContextIntegrationHub(repo_path)
        
        # Agent context profiles
        self.agent_profiles = self._initialize_agent_profiles()
        
        # Performance tracking
        self.delivery_metrics = {
            'total_deliveries': 0,
            'avg_delivery_time': 0.0,
            'agent_usage': {},
            'scope_usage': {},
            'satisfaction_scores': []
        }
        
        print("Unified Context Delivery System initialized")
    
    def _initialize_agent_profiles(self) -> Dict[str, AgentContextProfile]:
        """Initialize context profiles for each agent type"""
        return {
            'rif-analyst': AgentContextProfile(
                agent_type='rif-analyst',
                preferred_scope='comprehensive',
                max_context_tokens=2000,
                priority_keywords=['requirements', 'analysis', 'patterns', 'complexity'],
                requires_dependencies=False,
                requires_benchmarking=False,
                typical_tasks=[
                    'analyze issue requirements',
                    'identify patterns and complexity',
                    'extract specifications',
                    'assess implementation scope'
                ]
            ),
            'rif-planner': AgentContextProfile(
                agent_type='rif-planner',
                preferred_scope='comprehensive',
                max_context_tokens=2500,
                priority_keywords=['planning', 'strategy', 'workflow', 'phases', 'dependencies'],
                requires_dependencies=True,
                requires_benchmarking=False,
                typical_tasks=[
                    'create implementation strategy',
                    'plan workflow phases',
                    'assess dependencies',
                    'allocate resources'
                ]
            ),
            'rif-architect': AgentContextProfile(
                agent_type='rif-architect',
                preferred_scope='comprehensive',
                max_context_tokens=3000,
                priority_keywords=['architecture', 'design', 'structure', 'integration', 'components'],
                requires_dependencies=True,
                requires_benchmarking=False,
                typical_tasks=[
                    'design system architecture',
                    'plan component integration',
                    'define interfaces',
                    'create technical specifications'
                ]
            ),
            'rif-implementer': AgentContextProfile(
                agent_type='rif-implementer',
                preferred_scope='standard',
                max_context_tokens=1750,
                priority_keywords=['implementation', 'code', 'development', 'integration', 'testing'],
                requires_dependencies=True,
                requires_benchmarking=False,
                typical_tasks=[
                    'implement features',
                    'write code',
                    'integrate components',
                    'create tests'
                ]
            ),
            'rif-validator': AgentContextProfile(
                agent_type='rif-validator',
                preferred_scope='standard',
                max_context_tokens=2250,
                priority_keywords=['validation', 'testing', 'quality', 'compliance', 'benchmarking'],
                requires_dependencies=False,
                requires_benchmarking=True,
                typical_tasks=[
                    'validate implementations',
                    'run quality checks',
                    'verify compliance',
                    'benchmark performance'
                ]
            ),
            'rif-learner': AgentContextProfile(
                agent_type='rif-learner',
                preferred_scope='comprehensive',
                max_context_tokens=2000,
                priority_keywords=['learning', 'knowledge', 'patterns', 'metrics', 'improvement'],
                requires_dependencies=False,
                requires_benchmarking=True,
                typical_tasks=[
                    'extract learnings',
                    'update knowledge base',
                    'identify patterns',
                    'track metrics'
                ]
            ),
            'rif-pr-manager': AgentContextProfile(
                agent_type='rif-pr-manager',
                preferred_scope='minimal',
                max_context_tokens=1000,
                priority_keywords=['pull request', 'changes', 'summary', 'review'],
                requires_dependencies=False,
                requires_benchmarking=False,
                typical_tasks=[
                    'create pull requests',
                    'summarize changes',
                    'manage reviews'
                ]
            ),
            'rif-error-analyst': AgentContextProfile(
                agent_type='rif-error-analyst',
                preferred_scope='diagnostic',
                max_context_tokens=2500,
                priority_keywords=['error', 'failure', 'debugging', 'analysis', 'resolution'],
                requires_dependencies=True,
                requires_benchmarking=False,
                typical_tasks=[
                    'analyze errors',
                    'debug issues',
                    'identify root causes',
                    'recommend solutions'
                ]
            ),
            'rif-shadow-auditor': AgentContextProfile(
                agent_type='rif-shadow-auditor',
                preferred_scope='diagnostic',
                max_context_tokens=2750,
                priority_keywords=['audit', 'validation', 'adversarial', 'quality', 'compliance'],
                requires_dependencies=True,
                requires_benchmarking=True,
                typical_tasks=[
                    'adversarial testing',
                    'quality auditing',
                    'compliance checking',
                    'risk assessment'
                ]
            ),
            'rif-projectgen': AgentContextProfile(
                agent_type='rif-projectgen',
                preferred_scope='standard',
                max_context_tokens=2250,
                priority_keywords=['project', 'generation', 'structure', 'scaffolding'],
                requires_dependencies=False,
                requires_benchmarking=False,
                typical_tasks=[
                    'generate project structure',
                    'create scaffolding',
                    'setup configurations'
                ]
            )
        }
    
    def get_context_for_agent(self, 
                             agent_type: str,
                             task_description: str,
                             issue_number: Optional[int] = None,
                             priority: ContextPriority = ContextPriority.NORMAL,
                             custom_scope: Optional[str] = None,
                             specific_components: Optional[List[str]] = None) -> str:
        """
        Get formatted context for an agent with their specific profile
        
        Args:
            agent_type: RIF agent type (e.g., 'rif-implementer')
            task_description: Description of the task requiring context
            issue_number: GitHub issue number if applicable
            priority: Context request priority
            custom_scope: Override default scope for agent
            specific_components: Specific components to include context for
            
        Returns:
            Formatted context string ready for agent consumption
        """
        start_time = time.time()
        
        try:
            # Get agent profile
            profile = self.agent_profiles.get(agent_type)
            if not profile:
                return f"Error: Unknown agent type '{agent_type}'"
            
            # Determine context scope
            scope = custom_scope or profile.preferred_scope
            try:
                context_scope = ContextScope(scope.lower())
            except ValueError:
                context_scope = ContextScope.STANDARD
            
            # Convert agent type
            try:
                agent_enum = AgentType(agent_type)
            except ValueError:
                return f"Error: Invalid agent type '{agent_type}'"
            
            # Create context request
            request = ContextRequest(
                agent_type=agent_enum,
                task_description=task_description,
                issue_number=issue_number,
                scope=context_scope,
                specific_components=specific_components or [],
                include_dependencies=profile.requires_dependencies,
                include_benchmarking=profile.requires_benchmarking,
                max_context_size=profile.max_context_tokens * 4,  # Convert to chars
                priority_keywords=profile.priority_keywords,
                force_refresh=priority in [ContextPriority.URGENT, ContextPriority.HIGH]
            )
            
            # Get unified context
            response = self.hub.get_unified_context(request)
            
            # Format for agent
            formatted_context = self.hub.format_unified_context_for_agent(response)
            
            # Add agent-specific formatting
            final_context = self._apply_agent_specific_formatting(
                formatted_context, profile, task_description, priority
            )
            
            # Update metrics
            self._update_delivery_metrics(agent_type, scope, time.time() - start_time)
            
            return final_context
            
        except Exception as e:
            return f"Error generating context for {agent_type}: {str(e)}"
    
    def _apply_agent_specific_formatting(self, 
                                       context: str,
                                       profile: AgentContextProfile,
                                       task_description: str,
                                       priority: ContextPriority) -> str:
        """Apply agent-specific formatting and enhancements"""
        
        enhanced_context = []
        
        # Add agent-specific header
        enhanced_context.append(f"# Context for {profile.agent_type.upper()}")
        enhanced_context.append(f"**Task**: {task_description}")
        enhanced_context.append(f"**Priority**: {priority.value.upper()}")
        enhanced_context.append(f"**Profile**: {profile.preferred_scope} scope, {len(profile.priority_keywords)} focus areas")
        enhanced_context.append("")
        
        # Add agent-specific guidance
        enhanced_context.append("## Agent Focus Areas")
        enhanced_context.append("Based on your agent profile, focus on:")
        for keyword in profile.priority_keywords:
            enhanced_context.append(f"- {keyword.title()}")
        enhanced_context.append("")
        
        # Add typical tasks context
        enhanced_context.append("## Typical Tasks for Your Role")
        for task in profile.typical_tasks[:3]:  # Show top 3
            enhanced_context.append(f"- {task.title()}")
        enhanced_context.append("")
        
        # Add the main context
        enhanced_context.append("---")
        enhanced_context.append("")
        enhanced_context.append(context)
        
        # Add agent-specific footer
        enhanced_context.append("")
        enhanced_context.append("---")
        enhanced_context.append("## Agent Guidelines")
        
        # Agent-specific guidelines
        if profile.agent_type == 'rif-implementer':
            enhanced_context.extend([
                "- Focus on practical implementation details",
                "- Pay attention to dependency information",
                "- Look for integration patterns and code examples",
                "- Consider testing requirements early"
            ])
        elif profile.agent_type == 'rif-validator':
            enhanced_context.extend([
                "- Emphasize quality gates and compliance",
                "- Use benchmarking data for validation",
                "- Focus on testability and measurability",
                "- Look for specification adherence"
            ])
        elif profile.agent_type == 'rif-analyst':
            enhanced_context.extend([
                "- Extract all requirements and constraints",
                "- Identify complexity patterns and risks",
                "- Look for similar issues and solutions",
                "- Consider non-functional requirements"
            ])
        elif profile.agent_type == 'rif-architect':
            enhanced_context.extend([
                "- Focus on system structure and dependencies",
                "- Consider integration points and interfaces",
                "- Look for architectural constraints",
                "- Plan for scalability and maintainability"
            ])
        else:
            enhanced_context.extend([
                "- Follow your agent-specific responsibilities",
                "- Use the provided context to inform decisions",
                "- Focus on your priority keywords",
                "- Deliver high-quality output"
            ])
        
        enhanced_context.append("")
        enhanced_context.append(f"*Context delivered via Unified Context Delivery System*")
        
        return "\n".join(enhanced_context)
    
    def get_minimal_context(self, agent_type: str, task_description: str) -> str:
        """Get minimal context for quick tasks"""
        return self.get_context_for_agent(
            agent_type=agent_type,
            task_description=task_description,
            custom_scope='minimal',
            priority=ContextPriority.LOW
        )
    
    def get_comprehensive_context(self, agent_type: str, 
                                task_description: str,
                                issue_number: Optional[int] = None) -> str:
        """Get comprehensive context for complex tasks"""
        return self.get_context_for_agent(
            agent_type=agent_type,
            task_description=task_description,
            issue_number=issue_number,
            custom_scope='comprehensive',
            priority=ContextPriority.HIGH
        )
    
    def get_diagnostic_context(self, agent_type: str,
                             task_description: str,
                             specific_components: List[str]) -> str:
        """Get diagnostic context for troubleshooting"""
        return self.get_context_for_agent(
            agent_type=agent_type,
            task_description=task_description,
            custom_scope='diagnostic',
            priority=ContextPriority.URGENT,
            specific_components=specific_components
        )
    
    def get_context_templates(self) -> Dict[str, Any]:
        """Get context templates for different agent types"""
        templates = {}
        
        for agent_type, profile in self.agent_profiles.items():
            templates[agent_type] = {
                'preferred_scope': profile.preferred_scope,
                'max_tokens': profile.max_context_tokens,
                'priority_keywords': profile.priority_keywords,
                'requires_dependencies': profile.requires_dependencies,
                'requires_benchmarking': profile.requires_benchmarking,
                'typical_tasks': profile.typical_tasks,
                'example_request': {
                    'agent_type': agent_type,
                    'task_description': f"Example {profile.typical_tasks[0]}",
                    'scope': profile.preferred_scope,
                    'include_dependencies': profile.requires_dependencies,
                    'include_benchmarking': profile.requires_benchmarking
                }
            }
        
        return templates
    
    def update_agent_satisfaction(self, agent_type: str, 
                                satisfaction_score: float,
                                feedback: str = "") -> None:
        """Update agent satisfaction metrics for context delivery"""
        if 0 <= satisfaction_score <= 1.0:
            self.delivery_metrics['satisfaction_scores'].append({
                'agent_type': agent_type,
                'score': satisfaction_score,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_delivery_metrics(self) -> Dict[str, Any]:
        """Get context delivery performance metrics"""
        
        # Calculate satisfaction average
        if self.delivery_metrics['satisfaction_scores']:
            avg_satisfaction = sum(
                score['score'] for score in self.delivery_metrics['satisfaction_scores']
            ) / len(self.delivery_metrics['satisfaction_scores'])
        else:
            avg_satisfaction = 0.0
        
        return {
            'total_deliveries': self.delivery_metrics['total_deliveries'],
            'average_delivery_time_ms': self.delivery_metrics['avg_delivery_time'],
            'agent_usage': self.delivery_metrics['agent_usage'],
            'scope_usage': self.delivery_metrics['scope_usage'],
            'average_satisfaction': avg_satisfaction,
            'total_satisfaction_reports': len(self.delivery_metrics['satisfaction_scores']),
            'hub_status': self.hub.get_hub_status() if self.hub else 'unavailable'
        }
    
    def _update_delivery_metrics(self, agent_type: str, scope: str, 
                               delivery_time: float) -> None:
        """Update delivery performance metrics"""
        self.delivery_metrics['total_deliveries'] += 1
        
        # Update average delivery time
        total = self.delivery_metrics['total_deliveries']
        current_avg = self.delivery_metrics['avg_delivery_time']
        delivery_time_ms = delivery_time * 1000
        
        new_avg = ((current_avg * (total - 1)) + delivery_time_ms) / total
        self.delivery_metrics['avg_delivery_time'] = new_avg
        
        # Update agent usage
        if agent_type not in self.delivery_metrics['agent_usage']:
            self.delivery_metrics['agent_usage'][agent_type] = 0
        self.delivery_metrics['agent_usage'][agent_type] += 1
        
        # Update scope usage
        if scope not in self.delivery_metrics['scope_usage']:
            self.delivery_metrics['scope_usage'][scope] = 0
        self.delivery_metrics['scope_usage'][scope] += 1
    
    def test_agent_context(self, agent_type: str) -> Dict[str, Any]:
        """Test context delivery for an agent type"""
        profile = self.agent_profiles.get(agent_type)
        if not profile:
            return {'error': f'Unknown agent type: {agent_type}'}
        
        test_results = {}
        
        # Test each scope
        scopes = ['minimal', 'standard', 'comprehensive', 'diagnostic']
        for scope in scopes:
            start_time = time.time()
            
            try:
                context = self.get_context_for_agent(
                    agent_type=agent_type,
                    task_description=f"Test {scope} context for {agent_type}",
                    custom_scope=scope
                )
                
                delivery_time = (time.time() - start_time) * 1000
                context_size = len(context)
                
                test_results[scope] = {
                    'success': True,
                    'delivery_time_ms': delivery_time,
                    'context_size_chars': context_size,
                    'estimated_tokens': context_size // 4,
                    'within_limits': context_size <= (profile.max_context_tokens * 4)
                }
                
            except Exception as e:
                test_results[scope] = {
                    'success': False,
                    'error': str(e),
                    'delivery_time_ms': (time.time() - start_time) * 1000
                }
        
        return {
            'agent_type': agent_type,
            'profile': profile.__dict__,
            'test_results': test_results,
            'overall_success': all(result.get('success', False) 
                                 for result in test_results.values())
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.hub:
            self.hub.cleanup()
        print("Unified Context Delivery System cleaned up")

# Convenience functions for direct use
def get_implementer_context(task_description: str, 
                          issue_number: Optional[int] = None,
                          repo_path: str = "/Users/cal/DEV/RIF") -> str:
    """Get context optimized for RIF-Implementer"""
    delivery = UnifiedContextDelivery(repo_path)
    try:
        return delivery.get_context_for_agent('rif-implementer', task_description, issue_number)
    finally:
        delivery.cleanup()

def get_validator_context(task_description: str, 
                         issue_number: Optional[int] = None,
                         repo_path: str = "/Users/cal/DEV/RIF") -> str:
    """Get context optimized for RIF-Validator"""
    delivery = UnifiedContextDelivery(repo_path)
    try:
        return delivery.get_context_for_agent('rif-validator', task_description, issue_number)
    finally:
        delivery.cleanup()

def get_analyst_context(task_description: str, 
                       issue_number: Optional[int] = None,
                       repo_path: str = "/Users/cal/DEV/RIF") -> str:
    """Get context optimized for RIF-Analyst"""
    delivery = UnifiedContextDelivery(repo_path)
    try:
        return delivery.get_context_for_agent('rif-analyst', task_description, issue_number)
    finally:
        delivery.cleanup()

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Unified Context Delivery System")
    parser.add_argument("--agent", type=str, required=True,
                       help="Agent type (e.g., rif-implementer)")
    parser.add_argument("--task", type=str, required=True,
                       help="Task description")
    parser.add_argument("--issue", type=int,
                       help="GitHub issue number")
    parser.add_argument("--scope", type=str, default="standard",
                       choices=['minimal', 'standard', 'comprehensive', 'diagnostic'],
                       help="Context scope")
    parser.add_argument("--priority", type=str, default="normal",
                       choices=['urgent', 'high', 'normal', 'low', 'background'],
                       help="Request priority")
    parser.add_argument("--components", type=str, nargs='*',
                       help="Specific components to include")
    parser.add_argument("--templates", action="store_true",
                       help="Show context templates")
    parser.add_argument("--test", action="store_true",
                       help="Test agent context delivery")
    parser.add_argument("--metrics", action="store_true",
                       help="Show delivery metrics")
    parser.add_argument("--repo", type=str, default="/Users/cal/DEV/RIF",
                       help="Repository path")
    
    args = parser.parse_args()
    
    delivery = UnifiedContextDelivery(args.repo)
    
    try:
        if args.templates:
            templates = delivery.get_context_templates()
            print("=== Context Templates ===")
            print(json.dumps(templates, indent=2))
            
        elif args.test:
            test_results = delivery.test_agent_context(args.agent)
            print("=== Agent Context Test Results ===")
            print(json.dumps(test_results, indent=2, default=str))
            
        elif args.metrics:
            metrics = delivery.get_delivery_metrics()
            print("=== Context Delivery Metrics ===")
            print(json.dumps(metrics, indent=2, default=str))
            
        else:
            try:
                priority = ContextPriority(args.priority)
            except ValueError:
                priority = ContextPriority.NORMAL
            
            context = delivery.get_context_for_agent(
                agent_type=args.agent,
                task_description=args.task,
                issue_number=args.issue,
                priority=priority,
                custom_scope=args.scope,
                specific_components=args.components
            )
            
            print(context)
            
    finally:
        delivery.cleanup()