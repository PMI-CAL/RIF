#!/usr/bin/env python3
"""
RIF Context Optimization Engine

Implements intelligent context filtering and optimization for agents to provide
the "right" information without clogging context windows with irrelevant data.

Addresses Issue #112 and #113 requirements for intelligent agent context management.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import subprocess
import os

class AgentType(Enum):
    """RIF Agent Types"""
    ANALYST = "rif-analyst"
    PLANNER = "rif-planner" 
    ARCHITECT = "rif-architect"
    IMPLEMENTER = "rif-implementer"
    VALIDATOR = "rif-validator"
    LEARNER = "rif-learner"
    PR_MANAGER = "rif-pr-manager"
    ERROR_ANALYST = "rif-error-analyst"
    SHADOW_AUDITOR = "rif-shadow-auditor"
    PROJECTGEN = "rif-projectgen"

class ContextType(Enum):
    """Types of context information"""
    CLAUDE_CODE_CAPABILITIES = "claude_code_capabilities"
    SYSTEM_OVERVIEW = "system_overview"
    IMPLEMENTATION_PATTERNS = "implementation_patterns"
    ARCHITECTURAL_DECISIONS = "architectural_decisions"
    SIMILAR_ISSUES = "similar_issues"
    DEPENDENCY_INFO = "dependency_info"
    QUALITY_PATTERNS = "quality_patterns"
    ERROR_PATTERNS = "error_patterns"
    PERFORMANCE_DATA = "performance_data"
    SECURITY_PATTERNS = "security_patterns"

@dataclass
class ContextItem:
    """Individual context item with relevance metadata"""
    id: str
    type: ContextType
    content: str
    relevance_score: float
    last_updated: datetime
    source: str
    agent_relevance: Dict[AgentType, float]
    size_estimate: int  # Characters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        data['last_updated'] = self.last_updated.isoformat()
        data['agent_relevance'] = {k.value: v for k, v in self.agent_relevance.items()}
        return data

@dataclass
class SystemContext:
    """Live system context for big picture awareness"""
    overview: str
    purpose: str
    design_goals: List[str]
    constraints: List[str]
    dependencies: Dict[str, Any]
    architecture_summary: str
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data

@dataclass
class AgentContext:
    """Optimized context for specific agent type"""
    agent_type: AgentType
    relevant_knowledge: List[ContextItem]
    system_context: SystemContext
    task_specific_context: Dict[str, Any]
    context_window_utilization: float
    total_size: int
    
class ContextOptimizer:
    """
    Intelligent context optimization engine that provides relevant information
    to agents without overwhelming their context windows.
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.knowledge_base_path = knowledge_base_path
        self.context_cache = {}
        self.agent_context_limits = {
            AgentType.ANALYST: 4000,      # Characters for context
            AgentType.PLANNER: 5000,      # Needs more planning context
            AgentType.ARCHITECT: 6000,    # Needs architectural context
            AgentType.IMPLEMENTER: 3500,  # Focused implementation context
            AgentType.VALIDATOR: 4500,    # Needs validation patterns
            AgentType.LEARNER: 3000,      # Focused learning context
            AgentType.PR_MANAGER: 2500,   # Focused PR context
            AgentType.ERROR_ANALYST: 4000, # Error analysis context
            AgentType.SHADOW_AUDITOR: 5000, # Adversarial context
            AgentType.PROJECTGEN: 4500,   # Project generation context
        }
        self.relevance_weights = self._initialize_relevance_weights()
        
    def _initialize_relevance_weights(self) -> Dict[AgentType, Dict[ContextType, float]]:
        """Initialize relevance weights for each agent type and context type"""
        return {
            AgentType.ANALYST: {
                ContextType.CLAUDE_CODE_CAPABILITIES: 1.0,
                ContextType.SYSTEM_OVERVIEW: 0.8,
                ContextType.SIMILAR_ISSUES: 0.9,
                ContextType.IMPLEMENTATION_PATTERNS: 0.7,
                ContextType.ARCHITECTURAL_DECISIONS: 0.6,
                ContextType.DEPENDENCY_INFO: 0.5,
                ContextType.QUALITY_PATTERNS: 0.4,
                ContextType.ERROR_PATTERNS: 0.6,
                ContextType.PERFORMANCE_DATA: 0.3,
                ContextType.SECURITY_PATTERNS: 0.4,
            },
            AgentType.IMPLEMENTER: {
                ContextType.CLAUDE_CODE_CAPABILITIES: 1.0,
                ContextType.IMPLEMENTATION_PATTERNS: 0.9,
                ContextType.ARCHITECTURAL_DECISIONS: 0.8,
                ContextType.DEPENDENCY_INFO: 0.8,
                ContextType.SIMILAR_ISSUES: 0.7,
                ContextType.SYSTEM_OVERVIEW: 0.6,
                ContextType.QUALITY_PATTERNS: 0.7,
                ContextType.ERROR_PATTERNS: 0.5,
                ContextType.PERFORMANCE_DATA: 0.6,
                ContextType.SECURITY_PATTERNS: 0.6,
            },
            AgentType.VALIDATOR: {
                ContextType.CLAUDE_CODE_CAPABILITIES: 1.0,
                ContextType.QUALITY_PATTERNS: 0.9,
                ContextType.ERROR_PATTERNS: 0.9,
                ContextType.SECURITY_PATTERNS: 0.8,
                ContextType.PERFORMANCE_DATA: 0.8,
                ContextType.IMPLEMENTATION_PATTERNS: 0.7,
                ContextType.SIMILAR_ISSUES: 0.6,
                ContextType.SYSTEM_OVERVIEW: 0.5,
                ContextType.ARCHITECTURAL_DECISIONS: 0.6,
                ContextType.DEPENDENCY_INFO: 0.5,
            },
            # Add similar weights for other agents...
        }
    
    def optimize_for_agent(self, agent_type: AgentType, task_context: Dict[str, Any], 
                          issue_number: Optional[int] = None) -> AgentContext:
        """
        Optimize context for specific agent type and task.
        
        Args:
            agent_type: Type of RIF agent requesting context
            task_context: Specific task information (issue details, etc.)
            issue_number: GitHub issue number for context
            
        Returns:
            AgentContext: Optimized context for the agent
        """
        # 1. Gather all available knowledge
        available_knowledge = self._gather_available_knowledge(task_context, issue_number)
        
        # 2. Score knowledge relevance for this agent
        relevance_scores = self._score_knowledge_relevance(
            agent_type, task_context, available_knowledge
        )
        
        # 3. Filter by relevance and context window size
        # Handle cross-module enum instance issues
        if agent_type in self.agent_context_limits:
            context_limit = self.agent_context_limits[agent_type]
        else:
            # Fallback based on agent type value string
            agent_value = getattr(agent_type, 'value', str(agent_type))
            if 'analyst' in agent_value.lower():
                context_limit = 4000
            elif 'planner' in agent_value.lower():
                context_limit = 5000
            elif 'architect' in agent_value.lower():
                context_limit = 6000
            elif 'implementer' in agent_value.lower():
                context_limit = 3500
            elif 'validator' in agent_value.lower():
                context_limit = 4500
            else:
                context_limit = 4000  # Default
        
        filtered_context = self._filter_by_relevance_and_size(
            available_knowledge, relevance_scores, context_limit
        )
        
        # 4. Get live system context
        system_context = self._get_live_system_context()
        
        # 5. Format for agent consumption
        return AgentContext(
            agent_type=agent_type,
            relevant_knowledge=filtered_context,
            system_context=system_context,
            task_specific_context=task_context,
            context_window_utilization=self._calculate_utilization(filtered_context, system_context),
            total_size=sum(item.size_estimate for item in filtered_context)
        )
    
    def _gather_available_knowledge(self, task_context: Dict[str, Any], 
                                   issue_number: Optional[int] = None) -> List[ContextItem]:
        """Gather all available knowledge from various sources"""
        knowledge_items = []
        
        # 1. Claude Code capabilities (always include)
        claude_capabilities = self._query_claude_capabilities(task_context)
        knowledge_items.extend(claude_capabilities)
        
        # 2. System overview and big picture
        system_overview = self._get_system_overview_items()
        knowledge_items.extend(system_overview)
        
        # 3. Similar issues from knowledge base
        if issue_number:
            similar_issues = self._query_similar_issues(issue_number, task_context)
            knowledge_items.extend(similar_issues)
        
        # 4. Implementation patterns
        patterns = self._query_implementation_patterns(task_context)
        knowledge_items.extend(patterns)
        
        # 5. Architectural decisions
        decisions = self._query_architectural_decisions(task_context)
        knowledge_items.extend(decisions)
        
        return knowledge_items
    
    def _query_claude_capabilities(self, task_context: Dict[str, Any]) -> List[ContextItem]:
        """Query Claude Code capabilities from MCP knowledge server"""
        capabilities = []
        
        try:
            # Use MCP knowledge server to get Claude Code capabilities
            # This would use actual MCP calls in production
            claude_content = (
                "Claude Code is an AI-powered developer tool for terminal-based coding assistance. "
                "Core capabilities: Read/Write/Edit/MultiEdit files, Bash command execution, "
                "Git integration, Search (Grep/Glob), MCP server connections, WebFetch, "
                "thinking budget triggers ('think harder'), Task tool for subagent creation. "
                "NOT a multi-agent orchestration platform - it's a development assistant with "
                "performance optimization through caching and intelligent tool use."
            )
        except Exception as e:
            # Fallback for reliable operation
            claude_content = (
                "Claude Code: AI developer tool with file operations, bash commands, git integration, "
                "search tools, MCP servers, and Task tool for subagents. Performance-optimized."
            )
        
        claude_info = ContextItem(
            id="claude-core-capabilities-mcp",
            type=ContextType.CLAUDE_CODE_CAPABILITIES,
            content=claude_content,
            relevance_score=1.0,
            last_updated=datetime.now(),
            source="mcp-knowledge-server",
            agent_relevance={agent: 1.0 for agent in AgentType},
            size_estimate=len(claude_content)
        )
        capabilities.append(claude_info)
        
        return capabilities
    
    def _get_system_overview_items(self) -> List[ContextItem]:
        """Get live system overview context"""
        overview = ContextItem(
            id="rif-system-overview",
            type=ContextType.SYSTEM_OVERVIEW,
            content="RIF (Reactive Intelligence Framework) is an automatic intelligent development "
                   "system that orchestrates specialized agents through GitHub issues with zero "
                   "manual intervention. Uses state machine workflow, quality gates, and "
                   "knowledge base learning from every interaction.",
            relevance_score=0.8,
            last_updated=datetime.now(),
            source="live-system-analysis",
            agent_relevance={
                AgentType.ANALYST: 0.8,
                AgentType.PLANNER: 0.9,
                AgentType.ARCHITECT: 0.9,
                AgentType.IMPLEMENTER: 0.6,
                AgentType.VALIDATOR: 0.5,
                AgentType.LEARNER: 0.7,
                AgentType.PR_MANAGER: 0.4,
                AgentType.ERROR_ANALYST: 0.6,
                AgentType.SHADOW_AUDITOR: 0.5,
                AgentType.PROJECTGEN: 0.8,
            },
            size_estimate=280
        )
        return [overview]
    
    def _query_similar_issues(self, issue_number: int, task_context: Dict[str, Any]) -> List[ContextItem]:
        """Query similar issues from knowledge base"""
        # This would query the actual MCP knowledge server
        # For now, return simulated similar issues
        similar = ContextItem(
            id=f"similar-issues-{issue_number}",
            type=ContextType.SIMILAR_ISSUES,
            content="Found 3 similar issues with context optimization patterns. "
                   "Previous implementations focused on agent-specific filtering "
                   "and performance optimization approaches.",
            relevance_score=0.7,
            last_updated=datetime.now(),
            source="mcp-knowledge-server",
            agent_relevance={agent: 0.7 for agent in AgentType},
            size_estimate=150
        )
        return [similar]
    
    def _query_implementation_patterns(self, task_context: Dict[str, Any]) -> List[ContextItem]:
        """Query implementation patterns from knowledge base"""
        patterns = ContextItem(
            id="context-optimization-patterns",
            type=ContextType.IMPLEMENTATION_PATTERNS,
            content="Context optimization patterns: Multi-factor relevance scoring, "
                   "agent-specific filtering, dynamic window management, "
                   "performance-based caching, and knowledge base integration.",
            relevance_score=0.8,
            last_updated=datetime.now(),
            source="mcp-knowledge-server",
            agent_relevance={
                AgentType.ANALYST: 0.7,
                AgentType.PLANNER: 0.8,
                AgentType.ARCHITECT: 0.9,
                AgentType.IMPLEMENTER: 0.9,
                AgentType.VALIDATOR: 0.6,
                AgentType.LEARNER: 0.5,
                AgentType.PR_MANAGER: 0.3,
                AgentType.ERROR_ANALYST: 0.4,
                AgentType.SHADOW_AUDITOR: 0.4,
                AgentType.PROJECTGEN: 0.5,
            },
            size_estimate=200
        )
        return [patterns]
    
    def _query_architectural_decisions(self, task_context: Dict[str, Any]) -> List[ContextItem]:
        """Query architectural decisions from knowledge base"""
        decisions = ContextItem(
            id="context-architecture-decisions",
            type=ContextType.ARCHITECTURAL_DECISIONS,
            content="Key decisions: Agent-specific context limits, relevance-based filtering, "
                   "performance budgets <200ms, caching strategies, and MCP integration patterns.",
            relevance_score=0.6,
            last_updated=datetime.now(),
            source="mcp-knowledge-server",
            agent_relevance={
                AgentType.ANALYST: 0.6,
                AgentType.PLANNER: 0.8,
                AgentType.ARCHITECT: 0.9,
                AgentType.IMPLEMENTER: 0.7,
                AgentType.VALIDATOR: 0.5,
                AgentType.LEARNER: 0.4,
                AgentType.PR_MANAGER: 0.3,
                AgentType.ERROR_ANALYST: 0.4,
                AgentType.SHADOW_AUDITOR: 0.4,
                AgentType.PROJECTGEN: 0.6,
            },
            size_estimate=180
        )
        return [decisions]
    
    def _score_knowledge_relevance(self, agent_type: AgentType, task_context: Dict[str, Any],
                                  available_knowledge: List[ContextItem]) -> Dict[str, float]:
        """Enhanced multi-factor relevance scoring for specific agent and task"""
        scores = {}
        
        # Get base weights for this agent type
        agent_weights = self.relevance_weights.get(agent_type, {})
        
        for item in available_knowledge:
            # Factor 1: Agent-type specific base scoring (30%)
            base_score = agent_weights.get(item.type, 0.5)
            
            # Factor 2: Pre-computed agent relevance (25%)
            agent_relevance = item.agent_relevance.get(agent_type, 0.5)
            
            # Factor 3: Task-specific relevance adjustments (20%)
            task_adjustment = self._calculate_task_relevance(item, task_context)
            
            # Factor 4: Freshness factor - newer is more relevant (10%)
            freshness_factor = self._calculate_freshness_factor(item.last_updated)
            
            # Factor 5: Usage success rate from historical data (15%)
            success_factor = self._calculate_usage_success_factor(item, agent_type)
            
            # Multi-factor weighted combination
            final_score = (
                base_score * 0.30 + 
                agent_relevance * 0.25 + 
                task_adjustment * 0.20 + 
                freshness_factor * 0.10 +
                success_factor * 0.15
            )
            
            scores[item.id] = min(1.0, max(0.0, final_score))
        
        return scores
    
    def _calculate_task_relevance(self, item: ContextItem, task_context: Dict[str, Any]) -> float:
        """Calculate task-specific relevance adjustments"""
        # Simple keyword matching for task relevance
        task_description = task_context.get('description', '').lower()
        content_lower = item.content.lower()
        
        # Look for keyword overlaps
        task_keywords = set(re.findall(r'\b\w{4,}\b', task_description))
        content_keywords = set(re.findall(r'\b\w{4,}\b', content_lower))
        
        overlap = len(task_keywords & content_keywords)
        max_keywords = max(len(task_keywords), len(content_keywords), 1)
        
        return min(1.0, overlap / max_keywords * 2)
    
    def _calculate_freshness_factor(self, last_updated: datetime) -> float:
        """Calculate freshness factor (newer is more relevant)"""
        age_days = (datetime.now() - last_updated).days
        
        # Fresh within 7 days = 1.0, decay after that
        if age_days <= 7:
            return 1.0
        elif age_days <= 30:
            return 0.8
        elif age_days <= 90:
            return 0.6
        else:
            return 0.4
    
    def _calculate_usage_success_factor(self, item: ContextItem, agent_type: AgentType) -> float:
        """Calculate usage success rate from historical data"""
        # This would query actual usage metrics in production
        # For now, return simulated success rates based on context type
        
        success_rates = {
            ContextType.CLAUDE_CODE_CAPABILITIES: 0.95,  # Always highly successful
            ContextType.IMPLEMENTATION_PATTERNS: 0.85,   # Usually helpful
            ContextType.SIMILAR_ISSUES: 0.75,           # Often relevant
            ContextType.ARCHITECTURAL_DECISIONS: 0.70,   # Contextually helpful
            ContextType.SYSTEM_OVERVIEW: 0.80,          # Good foundational knowledge
            ContextType.DEPENDENCY_INFO: 0.65,          # Sometimes critical
            ContextType.QUALITY_PATTERNS: 0.70,         # Good for validation
            ContextType.ERROR_PATTERNS: 0.85,           # Very useful when relevant
            ContextType.PERFORMANCE_DATA: 0.60,         # Specialized use
            ContextType.SECURITY_PATTERNS: 0.75,        # Important when needed
        }
        
        base_success_rate = success_rates.get(item.type, 0.65)
        
        # Adjust based on agent type preferences
        if agent_type == AgentType.IMPLEMENTER:
            if item.type in [ContextType.IMPLEMENTATION_PATTERNS, ContextType.CLAUDE_CODE_CAPABILITIES]:
                return min(1.0, base_success_rate + 0.1)
        elif agent_type == AgentType.VALIDATOR:
            if item.type in [ContextType.QUALITY_PATTERNS, ContextType.ERROR_PATTERNS]:
                return min(1.0, base_success_rate + 0.1)
        elif agent_type == AgentType.ANALYST:
            if item.type in [ContextType.SIMILAR_ISSUES, ContextType.SYSTEM_OVERVIEW]:
                return min(1.0, base_success_rate + 0.1)
        
        return base_success_rate
    
    def _filter_by_relevance_and_size(self, available_knowledge: List[ContextItem],
                                     relevance_scores: Dict[str, float], 
                                     context_limit: int) -> List[ContextItem]:
        """Filter knowledge by relevance and context window size constraints"""
        # Sort by relevance score descending
        sorted_items = sorted(
            available_knowledge, 
            key=lambda x: relevance_scores.get(x.id, 0), 
            reverse=True
        )
        
        # Filter by minimum relevance threshold
        min_relevance = 0.3  # Configurable threshold
        relevant_items = [item for item in sorted_items 
                         if relevance_scores.get(item.id, 0) >= min_relevance]
        
        # Apply size constraints
        selected_items = []
        total_size = 0
        
        for item in relevant_items:
            if total_size + item.size_estimate <= context_limit:
                selected_items.append(item)
                total_size += item.size_estimate
            else:
                break
        
        return selected_items
    
    def _get_live_system_context(self) -> SystemContext:
        """Get live system context for big picture awareness"""
        return SystemContext(
            overview="RIF is an automatic intelligent development system",
            purpose="Orchestrate specialized agents through GitHub issues with zero manual intervention",
            design_goals=[
                "Automatic workflow management",
                "Quality gate enforcement", 
                "Knowledge base learning",
                "Zero manual configuration"
            ],
            constraints=[
                "Technology agnostic",
                "Claude Code integration required",
                "GitHub-driven workflow",
                "Performance budgets <200ms"
            ],
            dependencies={
                "claude_code": "AI-powered developer tool",
                "github": "Issue tracking and workflow",
                "mcp_servers": "Knowledge and tool integration",
                "git": "Version control"
            },
            architecture_summary="State machine driven with specialized agents, quality gates, and learning loops",
            last_updated=datetime.now()
        )
    
    def _calculate_utilization(self, filtered_context: List[ContextItem], 
                              system_context: SystemContext) -> float:
        """Calculate context window utilization percentage"""
        total_context_size = sum(item.size_estimate for item in filtered_context) + 500  # System context est.
        # Assuming average context window of 8000 characters
        return min(1.0, total_context_size / 8000)
    
    def format_context_for_agent(self, agent_context: AgentContext) -> str:
        """Format optimized context for agent consumption"""
        formatted_context = []
        
        # System overview (always first)
        formatted_context.append("## System Context")
        formatted_context.append(f"**RIF Overview**: {agent_context.system_context.overview}")
        formatted_context.append(f"**Purpose**: {agent_context.system_context.purpose}")
        
        # Claude Code capabilities (critical for Issue #113 fix)
        claude_items = [item for item in agent_context.relevant_knowledge 
                       if item.type == ContextType.CLAUDE_CODE_CAPABILITIES]
        if claude_items:
            formatted_context.append("\n## Claude Code Capabilities")
            for item in claude_items:
                formatted_context.append(f"**{item.id}**: {item.content}")
        
        # Other relevant knowledge
        other_items = [item for item in agent_context.relevant_knowledge 
                      if item.type != ContextType.CLAUDE_CODE_CAPABILITIES]
        if other_items:
            formatted_context.append("\n## Relevant Knowledge")
            for item in other_items:
                formatted_context.append(f"**{item.type.value}**: {item.content}")
        
        # Context utilization info
        formatted_context.append(f"\n## Context Optimization")
        formatted_context.append(f"**Window Utilization**: {agent_context.context_window_utilization:.1%}")
        formatted_context.append(f"**Items Provided**: {len(agent_context.relevant_knowledge)}")
        formatted_context.append(f"**Total Size**: {agent_context.total_size} characters")
        
        return "\n".join(formatted_context)
    
    def update_knowledge_utilization_metrics(self, agent_type: AgentType, 
                                           context_used: bool, decisions_made: int,
                                           problems_found: int = 0):
        """Update knowledge utilization metrics for continuous improvement"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type.value,
            "context_consultation": context_used,
            "decisions_made": decisions_made,
            "problems_found": problems_found,
            "knowledge_roi": problems_found / max(decisions_made, 1)
        }
        
        # Store metrics for analysis
        metrics_file = os.path.join(self.knowledge_base_path, "metrics", "context-utilization.jsonl")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def generate_agent_context(self, agent_type_str: str, issue_description: str) -> str:
        """Generate agent context for testing - convenience method for integration tests"""
        try:
            agent_type = AgentType(agent_type_str)
            task_context = {"description": issue_description}
            context = self.optimize_for_agent(agent_type, task_context, None)
            return self.format_context_for_agent(context)
        except Exception as e:
            return f"Error generating context for {agent_type_str}: {str(e)}"

# CLI Interface for testing and integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Context Optimization Engine")
    parser.add_argument("--agent", type=str, choices=[a.value for a in AgentType], 
                       help="Agent type requesting context")
    parser.add_argument("--issue", type=int, help="GitHub issue number")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--test", action="store_true", help="Run test optimization")
    
    args = parser.parse_args()
    
    optimizer = ContextOptimizer()
    
    if args.test:
        # Test context optimization for all agent types
        test_task = {
            "description": "Implement context optimization engine for intelligent agent filtering",
            "complexity": "high",
            "type": "implementation"
        }
        
        print("=== Context Optimization Engine Test ===\n")
        
        for agent_type in AgentType:
            context = optimizer.optimize_for_agent(agent_type, test_task, args.issue)
            formatted = optimizer.format_context_for_agent(context)
            
            print(f"## {agent_type.value.upper()} Context")
            print(formatted)
            print("\n" + "="*50 + "\n")
    
    elif args.agent and args.task:
        # Optimize context for specific agent and task
        agent_type = AgentType(args.agent)
        task_context = {"description": args.task}
        
        context = optimizer.optimize_for_agent(agent_type, task_context, args.issue)
        formatted = optimizer.format_context_for_agent(context)
        
        print(formatted)
    
    else:
        parser.print_help()