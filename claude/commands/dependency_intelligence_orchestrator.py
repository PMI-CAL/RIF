#!/usr/bin/env python3
"""
Dependency Intelligence Orchestrator - Issue #144 Implementation

This module implements the Enhanced Orchestration Intelligence Framework described in CLAUDE.md,
providing dependency analysis, critical path identification, and intelligent launch decision-making.

CRITICAL: This is NOT an orchestrator class - Claude Code IS the orchestrator.
This module provides intelligence utilities to support Claude Code's orchestration decisions.
"""

import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from enum import Enum

# Import existing RIF components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestration_utilities import (
    IssueContext, ContextAnalyzer, StateValidator, 
    OrchestrationHelper, GitHubStateManager
)
from dependency_manager import create_dependency_manager, DependencyCheckResult, Dependency

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IssuePhase(Enum):
    """Sequential phases in RIF workflow"""
    RESEARCH = "research"
    ARCHITECTURE = "architecture" 
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    LEARNING = "learning"
    UNKNOWN = "unknown"


class DependencyType(Enum):
    """Types of dependencies for critical path analysis"""
    BLOCKING = "blocking"          # Must complete before ANY other work
    FOUNDATION = "foundation"      # Core systems that others depend on
    SEQUENTIAL = "sequential"      # Research → Architecture → Implementation → Validation
    INTEGRATION = "integration"    # APIs before integrations that use them


@dataclass
class CriticalPathNode:
    """Node in the critical path dependency graph"""
    issue_number: int
    title: str
    phase: IssuePhase
    dependency_type: DependencyType
    dependencies: List[int] = field(default_factory=list)
    dependents: List[int] = field(default_factory=list)
    can_start: bool = False
    priority_score: float = 0.0
    complexity_score: int = 1


@dataclass
class OrchestrationDecision:
    """Result of intelligent orchestration analysis"""
    decision_type: str  # 'launch_blocking_only', 'launch_foundation_only', 'launch_research_only', 'launch_parallel'
    recommended_issues: List[int]
    blocked_issues: List[int]
    reasoning: str
    task_launch_codes: List[str] = field(default_factory=list)
    dependencies_analysis: Dict[str, Any] = field(default_factory=dict)


class DependencyIntelligenceOrchestrator:
    """
    Intelligence layer that provides dependency analysis and decision-making support 
    for Claude Code's orchestration decisions.
    
    CRITICAL: This is NOT an orchestrator - Claude Code IS the orchestrator.
    This class provides intelligence to help Claude Code make better decisions.
    """
    
    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.state_validator = StateValidator()
        self.dependency_manager = create_dependency_manager()
        self.orchestration_helper = OrchestrationHelper()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_critical_path(self, issue_numbers: List[int]) -> Dict[str, List[CriticalPathNode]]:
        """
        Analyze critical path and categorize issues by dependency type.
        
        Args:
            issue_numbers: List of GitHub issue numbers to analyze
            
        Returns:
            Dict with categorized critical path nodes
        """
        self.logger.info(f"Analyzing critical path for {len(issue_numbers)} issues")
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(issue_numbers)
        
        # Categorize by dependency type
        categorized = {
            'blocking_issues': [],
            'foundation_issues': [], 
            'sequential_research': [],
            'sequential_architecture': [],
            'sequential_implementation': [],
            'sequential_validation': [],
            'integration_ready': []
        }
        
        for issue_num, node in dependency_graph.items():
            category = self._categorize_issue_by_dependencies(node, dependency_graph)
            categorized[category].append(node)
        
        # Sort each category by priority
        for category in categorized:
            categorized[category].sort(key=lambda n: (n.priority_score, -n.complexity_score), reverse=True)
        
        self.logger.info(f"Critical path analysis complete: {sum(len(nodes) for nodes in categorized.values())} nodes categorized")
        return categorized
    
    def make_intelligent_orchestration_decision(self, issue_numbers: List[int]) -> OrchestrationDecision:
        """
        Implement the intelligent launch decision framework from CLAUDE.md.
        
        This follows the decision logic:
        if blocking_issues_exist:
            launch_agents_for_blocking_issues_ONLY()
        elif foundation_incomplete and has_dependent_issues:
            launch_agents_for_foundation_issues_ONLY()  
        elif research_phase_incomplete:
            launch_agents_for_research_issues_ONLY()
        else:
            launch_parallel_agents_for_ready_issues()
        
        Args:
            issue_numbers: List of issues to make orchestration decision for
            
        Returns:
            OrchestrationDecision with recommended action and reasoning
        """
        self.logger.info(f"Making intelligent orchestration decision for {len(issue_numbers)} issues")
        
        # Step 1: Analyze critical path
        critical_path = self.analyze_critical_path(issue_numbers)
        
        # Step 2: Apply decision framework logic
        blocking_issues = critical_path['blocking_issues']
        foundation_issues = critical_path['foundation_issues']
        research_issues = critical_path['sequential_research']
        
        # Decision Logic Implementation
        if blocking_issues:
            # BLOCKING ISSUES - highest priority, must complete before ANY other work
            recommended = [node.issue_number for node in blocking_issues]
            all_others = [node.issue_number for category in critical_path.values() 
                         for node in category if node not in blocking_issues]
            
            decision = OrchestrationDecision(
                decision_type='launch_blocking_only',
                recommended_issues=recommended,
                blocked_issues=all_others,
                reasoning=f"🚨 BLOCKING ISSUES DETECTED: {len(blocking_issues)} critical infrastructure issues must complete before any other work. These affect all agent operations and block the entire workflow."
            )
            
        elif foundation_issues and self._has_dependent_issues(foundation_issues, critical_path):
            # FOUNDATION ISSUES - core systems before dependent work
            recommended = [node.issue_number for node in foundation_issues]
            dependent_issues = self._get_dependent_issues(foundation_issues, critical_path)
            
            decision = OrchestrationDecision(
                decision_type='launch_foundation_only',
                recommended_issues=recommended,
                blocked_issues=dependent_issues,
                reasoning=f"🏗️ FOUNDATION PHASE: {len(foundation_issues)} core system issues must complete before {len(dependent_issues)} dependent issues can proceed. Building foundation first prevents integration conflicts."
            )
            
        elif research_issues and self._research_phase_incomplete(critical_path):
            # RESEARCH PHASE - must complete before implementation phases
            recommended = [node.issue_number for node in research_issues]
            implementation_issues = [node.issue_number for node in critical_path['sequential_implementation']]
            validation_issues = [node.issue_number for node in critical_path['sequential_validation']]
            
            decision = OrchestrationDecision(
                decision_type='launch_research_only',
                recommended_issues=recommended,
                blocked_issues=implementation_issues + validation_issues,
                reasoning=f"🔬 RESEARCH PHASE: {len(research_issues)} research issues must complete before implementation and validation can begin. Sequential phase discipline prevents rework."
            )
            
        else:
            # PARALLEL EXECUTION - all dependencies satisfied, can work in parallel
            ready_issues = []
            for category in ['sequential_architecture', 'sequential_implementation', 'sequential_validation', 'integration_ready']:
                ready_issues.extend([node.issue_number for node in critical_path[category] if node.can_start])
            
            decision = OrchestrationDecision(
                decision_type='launch_parallel',
                recommended_issues=ready_issues,
                blocked_issues=[],
                reasoning=f"✅ PARALLEL EXECUTION: All dependencies satisfied for {len(ready_issues)} issues. Can proceed with parallel agent execution for optimal throughput."
            )
        
        # Step 3: Generate Task launch codes for recommended issues
        decision.task_launch_codes = self._generate_task_launch_codes(decision.recommended_issues)
        
        # Step 4: Add detailed dependencies analysis
        decision.dependencies_analysis = {
            'total_issues_analyzed': len(issue_numbers),
            'blocking_issues_count': len(blocking_issues),
            'foundation_issues_count': len(foundation_issues),
            'research_issues_count': len(research_issues),
            'issues_ready_for_parallel': len([node for nodes in critical_path.values() for node in nodes if node.can_start]),
            'critical_path_depth': self._calculate_critical_path_depth(critical_path)
        }
        
        self.logger.info(f"Orchestration decision: {decision.decision_type} - {len(decision.recommended_issues)} issues recommended, {len(decision.blocked_issues)} blocked")
        return decision
    
    def _build_dependency_graph(self, issue_numbers: List[int]) -> Dict[int, CriticalPathNode]:
        """Build dependency graph with critical path analysis"""
        dependency_graph = {}
        
        for issue_num in issue_numbers:
            try:
                # Get issue context
                context = self.context_analyzer.analyze_issue(issue_num)
                
                # Check dependencies
                can_proceed, reason, dep_result = self.dependency_manager.can_work_on_issue(issue_num)
                
                # Determine phase and dependency type
                phase = self._determine_issue_phase(context)
                dep_type = self._determine_dependency_type(context, dep_result)
                
                # Extract dependencies and dependents
                dependencies = []
                if dep_result and dep_result.blocking_dependencies:
                    dependencies = [dep.issue_number for dep in dep_result.blocking_dependencies]
                
                node = CriticalPathNode(
                    issue_number=issue_num,
                    title=context.title,
                    phase=phase,
                    dependency_type=dep_type,
                    dependencies=dependencies,
                    can_start=can_proceed,
                    priority_score=self._calculate_priority_score(context),
                    complexity_score=context.complexity_score
                )
                
                dependency_graph[issue_num] = node
                
            except Exception as e:
                self.logger.warning(f"Error building dependency node for issue #{issue_num}: {e}")
        
        # Build dependents relationships
        for issue_num, node in dependency_graph.items():
            for dep_issue in node.dependencies:
                if dep_issue in dependency_graph:
                    dependency_graph[dep_issue].dependents.append(issue_num)
        
        return dependency_graph
    
    def _determine_issue_phase(self, context: IssueContext) -> IssuePhase:
        """Determine which sequential phase an issue belongs to"""
        title_lower = context.title.lower()
        body_lower = context.body.lower()
        combined = f"{title_lower} {body_lower}"
        
        # DPIBS pattern detection
        if 'research' in combined or 'analysis' in combined or 'investigate' in combined:
            return IssuePhase.RESEARCH
        elif 'architecture' in combined or 'design' in combined or 'schema' in combined:
            return IssuePhase.ARCHITECTURE  
        elif 'implement' in combined or 'build' in combined or 'create' in combined or 'develop' in combined:
            return IssuePhase.IMPLEMENTATION
        elif 'validat' in combined or 'test' in combined or 'verify' in combined:
            return IssuePhase.VALIDATION
        elif 'learn' in combined or 'extract' in combined or 'knowledge' in combined:
            return IssuePhase.LEARNING
        else:
            return IssuePhase.UNKNOWN
    
    def _determine_dependency_type(self, context: IssueContext, dep_result: Optional[DependencyCheckResult]) -> DependencyType:
        """Determine dependency type for critical path analysis"""
        title_lower = context.title.lower()
        body_lower = context.body.lower()
        combined = f"{title_lower} {body_lower}"
        
        # Check for true blocking indicators (infrastructure issues that prevent ALL work)
        critical_blocking_indicators = ['agent context reading', 'core system failure', 'infrastructure', 'critical bug']
        if any(indicator in combined for indicator in critical_blocking_indicators):
            return DependencyType.BLOCKING
        
        # Check for foundation indicators (core systems that others build upon)
        foundation_indicators = ['core api framework', 'database schema', 'base framework', 'foundation layer']
        if any(indicator in combined for indicator in foundation_indicators):
            return DependencyType.FOUNDATION
        
        # Check for integration indicators
        integration_indicators = ['integration architecture', 'api connector', 'interface', 'migration']
        if any(indicator in combined for indicator in integration_indicators):
            return DependencyType.INTEGRATION
        
        # DPIBS issues are sequential by nature (research → architecture → implementation → validation)
        # Check for DPIBS pattern or parent-child dependencies
        if ('dpibs' in combined or 
            'sub-issue' in combined or 
            'sub-research' in combined or
            'parent issue' in body_lower):
            return DependencyType.SEQUENTIAL
        
        # Default to sequential for standard workflow dependencies
        return DependencyType.SEQUENTIAL
    
    def _categorize_issue_by_dependencies(self, node: CriticalPathNode, dependency_graph: Dict[int, CriticalPathNode]) -> str:
        """Categorize issue by its dependency type and phase"""
        # Only issues that can actually start should be in blocking/foundation categories
        # Issues blocked by dependencies should be categorized by their phase
        
        if node.can_start and node.dependency_type == DependencyType.BLOCKING:
            return 'blocking_issues'
        elif node.can_start and node.dependency_type == DependencyType.FOUNDATION:
            return 'foundation_issues'
        elif node.phase == IssuePhase.RESEARCH:
            return 'sequential_research'
        elif node.phase == IssuePhase.ARCHITECTURE:
            return 'sequential_architecture'
        elif node.phase == IssuePhase.IMPLEMENTATION:
            return 'sequential_implementation'
        elif node.phase == IssuePhase.VALIDATION:
            return 'sequential_validation'
        else:
            return 'integration_ready'
    
    def _has_dependent_issues(self, foundation_issues: List[CriticalPathNode], critical_path: Dict[str, List[CriticalPathNode]]) -> bool:
        """Check if foundation issues have dependent issues waiting"""
        foundation_numbers = {node.issue_number for node in foundation_issues}
        
        for category in critical_path.values():
            for node in category:
                if any(dep in foundation_numbers for dep in node.dependencies):
                    return True
        return False
    
    def _get_dependent_issues(self, foundation_issues: List[CriticalPathNode], critical_path: Dict[str, List[CriticalPathNode]]) -> List[int]:
        """Get list of issues that depend on foundation issues"""
        foundation_numbers = {node.issue_number for node in foundation_issues}
        dependent_issues = []
        
        for category in critical_path.values():
            for node in category:
                if any(dep in foundation_numbers for dep in node.dependencies):
                    dependent_issues.append(node.issue_number)
        
        return dependent_issues
    
    def _research_phase_incomplete(self, critical_path: Dict[str, List[CriticalPathNode]]) -> bool:
        """Check if research phase has incomplete issues"""
        research_issues = critical_path['sequential_research']
        return len(research_issues) > 0 and any(not node.can_start for node in research_issues)
    
    def _calculate_priority_score(self, context: IssueContext) -> float:
        """Calculate priority score for issue ordering"""
        base_score = 0.5
        
        # Priority label scoring
        if 'priority:critical' in context.labels:
            base_score += 0.4
        elif 'priority:high' in context.labels:
            base_score += 0.3
        elif 'priority:medium' in context.labels:
            base_score += 0.2
        
        # Complexity scoring (higher complexity = slightly higher priority for early resolution)
        base_score += context.complexity_score * 0.1
        
        # Age scoring (older issues get slight priority boost)
        # This would need datetime parsing - simplified for now
        base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _calculate_critical_path_depth(self, critical_path: Dict[str, List[CriticalPathNode]]) -> int:
        """Calculate the depth of the critical path (maximum dependency chain length)"""
        max_depth = 0
        
        def calculate_node_depth(node: CriticalPathNode, visited: Set[int]) -> int:
            if node.issue_number in visited:
                return 0  # Prevent infinite recursion
            
            if not node.dependencies:
                return 1
            
            visited.add(node.issue_number)
            depths = []
            
            for dep_num in node.dependencies:
                # Find the dependency node across all categories
                dep_node = None
                for category_nodes in critical_path.values():
                    for n in category_nodes:
                        if n.issue_number == dep_num:
                            dep_node = n
                            break
                    if dep_node:
                        break
                
                if dep_node:
                    depths.append(calculate_node_depth(dep_node, visited.copy()))
            
            return 1 + max(depths) if depths else 1
        
        # Calculate depth for all nodes
        for category_nodes in critical_path.values():
            for node in category_nodes:
                depth = calculate_node_depth(node, set())
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _generate_task_launch_codes(self, recommended_issues: List[int]) -> List[str]:
        """Generate Task launch codes for recommended issues using orchestration helper"""
        task_codes = []
        
        for issue_num in recommended_issues:
            try:
                recommendation = self.orchestration_helper.recommend_orchestration_action(issue_num)
                if recommendation['action'] == 'launch_agent':
                    task_codes.append(recommendation['task_launch_code'])
            except Exception as e:
                self.logger.warning(f"Error generating task code for issue #{issue_num}: {e}")
        
        return task_codes
    
    def generate_orchestration_report(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """Generate detailed orchestration intelligence report"""
        decision = self.make_intelligent_orchestration_decision(issue_numbers)
        critical_path = self.analyze_critical_path(issue_numbers)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {
                'total_issues': len(issue_numbers),
                'decision_type': decision.decision_type,
                'recommended_issues': len(decision.recommended_issues),
                'blocked_issues': len(decision.blocked_issues),
                'reasoning': decision.reasoning
            },
            'critical_path_analysis': {
                'blocking_issues': len(critical_path['blocking_issues']),
                'foundation_issues': len(critical_path['foundation_issues']),
                'research_phase': len(critical_path['sequential_research']),
                'architecture_phase': len(critical_path['sequential_architecture']),
                'implementation_phase': len(critical_path['sequential_implementation']),
                'validation_phase': len(critical_path['sequential_validation']),
                'integration_ready': len(critical_path['integration_ready'])
            },
            'orchestration_decision': {
                'action': decision.decision_type,
                'launch_immediately': decision.recommended_issues,
                'wait_for_dependencies': decision.blocked_issues,
                'task_launch_codes_count': len(decision.task_launch_codes)
            },
            'dependencies_intelligence': decision.dependencies_analysis,
            'recommendations': self._generate_recommendations(decision, critical_path)
        }
        
        return report
    
    def _generate_recommendations(self, decision: OrchestrationDecision, critical_path: Dict[str, List[CriticalPathNode]]) -> List[str]:
        """Generate actionable recommendations for Claude Code"""
        recommendations = []
        
        if decision.decision_type == 'launch_blocking_only':
            recommendations.append("🚨 Focus all resources on blocking issues - do not launch any other agents until these complete")
            recommendations.append("⏰ Monitor blocking issues closely and run dependency check immediately when they complete")
            
        elif decision.decision_type == 'launch_foundation_only':
            recommendations.append("🏗️ Complete foundation issues before dependent work to prevent integration conflicts")
            recommendations.append("🔄 Run dependency unblocking check after foundation completion")
            
        elif decision.decision_type == 'launch_research_only':
            recommendations.append("🔬 Complete research phase before moving to implementation to prevent rework")
            recommendations.append("📋 Use research findings to inform architecture and implementation phases")
            
        else:
            recommendations.append("✅ Dependencies satisfied - launch all recommended issues in parallel for optimal throughput")
            recommendations.append("📊 Monitor agent execution and use checkpoints for progress tracking")
        
        # Add specific recommendations based on critical path analysis
        if critical_path['blocking_issues']:
            blocking_issues = [str(node.issue_number) for node in critical_path['blocking_issues']]
            recommendations.append(f"⚡ Prioritize blocking issues: #{', #'.join(blocking_issues)}")
        
        return recommendations


def main():
    """Example usage of dependency intelligence orchestrator"""
    print("🧠 Dependency Intelligence Orchestrator - Issue #144 Implementation")
    print("=" * 80)
    
    orchestrator = DependencyIntelligenceOrchestrator()
    
    # Test with DPIBS issues
    dpibs_issues = [133, 134, 135, 136, 137, 138, 139, 140, 141, 142]
    
    print(f"🔍 Analyzing {len(dpibs_issues)} DPIBS issues for intelligent orchestration...")
    
    try:
        # Generate orchestration decision
        decision = orchestrator.make_intelligent_orchestration_decision(dpibs_issues)
        
        print(f"\n🎯 Orchestration Decision: {decision.decision_type.upper()}")
        print(f"📋 Reasoning: {decision.reasoning}")
        print(f"✅ Recommended Issues: {decision.recommended_issues}")
        print(f"⏸️  Blocked Issues: {decision.blocked_issues}")
        
        print(f"\n🚀 Task Launch Codes ({len(decision.task_launch_codes)}):")
        print("=" * 60)
        for i, code in enumerate(decision.task_launch_codes, 1):
            print(f"# Task {i}:")
            print(code)
            print()
        
        # Generate full report
        report = orchestrator.generate_orchestration_report(dpibs_issues)
        
        print("📊 Intelligence Report Summary:")
        print(f"- Total Issues: {report['analysis_summary']['total_issues']}")
        print(f"- Decision Type: {report['analysis_summary']['decision_type']}")
        print(f"- Recommended: {report['analysis_summary']['recommended_issues']}")
        print(f"- Blocked: {report['analysis_summary']['blocked_issues']}")
        
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
    except Exception as e:
        logger.error(f"Error in orchestration analysis: {e}")
        print(f"❌ Error: {e}")
        print("This may occur if GitHub CLI is not authenticated or issues don't exist")


if __name__ == "__main__":
    main()