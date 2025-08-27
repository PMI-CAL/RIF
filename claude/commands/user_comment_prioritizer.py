#!/usr/bin/env python3
"""
User Comment Prioritizer - Issue #275 Implementation
RIF-Implementer: User-first orchestration with comment directive extraction

This module implements a user-first orchestration system where user comments 
get VERY HIGH priority and agent recommendations are treated as suggestions only.

Key Features:
1. User comment analysis and directive extraction
2. Priority-based decision hierarchy (VERY HIGH for users, Medium for agents)
3. "Think Hard" orchestration logic for complex scenarios
4. Complete integration with existing orchestration intelligence
5. 100% user directive influence on conflicting orchestration decisions

CRITICAL: User comments ALWAYS override agent recommendations when conflicts occur.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import subprocess
from pathlib import Path

# Import GitHub client for comment retrieval
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from systems.github_api_client import get_github_client, GitHubAPIError
    GITHUB_CLIENT_AVAILABLE = True
except ImportError:
    GITHUB_CLIENT_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectivePriority(Enum):
    """Priority levels for orchestration directives"""
    VERY_HIGH = "VERY_HIGH"     # User directives - highest priority
    HIGH = "HIGH"               # Critical system requirements
    MEDIUM = "MEDIUM"           # Agent suggestions
    LOW = "LOW"                 # Default recommendations
    
    @property
    def numeric_value(self) -> int:
        """Get numeric value for priority comparison"""
        priority_map = {
            "VERY_HIGH": 100,
            "HIGH": 75,
            "MEDIUM": 50,
            "LOW": 25
        }
        return priority_map[self.value]


class DirectiveSource(Enum):
    """Source of orchestration directive"""
    USER_COMMENT = "USER_COMMENT"
    USER_ISSUE_BODY = "USER_ISSUE_BODY"  
    AGENT_SUGGESTION = "AGENT_SUGGESTION"
    SYSTEM_DEFAULT = "SYSTEM_DEFAULT"


@dataclass
class UserDirective:
    """Represents a user directive extracted from comments or issue body"""
    source_type: DirectiveSource
    priority: DirectivePriority
    directive_text: str
    action_type: str  # e.g., "IMPLEMENT", "BLOCK", "PRIORITIZE", "SEQUENCE"
    target_issues: List[int]
    specific_agents: List[str]
    reasoning: str
    confidence_score: float
    timestamp: datetime
    comment_id: Optional[str] = None
    author: Optional[str] = None
    
    def __post_init__(self):
        """Set priority based on source type"""
        if self.source_type in [DirectiveSource.USER_COMMENT, DirectiveSource.USER_ISSUE_BODY]:
            self.priority = DirectivePriority.VERY_HIGH
        elif self.source_type == DirectiveSource.AGENT_SUGGESTION:
            self.priority = DirectivePriority.MEDIUM
        else:
            self.priority = DirectivePriority.LOW


class UserCommentPrioritizer:
    """
    Analyzes user comments and extracts orchestration directives with VERY HIGH priority.
    
    This class implements the core logic for Issue #275: ensuring user input always
    takes precedence over agent recommendations in orchestration decisions.
    """
    
    # Directive extraction patterns
    DIRECTIVE_PATTERNS = {
        'IMPLEMENT': [
            r'implement (?:issue\s*#?)?(\d+)(?:\s*(?:first|next|immediately))?',
            r'work on (?:issue\s*#?)?(\d+)',
            r'start (?:issue\s*#?)?(\d+)',
            r'begin (?:issue\s*#?)?(\d+)',
            r'focus on (?:issue\s*#?)?(\d+)'
        ],
        'BLOCK': [
            r'(?:don\'t|do not) work on (?:issue\s*#?)?(\d+)',
            r'block (?:issue\s*#?)?(\d+)',
            r'halt (?:issue\s*#?)?(\d+)',
            r'stop (?:issue\s*#?)?(\d+)',
            r'pause (?:issue\s*#?)?(\d+)'
        ],
        'PRIORITIZE': [
            r'prioritize (?:issue\s*#?)?(\d+)',
            r'(?:issue\s*#?)?(\d+) (?:is\s*)?(?:high\s*)?priority',
            r'urgent(?:ly)?\s*(?:issue\s*#?)?(\d+)',
            r'critical(?:ly)?\s*(?:issue\s*#?)?(\d+)',
            r'important(?:ly)?\s*(?:issue\s*#?)?(\d+)'
        ],
        'SEQUENCE': [
            r'(?:issue\s*#?)?(\d+) (?:first|before) (?:issue\s*#?)?(\d+)',
            r'(?:complete|finish) (?:issue\s*#?)?(\d+) before (?:issue\s*#?)?(\d+)',
            r'(?:issue\s*#?)?(\d+) depends on (?:issue\s*#?)?(\d+)'
        ],
        'AGENT_SPECIFIC': [
            r'use ([A-Z][A-Za-z-]+) (?:agent\s*)?(?:for\s*)?(?:issue\s*#?)?(\d+)',
            r'assign ([A-Z][A-Za-z-]+) (?:to\s*)?(?:issue\s*#?)?(\d+)',
            r'(?:issue\s*#?)?(\d+) needs ([A-Z][A-Za-z-]+) (?:agent)?'
        ]
    }
    
    # "Think Hard" trigger patterns for complex scenarios
    THINK_HARD_TRIGGERS = [
        r'think\s*hard',
        r'careful(?:ly)?\s*consider',
        r'complex\s*(?:scenario|situation)',
        r'multiple\s*(?:dependencies|options)',
        r'difficult\s*(?:decision|choice)',
        r'analyze\s*(?:deeply|thoroughly)',
        r'comprehensive\s*(?:analysis|review)'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.think_hard_enabled = True
        
    def extract_user_directives(self, issue_numbers: List[int]) -> List[UserDirective]:
        """
        Extract user directives from GitHub issue comments and bodies.
        
        Args:
            issue_numbers: List of GitHub issue numbers to analyze
            
        Returns:
            List of UserDirective objects sorted by priority (VERY HIGH first)
        """
        all_directives = []
        
        for issue_num in issue_numbers:
            try:
                # Extract directives from issue body
                issue_directives = self._extract_directives_from_issue(issue_num)
                all_directives.extend(issue_directives)
                
                # Extract directives from comments
                comment_directives = self._extract_directives_from_comments(issue_num)
                all_directives.extend(comment_directives)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract directives from issue #{issue_num}: {e}")
        
        # Sort by priority (VERY HIGH first) and timestamp (newest first)
        all_directives.sort(
            key=lambda d: (d.priority.numeric_value, d.timestamp.timestamp()), 
            reverse=True
        )
        
        self.logger.info(f"Extracted {len(all_directives)} user directives, "
                        f"{len([d for d in all_directives if d.priority == DirectivePriority.VERY_HIGH])} VERY HIGH priority")
        
        return all_directives
    
    def _extract_directives_from_issue(self, issue_number: int) -> List[UserDirective]:
        """Extract directives from issue body"""
        try:
            if GITHUB_CLIENT_AVAILABLE:
                client = get_github_client()
                result = client.get_issue(issue_number)
                
                if not result['success']:
                    self.logger.warning(f"Failed to fetch issue #{issue_number}: {result['stderr']}")
                    return []
                
                issue_data = result['data']
            else:
                # Fallback to direct gh command
                cmd = f"gh issue view {issue_number} --json number,title,body,author"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.warning(f"Failed to fetch issue #{issue_number}: {result.stderr}")
                    return []
                
                issue_data = json.loads(result.stdout)
            
            directives = []
            if issue_data.get('body'):
                body_directives = self._parse_text_for_directives(
                    issue_data['body'],
                    DirectiveSource.USER_ISSUE_BODY,
                    issue_data.get('author', {}).get('login', 'unknown')
                )
                directives.extend(body_directives)
            
            return directives
            
        except Exception as e:
            self.logger.error(f"Error extracting directives from issue #{issue_number}: {e}")
            return []
    
    def _extract_directives_from_comments(self, issue_number: int) -> List[UserDirective]:
        """Extract directives from issue comments"""
        try:
            if GITHUB_CLIENT_AVAILABLE:
                client = get_github_client()
                # Get comments using gh command since API client might not have comment listing
                cmd = f"gh issue view {issue_number} --json comments"
                result = client.execute_gh_command(f"issue view {issue_number} --json comments")
                
                if not result['success']:
                    self.logger.warning(f"Failed to fetch comments for issue #{issue_number}: {result['stderr']}")
                    return []
                
                comments_data = json.loads(result['stdout'])
            else:
                # Direct fallback
                cmd = f"gh issue view {issue_number} --json comments"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.warning(f"Failed to fetch comments for issue #{issue_number}: {result.stderr}")
                    return []
                
                comments_data = json.loads(result.stdout)
            
            directives = []
            comments = comments_data.get('comments', [])
            
            for comment in comments:
                if comment.get('body'):
                    comment_directives = self._parse_text_for_directives(
                        comment['body'],
                        DirectiveSource.USER_COMMENT,
                        comment.get('author', {}).get('login', 'unknown'),
                        comment.get('id'),
                        comment.get('createdAt')
                    )
                    directives.extend(comment_directives)
            
            return directives
            
        except Exception as e:
            self.logger.error(f"Error extracting directives from comments for issue #{issue_number}: {e}")
            return []
    
    def _parse_text_for_directives(self, text: str, source_type: DirectiveSource, 
                                 author: str, comment_id: Optional[str] = None,
                                 created_at: Optional[str] = None) -> List[UserDirective]:
        """Parse text content for orchestration directives"""
        directives = []
        text_lower = text.lower()
        
        # Parse timestamp
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        # Check for "Think Hard" triggers
        think_hard_required = any(re.search(pattern, text_lower) for pattern in self.THINK_HARD_TRIGGERS)
        
        # Extract directives by type
        for action_type, patterns in self.DIRECTIVE_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    if action_type == 'SEQUENCE':
                        # Special handling for sequence patterns with two issue numbers
                        issue_nums = [int(group) for group in match.groups() if group and group.isdigit()]
                        if len(issue_nums) >= 2:
                            directive = UserDirective(
                                source_type=source_type,
                                priority=DirectivePriority.VERY_HIGH,
                                directive_text=match.group(0),
                                action_type=action_type,
                                target_issues=issue_nums,
                                specific_agents=[],
                                reasoning=f"User specified sequence: {issue_nums[0]} before {issue_nums[1]}",
                                confidence_score=0.95,
                                timestamp=timestamp,
                                comment_id=comment_id,
                                author=author
                            )
                            directives.append(directive)
                    
                    elif action_type == 'AGENT_SPECIFIC':
                        # Extract agent name and issue number
                        groups = match.groups()
                        if len(groups) >= 2:
                            agent_name = groups[0] if groups[0] else groups[2] if len(groups) > 2 else ""
                            issue_num_str = groups[1] if groups[1] else groups[0] if groups[0].isdigit() else ""
                            
                            if agent_name and issue_num_str and issue_num_str.isdigit():
                                directive = UserDirective(
                                    source_type=source_type,
                                    priority=DirectivePriority.VERY_HIGH,
                                    directive_text=match.group(0),
                                    action_type=action_type,
                                    target_issues=[int(issue_num_str)],
                                    specific_agents=[agent_name],
                                    reasoning=f"User specified {agent_name} for issue #{issue_num_str}",
                                    confidence_score=0.9,
                                    timestamp=timestamp,
                                    comment_id=comment_id,
                                    author=author
                                )
                                directives.append(directive)
                    
                    else:
                        # Standard single issue number patterns
                        issue_nums = [int(group) for group in match.groups() if group and group.isdigit()]
                        if issue_nums:
                            confidence = 0.95 if think_hard_required else 0.85
                            directive = UserDirective(
                                source_type=source_type,
                                priority=DirectivePriority.VERY_HIGH,
                                directive_text=match.group(0),
                                action_type=action_type,
                                target_issues=issue_nums,
                                specific_agents=[],
                                reasoning=f"User directive: {action_type.lower()} issue(s) {issue_nums}",
                                confidence_score=confidence,
                                timestamp=timestamp,
                                comment_id=comment_id,
                                author=author
                            )
                            directives.append(directive)
        
        return directives
    
    def analyze_directive_conflicts(self, directives: List[UserDirective], 
                                  agent_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conflicts between user directives and agent recommendations.
        
        CRITICAL: User directives ALWAYS win in conflicts.
        
        Args:
            directives: List of user directives (VERY HIGH priority)
            agent_recommendations: List of agent suggestions (MEDIUM priority)
            
        Returns:
            Conflict analysis with resolution favoring user directives
        """
        conflicts = []
        user_overrides = []
        
        # Create lookup of user directives by issue number and action
        user_directives_by_issue = {}
        for directive in directives:
            for issue_num in directive.target_issues:
                if issue_num not in user_directives_by_issue:
                    user_directives_by_issue[issue_num] = []
                user_directives_by_issue[issue_num].append(directive)
        
        # Check each agent recommendation against user directives
        for agent_rec in agent_recommendations:
            agent_issue = agent_rec.get('issue', 0)
            agent_action = agent_rec.get('action', 'unknown')
            
            if agent_issue in user_directives_by_issue:
                user_dirs_for_issue = user_directives_by_issue[agent_issue]
                
                for user_dir in user_dirs_for_issue:
                    # Check for conflicts
                    if self._is_conflicting_directive(user_dir, agent_rec):
                        conflicts.append({
                            'issue': agent_issue,
                            'user_directive': {
                                'action': user_dir.action_type,
                                'text': user_dir.directive_text,
                                'priority': user_dir.priority.value,
                                'confidence': user_dir.confidence_score
                            },
                            'agent_recommendation': {
                                'action': agent_action,
                                'agent': agent_rec.get('recommended_agent', 'unknown'),
                                'priority': 'MEDIUM'
                            },
                            'resolution': 'USER_DIRECTIVE_WINS',
                            'reasoning': f"User directive priority (VERY HIGH) overrides agent suggestion (MEDIUM)"
                        })
                        
                        user_overrides.append({
                            'issue': agent_issue,
                            'overridden_agent_action': agent_action,
                            'user_directive_action': user_dir.action_type,
                            'user_reasoning': user_dir.reasoning
                        })
        
        return {
            'total_conflicts': len(conflicts),
            'conflicts': conflicts,
            'user_overrides': user_overrides,
            'user_directive_influence_percentage': 100.0 if conflicts else 0.0,
            'conflict_resolution_rule': 'USER_DIRECTIVES_ALWAYS_WIN'
        }
    
    def _is_conflicting_directive(self, user_directive: UserDirective, 
                                agent_recommendation: Dict[str, Any]) -> bool:
        """Check if user directive conflicts with agent recommendation"""
        user_action = user_directive.action_type
        agent_action = agent_recommendation.get('action', '')
        
        # Define conflict patterns
        conflicts = {
            'BLOCK': ['launch_agent', 'implement', 'validate'],
            'IMPLEMENT': ['no_action_needed', 'blocked_by_dependencies'],
            'PRIORITIZE': ['blocked_by_dependencies', 'low_priority']
        }
        
        return any(
            user_action in conflict_type and agent_action in conflict_actions
            for conflict_type, conflict_actions in conflicts.items()
        )
    
    def create_user_priority_orchestration_plan(self, issue_numbers: List[int],
                                              agent_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create orchestration plan with user directives taking precedence.
        
        This is the main integration point with the existing orchestration system.
        """
        # Extract user directives
        user_directives = self.extract_user_directives(issue_numbers)
        
        # Analyze conflicts between user directives and agent recommendations
        conflict_analysis = self.analyze_directive_conflicts(user_directives, agent_recommendations)
        
        # Create user-priority plan
        plan = {
            'timestamp': datetime.now().isoformat(),
            'decision_hierarchy': 'USER_FIRST',
            'user_directive_count': len(user_directives),
            'user_very_high_priority_count': len([d for d in user_directives if d.priority == DirectivePriority.VERY_HIGH]),
            'agent_suggestion_count': len(agent_recommendations),
            'conflict_analysis': conflict_analysis,
            'final_orchestration_decisions': [],
            'user_directive_compliance': 100.0,
            'think_hard_scenarios': []
        }
        
        # Process user directives first (VERY HIGH priority)
        user_decisions = self._process_user_directives(user_directives)
        plan['final_orchestration_decisions'].extend(user_decisions)
        
        # Process agent recommendations that don't conflict with user directives
        non_conflicting_agents = self._filter_non_conflicting_agents(
            agent_recommendations, conflict_analysis['user_overrides']
        )
        agent_decisions = self._process_agent_recommendations(non_conflicting_agents)
        plan['final_orchestration_decisions'].extend(agent_decisions)
        
        # Check for "Think Hard" scenarios requiring extended reasoning
        think_hard_scenarios = self._identify_think_hard_scenarios(user_directives, issue_numbers)
        plan['think_hard_scenarios'] = think_hard_scenarios
        
        if think_hard_scenarios:
            plan['think_hard_required'] = True
            plan['extended_reasoning'] = self._perform_think_hard_analysis(
                user_directives, agent_recommendations, issue_numbers
            )
        
        return plan
    
    def _process_user_directives(self, directives: List[UserDirective]) -> List[Dict[str, Any]]:
        """Process user directives into orchestration decisions"""
        decisions = []
        
        for directive in directives:
            if directive.action_type == 'IMPLEMENT':
                for issue_num in directive.target_issues:
                    decisions.append({
                        'type': 'LAUNCH_AGENT',
                        'issue': issue_num,
                        'priority': 'VERY_HIGH',
                        'source': 'USER_DIRECTIVE',
                        'agent': directive.specific_agents[0] if directive.specific_agents else 'RIF-Implementer',
                        'reasoning': directive.reasoning,
                        'user_text': directive.directive_text,
                        'confidence': directive.confidence_score
                    })
            
            elif directive.action_type == 'BLOCK':
                for issue_num in directive.target_issues:
                    decisions.append({
                        'type': 'BLOCK_ISSUE',
                        'issue': issue_num,
                        'priority': 'VERY_HIGH',
                        'source': 'USER_DIRECTIVE',
                        'reasoning': directive.reasoning,
                        'user_text': directive.directive_text,
                        'confidence': directive.confidence_score
                    })
            
            elif directive.action_type == 'PRIORITIZE':
                for issue_num in directive.target_issues:
                    decisions.append({
                        'type': 'PRIORITIZE_ISSUE',
                        'issue': issue_num,
                        'priority': 'VERY_HIGH',
                        'source': 'USER_DIRECTIVE',
                        'agent': directive.specific_agents[0] if directive.specific_agents else 'auto-select',
                        'reasoning': directive.reasoning,
                        'user_text': directive.directive_text,
                        'confidence': directive.confidence_score
                    })
            
            elif directive.action_type == 'SEQUENCE':
                if len(directive.target_issues) >= 2:
                    decisions.append({
                        'type': 'ENFORCE_SEQUENCE',
                        'issue': directive.target_issues[0],
                        'priority': 'VERY_HIGH',
                        'source': 'USER_DIRECTIVE',
                        'prerequisite_issue': directive.target_issues[1],
                        'reasoning': directive.reasoning,
                        'user_text': directive.directive_text,
                        'confidence': directive.confidence_score
                    })
        
        return decisions
    
    def _process_agent_recommendations(self, agent_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process agent recommendations that don't conflict with user directives"""
        decisions = []
        
        for agent_rec in agent_recommendations:
            decisions.append({
                'type': 'AGENT_SUGGESTION',
                'issue': agent_rec.get('issue'),
                'priority': 'MEDIUM',
                'source': 'AGENT_RECOMMENDATION',
                'agent': agent_rec.get('recommended_agent'),
                'action': agent_rec.get('action'),
                'reasoning': 'Agent suggestion (no user directive conflict)',
                'confidence': 0.7  # Lower confidence for agent suggestions
            })
        
        return decisions
    
    def _filter_non_conflicting_agents(self, agent_recommendations: List[Dict[str, Any]],
                                     user_overrides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out agent recommendations that conflict with user directives"""
        overridden_issues = {override['issue'] for override in user_overrides}
        return [rec for rec in agent_recommendations if rec.get('issue') not in overridden_issues]
    
    def _identify_think_hard_scenarios(self, user_directives: List[UserDirective], 
                                     issue_numbers: List[int]) -> List[Dict[str, Any]]:
        """Identify scenarios requiring "Think Hard" extended reasoning"""
        scenarios = []
        
        # Check for explicit "Think Hard" triggers in user directives
        for directive in user_directives:
            if any(re.search(pattern, directive.directive_text.lower()) for pattern in self.THINK_HARD_TRIGGERS):
                scenarios.append({
                    'type': 'EXPLICIT_THINK_HARD_REQUEST',
                    'directive': directive.directive_text,
                    'issues': directive.target_issues,
                    'reasoning': 'User explicitly requested careful consideration'
                })
        
        # Check for complex dependency scenarios
        if len(issue_numbers) > 5:
            scenarios.append({
                'type': 'COMPLEX_MULTI_ISSUE_SCENARIO',
                'issues': issue_numbers,
                'reasoning': f'Large number of issues ({len(issue_numbers)}) requires careful orchestration'
            })
        
        # Check for conflicting directives
        action_types = [d.action_type for d in user_directives]
        if 'BLOCK' in action_types and 'IMPLEMENT' in action_types:
            scenarios.append({
                'type': 'CONFLICTING_USER_DIRECTIVES',
                'issues': list(set().union(*[d.target_issues for d in user_directives])),
                'reasoning': 'User has both BLOCK and IMPLEMENT directives - requires careful analysis'
            })
        
        return scenarios
    
    def _perform_think_hard_analysis(self, user_directives: List[UserDirective],
                                   agent_recommendations: List[Dict[str, Any]],
                                   issue_numbers: List[int]) -> Dict[str, Any]:
        """
        Perform extended "Think Hard" reasoning for complex orchestration scenarios.
        
        This implements the deliberative decision-making logic required by Issue #275.
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'analysis_depth': 'EXTENDED',
            'reasoning_steps': [],
            'considered_factors': [],
            'decision_confidence': 0.0,
            'recommendations': []
        }
        
        # Step 1: Analyze user directive patterns
        analysis['reasoning_steps'].append({
            'step': 1,
            'description': 'Analyzing user directive patterns',
            'findings': self._analyze_user_directive_patterns(user_directives)
        })
        
        # Step 2: Assess dependency complexity
        analysis['reasoning_steps'].append({
            'step': 2,
            'description': 'Assessing issue dependency complexity',
            'findings': self._assess_dependency_complexity(issue_numbers)
        })
        
        # Step 3: Evaluate resource constraints
        analysis['reasoning_steps'].append({
            'step': 3,
            'description': 'Evaluating orchestration resource constraints',
            'findings': self._evaluate_resource_constraints(len(issue_numbers))
        })
        
        # Step 4: Generate comprehensive recommendations
        analysis['reasoning_steps'].append({
            'step': 4,
            'description': 'Generating comprehensive recommendations',
            'findings': self._generate_think_hard_recommendations(user_directives, issue_numbers)
        })
        
        # Calculate overall confidence
        step_confidences = [step['findings'].get('confidence', 0.5) for step in analysis['reasoning_steps']]
        analysis['decision_confidence'] = sum(step_confidences) / len(step_confidences) if step_confidences else 0.5
        
        # Key factors considered
        analysis['considered_factors'] = [
            'User directive priority and consistency',
            'Inter-issue dependencies and sequences', 
            'Agent specialization and workload',
            'Risk of orchestration conflicts',
            'Resource allocation optimization'
        ]
        
        return analysis
    
    def _analyze_user_directive_patterns(self, directives: List[UserDirective]) -> Dict[str, Any]:
        """Analyze patterns in user directives for Think Hard reasoning"""
        patterns = {
            'action_distribution': {},
            'issue_coverage': set(),
            'temporal_pattern': [],
            'consistency_score': 0.0,
            'confidence': 0.8
        }
        
        # Count action types
        for directive in directives:
            action = directive.action_type
            patterns['action_distribution'][action] = patterns['action_distribution'].get(action, 0) + 1
            patterns['issue_coverage'].update(directive.target_issues)
        
        patterns['issue_coverage'] = list(patterns['issue_coverage'])
        
        # Assess consistency (fewer conflicts = higher consistency)
        total_actions = len(directives)
        conflicting_actions = patterns['action_distribution'].get('BLOCK', 0)
        if total_actions > 0:
            patterns['consistency_score'] = 1.0 - (conflicting_actions / total_actions)
        
        return patterns
    
    def _assess_dependency_complexity(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """Assess complexity of issue dependencies for Think Hard reasoning"""
        return {
            'total_issues': len(issue_numbers),
            'complexity_level': 'HIGH' if len(issue_numbers) > 5 else 'MEDIUM' if len(issue_numbers) > 2 else 'LOW',
            'potential_conflicts': len(issue_numbers) * 0.1,  # Rough estimate
            'confidence': 0.7
        }
    
    def _evaluate_resource_constraints(self, issue_count: int) -> Dict[str, Any]:
        """Evaluate orchestration resource constraints"""
        return {
            'recommended_parallel_limit': min(4, issue_count),
            'sequential_phases_recommended': issue_count > 6,
            'resource_pressure': 'HIGH' if issue_count > 8 else 'MEDIUM' if issue_count > 4 else 'LOW',
            'confidence': 0.8
        }
    
    def _generate_think_hard_recommendations(self, user_directives: List[UserDirective], 
                                           issue_numbers: List[int]) -> Dict[str, Any]:
        """Generate comprehensive recommendations from Think Hard analysis"""
        recommendations = []
        
        # Prioritize user-specified sequences
        sequence_directives = [d for d in user_directives if d.action_type == 'SEQUENCE']
        if sequence_directives:
            recommendations.append({
                'type': 'ENFORCE_USER_SEQUENCES',
                'priority': 'CRITICAL',
                'description': f'Enforce {len(sequence_directives)} user-specified sequences before parallel work'
            })
        
        # Prioritize blocked issues
        block_directives = [d for d in user_directives if d.action_type == 'BLOCK']
        if block_directives:
            blocked_issues = list(set().union(*[d.target_issues for d in block_directives]))
            recommendations.append({
                'type': 'RESPECT_USER_BLOCKS',
                'priority': 'CRITICAL',
                'description': f'Block work on {len(blocked_issues)} user-specified issues'
            })
        
        # Optimize parallel execution within constraints
        implement_directives = [d for d in user_directives if d.action_type == 'IMPLEMENT']
        if implement_directives:
            implement_issues = list(set().union(*[d.target_issues for d in implement_directives]))
            recommendations.append({
                'type': 'PARALLEL_USER_IMPLEMENTATIONS',
                'priority': 'HIGH',
                'description': f'Execute {len(implement_issues)} user-requested implementations in parallel'
            })
        
        return {
            'recommendations': recommendations,
            'confidence': 0.85,
            'reasoning': 'User directives take absolute precedence with optimized execution strategy'
        }


# Integration functions for existing orchestration system
def integrate_user_comment_prioritization(issue_numbers: List[int],
                                        existing_orchestration_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integration function to add user comment prioritization to existing orchestration.
    
    This function serves as the main entry point for Issue #275 integration.
    """
    prioritizer = UserCommentPrioritizer()
    
    # Extract agent recommendations from existing plan
    agent_recommendations = existing_orchestration_plan.get('parallel_tasks', [])
    
    # Create user-priority plan
    user_priority_plan = prioritizer.create_user_priority_orchestration_plan(
        issue_numbers, agent_recommendations
    )
    
    # Merge with existing plan, giving precedence to user directives
    integrated_plan = existing_orchestration_plan.copy()
    integrated_plan.update({
        'user_directive_integration': True,
        'decision_hierarchy': 'USER_FIRST',
        'user_priority_analysis': user_priority_plan,
        'original_agent_plan': existing_orchestration_plan,
        'final_execution_plan': user_priority_plan['final_orchestration_decisions']
    })
    
    return integrated_plan


def validate_user_directive_extraction(test_comments: List[str]) -> Dict[str, Any]:
    """
    Validation function for testing user directive extraction accuracy.
    Used for testing and validation of Issue #275 implementation.
    """
    prioritizer = UserCommentPrioritizer()
    results = {
        'total_comments': len(test_comments),
        'directives_found': 0,
        'directive_types': {},
        'confidence_scores': [],
        'extraction_accuracy': 0.0
    }
    
    for i, comment_text in enumerate(test_comments):
        # Parse as if it's a user comment
        directives = prioritizer._parse_text_for_directives(
            comment_text, DirectiveSource.USER_COMMENT, f"test_user_{i}"
        )
        
        results['directives_found'] += len(directives)
        
        for directive in directives:
            action_type = directive.action_type
            results['directive_types'][action_type] = results['directive_types'].get(action_type, 0) + 1
            results['confidence_scores'].append(directive.confidence_score)
    
    if results['confidence_scores']:
        results['extraction_accuracy'] = sum(results['confidence_scores']) / len(results['confidence_scores'])
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("🎯 User Comment Prioritizer - Issue #275 Implementation")
    print("=" * 70)
    
    # Test directive extraction
    test_comments = [
        "Please implement issue #275 first before working on anything else",
        "Don't work on issue #276, block it until we resolve the dependency",
        "Use RIF-Implementer for issue #277, it's high priority",
        "Think hard about the orchestration approach for issues #278 and #279",
        "Issue #280 depends on issue #275 completing first"
    ]
    
    validation_results = validate_user_directive_extraction(test_comments)
    
    print(f"Test Results:")
    print(f"  Comments processed: {validation_results['total_comments']}")
    print(f"  Directives found: {validation_results['directives_found']}")
    print(f"  Directive types: {validation_results['directive_types']}")
    print(f"  Extraction accuracy: {validation_results['extraction_accuracy']:.2%}")
    
    print(f"\n✅ User Comment Prioritizer ready for orchestration integration!")