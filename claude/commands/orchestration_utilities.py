#!/usr/bin/env python3
"""
RIF Orchestration Utilities - Pattern-Compliant Support for Claude Code

This module provides utility classes that support Claude Code as the orchestrator,
following the correct RIF pattern where Claude Code IS the orchestrator and launches
Task agents directly.

CRITICAL: This module does NOT contain an orchestrator class - Claude Code IS the orchestrator.
These are helper utilities for GitHub state analysis, context modeling, and Task() launching.
"""

import json
import re
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IssueContext:
    """Rich context representation for GitHub issues"""
    number: int
    title: str
    body: str
    labels: List[str]
    state: str
    complexity: str
    priority: str
    agent_history: List[str]
    created_at: str
    updated_at: str
    comments_count: int
    
    @property
    def current_state_label(self) -> Optional[str]:
        """Extract current state label from issue labels"""
        state_labels = [label for label in self.labels if label.startswith('state:')]
        return state_labels[0] if state_labels else None
    
    @property  
    def complexity_score(self) -> int:
        """Convert complexity label to numeric score"""
        complexity_map = {
            'complexity:low': 1,
            'complexity:medium': 2, 
            'complexity:high': 3,
            'complexity:very-high': 4
        }
        for label in self.labels:
            if label in complexity_map:
                return complexity_map[label]
        return 2  # default medium
    
    @property
    def requires_planning(self) -> bool:
        """Determine if issue complexity requires planning phase"""
        return self.complexity_score >= 3


class ContextAnalyzer:
    """
    Analyzes GitHub issues and provides rich context for orchestration decisions.
    Supports Claude Code by providing the analysis needed to launch appropriate agents.
    """
    
    def __init__(self):
        self.knowledge_base_path = Path("knowledge")
        self.patterns_path = self.knowledge_base_path / "patterns"
        self.decisions_path = self.knowledge_base_path / "decisions"
        
    def analyze_issue(self, issue_number: int) -> IssueContext:
        """
        Analyze a GitHub issue and return rich context for orchestration decisions.
        
        Args:
            issue_number: GitHub issue number to analyze
            
        Returns:
            IssueContext with comprehensive analysis
        """
        try:
            # Get issue data from GitHub
            cmd = f"gh issue view {issue_number} --json number,title,body,labels,state,createdAt,updatedAt,comments"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to fetch issue {issue_number}: {result.stderr}")
            
            issue_data = json.loads(result.stdout)
            
            # Extract label information
            labels = [label['name'] for label in issue_data.get('labels', [])]
            
            # Determine complexity from labels or content analysis
            complexity = self._analyze_complexity(issue_data['body'], labels)
            
            # Determine priority from labels or content analysis  
            priority = self._analyze_priority(issue_data['body'], labels)
            
            # Extract agent history from comments
            agent_history = self._extract_agent_history(issue_number)
            
            return IssueContext(
                number=issue_number,
                title=issue_data['title'],
                body=issue_data['body'] or '',
                labels=labels,
                state=issue_data['state'],
                complexity=complexity,
                priority=priority,
                agent_history=agent_history,
                created_at=issue_data['createdAt'],
                updated_at=issue_data['updatedAt'],
                comments_count=len(issue_data.get('comments', []))
            )
            
        except Exception as e:
            logger.error(f"Error analyzing issue {issue_number}: {e}")
            raise
    
    def analyze_multiple_issues(self, issue_numbers: List[int]) -> List[IssueContext]:
        """Analyze multiple issues in batch for orchestration planning"""
        contexts = []
        for issue_num in issue_numbers:
            try:
                context = self.analyze_issue(issue_num)
                contexts.append(context)
            except Exception as e:
                logger.warning(f"Skipping issue {issue_num} due to error: {e}")
        return contexts
    
    def _analyze_complexity(self, body: str, labels: List[str]) -> str:
        """Analyze issue complexity from content and labels"""
        # Check for explicit complexity labels
        for label in labels:
            if label.startswith('complexity:'):
                return label.split(':', 1)[1]
        
        # Analyze content for complexity indicators
        complexity_indicators = {
            'very-high': ['microservice', 'distributed', 'architecture', 'migration', 'refactor'],
            'high': ['integration', 'api', 'database', 'security', 'performance'],
            'medium': ['feature', 'enhancement', 'component', 'module'],
            'low': ['fix', 'bug', 'documentation', 'typo', 'minor']
        }
        
        body_lower = body.lower()
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in body_lower for indicator in indicators):
                return complexity
                
        return 'medium'  # default
    
    def _analyze_priority(self, body: str, labels: List[str]) -> str:
        """Analyze issue priority from content and labels"""
        # Check for explicit priority labels
        for label in labels:
            if label.startswith('priority:'):
                return label.split(':', 1)[1]
        
        # Analyze content for priority indicators
        body_lower = body.lower()
        if any(word in body_lower for word in ['critical', 'urgent', 'blocking', 'production']):
            return 'high'
        elif any(word in body_lower for word in ['important', 'needed', 'required']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_agent_history(self, issue_number: int) -> List[str]:
        """Extract history of RIF agents that have worked on this issue"""
        try:
            cmd = f"gh issue view {issue_number} --json comments --jq '.comments[] | select(.body | contains(\"**Agent**:\")) | .body'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                agent_comments = result.stdout.strip().split('\n')
                agents = []
                for comment in agent_comments:
                    if '**Agent**:' in comment:
                        # Extract agent name from comment
                        match = re.search(r'\*\*Agent\*\*:\s*([^\n]*)', comment)
                        if match:
                            agents.append(match.group(1).strip())
                return agents
            
        except Exception as e:
            logger.warning(f"Could not extract agent history for issue {issue_number}: {e}")
        
        return []


class StateValidator:
    """
    Validates GitHub issue states and ensures proper workflow progression.
    Supports Claude Code by validating states before agent launches.
    """
    
    # Valid state transitions according to RIF workflow
    VALID_TRANSITIONS = {
        'new': ['analyzing', 'planning'],
        'analyzing': ['planning', 'architecting', 'implementing'],
        'planning': ['architecting', 'implementing'],
        'architecting': ['implementing'],
        'implementing': ['validating', 'analyzing'],  # can loop back
        'validating': ['learning', 'implementing'],   # can return for fixes
        'learning': ['complete'],
        'complete': []  # terminal state
    }
    
    # Required agents for each state
    STATE_AGENTS = {
        'new': ['RIF-Analyst'],
        'analyzing': ['RIF-Analyst'],
        'planning': ['RIF-Planner'],
        'architecting': ['RIF-Architect'],
        'implementing': ['RIF-Implementer'],
        'validating': ['RIF-Validator'],
        'learning': ['RIF-Learner']
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_state_transition(self, current_state: str, target_state: str) -> Tuple[bool, str]:
        """
        Validate if a state transition is allowed.
        
        Args:
            current_state: Current issue state (without 'state:' prefix)
            target_state: Target state to transition to
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if current_state not in self.VALID_TRANSITIONS:
            return False, f"Unknown current state: {current_state}"
        
        valid_next_states = self.VALID_TRANSITIONS[current_state]
        if target_state not in valid_next_states:
            return False, f"Invalid transition from {current_state} to {target_state}. Valid: {valid_next_states}"
        
        return True, "Valid state transition"
    
    def get_required_agent(self, state: str) -> Optional[str]:
        """Get the agent required for a given state"""
        agents = self.STATE_AGENTS.get(state, [])
        return agents[0] if agents else None
    
    def validate_issue_ready_for_state(self, context: IssueContext, target_state: str) -> Tuple[bool, List[str]]:
        """
        Validate if an issue is ready to transition to target state.
        
        Args:
            context: Issue context from ContextAnalyzer
            target_state: State to validate readiness for
            
        Returns:
            Tuple of (is_ready, list_of_blocking_issues)
        """
        blocking_issues = []
        
        # Check if transition is valid
        current_state = context.current_state_label
        if current_state:
            current_state = current_state.replace('state:', '')
            is_valid, reason = self.validate_state_transition(current_state, target_state)
            if not is_valid:
                blocking_issues.append(reason)
        
        # State-specific validation
        if target_state == 'implementing':
            if context.complexity_score >= 3 and 'architecting' not in [h.lower() for h in context.agent_history]:
                blocking_issues.append("High complexity issue needs architecture phase before implementation")
        
        elif target_state == 'validating':
            if 'RIF-Implementer' not in context.agent_history:
                blocking_issues.append("Issue needs implementation before validation")
        
        return len(blocking_issues) == 0, blocking_issues
    
    def get_next_recommended_state(self, context: IssueContext) -> Optional[str]:
        """Get the next recommended state based on issue context and history"""
        current_state = context.current_state_label
        if not current_state:
            return 'new'
        
        current_state = current_state.replace('state:', '')
        valid_next = self.VALID_TRANSITIONS.get(current_state, [])
        
        if not valid_next:
            return None
        
        # Apply intelligent decision making
        if current_state == 'analyzing':
            # High complexity needs architecture, others can go straight to implementation
            return 'architecting' if context.complexity_score >= 3 else 'implementing'
        
        elif current_state == 'validating':
            # Check if validation passed or failed based on recent comments
            # This would need more sophisticated analysis
            return 'learning'  # Assume validation passed
        
        # Default to first valid next state
        return valid_next[0]


class OrchestrationHelper:
    """
    Helper class that provides Task() launching support and orchestration decision making.
    This is the key utility that helps Claude Code orchestrate agents correctly.
    """
    
    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.state_validator = StateValidator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_task_launch_code(self, issue_context: IssueContext, agent_name: str) -> str:
        """
        Generate the Task() launch code for Claude Code to execute.
        
        Args:
            issue_context: Context from ContextAnalyzer
            agent_name: Name of agent to launch (e.g., 'RIF-Analyst')
            
        Returns:
            String containing Task() function call code
        """
        agent_file = agent_name.lower().replace('-', '-')
        
        # Generate appropriate prompt based on agent and issue context
        prompt = self._generate_agent_prompt(agent_name, issue_context)
        
        task_code = f'''Task(
    description="{agent_name}: {self._generate_task_description(agent_name, issue_context)}",
    subagent_type="general-purpose",
    prompt="{prompt}"
)'''
        
        return task_code
    
    def generate_orchestration_plan(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """
        Generate a complete orchestration plan for multiple issues.
        Returns Task launch codes for Claude Code to execute.
        
        Args:
            issue_numbers: List of GitHub issue numbers to orchestrate
            
        Returns:
            Dict containing orchestration plan with Task launch codes
        """
        plan = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(issue_numbers),
            'parallel_tasks': [],
            'sequential_tasks': [],
            'recommendations': []
        }
        
        # Analyze all issues
        contexts = self.context_analyzer.analyze_multiple_issues(issue_numbers)
        
        # Group by required actions
        immediate_tasks = []
        
        for context in contexts:
            # Determine what agent is needed
            current_state = context.current_state_label
            if current_state:
                state_name = current_state.replace('state:', '')
                required_agent = self.state_validator.get_required_agent(state_name)
                
                if required_agent:
                    task_code = self.generate_task_launch_code(context, required_agent)
                    immediate_tasks.append({
                        'issue': context.number,
                        'agent': required_agent,
                        'state': state_name,
                        'priority': context.priority,
                        'task_code': task_code
                    })
        
        # Sort by priority and complexity for optimal execution
        immediate_tasks.sort(key=lambda x: (x['priority'] != 'high', -contexts[0].complexity_score))
        
        plan['parallel_tasks'] = immediate_tasks
        plan['task_launch_codes'] = [task['task_code'] for task in immediate_tasks]
        
        return plan
    
    def _generate_agent_prompt(self, agent_name: str, context: IssueContext) -> str:
        """Generate appropriate prompt for agent based on context"""
        base_prompt = f"You are {agent_name}. "
        
        if agent_name == 'RIF-Analyst':
            base_prompt += f"Analyze GitHub issue #{context.number} titled '{context.title}'. Extract requirements, identify patterns from knowledge base, assess complexity and create detailed analysis."
        
        elif agent_name == 'RIF-Planner':
            base_prompt += f"Create detailed plan for GitHub issue #{context.number}. Assess complexity, create workflow, identify dependencies and plan implementation approach."
        
        elif agent_name == 'RIF-Architect':
            base_prompt += f"Design system architecture for GitHub issue #{context.number}. Create detailed technical design based on planning phase results."
        
        elif agent_name == 'RIF-Implementer':
            base_prompt += f"Implement solution for GitHub issue #{context.number}. Use checkpoints for progress tracking and ensure quality standards."
        
        elif agent_name == 'RIF-Validator':
            base_prompt += f"Validate implementation for GitHub issue #{context.number}. Run tests, check quality gates, ensure standards compliance."
        
        elif agent_name == 'RIF-Learner':
            base_prompt += f"Extract learnings from completed GitHub issue #{context.number}. Update knowledge base with patterns, decisions, metrics."
        
        # Add agent instructions reference
        agent_file = agent_name.lower().replace('rif-', '') + '.md'
        base_prompt += f" Follow all instructions in claude/agents/{agent_file}."
        
        return base_prompt
    
    def _generate_task_description(self, agent_name: str, context: IssueContext) -> str:
        """Generate concise task description"""
        action_map = {
            'RIF-Analyst': 'Analyze requirements',
            'RIF-Planner': 'Plan implementation',
            'RIF-Architect': 'Design architecture',
            'RIF-Implementer': 'Implement solution',
            'RIF-Validator': 'Validate implementation',
            'RIF-Learner': 'Extract learnings'
        }
        
        action = action_map.get(agent_name, 'Process issue')
        return f"{action} for issue #{context.number}"
    
    def recommend_orchestration_action(self, issue_number: int) -> Dict[str, Any]:
        """
        Analyze an issue and recommend what orchestration action Claude Code should take.
        
        Args:
            issue_number: GitHub issue number to analyze
            
        Returns:
            Dict with recommended action and Task launch code
        """
        context = self.context_analyzer.analyze_issue(issue_number)
        
        # Determine current state and next action
        current_state = context.current_state_label
        if not current_state:
            # New issue, needs initial analysis
            recommended_agent = 'RIF-Analyst'
            next_state = 'analyzing'
        else:
            state_name = current_state.replace('state:', '')
            recommended_agent = self.state_validator.get_required_agent(state_name)
            next_state = self.state_validator.get_next_recommended_state(context)
        
        if not recommended_agent:
            return {
                'action': 'no_action_needed',
                'reason': f"Issue #{issue_number} is in state {current_state} with no required agent"
            }
        
        task_code = self.generate_task_launch_code(context, recommended_agent)
        
        return {
            'action': 'launch_agent',
            'issue': issue_number,
            'current_state': current_state,
            'recommended_agent': recommended_agent,
            'next_state': next_state,
            'task_launch_code': task_code,
            'context': asdict(context)
        }


class GitHubStateManager:
    """
    Utility for managing GitHub issue states and labels.
    Supports Claude Code by handling GitHub interactions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update_issue_state(self, issue_number: int, new_state: str, comment: Optional[str] = None) -> bool:
        """
        Update GitHub issue state label.
        
        Args:
            issue_number: Issue number to update
            new_state: New state (without 'state:' prefix)
            comment: Optional comment to add when changing state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove old state labels
            cmd_remove = f"gh issue edit {issue_number} --remove-label 'state:new,state:analyzing,state:planning,state:architecting,state:implementing,state:validating,state:learning,state:complete'"
            subprocess.run(cmd_remove, shell=True, capture_output=True)
            
            # Add new state label
            cmd_add = f"gh issue edit {issue_number} --add-label 'state:{new_state}'"
            result = subprocess.run(cmd_add, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to update state for issue {issue_number}: {result.stderr}")
                return False
            
            # Add comment if provided
            if comment:
                cmd_comment = f"gh issue comment {issue_number} --body {json.dumps(comment)}"
                subprocess.run(cmd_comment, shell=True, capture_output=True)
            
            self.logger.info(f"Updated issue {issue_number} to state:{new_state}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating issue {issue_number} state: {e}")
            return False
    
    def get_active_issues(self) -> List[Dict[str, Any]]:
        """Get all active issues with state labels"""
        try:
            cmd = "gh issue list --state open --label 'state:*' --json number,title,labels --limit 50"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                self.logger.error(f"Failed to fetch active issues: {result.stderr}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching active issues: {e}")
            return []
    
    def add_agent_tracking_label(self, issue_number: int, agent_name: str) -> bool:
        """Add agent tracking label to issue"""
        try:
            agent_label = f"agent:{agent_name.lower()}"
            cmd = f"gh issue edit {issue_number} --add-label '{agent_label}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Error adding agent label to issue {issue_number}: {e}")
            return False


# Example usage for Claude Code:
def main():
    """
    Example of how Claude Code should use these utilities for orchestration.
    This is for demonstration - Claude Code would use these utilities directly.
    """
    print("RIF Orchestration Utilities - Pattern-Compliant Support")
    print("=" * 60)
    
    # Initialize utilities
    helper = OrchestrationHelper()
    
    # Example: Get orchestration recommendation for an issue
    issue_num = 52  # This issue
    recommendation = helper.recommend_orchestration_action(issue_num)
    
    print(f"\nOrchestration Recommendation for Issue #{issue_num}:")
    print(f"Action: {recommendation['action']}")
    if recommendation['action'] == 'launch_agent':
        print(f"Recommended Agent: {recommendation['recommended_agent']}")
        print(f"Current State: {recommendation['current_state']}")
        print(f"Next State: {recommendation['next_state']}")
        print("\nTask Launch Code:")
        print(recommendation['task_launch_code'])


if __name__ == "__main__":
    main()