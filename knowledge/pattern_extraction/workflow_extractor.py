"""
Workflow Pattern Extractor - State transition and workflow analysis.

This module extracts workflow patterns from GitHub issue histories, state transitions,
and agent interaction sequences to identify successful process patterns.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

from .discovery_engine import ExtractedPattern, PatternSignature


@dataclass
class StateTransition:
    """Represents a single state transition."""
    from_state: str
    to_state: str
    timestamp: datetime
    trigger: str
    duration: Optional[float] = None  # seconds
    agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class WorkflowSequence:
    """Represents a complete workflow sequence."""
    sequence_id: str
    states: List[str]
    transitions: List[StateTransition]
    total_duration: float
    success: bool
    complexity: str
    parallel_activities: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['transitions'] = [t.to_dict() for t in self.transitions]
        return data


@dataclass
class AgentInteractionPattern:
    """Represents agent interaction patterns."""
    interaction_type: str
    agents_involved: List[str]
    sequence: List[Dict[str, Any]]
    success_rate: float
    average_duration: float
    coordination_complexity: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowPatternExtractor:
    """
    Extracts workflow patterns from issue histories and state transitions.
    
    This extractor analyzes GitHub issue workflows, agent interactions,
    and development process patterns to identify successful workflow patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Known workflow patterns to detect
        self.workflow_patterns = {
            'linear_workflow': self._detect_linear_workflow,
            'parallel_workflow': self._detect_parallel_workflow,
            'iterative_workflow': self._detect_iterative_workflow,
            'cascade_workflow': self._detect_cascade_workflow,
            'conditional_workflow': self._detect_conditional_workflow,
            'recovery_workflow': self._detect_recovery_workflow,
            'optimization_workflow': self._detect_optimization_workflow,
            'validation_workflow': self._detect_validation_workflow
        }
        
        # Agent coordination patterns
        self.coordination_patterns = {
            'sequential_handoff': self._detect_sequential_handoff,
            'parallel_execution': self._detect_parallel_execution,
            'master_worker': self._detect_master_worker,
            'consensus_building': self._detect_consensus_building,
            'escalation_hierarchy': self._detect_escalation_hierarchy,
            'collaborative_review': self._detect_collaborative_review
        }
        
        # Process quality patterns
        self.quality_patterns = {
            'checkpoint_pattern': self._detect_checkpoint_pattern,
            'rollback_pattern': self._detect_rollback_pattern,
            'validation_gates': self._detect_validation_gates,
            'monitoring_pattern': self._detect_monitoring_pattern,
            'feedback_loops': self._detect_feedback_loops
        }
        
        # Standard RIF workflow states
        self.rif_states = {
            'new', 'analyzing', 'planning', 'architecting', 'implementing',
            'validating', 'learning', 'complete', 'blocked', 'failed'
        }
    
    def extract_patterns(self, completed_issue: Dict[str, Any]) -> List[ExtractedPattern]:
        """
        Extract workflow patterns from completed issue data.
        
        Args:
            completed_issue: Issue data containing workflow history and state transitions
            
        Returns:
            List of extracted workflow patterns
        """
        patterns = []
        issue_id = completed_issue.get('issue_number', 'unknown')
        
        self.logger.info(f"Extracting workflow patterns from issue #{issue_id}")
        
        try:
            # Extract workflow sequence from issue history
            workflow_sequence = self._extract_workflow_sequence(completed_issue)
            
            if workflow_sequence:
                # Detect workflow patterns
                patterns.extend(self._detect_workflow_patterns(workflow_sequence, completed_issue))
                
                # Detect agent coordination patterns
                patterns.extend(self._detect_coordination_patterns(workflow_sequence, completed_issue))
                
                # Detect quality patterns
                patterns.extend(self._detect_quality_patterns(workflow_sequence, completed_issue))
            
            # Extract from agent interactions
            if 'agent_interactions' in completed_issue:
                patterns.extend(self._extract_agent_interaction_patterns(completed_issue))
            
            # Extract from timing patterns
            patterns.extend(self._extract_timing_patterns(completed_issue))
            
            self.logger.info(f"Extracted {len(patterns)} workflow patterns from issue #{issue_id}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error extracting workflow patterns from issue #{issue_id}: {e}")
            return []
    
    def _extract_workflow_sequence(self, issue_data: Dict[str, Any]) -> Optional[WorkflowSequence]:
        """Extract workflow sequence from issue history."""
        try:
            history = issue_data.get('history', [])
            if not history:
                return None
            
            transitions = []
            states = []
            current_state = 'new'
            states.append(current_state)
            
            start_time = None
            
            # Process history events
            for event in history:
                if 'timestamp' in event:
                    event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                    if start_time is None:
                        start_time = event_time
                
                # Look for state changes
                if 'label_added' in event or 'label_removed' in event:
                    new_state = self._extract_state_from_label_event(event)
                    if new_state and new_state != current_state:
                        transition = StateTransition(
                            from_state=current_state,
                            to_state=new_state,
                            timestamp=event_time,
                            trigger=event.get('trigger', 'manual'),
                            agent=event.get('agent', 'unknown'),
                            metadata=event.get('metadata', {})
                        )
                        
                        # Calculate duration from previous transition
                        if transitions:
                            prev_time = transitions[-1].timestamp
                            transition.duration = (event_time - prev_time).total_seconds()
                        
                        transitions.append(transition)
                        states.append(new_state)
                        current_state = new_state
            
            # Calculate total duration
            total_duration = 0
            if start_time and transitions:
                end_time = transitions[-1].timestamp
                total_duration = (end_time - start_time).total_seconds()
            
            # Determine success
            success = current_state in ['complete', 'learning'] or issue_data.get('state') == 'closed'
            
            # Determine complexity
            complexity = self._calculate_workflow_complexity(transitions, states)
            
            # Identify parallel activities and bottlenecks
            parallel_activities = self._identify_parallel_activities(issue_data, transitions)
            bottlenecks = self._identify_bottlenecks(transitions)
            
            return WorkflowSequence(
                sequence_id=f"workflow_{issue_data.get('issue_number', 'unknown')}",
                states=states,
                transitions=transitions,
                total_duration=total_duration,
                success=success,
                complexity=complexity,
                parallel_activities=parallel_activities,
                bottlenecks=bottlenecks
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting workflow sequence: {e}")
            return None
    
    def _extract_state_from_label_event(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract state from label event."""
        if 'label_added' in event:
            label = event['label_added']
        elif 'label_removed' in event:
            # For removed labels, we need to infer the new state
            return None
        else:
            return None
        
        # Extract state from label (format: state:xxx)
        if isinstance(label, str) and label.startswith('state:'):
            return label[6:]  # Remove 'state:' prefix
        elif isinstance(label, dict) and 'name' in label:
            label_name = label['name']
            if label_name.startswith('state:'):
                return label_name[6:]
        
        return None
    
    def _calculate_workflow_complexity(self, transitions: List[StateTransition], states: List[str]) -> str:
        """Calculate workflow complexity based on transitions and states."""
        unique_states = len(set(states))
        transition_count = len(transitions)
        
        # Look for loops (revisiting states)
        state_visits = Counter(states)
        has_loops = any(count > 1 for count in state_visits.values())
        
        # Calculate complexity score
        complexity_score = unique_states + transition_count
        if has_loops:
            complexity_score += 5
        
        if complexity_score < 5:
            return 'low'
        elif complexity_score < 10:
            return 'medium'
        elif complexity_score < 20:
            return 'high'
        else:
            return 'very-high'
    
    def _identify_parallel_activities(self, issue_data: Dict[str, Any], 
                                    transitions: List[StateTransition]) -> List[Dict[str, Any]]:
        """Identify parallel activities in the workflow."""
        parallel_activities = []
        
        # Look for simultaneous agent activities
        if 'agent_interactions' in issue_data:
            agent_activities = defaultdict(list)
            
            for interaction in issue_data['agent_interactions']:
                timestamp = interaction.get('timestamp')
                agent = interaction.get('agent')
                if timestamp and agent:
                    agent_activities[timestamp].append(agent)
            
            # Find timestamps with multiple agents
            for timestamp, agents in agent_activities.items():
                if len(agents) > 1:
                    parallel_activities.append({
                        'timestamp': timestamp,
                        'agents': agents,
                        'activity_type': 'parallel_execution'
                    })
        
        return parallel_activities
    
    def _identify_bottlenecks(self, transitions: List[StateTransition]) -> List[Dict[str, Any]]:
        """Identify bottlenecks in the workflow."""
        bottlenecks = []
        
        if not transitions:
            return bottlenecks
        
        # Calculate average duration
        durations = [t.duration for t in transitions if t.duration is not None]
        if not durations:
            return bottlenecks
        
        avg_duration = statistics.mean(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
        
        # Identify transitions that are significantly longer than average
        threshold = avg_duration + 2 * std_duration
        
        for transition in transitions:
            if transition.duration and transition.duration > threshold:
                bottlenecks.append({
                    'from_state': transition.from_state,
                    'to_state': transition.to_state,
                    'duration': transition.duration,
                    'severity': 'high' if transition.duration > threshold * 1.5 else 'medium'
                })
        
        return bottlenecks
    
    def _detect_workflow_patterns(self, workflow: WorkflowSequence, 
                                issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Detect workflow patterns in the sequence."""
        patterns = []
        
        for pattern_name, detector in self.workflow_patterns.items():
            if detector(workflow, issue_data):
                pattern = self._create_workflow_pattern(
                    pattern_name, workflow, issue_data
                )
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_coordination_patterns(self, workflow: WorkflowSequence,
                                   issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Detect agent coordination patterns."""
        patterns = []
        
        for pattern_name, detector in self.coordination_patterns.items():
            if detector(workflow, issue_data):
                pattern = self._create_coordination_pattern(
                    pattern_name, workflow, issue_data
                )
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_quality_patterns(self, workflow: WorkflowSequence,
                               issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Detect quality patterns in the workflow."""
        patterns = []
        
        for pattern_name, detector in self.quality_patterns.items():
            if detector(workflow, issue_data):
                pattern = self._create_quality_pattern(
                    pattern_name, workflow, issue_data
                )
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    # Workflow pattern detectors
    def _detect_linear_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect linear workflow pattern (no loops, straight progression)."""
        state_visits = Counter(workflow.states)
        return all(count == 1 for count in state_visits.values())
    
    def _detect_parallel_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect parallel workflow pattern."""
        return len(workflow.parallel_activities) > 0
    
    def _detect_iterative_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect iterative workflow pattern (loops and cycles)."""
        state_visits = Counter(workflow.states)
        return any(count > 1 for count in state_visits.values())
    
    def _detect_cascade_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect cascade workflow pattern (systematic progression)."""
        expected_sequence = ['new', 'analyzing', 'planning', 'implementing', 'validating', 'complete']
        workflow_states = [s for s in workflow.states if s in expected_sequence]
        
        # Check if states follow expected order
        if len(workflow_states) < 3:
            return False
        
        for i in range(len(workflow_states) - 1):
            current_idx = expected_sequence.index(workflow_states[i])
            next_idx = expected_sequence.index(workflow_states[i + 1])
            if next_idx <= current_idx:
                return False
        
        return True
    
    def _detect_conditional_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect conditional workflow pattern (branching paths)."""
        # Look for evidence of conditional branching
        complexity = issue_data.get('complexity', 'medium')
        has_architecture_state = 'architecting' in workflow.states
        
        return complexity in ['high', 'very-high'] and has_architecture_state
    
    def _detect_recovery_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect recovery workflow pattern (error recovery and retry)."""
        has_blocked = 'blocked' in workflow.states
        has_failed = 'failed' in workflow.states
        
        # Check for recovery (blocked/failed followed by progress)
        if has_blocked or has_failed:
            for i, state in enumerate(workflow.states[:-1]):
                if state in ['blocked', 'failed']:
                    next_state = workflow.states[i + 1]
                    if next_state not in ['blocked', 'failed']:
                        return True
        
        return False
    
    def _detect_optimization_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect optimization workflow pattern (performance improvements)."""
        # Look for optimization indicators
        title = issue_data.get('title', '').lower()
        description = issue_data.get('body', '').lower()
        
        optimization_keywords = ['optimize', 'performance', 'speed', 'efficiency', 'cache', 'improve']
        return any(keyword in title or keyword in description for keyword in optimization_keywords)
    
    def _detect_validation_workflow(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect validation workflow pattern (comprehensive validation steps)."""
        return 'validating' in workflow.states and workflow.success
    
    # Coordination pattern detectors
    def _detect_sequential_handoff(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect sequential handoff pattern between agents."""
        # First try to get agents from workflow transitions
        agents = [t.agent for t in workflow.transitions if t.agent and t.agent != 'unknown']
        unique_agents = list(set(agents))
        
        # If no agents found in transitions, check agent_interactions
        if len(unique_agents) < 2 and 'agent_interactions' in issue_data:
            interaction_agents = [
                interaction.get('agent') 
                for interaction in issue_data['agent_interactions'] 
                if interaction.get('agent')
            ]
            unique_agents = list(set(interaction_agents))
        
        # Sequential handoff: multiple agents, minimal overlap
        return len(unique_agents) >= 2 and len(workflow.parallel_activities) == 0
    
    def _detect_parallel_execution(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect parallel execution pattern."""
        return len(workflow.parallel_activities) > 0
    
    def _detect_master_worker(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect master-worker coordination pattern."""
        # Look for one dominant agent with others in supporting roles
        agents = [t.agent for t in workflow.transitions if t.agent and t.agent != 'unknown']
        if not agents:
            return False
        
        agent_counts = Counter(agents)
        most_common_count = agent_counts.most_common(1)[0][1]
        total_count = sum(agent_counts.values())
        
        # Master agent handles >60% of transitions
        return most_common_count / total_count > 0.6 and len(agent_counts) > 1
    
    def _detect_consensus_building(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect consensus building pattern."""
        # Look for consensus indicators in title/description
        text = f"{issue_data.get('title', '')} {issue_data.get('body', '')}".lower()
        consensus_keywords = ['consensus', 'agreement', 'vote', 'decision', 'discussion']
        return any(keyword in text for keyword in consensus_keywords)
    
    def _detect_escalation_hierarchy(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect escalation hierarchy pattern."""
        # Look for escalation from blocked state
        has_escalation = False
        for i, transition in enumerate(workflow.transitions[:-1]):
            if transition.from_state == 'blocked':
                next_transition = workflow.transitions[i + 1]
                if next_transition.agent != transition.agent:
                    has_escalation = True
        
        return has_escalation
    
    def _detect_collaborative_review(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect collaborative review pattern."""
        # Look for multiple agents in validating state
        validating_agents = set()
        for transition in workflow.transitions:
            if transition.to_state == 'validating' and transition.agent:
                validating_agents.add(transition.agent)
        
        return len(validating_agents) > 1
    
    # Quality pattern detectors
    def _detect_checkpoint_pattern(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect checkpoint pattern (regular progress checkpoints)."""
        # Look for checkpoint-related keywords or regular state progressions
        has_checkpoints = 'checkpoint' in str(issue_data).lower()
        has_regular_progression = len(workflow.transitions) >= 4
        
        return has_checkpoints or has_regular_progression
    
    def _detect_rollback_pattern(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect rollback pattern (reversing to previous states)."""
        # Look for backward state transitions
        expected_order = ['new', 'analyzing', 'planning', 'architecting', 'implementing', 'validating', 'learning', 'complete']
        
        for i, transition in enumerate(workflow.transitions):
            try:
                from_idx = expected_order.index(transition.from_state)
                to_idx = expected_order.index(transition.to_state)
                if to_idx < from_idx:  # Backward transition
                    return True
            except ValueError:
                continue
        
        return False
    
    def _detect_validation_gates(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect validation gates pattern (quality checkpoints)."""
        return 'validating' in workflow.states and workflow.success
    
    def _detect_monitoring_pattern(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect monitoring pattern (continuous monitoring)."""
        monitoring_keywords = ['monitor', 'track', 'observe', 'metrics', 'alert']
        text = f"{issue_data.get('title', '')} {issue_data.get('body', '')}".lower()
        return any(keyword in text for keyword in monitoring_keywords)
    
    def _detect_feedback_loops(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> bool:
        """Detect feedback loops pattern."""
        # Look for iterative patterns and feedback
        has_loops = any(workflow.states.count(state) > 1 for state in workflow.states)
        feedback_keywords = ['feedback', 'iteration', 'review', 'refine']
        text = f"{issue_data.get('title', '')} {issue_data.get('body', '')}".lower()
        has_feedback_keywords = any(keyword in text for keyword in feedback_keywords)
        
        return has_loops and has_feedback_keywords
    
    def _extract_agent_interaction_patterns(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract patterns from agent interactions."""
        patterns = []
        
        try:
            interactions = issue_data.get('agent_interactions', [])
            if not interactions:
                return patterns
            
            # Analyze interaction sequences
            interaction_pattern = self._analyze_interaction_sequences(interactions)
            
            if interaction_pattern:
                pattern = ExtractedPattern(
                    title="Agent Interaction Pattern",
                    description=f"Agent coordination pattern with {len(interaction_pattern.agents_involved)} agents",
                    pattern_type='coordination',
                    source=f"issue-{issue_data.get('issue_number', 'unknown')}",
                    content={
                        'interaction_pattern': interaction_pattern.to_dict(),
                        'coordination_effectiveness': self._calculate_coordination_effectiveness(interaction_pattern),
                        'communication_flow': self._analyze_communication_flow(interactions)
                    },
                    context={
                        'issue_complexity': issue_data.get('complexity', 'medium'),
                        'number_of_agents': len(interaction_pattern.agents_involved),
                        'interaction_duration': interaction_pattern.average_duration
                    },
                    signature=PatternSignature.from_pattern({
                        'title': 'Agent Interaction Pattern',
                        'description': f"Coordination with {len(interaction_pattern.agents_involved)} agents",
                        'complexity': issue_data.get('complexity', 'medium'),
                        'domain': 'agent_coordination'
                    }),
                    extraction_method='interaction_analysis',
                    confidence=0.7,
                    created_at=datetime.now()
                )
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting agent interaction patterns: {e}")
        
        return patterns
    
    def _extract_timing_patterns(self, issue_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract timing-related patterns."""
        patterns = []
        
        try:
            history = issue_data.get('history', [])
            if not history:
                return patterns
            
            # Analyze timing patterns
            timing_analysis = self._analyze_timing_patterns(history)
            
            if timing_analysis['has_patterns']:
                pattern = ExtractedPattern(
                    title="Workflow Timing Pattern",
                    description=f"Timing pattern with {timing_analysis['pattern_type']} characteristics",
                    pattern_type='timing',
                    source=f"issue-{issue_data.get('issue_number', 'unknown')}",
                    content={
                        'timing_analysis': timing_analysis,
                        'efficiency_metrics': self._calculate_efficiency_metrics(history),
                        'time_distribution': self._analyze_time_distribution(history)
                    },
                    context={
                        'total_duration': timing_analysis.get('total_duration', 0),
                        'complexity': issue_data.get('complexity', 'medium'),
                        'success': issue_data.get('state') == 'closed'
                    },
                    signature=PatternSignature.from_pattern({
                        'title': 'Workflow Timing Pattern',
                        'description': timing_analysis['pattern_type'],
                        'complexity': issue_data.get('complexity', 'medium'),
                        'domain': 'workflow_timing'
                    }),
                    extraction_method='timing_analysis',
                    confidence=0.6,
                    created_at=datetime.now()
                )
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting timing patterns: {e}")
        
        return patterns
    
    def _create_workflow_pattern(self, pattern_name: str, workflow: WorkflowSequence,
                               issue_data: Dict[str, Any]) -> Optional[ExtractedPattern]:
        """Create workflow pattern from detected pattern."""
        try:
            return ExtractedPattern(
                title=f"{pattern_name.replace('_', ' ').title()} Workflow",
                description=f"Workflow pattern with {len(workflow.states)} states and {len(workflow.transitions)} transitions",
                pattern_type='workflow',
                source=f"issue-{issue_data.get('issue_number', 'unknown')}",
                content={
                    'pattern_name': pattern_name,
                    'workflow_sequence': workflow.to_dict(),
                    'success_indicators': self._identify_success_indicators(workflow, issue_data),
                    'efficiency_metrics': self._calculate_workflow_efficiency(workflow),
                    'scalability_factors': self._assess_scalability(workflow)
                },
                context={
                    'complexity': workflow.complexity,
                    'success': workflow.success,
                    'duration': workflow.total_duration,
                    'bottleneck_count': len(workflow.bottlenecks)
                },
                signature=PatternSignature.from_pattern({
                    'title': f"{pattern_name} Workflow",
                    'description': f"Workflow with {len(workflow.states)} states",
                    'complexity': workflow.complexity,
                    'domain': 'workflow_management'
                }),
                extraction_method='workflow_analysis',
                confidence=0.8,
                created_at=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error creating workflow pattern {pattern_name}: {e}")
            return None
    
    def _create_coordination_pattern(self, pattern_name: str, workflow: WorkflowSequence,
                                   issue_data: Dict[str, Any]) -> Optional[ExtractedPattern]:
        """Create coordination pattern from detected pattern."""
        try:
            agents = set(t.agent for t in workflow.transitions if t.agent and t.agent != 'unknown')
            
            return ExtractedPattern(
                title=f"{pattern_name.replace('_', ' ').title()} Coordination",
                description=f"Agent coordination pattern involving {len(agents)} agents",
                pattern_type='coordination',
                source=f"issue-{issue_data.get('issue_number', 'unknown')}",
                content={
                    'pattern_name': pattern_name,
                    'agents_involved': list(agents),
                    'coordination_sequence': [t.to_dict() for t in workflow.transitions],
                    'parallel_activities': workflow.parallel_activities,
                    'coordination_effectiveness': self._assess_coordination_effectiveness(workflow)
                },
                context={
                    'agent_count': len(agents),
                    'success': workflow.success,
                    'complexity': workflow.complexity,
                    'coordination_overhead': len(workflow.parallel_activities)
                },
                signature=PatternSignature.from_pattern({
                    'title': f"{pattern_name} Coordination",
                    'description': f"Coordination with {len(agents)} agents",
                    'complexity': workflow.complexity,
                    'domain': 'agent_coordination'
                }),
                extraction_method='coordination_analysis',
                confidence=0.7,
                created_at=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error creating coordination pattern {pattern_name}: {e}")
            return None
    
    def _create_quality_pattern(self, pattern_name: str, workflow: WorkflowSequence,
                              issue_data: Dict[str, Any]) -> Optional[ExtractedPattern]:
        """Create quality pattern from detected pattern."""
        try:
            return ExtractedPattern(
                title=f"{pattern_name.replace('_', ' ').title()} Quality Pattern",
                description=f"Quality assurance pattern in workflow management",
                pattern_type='quality',
                source=f"issue-{issue_data.get('issue_number', 'unknown')}",
                content={
                    'pattern_name': pattern_name,
                    'quality_indicators': self._identify_quality_indicators(workflow, issue_data),
                    'validation_steps': self._extract_validation_steps(workflow),
                    'quality_metrics': self._calculate_quality_metrics(workflow, issue_data)
                },
                context={
                    'success': workflow.success,
                    'has_validation': 'validating' in workflow.states,
                    'has_checkpoints': len(workflow.transitions) >= 4,
                    'quality_gates_passed': workflow.success
                },
                signature=PatternSignature.from_pattern({
                    'title': f"{pattern_name} Quality Pattern",
                    'description': "Quality assurance workflow pattern",
                    'complexity': 'medium',
                    'domain': 'quality_assurance'
                }),
                extraction_method='quality_analysis',
                confidence=0.6,
                created_at=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error creating quality pattern {pattern_name}: {e}")
            return None
    
    # Helper methods for pattern creation
    def _identify_success_indicators(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> List[str]:
        """Identify indicators that contributed to workflow success."""
        indicators = []
        
        if workflow.success:
            if 'validating' in workflow.states:
                indicators.append('comprehensive_validation')
            if len(workflow.bottlenecks) == 0:
                indicators.append('no_bottlenecks')
            if workflow.total_duration < 86400:  # Less than 1 day
                indicators.append('efficient_timing')
            if len(set(workflow.states)) >= 4:
                indicators.append('thorough_process')
        
        return indicators
    
    def _calculate_workflow_efficiency(self, workflow: WorkflowSequence) -> Dict[str, Any]:
        """Calculate workflow efficiency metrics."""
        return {
            'total_duration': workflow.total_duration,
            'average_transition_time': workflow.total_duration / max(len(workflow.transitions), 1),
            'bottleneck_count': len(workflow.bottlenecks),
            'parallel_efficiency': len(workflow.parallel_activities) / max(len(workflow.transitions), 1),
            'state_efficiency': len(set(workflow.states)) / len(workflow.states) if workflow.states else 1
        }
    
    def _assess_scalability(self, workflow: WorkflowSequence) -> Dict[str, Any]:
        """Assess workflow scalability factors."""
        return {
            'complexity_scalability': 'high' if workflow.complexity in ['low', 'medium'] else 'medium',
            'agent_scalability': 'high' if len(workflow.parallel_activities) > 0 else 'medium',
            'process_scalability': 'high' if len(workflow.bottlenecks) == 0 else 'low',
            'time_scalability': 'high' if workflow.total_duration < 86400 else 'medium'
        }
    
    def _assess_coordination_effectiveness(self, workflow: WorkflowSequence) -> Dict[str, Any]:
        """Assess coordination effectiveness."""
        agents = set(t.agent for t in workflow.transitions if t.agent and t.agent != 'unknown')
        
        return {
            'agent_utilization': len(agents) / max(len(workflow.transitions), 1),
            'parallel_coordination': len(workflow.parallel_activities) > 0,
            'handoff_efficiency': workflow.success and len(workflow.bottlenecks) == 0,
            'communication_overhead': len(workflow.parallel_activities) / max(len(workflow.transitions), 1)
        }
    
    def _identify_quality_indicators(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> List[str]:
        """Identify quality indicators in the workflow."""
        indicators = []
        
        if 'validating' in workflow.states:
            indicators.append('validation_step')
        if workflow.success:
            indicators.append('successful_completion')
        if len(workflow.bottlenecks) == 0:
            indicators.append('smooth_execution')
        if 'learning' in workflow.states:
            indicators.append('knowledge_capture')
        
        return indicators
    
    def _extract_validation_steps(self, workflow: WorkflowSequence) -> List[Dict[str, Any]]:
        """Extract validation steps from workflow."""
        validation_steps = []
        
        for transition in workflow.transitions:
            if transition.to_state == 'validating':
                validation_steps.append({
                    'timestamp': transition.timestamp.isoformat(),
                    'agent': transition.agent,
                    'duration': transition.duration,
                    'metadata': transition.metadata
                })
        
        return validation_steps
    
    def _calculate_quality_metrics(self, workflow: WorkflowSequence, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the workflow."""
        return {
            'completeness': workflow.success,
            'efficiency': workflow.total_duration < 86400,  # Less than 1 day
            'reliability': len(workflow.bottlenecks) == 0,
            'predictability': len(set(workflow.states)) <= 6,  # Reasonable number of states
            'maintainability': 'validating' in workflow.states
        }
    
    def _analyze_interaction_sequences(self, interactions: List[Dict[str, Any]]) -> Optional[AgentInteractionPattern]:
        """Analyze agent interaction sequences."""
        try:
            if not interactions:
                return None
            
            agents = set(i.get('agent') for i in interactions if i.get('agent'))
            
            # Calculate success rate (simplified)
            success_rate = 0.8  # Default assumption
            
            # Calculate average duration
            durations = []
            for i in range(len(interactions) - 1):
                try:
                    t1 = datetime.fromisoformat(interactions[i]['timestamp'])
                    t2 = datetime.fromisoformat(interactions[i + 1]['timestamp'])
                    durations.append((t2 - t1).total_seconds())
                except:
                    continue
            
            avg_duration = statistics.mean(durations) if durations else 3600  # 1 hour default
            
            return AgentInteractionPattern(
                interaction_type='collaborative',
                agents_involved=list(agents),
                sequence=interactions,
                success_rate=success_rate,
                average_duration=avg_duration,
                coordination_complexity='medium' if len(agents) <= 3 else 'high'
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing interaction sequences: {e}")
            return None
    
    def _calculate_coordination_effectiveness(self, pattern: AgentInteractionPattern) -> Dict[str, Any]:
        """Calculate coordination effectiveness metrics."""
        return {
            'agent_efficiency': pattern.success_rate,
            'communication_quality': 0.8,  # Simplified
            'task_distribution': len(pattern.agents_involved) > 1,
            'coordination_overhead': pattern.coordination_complexity
        }
    
    def _analyze_communication_flow(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze communication flow between agents."""
        flow = {
            'total_interactions': len(interactions),
            'unique_agents': len(set(i.get('agent') for i in interactions if i.get('agent'))),
            'interaction_types': list(set(i.get('type') for i in interactions if i.get('type'))),
            'communication_density': len(interactions) / max(len(set(i.get('agent') for i in interactions if i.get('agent'))), 1)
        }
        return flow
    
    def _analyze_timing_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing patterns in workflow history."""
        try:
            if len(history) < 2:
                return {'has_patterns': False}
            
            timestamps = []
            for event in history:
                if 'timestamp' in event:
                    try:
                        ts = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                        timestamps.append(ts)
                    except:
                        continue
            
            if len(timestamps) < 2:
                return {'has_patterns': False}
            
            timestamps.sort()
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            
            total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
            avg_interval = statistics.mean(intervals)
            
            # Determine pattern type
            pattern_type = 'regular'
            if len(intervals) > 2:
                std_dev = statistics.stdev(intervals)
                if std_dev / avg_interval > 1.0:
                    pattern_type = 'irregular'
                elif std_dev / avg_interval < 0.3:
                    pattern_type = 'consistent'
            
            return {
                'has_patterns': True,
                'pattern_type': pattern_type,
                'total_duration': total_duration,
                'average_interval': avg_interval,
                'interval_count': len(intervals)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing timing patterns: {e}")
            return {'has_patterns': False}
    
    def _calculate_efficiency_metrics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate efficiency metrics from workflow history."""
        return {
            'event_density': len(history) / max((datetime.now() - datetime.now()).total_seconds() / 3600, 1),  # events per hour
            'progress_consistency': 0.8,  # Simplified metric
            'resource_utilization': 0.7   # Simplified metric
        }
    
    def _analyze_time_distribution(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time distribution patterns."""
        return {
            'peak_activity_hours': [10, 14, 16],  # Simplified
            'activity_pattern': 'business_hours',
            'weekend_activity': False
        }