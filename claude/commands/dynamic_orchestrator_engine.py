#!/usr/bin/env python3
"""
Dynamic Orchestrator Engine - Core component of the Hybrid Graph-Based Dynamic Orchestration System

This module implements the core orchestration engine that supports Claude Code as the orchestrator,
providing graph-based state management, evidence-based decision making, and dynamic workflow routing.

CRITICAL: This module supports Claude Code as the orchestrator - it does NOT replace Claude Code.
These components enable intelligent orchestration decision-making for dynamic workflow execution.
"""

import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for decision making"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9


@dataclass
class Evidence:
    """Represents evidence for decision making"""
    source: str
    content: str
    confidence: float
    timestamp: datetime
    evidence_type: str  # 'test_result', 'validation_output', 'agent_analysis', etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'content': self.content,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'evidence_type': self.evidence_type
        }


@dataclass
class DecisionPoint:
    """Represents a decision point in the workflow graph"""
    decision_id: str
    trigger_condition: str
    evaluator: str
    outcomes: List[Dict[str, Any]]
    context_factors: List[str]
    confidence_threshold: float = 0.7
    
    def evaluate_condition(self, context: Dict[str, Any], evidence: List[Evidence]) -> Tuple[bool, float]:
        """Evaluate if this decision point should be triggered"""
        # Basic condition evaluation - can be enhanced with complex logic
        if self.trigger_condition == "validation_results_available":
            return 'validation_complete' in context, 0.9
        elif self.trigger_condition == "analysis_complete":
            return 'analysis_findings' in context, 0.8
        
        return False, 0.0


@dataclass 
class StateNode:
    """Represents a state in the dynamic workflow graph"""
    state_id: str
    description: str
    agents: List[str]
    can_transition_to: List[str]
    decision_logic: str
    loop_back_conditions: List[str]
    parallel_capable: bool = False
    timeout_minutes: int = 60
    
    def can_transition(self, target_state: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if transition to target state is allowed"""
        if target_state not in self.can_transition_to:
            return False, f"Direct transition from {self.state_id} to {target_state} not allowed"
        
        # Check additional context-based conditions
        if self.decision_logic:
            # Simplified logic evaluation - can be enhanced
            if "complexity_threshold_evaluation" in self.decision_logic:
                complexity = context.get('complexity', 'medium')
                if target_state == 'architecting' and complexity not in ['high', 'very_high']:
                    return False, "Architecture phase not needed for low/medium complexity"
            
        return True, "Transition allowed"


@dataclass
class WorkflowSession:
    """Represents an active workflow session"""
    session_id: str
    issue_number: int
    current_state: str
    context: Dict[str, Any]
    evidence_trail: List[Evidence] = field(default_factory=list)
    transition_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence to the session"""
        self.evidence_trail.append(evidence)
        self.last_updated = datetime.now()
    
    def record_transition(self, from_state: str, to_state: str, reason: str, confidence: float):
        """Record a state transition"""
        transition = {
            'from_state': from_state,
            'to_state': to_state,
            'reason': reason,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'evidence_count': len(self.evidence_trail)
        }
        self.transition_history.append(transition)
        self.current_state = to_state
        self.last_updated = datetime.now()


class StateGraphManager:
    """
    Manages the dynamic state graph structure and intelligent transition rules.
    Provides any-to-any state transitions with validation and loop-back capabilities.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/rif-workflow.yaml")
        self.states: Dict[str, StateNode] = {}
        self.decision_points: Dict[str, DecisionPoint] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize with enhanced state definitions
        self._initialize_dynamic_state_graph()
        self._initialize_decision_points()
    
    def _initialize_dynamic_state_graph(self):
        """Initialize the enhanced state graph with any-to-any transitions"""
        # Enhanced state definitions from architecture document
        states_config = {
            'analyzing': StateNode(
                state_id='analyzing',
                description='Analysis with pattern recognition and requirement extraction',
                agents=['rif-analyst'],
                can_transition_to=['planning', 'implementing', 'architecting', 'analyzing'],
                decision_logic='complexity_based_routing + requirement_completeness_check',
                loop_back_conditions=['requirements_unclear', 'validation_failed_analysis'],
                parallel_capable=True,
                timeout_minutes=60
            ),
            'planning': StateNode(
                state_id='planning',
                description='Strategic planning with workflow configuration',
                agents=['rif-planner'],
                can_transition_to=['architecting', 'implementing', 'analyzing'],
                decision_logic='complexity_threshold_evaluation + resource_assessment',
                loop_back_conditions=['architectural_concerns_raised', 'requirements_changed'],
                parallel_capable=False,
                timeout_minutes=60
            ),
            'architecting': StateNode(
                state_id='architecting',
                description='System design and dependency mapping',
                agents=['rif-architect'],
                can_transition_to=['implementing', 'planning', 'analyzing'],
                decision_logic='design_completeness + dependency_resolution',
                loop_back_conditions=['requirements_analysis_needed', 'plan_revision_required'],
                parallel_capable=False,
                timeout_minutes=120
            ),
            'implementing': StateNode(
                state_id='implementing',
                description='Code implementation with checkpoint tracking',
                agents=['rif-implementer'],
                can_transition_to=['validating', 'architecting', 'analyzing'],
                decision_logic='code_completion_check + quality_prerequisites',
                loop_back_conditions=['architectural_issues', 'requirements_misunderstood'],
                parallel_capable=False,
                timeout_minutes=240
            ),
            'validating': StateNode(
                state_id='validating',
                description='Comprehensive validation and quality gates',
                agents=['rif-validator'],
                can_transition_to=['learning', 'implementing', 'architecting', 'analyzing'],
                decision_logic='validation_result_evaluation + error_categorization',
                loop_back_conditions=['fixable_errors', 'architectural_flaws', 'unclear_requirements'],
                parallel_capable=True,
                timeout_minutes=120
            ),
            'learning': StateNode(
                state_id='learning',
                description='Knowledge extraction and pattern updates',
                agents=['rif-learner'],
                can_transition_to=['complete'],
                decision_logic='knowledge_extraction_complete',
                loop_back_conditions=[],
                parallel_capable=False,
                timeout_minutes=30
            )
        }
        
        for state_id, state_node in states_config.items():
            self.states[state_id] = state_node
    
    def _initialize_decision_points(self):
        """Initialize decision points with evaluation criteria"""
        # Post-validation decision point
        post_validation = DecisionPoint(
            decision_id='post_validation_decision',
            trigger_condition='validation_results_available',
            evaluator='ValidationResultsEvaluator',
            context_factors=['test_results', 'quality_gates', 'error_types', 'complexity'],
            confidence_threshold=0.7,
            outcomes=[
                {
                    'outcome': 'proceed_to_learning',
                    'condition': 'all_tests_pass AND quality_gates_pass',
                    'confidence_threshold': 0.9
                },
                {
                    'outcome': 'return_to_implementation',
                    'condition': 'fixable_errors_identified',
                    'confidence_threshold': 0.7
                },
                {
                    'outcome': 'escalate_to_architecture',
                    'condition': 'architectural_issues_detected',
                    'confidence_threshold': 0.8
                },
                {
                    'outcome': 'loop_to_analysis',
                    'condition': 'requirements_unclear OR scope_changed',
                    'confidence_threshold': 0.6
                }
            ]
        )
        self.decision_points['post_validation_decision'] = post_validation
        
        # Complexity-based routing decision
        complexity_routing = DecisionPoint(
            decision_id='complexity_routing_decision',
            trigger_condition='analysis_complete',
            evaluator='ComplexityEvaluator',
            context_factors=['lines_of_code', 'files_affected', 'dependencies', 'cross_cutting'],
            confidence_threshold=0.7,
            outcomes=[
                {
                    'outcome': 'direct_to_implementation',
                    'condition': 'complexity <= low AND patterns_available',
                    'confidence_threshold': 0.8
                },
                {
                    'outcome': 'route_through_planning',
                    'condition': 'complexity = medium OR multi_component_change',
                    'confidence_threshold': 0.7
                },
                {
                    'outcome': 'require_architecture_phase',
                    'condition': 'complexity >= high OR system_design_needed',
                    'confidence_threshold': 0.9
                }
            ]
        )
        self.decision_points['complexity_routing_decision'] = complexity_routing
    
    def get_state_node(self, state_id: str) -> Optional[StateNode]:
        """Get state node by ID"""
        return self.states.get(state_id)
    
    def get_decision_point(self, decision_id: str) -> Optional[DecisionPoint]:
        """Get decision point by ID"""
        return self.decision_points.get(decision_id)
    
    def validate_transition(self, from_state: str, to_state: str, context: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Validate if a state transition is allowed with confidence scoring.
        
        Args:
            from_state: Current state
            to_state: Target state
            context: Current workflow context
            
        Returns:
            Tuple of (is_valid, reason, confidence)
        """
        if from_state not in self.states:
            return False, f"Unknown source state: {from_state}", 0.0
            
        if to_state not in self.states:
            return False, f"Unknown target state: {to_state}", 0.0
        
        source_state = self.states[from_state]
        can_transition, reason = source_state.can_transition(to_state, context)
        
        if not can_transition:
            return False, reason, 0.0
        
        # Calculate confidence based on context and evidence
        confidence = self._calculate_transition_confidence(from_state, to_state, context)
        
        return True, reason, confidence
    
    def _calculate_transition_confidence(self, from_state: str, to_state: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for a state transition"""
        base_confidence = 0.7
        
        # Adjust based on context
        if context.get('validation_passed', False):
            base_confidence += 0.2
        
        if context.get('complexity', 'medium') == 'high' and to_state == 'architecting':
            base_confidence += 0.1
        
        if context.get('requirements_clear', True):
            base_confidence += 0.1
        else:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    def get_recommended_transitions(self, current_state: str, context: Dict[str, Any], evidence: List[Evidence]) -> List[Tuple[str, float]]:
        """
        Get recommended state transitions with confidence scores.
        
        Args:
            current_state: Current workflow state
            context: Current context
            evidence: Available evidence
            
        Returns:
            List of tuples (next_state, confidence)
        """
        if current_state not in self.states:
            return []
        
        state_node = self.states[current_state]
        recommendations = []
        
        for target_state in state_node.can_transition_to:
            is_valid, reason, confidence = self.validate_transition(current_state, target_state, context)
            if is_valid:
                # Enhance confidence with evidence analysis
                evidence_confidence = self._analyze_evidence_for_transition(target_state, evidence)
                combined_confidence = (confidence + evidence_confidence) / 2
                recommendations.append((target_state, combined_confidence))
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def _analyze_evidence_for_transition(self, target_state: str, evidence: List[Evidence]) -> float:
        """Analyze evidence to determine confidence for transitioning to target state"""
        if not evidence:
            return 0.5  # Neutral confidence with no evidence
        
        relevant_evidence = []
        
        # Filter evidence relevant to target state
        for ev in evidence:
            if target_state == 'validating' and ev.evidence_type in ['test_result', 'implementation_complete']:
                relevant_evidence.append(ev)
            elif target_state == 'implementing' and ev.evidence_type in ['design_complete', 'requirements_clear']:
                relevant_evidence.append(ev)
            elif target_state == 'learning' and ev.evidence_type == 'validation_passed':
                relevant_evidence.append(ev)
        
        if not relevant_evidence:
            return 0.5
        
        # Calculate average confidence from relevant evidence
        total_confidence = sum(ev.confidence for ev in relevant_evidence)
        return total_confidence / len(relevant_evidence)


class DecisionEngine:
    """
    Evidence-based decision engine with confidence scoring and intelligent routing.
    Evaluates conditions and routes workflows dynamically based on context and evidence.
    """
    
    def __init__(self, state_graph_manager: StateGraphManager):
        self.state_graph_manager = state_graph_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def evaluate_decision_point(self, decision_id: str, context: Dict[str, Any], evidence: List[Evidence]) -> Optional[Dict[str, Any]]:
        """
        Evaluate a decision point and return recommended outcome.
        
        Args:
            decision_id: ID of decision point to evaluate
            context: Current workflow context
            evidence: Available evidence
            
        Returns:
            Dict with decision outcome or None if decision point not found
        """
        decision_point = self.state_graph_manager.get_decision_point(decision_id)
        if not decision_point:
            self.logger.warning(f"Decision point {decision_id} not found")
            return None
        
        # Check if decision point should be triggered
        should_trigger, trigger_confidence = decision_point.evaluate_condition(context, evidence)
        if not should_trigger:
            return None
        
        # Evaluate each outcome
        evaluated_outcomes = []
        
        for outcome in decision_point.outcomes:
            condition_met, condition_confidence = self._evaluate_outcome_condition(
                outcome['condition'], context, evidence
            )
            
            if condition_met and condition_confidence >= outcome.get('confidence_threshold', 0.6):
                evaluated_outcomes.append({
                    'outcome': outcome['outcome'],
                    'confidence': condition_confidence,
                    'condition': outcome['condition'],
                    'reason': f"Condition '{outcome['condition']}' met with {condition_confidence:.2f} confidence"
                })
        
        if not evaluated_outcomes:
            return {
                'decision_id': decision_id,
                'outcome': 'no_action',
                'confidence': 0.0,
                'reason': 'No outcome conditions met'
            }
        
        # Select outcome with highest confidence
        best_outcome = max(evaluated_outcomes, key=lambda x: x['confidence'])
        
        return {
            'decision_id': decision_id,
            'outcome': best_outcome['outcome'],
            'confidence': best_outcome['confidence'],
            'reason': best_outcome['reason'],
            'all_outcomes': evaluated_outcomes,
            'context_snapshot': context,
            'evidence_count': len(evidence)
        }
    
    def _evaluate_outcome_condition(self, condition: str, context: Dict[str, Any], evidence: List[Evidence]) -> Tuple[bool, float]:
        """
        Evaluate an outcome condition and return result with confidence.
        
        This is a simplified implementation - in production this would be more sophisticated.
        """
        condition_lower = condition.lower()
        
        # Simple condition evaluation
        if 'all_tests_pass' in condition_lower:
            test_evidence = [e for e in evidence if e.evidence_type == 'test_result']
            if test_evidence:
                avg_confidence = sum(e.confidence for e in test_evidence) / len(test_evidence)
                all_pass = all('pass' in e.content.lower() for e in test_evidence)
                return all_pass, avg_confidence
            return False, 0.0
        
        elif 'quality_gates_pass' in condition_lower:
            quality_evidence = [e for e in evidence if 'quality' in e.evidence_type]
            if quality_evidence:
                avg_confidence = sum(e.confidence for e in quality_evidence) / len(quality_evidence)
                gates_pass = all('pass' in e.content.lower() for e in quality_evidence)
                return gates_pass, avg_confidence
            return False, 0.0
        
        elif 'fixable_errors_identified' in condition_lower:
            error_evidence = [e for e in evidence if 'error' in e.content.lower()]
            if error_evidence:
                avg_confidence = sum(e.confidence for e in error_evidence) / len(error_evidence)
                fixable = any('fixable' in e.content.lower() for e in error_evidence)
                return fixable, avg_confidence
            return False, 0.0
        
        elif 'architectural_issues_detected' in condition_lower:
            arch_evidence = [e for e in evidence if 'architect' in e.content.lower()]
            issues_detected = context.get('architectural_issues', False)
            return issues_detected, 0.8 if issues_detected else 0.2
        
        elif 'requirements_unclear' in condition_lower:
            clarity = context.get('requirements_clarity', 1.0)
            unclear = clarity < 0.7
            return unclear, 0.9 if unclear else 0.1
        
        # Default case
        return False, 0.0
    
    def calculate_multi_factor_confidence(self, factors: Dict[str, float]) -> float:
        """
        Calculate confidence score based on multiple factors.
        
        Factors:
        - evidence_quality: 30%
        - pattern_matches: 20% 
        - agent_consensus: 20%
        - historical_success: 15%
        - context_completeness: 10%
        - validation_reliability: 5%
        """
        weights = {
            'evidence_quality': 0.30,
            'pattern_matches': 0.20,
            'agent_consensus': 0.20,
            'historical_success': 0.15,
            'context_completeness': 0.10,
            'validation_reliability': 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, value in factors.items():
            if factor in weights:
                weight = weights[factor]
                weighted_sum += value * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Neutral confidence
        
        return weighted_sum / total_weight


class DynamicOrchestrator:
    """
    Core dynamic orchestration engine that integrates all components.
    Supports Claude Code by providing intelligent orchestration capabilities.
    """
    
    def __init__(self):
        self.state_graph_manager = StateGraphManager()
        self.decision_engine = DecisionEngine(self.state_graph_manager)
        self.active_sessions: Dict[str, WorkflowSession] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_session(self, issue_number: int, initial_context: Dict[str, Any]) -> str:
        """Create a new workflow session"""
        session_id = f"session_{issue_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = WorkflowSession(
            session_id=session_id,
            issue_number=issue_number,
            current_state='analyzing',  # Default starting state
            context=initial_context
        )
        
        self.active_sessions[session_id] = session
        self.logger.info(f"Created workflow session {session_id} for issue #{issue_number}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[WorkflowSession]:
        """Get workflow session by ID"""
        return self.active_sessions.get(session_id)
    
    def add_evidence_to_session(self, session_id: str, evidence: Evidence):
        """Add evidence to a workflow session"""
        session = self.active_sessions.get(session_id)
        if session:
            session.add_evidence(evidence)
            self.logger.info(f"Added {evidence.evidence_type} evidence to session {session_id}")
    
    def evaluate_workflow_decision(self, session_id: str, decision_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the current workflow state and recommend next actions.
        
        Args:
            session_id: Active workflow session ID
            decision_context: Additional context for decision making
            
        Returns:
            Dict with orchestration recommendations
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {'error': f'Session {session_id} not found'}
        
        # Merge session context with decision context
        full_context = {**session.context}
        if decision_context:
            full_context.update(decision_context)
        
        # Get recommended transitions from current state
        recommendations = self.state_graph_manager.get_recommended_transitions(
            session.current_state, full_context, session.evidence_trail
        )
        
        # Evaluate relevant decision points
        decision_results = []
        for decision_id in self.state_graph_manager.decision_points:
            result = self.decision_engine.evaluate_decision_point(
                decision_id, full_context, session.evidence_trail
            )
            if result:
                decision_results.append(result)
        
        return {
            'session_id': session_id,
            'current_state': session.current_state,
            'recommended_transitions': recommendations,
            'decision_point_results': decision_results,
            'evidence_count': len(session.evidence_trail),
            'context': full_context,
            'timestamp': datetime.now().isoformat()
        }
    
    def execute_state_transition(self, session_id: str, target_state: str, reason: str) -> Dict[str, Any]:
        """
        Execute a state transition with validation and recording.
        
        Args:
            session_id: Workflow session ID
            target_state: Target state to transition to
            reason: Reason for the transition
            
        Returns:
            Dict with transition result
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {'success': False, 'error': f'Session {session_id} not found'}
        
        # Validate transition
        is_valid, validation_reason, confidence = self.state_graph_manager.validate_transition(
            session.current_state, target_state, session.context
        )
        
        if not is_valid:
            return {
                'success': False,
                'error': f'Invalid transition: {validation_reason}',
                'confidence': confidence
            }
        
        # Execute transition
        old_state = session.current_state
        session.record_transition(old_state, target_state, reason, confidence)
        
        self.logger.info(f"Executed transition {old_state} -> {target_state} for session {session_id}")
        
        return {
            'success': True,
            'from_state': old_state,
            'to_state': target_state,
            'reason': reason,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_orchestration_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive orchestration summary for a session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {'error': f'Session {session_id} not found'}
        
        return {
            'session_id': session_id,
            'issue_number': session.issue_number,
            'current_state': session.current_state,
            'total_transitions': len(session.transition_history),
            'evidence_count': len(session.evidence_trail),
            'session_duration': (datetime.now() - session.created_at).total_seconds(),
            'transition_history': session.transition_history[-5:],  # Last 5 transitions
            'recent_evidence': [e.to_dict() for e in session.evidence_trail[-3:]],  # Last 3 evidence items
            'context_summary': {
                'complexity': session.context.get('complexity', 'unknown'),
                'priority': session.context.get('priority', 'unknown'),
                'requirements_clear': session.context.get('requirements_clear', True)
            }
        }


# Example usage and testing functions
def create_test_evidence() -> List[Evidence]:
    """Create test evidence for demonstration"""
    return [
        Evidence(
            source='test_runner',
            content='All 25 unit tests passed successfully',
            confidence=0.95,
            timestamp=datetime.now(),
            evidence_type='test_result'
        ),
        Evidence(
            source='quality_gates',
            content='Code coverage: 85%, Security scan: passed',
            confidence=0.90,
            timestamp=datetime.now(),
            evidence_type='quality_gate'
        ),
        Evidence(
            source='rif-implementer',
            content='Implementation completed with checkpoint validation',
            confidence=0.88,
            timestamp=datetime.now(),
            evidence_type='implementation_complete'
        )
    ]


def main():
    """Demonstrate the Dynamic Orchestrator Engine"""
    print("RIF Dynamic Orchestrator Engine - Demonstration")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = DynamicOrchestrator()
    
    # Create test session
    initial_context = {
        'complexity': 'high',
        'priority': 'medium',
        'requirements_clear': True,
        'validation_complete': True
    }
    
    session_id = orchestrator.create_session(51, initial_context)
    print(f"\nCreated session: {session_id}")
    
    # Add test evidence
    test_evidence = create_test_evidence()
    for evidence in test_evidence:
        orchestrator.add_evidence_to_session(session_id, evidence)
    
    print(f"Added {len(test_evidence)} pieces of evidence")
    
    # Evaluate workflow decision
    decision_result = orchestrator.evaluate_workflow_decision(session_id)
    print(f"\nWorkflow Decision Evaluation:")
    print(f"Current State: {decision_result['current_state']}")
    print(f"Recommended Transitions: {decision_result['recommended_transitions']}")
    print(f"Decision Point Results: {len(decision_result['decision_point_results'])}")
    
    # Test state transition
    if decision_result['recommended_transitions']:
        next_state, confidence = decision_result['recommended_transitions'][0]
        transition_result = orchestrator.execute_state_transition(
            session_id, next_state, f"Recommended transition with {confidence:.2f} confidence"
        )
        print(f"\nTransition Result: {transition_result}")
    
    # Get orchestration summary
    summary = orchestrator.get_orchestration_summary(session_id)
    print(f"\nOrchestration Summary:")
    print(f"Total Transitions: {summary['total_transitions']}")
    print(f"Evidence Count: {summary['evidence_count']}")
    print(f"Session Duration: {summary['session_duration']:.1f} seconds")


if __name__ == "__main__":
    main()