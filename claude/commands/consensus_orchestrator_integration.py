#!/usr/bin/env python3
"""
Consensus Orchestrator Integration - Critical Integration for Issue #148

This module integrates the consensus architecture with the enhanced orchestration intelligence
to provide multi-agent consensus validation for critical decisions. It implements the
risk-based consensus triggers and decision criteria specified in the planning phase.

CRITICAL: This integration connects the 674-line consensus architecture to the orchestration
decision-making process, enabling parallel agent consensus for critical decisions.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import consensus and orchestration components
try:
    from claude.commands.consensus_architecture import (
        ConsensusArchitecture, VotingMechanism, RiskLevel, ConfidenceLevel,
        AgentVote, VotingConfig, ConsensusResult
    )
    from claude.commands.parallel_execution_coordinator import (
        ParallelExecutionCoordinator, ParallelPath, ResourceRequirement, ResourceType
    )
except ImportError as e:
    logging.warning(f"Could not import consensus components: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DecisionContext:
    """Context for orchestration decisions requiring consensus"""
    issue_number: int
    issue_title: str
    risk_level: str
    complexity: str
    security_critical: bool
    agent_confidence: float
    previous_failures: bool
    multi_system_impact: bool
    emergency_protocol: bool
    estimated_impact: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'issue_number': self.issue_number,
            'issue_title': self.issue_title,
            'risk_level': self.risk_level,
            'complexity': self.complexity,
            'security_critical': self.security_critical,
            'agent_confidence': self.agent_confidence,
            'previous_failures': self.previous_failures,
            'multi_system_impact': self.multi_system_impact,
            'emergency_protocol': self.emergency_protocol,
            'estimated_impact': self.estimated_impact
        }

@dataclass
class ConsensusDecision:
    """Result of consensus-based orchestration decision"""
    decision_id: str
    consensus_required: bool
    consensus_result: Optional[ConsensusResult]
    voting_mechanism: Optional[VotingMechanism]
    decision_rationale: str
    recommended_agents: List[str]
    parallel_execution_required: bool
    escalation_required: bool
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'consensus_required': self.consensus_required,
            'consensus_result': self.consensus_result.__dict__ if self.consensus_result else None,
            'voting_mechanism': self.voting_mechanism.value if self.voting_mechanism else None,
            'decision_rationale': self.decision_rationale,
            'recommended_agents': self.recommended_agents,
            'parallel_execution_required': self.parallel_execution_required,
            'escalation_required': self.escalation_required,
            'timestamp': self.timestamp.isoformat()
        }

class ConsensusRequirementEvaluator:
    """
    Evaluates whether consensus is required based on decision context.
    Implements the risk-based consensus triggers from the planning phase.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Consensus trigger thresholds
        self.risk_thresholds = {
            'critical': True,  # Always requires consensus
            'high': True,      # Always requires consensus
            'medium': False,   # Conditional consensus
            'low': False       # No consensus required
        }
        
        self.complexity_thresholds = {
            'very_high': True,  # Always requires consensus
            'high': True,       # Always requires consensus
            'medium': False,    # Conditional consensus
            'low': False        # No consensus required
        }
        
        self.confidence_threshold = 0.7  # Below this triggers consensus
        
    def evaluate_consensus_requirement(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Evaluate if consensus is required for a decision context.
        
        Args:
            context: Decision context to evaluate
            
        Returns:
            Dict with consensus requirement evaluation
        """
        evaluation = {
            'consensus_required': False,
            'trigger_reasons': [],
            'recommended_mechanism': None,
            'urgency_level': 'normal',
            'timeout_minutes': 30
        }
        
        # Critical risk factors (automatic consensus)
        if context.risk_level.lower() == 'critical':
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append('critical_risk_level')
            evaluation['recommended_mechanism'] = VotingMechanism.UNANIMOUS
            evaluation['urgency_level'] = 'high'
            evaluation['timeout_minutes'] = 15
        
        elif context.risk_level.lower() == 'high':
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append('high_risk_level')
            evaluation['recommended_mechanism'] = VotingMechanism.SUPERMAJORITY
            evaluation['urgency_level'] = 'high'
            evaluation['timeout_minutes'] = 20
        
        # Security critical decisions (automatic consensus)
        if context.security_critical:
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append('security_critical')
            evaluation['recommended_mechanism'] = VotingMechanism.VETO_POWER
            evaluation['urgency_level'] = 'high'
            evaluation['timeout_minutes'] = 15
        
        # Emergency protocols (automatic consensus with timeout)
        if context.emergency_protocol:
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append('emergency_protocol')
            evaluation['recommended_mechanism'] = VotingMechanism.SUPERMAJORITY
            evaluation['urgency_level'] = 'critical'
            evaluation['timeout_minutes'] = 10
        
        # Complexity-based triggers
        if context.complexity.lower() in ['high', 'very_high']:
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append(f'{context.complexity}_complexity')
            if not evaluation['recommended_mechanism']:
                evaluation['recommended_mechanism'] = VotingMechanism.WEIGHTED_VOTING
        
        # Confidence-based triggers
        if context.agent_confidence < self.confidence_threshold:
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append('low_agent_confidence')
            if not evaluation['recommended_mechanism']:
                evaluation['recommended_mechanism'] = VotingMechanism.SIMPLE_MAJORITY
        
        # Multi-system impact triggers
        if context.multi_system_impact:
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append('multi_system_impact')
            if not evaluation['recommended_mechanism']:
                evaluation['recommended_mechanism'] = VotingMechanism.WEIGHTED_VOTING
        
        # Previous failures trigger
        if context.previous_failures:
            evaluation['consensus_required'] = True
            evaluation['trigger_reasons'].append('previous_failures')
            if not evaluation['recommended_mechanism']:
                evaluation['recommended_mechanism'] = VotingMechanism.SUPERMAJORITY
        
        # Default mechanism if consensus required but no specific mechanism set
        if evaluation['consensus_required'] and not evaluation['recommended_mechanism']:
            evaluation['recommended_mechanism'] = VotingMechanism.SIMPLE_MAJORITY
        
        self.logger.info(f"Consensus evaluation for issue #{context.issue_number}: "
                        f"Required={evaluation['consensus_required']}, "
                        f"Triggers={evaluation['trigger_reasons']}")
        
        return evaluation

class ConsensusOrchestratorIntegration:
    """
    Main integration class connecting consensus architecture to orchestration intelligence.
    Provides consensus-based decision making for critical orchestration decisions.
    """
    
    def __init__(self):
        try:
            self.consensus_architecture = ConsensusArchitecture()
            self.parallel_coordinator = ParallelExecutionCoordinator()
        except Exception as e:
            logger.warning(f"Could not initialize consensus components: {e}")
            self.consensus_architecture = None
            self.parallel_coordinator = None
        
        self.requirement_evaluator = ConsensusRequirementEvaluator()
        
        # Decision tracking
        self.active_decisions: Dict[str, DecisionContext] = {}
        self.decision_history: List[ConsensusDecision] = []
        
        # Performance metrics
        self.metrics = {
            'total_decisions': 0,
            'consensus_decisions': 0,
            'consensus_success_rate': 0.0,
            'average_consensus_time': 0.0,
            'escalation_rate': 0.0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_orchestration_decision(self, issue_context: Dict[str, Any]) -> ConsensusDecision:
        """
        Evaluate whether an orchestration decision requires consensus and execute if needed.
        
        Args:
            issue_context: Context about the issue requiring orchestration decision
            
        Returns:
            ConsensusDecision with recommendation
        """
        decision_id = f"decision_{issue_context.get('issue_number', 'unknown')}_{int(time.time())}"
        
        # Create decision context
        decision_context = DecisionContext(
            issue_number=issue_context.get('issue_number', 0),
            issue_title=issue_context.get('title', 'Unknown'),
            risk_level=issue_context.get('risk_level', 'medium'),
            complexity=issue_context.get('complexity', 'medium'),
            security_critical=issue_context.get('security_critical', False),
            agent_confidence=issue_context.get('agent_confidence', 1.0),
            previous_failures=issue_context.get('previous_failures', False),
            multi_system_impact=issue_context.get('multi_system_impact', False),
            emergency_protocol=issue_context.get('emergency_protocol', False),
            estimated_impact=issue_context.get('estimated_impact', 'medium')
        )
        
        # Evaluate consensus requirement
        evaluation = self.requirement_evaluator.evaluate_consensus_requirement(decision_context)
        
        consensus_decision = ConsensusDecision(
            decision_id=decision_id,
            consensus_required=evaluation['consensus_required'],
            consensus_result=None,
            voting_mechanism=evaluation.get('recommended_mechanism'),
            decision_rationale=f"Triggered by: {', '.join(evaluation['trigger_reasons'])}",
            recommended_agents=[],
            parallel_execution_required=evaluation['consensus_required'],
            escalation_required=False,
            timestamp=datetime.now()
        )
        
        if evaluation['consensus_required'] and self.consensus_architecture:
            # Execute consensus process
            consensus_result = self._execute_consensus_process(decision_context, evaluation)
            consensus_decision.consensus_result = consensus_result
            
            # Determine recommended agents based on consensus
            consensus_decision.recommended_agents = self._determine_consensus_agents(
                decision_context, evaluation['recommended_mechanism']
            )
            
            # Check if escalation is needed
            if consensus_result and consensus_result.arbitration_triggered:
                consensus_decision.escalation_required = True
        
        else:
            # No consensus required - single agent decision
            consensus_decision.recommended_agents = self._determine_single_agent(decision_context)
            consensus_decision.decision_rationale = "Single agent sufficient - no consensus triggers"
        
        # Update metrics
        self._update_metrics(consensus_decision)
        
        # Store decision
        self.decision_history.append(consensus_decision)
        self.active_decisions[decision_id] = decision_context
        
        self.logger.info(f"Orchestration decision {decision_id}: "
                        f"Consensus={consensus_decision.consensus_required}, "
                        f"Agents={consensus_decision.recommended_agents}")
        
        return consensus_decision
    
    def _execute_consensus_process(self, context: DecisionContext, 
                                 evaluation: Dict[str, Any]) -> Optional[ConsensusResult]:
        """Execute the consensus voting process"""
        if not self.consensus_architecture:
            self.logger.warning("Consensus architecture not available")
            return None
        
        try:
            # Create consensus context
            consensus_context = {
                'risk_level': context.risk_level,
                'domain': 'orchestration',
                'security_critical': context.security_critical,
                'complexity': context.complexity
            }
            
            # Start consensus vote
            decision_id = f"orchestration_{context.issue_number}_{int(time.time())}"
            consensus_result, votes = self.consensus_architecture.run_consensus_vote(
                decision_id, consensus_context, evaluation.get('timeout_minutes')
            )
            
            self.logger.info(f"Consensus completed for issue #{context.issue_number}: "
                           f"Decision={consensus_result.decision}, "
                           f"Confidence={consensus_result.confidence_score:.2f}")
            
            return consensus_result
        
        except Exception as e:
            self.logger.error(f"Error executing consensus process: {e}")
            return None
    
    def _determine_consensus_agents(self, context: DecisionContext, 
                                  mechanism: VotingMechanism) -> List[str]:
        """Determine which agents should participate in consensus"""
        agents = []
        
        # Base agents for all consensus decisions
        agents.append("rif-analyst")  # Requirements analysis
        agents.append("rif-validator")  # Quality validation
        
        # Risk-based agent selection
        if context.risk_level.lower() in ['high', 'critical']:
            agents.append("rif-architect")  # Architecture review
        
        if context.security_critical:
            agents.append("rif-security")  # Security review
        
        if context.complexity.lower() in ['high', 'very_high']:
            agents.append("rif-planner")  # Complex planning
        
        if context.emergency_protocol:
            agents.append("rif-emergency")  # Emergency protocols
        
        # Mechanism-specific additions
        if mechanism == VotingMechanism.WEIGHTED_VOTING:
            # Ensure key weighted agents are included
            if "rif-implementer" not in agents:
                agents.append("rif-implementer")
        
        elif mechanism == VotingMechanism.VETO_POWER:
            # Ensure veto power agents are included
            if "rif-security" not in agents:
                agents.append("rif-security")
        
        return list(set(agents))  # Remove duplicates
    
    def _determine_single_agent(self, context: DecisionContext) -> List[str]:
        """Determine single agent for non-consensus decisions"""
        # Simple agent selection based on context
        if context.complexity.lower() == 'low':
            return ["rif-implementer"]
        elif context.risk_level.lower() == 'low':
            return ["rif-analyst"]
        else:
            return ["rif-planner"]
    
    def _update_metrics(self, decision: ConsensusDecision):
        """Update consensus metrics"""
        self.metrics['total_decisions'] += 1
        
        if decision.consensus_required:
            self.metrics['consensus_decisions'] += 1
        
        if decision.escalation_required:
            escalations = sum(1 for d in self.decision_history if d.escalation_required)
            self.metrics['escalation_rate'] = escalations / len(self.decision_history)
        
        # Calculate success rate
        successful_decisions = sum(
            1 for d in self.decision_history 
            if d.consensus_result and d.consensus_result.decision is True
        )
        if self.metrics['consensus_decisions'] > 0:
            self.metrics['consensus_success_rate'] = successful_decisions / self.metrics['consensus_decisions']
    
    def create_parallel_consensus_paths(self, decision: ConsensusDecision) -> List[ParallelPath]:
        """
        Create parallel execution paths for consensus agents.
        
        Args:
            decision: Consensus decision requiring parallel execution
            
        Returns:
            List of parallel paths for consensus agents
        """
        if not decision.parallel_execution_required or not self.parallel_coordinator:
            return []
        
        paths = []
        
        for i, agent in enumerate(decision.recommended_agents):
            path = ParallelPath(
                path_id=f"consensus_{decision.decision_id}_{agent}",
                description=f"Consensus validation by {agent}",
                agents=[agent],
                resource_requirements=ResourceRequirement(
                    cpu_cores=1.0,
                    memory_mb=1024,
                    io_bandwidth=2.0,
                    execution_time_estimate=300,  # 5 minutes
                    resource_type=ResourceType.MIXED
                ),
                synchronization_points=[f"consensus_sync_{decision.decision_id}"]
            )
            paths.append(path)
        
        self.logger.info(f"Created {len(paths)} parallel consensus paths for decision {decision.decision_id}")
        return paths
    
    def generate_orchestration_recommendations(self, decision: ConsensusDecision) -> Dict[str, Any]:
        """
        Generate orchestration recommendations based on consensus decision.
        
        Args:
            decision: Consensus decision to base recommendations on
            
        Returns:
            Dict with orchestration recommendations
        """
        recommendations = {
            'decision_id': decision.decision_id,
            'orchestration_approach': 'single_agent',
            'agent_launch_strategy': 'sequential',
            'priority_level': 'normal',
            'monitoring_required': False,
            'timeout_minutes': 60,
            'success_criteria': [],
            'risk_mitigation': []
        }
        
        if decision.consensus_required:
            recommendations.update({
                'orchestration_approach': 'multi_agent_consensus',
                'agent_launch_strategy': 'parallel',
                'priority_level': 'high',
                'monitoring_required': True,
                'timeout_minutes': 30,
                'success_criteria': [
                    'Consensus achieved among participating agents',
                    'No arbitration escalation required',
                    'Decision confidence above 0.7'
                ],
                'risk_mitigation': [
                    'Multiple agent perspectives reduce single-point-of-failure',
                    'Consensus validation catches edge cases',
                    'Arbitration process handles conflicts'
                ]
            })
        
        if decision.escalation_required:
            recommendations.update({
                'priority_level': 'critical',
                'monitoring_required': True,
                'timeout_minutes': 15,
                'success_criteria': recommendations['success_criteria'] + [
                    'Escalation resolved within timeout',
                    'Human intervention if needed'
                ],
                'risk_mitigation': recommendations['risk_mitigation'] + [
                    'Escalation process prevents deadlocks',
                    'Human oversight for complex conflicts'
                ]
            })
        
        return recommendations
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics"""
        return {
            'integration_active': self.consensus_architecture is not None,
            'parallel_coordinator_available': self.parallel_coordinator is not None,
            'active_decisions': len(self.active_decisions),
            'decision_history_count': len(self.decision_history),
            'metrics': self.metrics,
            'recent_decisions': [
                decision.to_dict() for decision in self.decision_history[-5:]
            ]
        }

# Factory function for integration
def get_consensus_orchestrator_integration() -> ConsensusOrchestratorIntegration:
    """
    Factory function to get the consensus orchestrator integration.
    
    Returns:
        ConsensusOrchestratorIntegration instance
    """
    return ConsensusOrchestratorIntegration()

# Convenience functions for orchestration integration
def evaluate_consensus_requirement(issue_context: Dict[str, Any]) -> ConsensusDecision:
    """Convenience function for consensus requirement evaluation"""
    integration = get_consensus_orchestrator_integration()
    return integration.evaluate_orchestration_decision(issue_context)

def create_consensus_parallel_paths(consensus_decision: ConsensusDecision) -> List[ParallelPath]:
    """Convenience function for creating parallel consensus paths"""
    integration = get_consensus_orchestrator_integration()
    return integration.create_parallel_consensus_paths(consensus_decision)

# Testing and demonstration
def main():
    """Demonstrate the consensus orchestrator integration"""
    print("RIF Consensus Orchestrator Integration - Demonstration")
    print("=" * 60)
    
    # Initialize integration
    integration = get_consensus_orchestrator_integration()
    
    # Test critical decision scenario
    critical_issue_context = {
        'issue_number': 148,
        'title': 'CRITICAL FIX: Parallel Agent Consensus System Non-Functional',
        'risk_level': 'critical',
        'complexity': 'high',
        'security_critical': True,
        'agent_confidence': 0.6,
        'previous_failures': True,
        'multi_system_impact': True,
        'emergency_protocol': False,
        'estimated_impact': 'high'
    }
    
    print(f"\n=== Critical Decision Scenario ===")
    decision = integration.evaluate_orchestration_decision(critical_issue_context)
    print(f"Decision ID: {decision.decision_id}")
    print(f"Consensus Required: {decision.consensus_required}")
    print(f"Voting Mechanism: {decision.voting_mechanism}")
    print(f"Rationale: {decision.decision_rationale}")
    print(f"Recommended Agents: {decision.recommended_agents}")
    print(f"Parallel Execution: {decision.parallel_execution_required}")
    
    # Generate recommendations
    recommendations = integration.generate_orchestration_recommendations(decision)
    print(f"\n=== Orchestration Recommendations ===")
    print(f"Approach: {recommendations['orchestration_approach']}")
    print(f"Strategy: {recommendations['agent_launch_strategy']}")
    print(f"Priority: {recommendations['priority_level']}")
    print(f"Success Criteria: {len(recommendations['success_criteria'])} criteria defined")
    
    # Create parallel paths if needed
    if decision.parallel_execution_required:
        parallel_paths = integration.create_parallel_consensus_paths(decision)
        print(f"\n=== Parallel Consensus Paths ===")
        print(f"Created {len(parallel_paths)} parallel paths for consensus")
        for path in parallel_paths:
            print(f"- {path.path_id}: {path.description}")
    
    # Display integration status
    status = integration.get_integration_status()
    print(f"\n=== Integration Status ===")
    print(f"Integration Active: {status['integration_active']}")
    print(f"Total Decisions: {status['metrics']['total_decisions']}")
    print(f"Consensus Decisions: {status['metrics']['consensus_decisions']}")
    print(f"Success Rate: {status['metrics']['consensus_success_rate']:.1%}")

if __name__ == "__main__":
    main()