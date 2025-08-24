#!/usr/bin/env python3
"""
RIF Escalation Engine
Multi-level escalation system for systematic conflict resolution.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Import arbitration components
from .conflict_detector import ConflictAnalysis, ConflictSeverity, ConflictPattern
from .arbitration_system import ArbitrationDecision, ArbitrationType

# Import consensus components
import sys
sys.path.insert(0, '/Users/cal/DEV/RIF/claude/commands')

from consensus_architecture import AgentVote, ConfidenceLevel, ConsensusArchitecture
from voting_aggregator import VoteConflict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EscalationLevel(Enum):
    """Escalation levels in order of severity"""
    AUTOMATED_RESOLUTION = "automated_resolution"
    WEIGHTED_CONSENSUS = "weighted_consensus"
    EXPERT_PANEL = "expert_panel"
    ARBITRATOR_AGENT = "arbitrator_agent"
    EVIDENCE_REVIEW = "evidence_review"
    HUMAN_INTERVENTION = "human_intervention"
    EXECUTIVE_DECISION = "executive_decision"

class EscalationStrategy(Enum):
    """Strategies for escalation path determination"""
    CONSERVATIVE = "conservative"       # Escalates quickly to ensure safety
    BALANCED = "balanced"              # Standard escalation based on severity
    AGGRESSIVE = "aggressive"          # Tries automated resolution longer
    SECURITY_FOCUSED = "security_focused"  # Prioritizes security concerns
    EFFICIENCY_FOCUSED = "efficiency_focused"  # Prioritizes fast resolution

class EscalationTrigger(Enum):
    """Triggers that cause escalation"""
    SEVERITY_THRESHOLD = "severity_threshold"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
    METHOD_FAILED = "method_failed"
    SECURITY_CONCERN = "security_concern"
    EXPERT_OVERRIDE = "expert_override"

@dataclass
class EscalationStep:
    """Individual step in escalation process"""
    level: EscalationLevel
    method_name: str
    timeout_minutes: int
    required_confidence: float
    success_criteria: Dict[str, Any]
    fallback_level: Optional[EscalationLevel]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationPath:
    """Complete escalation path for a conflict"""
    path_id: str
    strategy: EscalationStrategy
    steps: List[EscalationStep]
    max_total_time_minutes: int
    created_for_context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EscalationResult:
    """Result of an escalation attempt"""
    step: EscalationStep
    success: bool
    decision: Optional[ArbitrationDecision]
    execution_time_seconds: float
    trigger: EscalationTrigger
    next_level: Optional[EscalationLevel]
    metadata: Dict[str, Any] = field(default_factory=dict)

class EscalationEngine:
    """
    Advanced escalation engine that manages multi-level conflict resolution
    """
    
    def __init__(self, consensus: Optional[ConsensusArchitecture] = None,
                 knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """Initialize escalation engine"""
        self.consensus = consensus or ConsensusArchitecture()
        self.knowledge_base_path = Path(knowledge_base_path)
        
        # Escalation configuration
        self.config = {
            "default_strategy": EscalationStrategy.BALANCED,
            "max_escalation_time_minutes": 120,  # 2 hours total
            "confidence_thresholds": {
                EscalationLevel.AUTOMATED_RESOLUTION: 0.6,
                EscalationLevel.WEIGHTED_CONSENSUS: 0.7,
                EscalationLevel.EXPERT_PANEL: 0.8,
                EscalationLevel.ARBITRATOR_AGENT: 0.75,
                EscalationLevel.EVIDENCE_REVIEW: 0.85,
                EscalationLevel.HUMAN_INTERVENTION: 0.9
            },
            "timeout_minutes": {
                EscalationLevel.AUTOMATED_RESOLUTION: 5,
                EscalationLevel.WEIGHTED_CONSENSUS: 10,
                EscalationLevel.EXPERT_PANEL: 30,
                EscalationLevel.ARBITRATOR_AGENT: 45,
                EscalationLevel.EVIDENCE_REVIEW: 30,
                EscalationLevel.HUMAN_INTERVENTION: 1440  # 24 hours
            }
        }
        
        # Strategy definitions
        self.strategies = self._initialize_strategies()
        
        # Escalation methods registry
        self.escalation_methods: Dict[EscalationLevel, Callable] = {
            EscalationLevel.AUTOMATED_RESOLUTION: self._automated_resolution,
            EscalationLevel.WEIGHTED_CONSENSUS: self._weighted_consensus,
            EscalationLevel.EXPERT_PANEL: self._expert_panel,
            EscalationLevel.ARBITRATOR_AGENT: self._arbitrator_agent,
            EscalationLevel.EVIDENCE_REVIEW: self._evidence_review,
            EscalationLevel.HUMAN_INTERVENTION: self._human_intervention,
            EscalationLevel.EXECUTIVE_DECISION: self._executive_decision
        }
        
        # Performance tracking
        self.metrics = {
            "total_escalations": 0,
            "successful_resolutions_by_level": {level.value: 0 for level in EscalationLevel},
            "average_resolution_time_by_level": {level.value: 0.0 for level in EscalationLevel},
            "escalation_triggers": {trigger.value: 0 for trigger in EscalationTrigger},
            "strategy_effectiveness": {strategy.value: {"attempts": 0, "successes": 0} for strategy in EscalationStrategy}
        }

    def create_escalation_path(self, conflict_analysis: ConflictAnalysis, 
                             context: Dict[str, Any],
                             strategy: Optional[EscalationStrategy] = None) -> EscalationPath:
        """
        Create appropriate escalation path based on conflict analysis
        
        Args:
            conflict_analysis: Detailed conflict analysis
            context: Decision context
            strategy: Optional escalation strategy override
            
        Returns:
            EscalationPath: Customized escalation path
        """
        strategy = strategy or self._determine_optimal_strategy(conflict_analysis, context)
        path_id = f"escalation-{int(time.time())}-{conflict_analysis.conflict_id[:8]}"
        
        logger.info(f"Creating escalation path {path_id} with strategy {strategy.value}")
        
        # Get base steps for strategy
        base_steps = self.strategies[strategy].copy()
        
        # Customize steps based on conflict analysis
        customized_steps = self._customize_escalation_steps(base_steps, conflict_analysis, context)
        
        # Calculate total time limit
        max_time = sum(step.timeout_minutes for step in customized_steps)
        max_time = min(max_time, self.config["max_escalation_time_minutes"])
        
        return EscalationPath(
            path_id=path_id,
            strategy=strategy,
            steps=customized_steps,
            max_total_time_minutes=max_time,
            created_for_context=context
        )

    def execute_escalation_path(self, path: EscalationPath, votes: List[AgentVote],
                              conflict_analysis: ConflictAnalysis,
                              context: Dict[str, Any]) -> Tuple[ArbitrationDecision, List[EscalationResult]]:
        """
        Execute escalation path until resolution or exhaustion
        
        Args:
            path: Escalation path to execute
            votes: Agent votes to resolve
            conflict_analysis: Conflict analysis
            context: Decision context
            
        Returns:
            Tuple of final decision and escalation results
        """
        start_time = datetime.now()
        escalation_results = []
        
        logger.info(f"Executing escalation path {path.path_id} with {len(path.steps)} steps")
        
        for i, step in enumerate(path.steps):
            step_start_time = time.time()
            
            # Check total time limit
            elapsed_total = (datetime.now() - start_time).total_seconds() / 60
            if elapsed_total > path.max_total_time_minutes:
                logger.warning(f"Escalation path {path.path_id} exceeded time limit")
                break
            
            logger.info(f"Executing escalation step {i+1}/{len(path.steps)}: {step.level.value}")
            
            try:
                # Execute escalation method
                method = self.escalation_methods[step.level]
                decision = method(votes, conflict_analysis, context, step)
                
                execution_time = time.time() - step_start_time
                
                # Evaluate success
                success = self._evaluate_step_success(decision, step, conflict_analysis, context)
                
                # Determine trigger for next step
                trigger = self._determine_escalation_trigger(decision, step, success)
                
                # Create escalation result
                result = EscalationResult(
                    step=step,
                    success=success,
                    decision=decision,
                    execution_time_seconds=execution_time,
                    trigger=trigger,
                    next_level=path.steps[i+1].level if i+1 < len(path.steps) else None
                )
                
                escalation_results.append(result)
                
                # Update metrics
                self._update_escalation_metrics(step.level, success, execution_time)
                
                # Check if resolution is satisfactory
                if success and self._is_final_resolution(decision, step, context):
                    logger.info(f"Escalation path {path.path_id} resolved at level {step.level.value}")
                    return decision, escalation_results
                
                # Log step completion
                logger.info(f"Step {step.level.value} completed - Success: {success}, "
                          f"Confidence: {decision.confidence_score:.2f}")
                
            except Exception as e:
                logger.error(f"Escalation step {step.level.value} failed: {str(e)}")
                
                # Create failed result
                result = EscalationResult(
                    step=step,
                    success=False,
                    decision=None,
                    execution_time_seconds=time.time() - step_start_time,
                    trigger=EscalationTrigger.METHOD_FAILED,
                    next_level=path.steps[i+1].level if i+1 < len(path.steps) else None,
                    metadata={"error": str(e)}
                )
                
                escalation_results.append(result)
                continue
        
        # If we reach here, all steps completed but no satisfactory resolution
        logger.warning(f"Escalation path {path.path_id} exhausted without satisfactory resolution")
        
        # Return the best decision we found, or create fallback
        best_decision = self._select_best_decision(escalation_results)
        if not best_decision:
            best_decision = self._create_exhaustion_fallback(votes, context, path.path_id)
        
        return best_decision, escalation_results

    def _determine_optimal_strategy(self, conflict_analysis: ConflictAnalysis, 
                                  context: Dict[str, Any]) -> EscalationStrategy:
        """Determine optimal escalation strategy based on analysis and context"""
        
        # Security-critical decisions use security-focused strategy
        if context.get("security_critical", False):
            return EscalationStrategy.SECURITY_FOCUSED
        
        # High severity conflicts need conservative approach
        if conflict_analysis.severity == ConflictSeverity.CRITICAL:
            return EscalationStrategy.CONSERVATIVE
        
        # Low severity conflicts can try aggressive resolution
        if conflict_analysis.severity == ConflictSeverity.LOW:
            return EscalationStrategy.EFFICIENCY_FOCUSED
        
        # High risk contexts use conservative approach
        risk_level = context.get("risk_level", "medium")
        if risk_level == "critical" or risk_level == "high":
            return EscalationStrategy.CONSERVATIVE
        
        # Expertise gap patterns benefit from expert-focused approach
        if ConflictPattern.EXPERTISE_GAP in conflict_analysis.patterns_detected:
            return EscalationStrategy.BALANCED
        
        # Default to balanced strategy
        return EscalationStrategy.BALANCED

    def _customize_escalation_steps(self, base_steps: List[EscalationStep],
                                  conflict_analysis: ConflictAnalysis,
                                  context: Dict[str, Any]) -> List[EscalationStep]:
        """Customize escalation steps based on specific conflict characteristics"""
        customized_steps = []
        
        for step in base_steps:
            # Adjust confidence thresholds based on context
            required_confidence = step.required_confidence
            
            if context.get("security_critical", False):
                required_confidence = min(1.0, required_confidence + 0.1)
            
            if conflict_analysis.severity == ConflictSeverity.HIGH:
                required_confidence = min(1.0, required_confidence + 0.05)
            
            # Adjust timeouts based on conflict complexity
            timeout = step.timeout_minutes
            
            if ConflictPattern.EXPERTISE_GAP in conflict_analysis.patterns_detected:
                if step.level == EscalationLevel.EXPERT_PANEL:
                    timeout = int(timeout * 1.5)  # More time for expert consultation
            
            if ConflictPattern.EVIDENCE_QUALITY in conflict_analysis.patterns_detected:
                if step.level == EscalationLevel.EVIDENCE_REVIEW:
                    timeout = int(timeout * 2.0)  # More time for evidence gathering
            
            # Create customized step
            customized_step = EscalationStep(
                level=step.level,
                method_name=step.method_name,
                timeout_minutes=timeout,
                required_confidence=required_confidence,
                success_criteria=step.success_criteria.copy(),
                fallback_level=step.fallback_level,
                metadata={
                    **step.metadata,
                    "customized_for": conflict_analysis.conflict_id,
                    "original_timeout": step.timeout_minutes,
                    "original_confidence": step.required_confidence
                }
            )
            
            customized_steps.append(customized_step)
        
        return customized_steps

    # Escalation method implementations
    def _automated_resolution(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                            context: Dict[str, Any], step: EscalationStep) -> ArbitrationDecision:
        """Attempt automated resolution using basic algorithms"""
        logger.info("Attempting automated resolution")
        
        # Simple majority vote with confidence weighting
        vote_counts = {}
        confidence_weights = {}
        
        for vote in votes:
            vote_key = str(vote.vote)
            vote_counts[vote_key] = vote_counts.get(vote_key, 0) + 1
            
            conf_score = self._confidence_to_score(vote.confidence)
            confidence_weights[vote_key] = confidence_weights.get(vote_key, 0) + conf_score
        
        # Find decision with highest confidence-weighted support
        best_decision = max(confidence_weights.keys(), key=lambda k: confidence_weights[k])
        confidence = confidence_weights[best_decision] / len(votes)
        
        return ArbitrationDecision(
            decision_id=f"auto-{int(time.time())}",
            final_decision=self._parse_vote_value(best_decision),
            confidence_score=confidence,
            arbitration_type=ArbitrationType.WEIGHTED_RESOLUTION,
            reasoning="Automated resolution using confidence-weighted majority",
            supporting_evidence={
                "vote_counts": vote_counts,
                "confidence_weights": confidence_weights,
                "method": "confidence_weighted_majority"
            },
            dissenting_opinions=[],
            resolution_method="automated"
        )

    def _weighted_consensus(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                          context: Dict[str, Any], step: EscalationStep) -> ArbitrationDecision:
        """Use weighted consensus from consensus architecture"""
        logger.info("Attempting weighted consensus resolution")
        
        weighted_config = self.consensus.voting_configs["weighted_voting"]
        consensus_result = self.consensus.calculate_consensus(votes, weighted_config, context)
        
        return ArbitrationDecision(
            decision_id=f"weighted-{int(time.time())}",
            final_decision=consensus_result.decision,
            confidence_score=consensus_result.confidence_score,
            arbitration_type=ArbitrationType.WEIGHTED_RESOLUTION,
            reasoning="Weighted consensus using agent expertise weights",
            supporting_evidence={
                "agreement_level": consensus_result.agreement_level,
                "weights_used": weighted_config.weights,
                "mechanism": consensus_result.mechanism_used.value
            },
            dissenting_opinions=[],
            resolution_method="weighted_consensus"
        )

    def _expert_panel(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                     context: Dict[str, Any], step: EscalationStep) -> ArbitrationDecision:
        """Resolve using domain expert panel"""
        logger.info("Attempting expert panel resolution")
        
        domain = context.get("domain", "general")
        
        # Identify expert votes
        expert_votes = []
        for vote in votes:
            expertise = self.consensus.agent_expertise.get(vote.agent_id, {}).get(domain, 0.5)
            if expertise > 0.7:  # Expert threshold
                expert_votes.append((vote, expertise))
        
        if not expert_votes:
            raise ValueError(f"No experts available for domain: {domain}")
        
        # Calculate expert-weighted decision
        total_expertise = sum(expertise for _, expertise in expert_votes)
        expert_decisions = {}
        
        for vote, expertise in expert_votes:
            decision_key = str(vote.vote)
            weight = expertise / total_expertise
            expert_decisions[decision_key] = expert_decisions.get(decision_key, 0) + weight
        
        best_decision = max(expert_decisions.keys(), key=lambda k: expert_decisions[k])
        confidence = expert_decisions[best_decision]
        
        return ArbitrationDecision(
            decision_id=f"expert-{int(time.time())}",
            final_decision=self._parse_vote_value(best_decision),
            confidence_score=confidence,
            arbitration_type=ArbitrationType.EXPERT_PANEL,
            reasoning=f"Expert panel consensus from {len(expert_votes)} domain experts",
            supporting_evidence={
                "expert_count": len(expert_votes),
                "domain": domain,
                "expert_decisions": expert_decisions,
                "total_expertise": total_expertise
            },
            dissenting_opinions=[],
            resolution_method="expert_panel"
        )

    def _arbitrator_agent(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                         context: Dict[str, Any], step: EscalationStep) -> ArbitrationDecision:
        """Simulate specialized arbitrator agent (would spawn actual agent in full implementation)"""
        logger.info("Attempting arbitrator agent resolution")
        
        # Advanced decision logic considering multiple factors
        decision_factors = {}
        
        # Factor 1: Evidence quality
        evidence_weighted = {}
        total_evidence = sum(vote.evidence_quality for vote in votes)
        
        for vote in votes:
            decision_key = str(vote.vote)
            if total_evidence > 0:
                evidence_weight = vote.evidence_quality / total_evidence
                evidence_weighted[decision_key] = evidence_weighted.get(decision_key, 0) + evidence_weight
        
        # Factor 2: Reasoning depth (length as proxy)
        reasoning_weighted = {}
        total_reasoning_length = sum(len(vote.reasoning) for vote in votes)
        
        for vote in votes:
            decision_key = str(vote.vote)
            if total_reasoning_length > 0:
                reasoning_weight = len(vote.reasoning) / total_reasoning_length
                reasoning_weighted[decision_key] = reasoning_weighted.get(decision_key, 0) + reasoning_weight
        
        # Factor 3: Agent expertise
        expertise_weighted = {}
        domain = context.get("domain", "general")
        total_expertise = sum(self.consensus.agent_expertise.get(vote.agent_id, {}).get(domain, 0.5) for vote in votes)
        
        for vote in votes:
            decision_key = str(vote.vote)
            expertise = self.consensus.agent_expertise.get(vote.agent_id, {}).get(domain, 0.5)
            if total_expertise > 0:
                expertise_weight = expertise / total_expertise
                expertise_weighted[decision_key] = expertise_weighted.get(decision_key, 0) + expertise_weight
        
        # Combine factors with weights
        combined_scores = {}
        decision_keys = set(evidence_weighted.keys()) | set(reasoning_weighted.keys()) | set(expertise_weighted.keys())
        
        for key in decision_keys:
            evidence_score = evidence_weighted.get(key, 0) * 0.4
            reasoning_score = reasoning_weighted.get(key, 0) * 0.3
            expertise_score = expertise_weighted.get(key, 0) * 0.3
            combined_scores[key] = evidence_score + reasoning_score + expertise_score
        
        best_decision = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        confidence = combined_scores[best_decision]
        
        return ArbitrationDecision(
            decision_id=f"arbitrator-{int(time.time())}",
            final_decision=self._parse_vote_value(best_decision),
            confidence_score=confidence,
            arbitration_type=ArbitrationType.ARBITRATOR_AGENT,
            reasoning="Arbitrator agent multi-factor analysis",
            supporting_evidence={
                "evidence_weights": evidence_weighted,
                "reasoning_weights": reasoning_weighted,
                "expertise_weights": expertise_weighted,
                "combined_scores": combined_scores,
                "factors": ["evidence_quality", "reasoning_depth", "agent_expertise"]
            },
            dissenting_opinions=[],
            resolution_method="arbitrator_agent"
        )

    def _evidence_review(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                        context: Dict[str, Any], step: EscalationStep) -> ArbitrationDecision:
        """Focus on evidence quality for resolution"""
        logger.info("Attempting evidence-based resolution")
        
        # Filter votes by evidence quality threshold
        evidence_threshold = step.success_criteria.get("evidence_threshold", 0.7)
        high_quality_votes = [vote for vote in votes if vote.evidence_quality >= evidence_threshold]
        
        if not high_quality_votes:
            # Lower threshold if no votes meet criteria
            evidence_threshold = 0.5
            high_quality_votes = [vote for vote in votes if vote.evidence_quality >= evidence_threshold]
        
        if not high_quality_votes:
            raise ValueError("No votes meet minimum evidence quality standards")
        
        # Use consensus on high-quality votes only
        config = self.consensus.select_voting_mechanism(context)
        consensus_result = self.consensus.calculate_consensus(high_quality_votes, config, context)
        
        return ArbitrationDecision(
            decision_id=f"evidence-{int(time.time())}",
            final_decision=consensus_result.decision,
            confidence_score=consensus_result.confidence_score,
            arbitration_type=ArbitrationType.EVIDENCE_BASED,
            reasoning=f"Evidence-based resolution using {len(high_quality_votes)} high-quality votes",
            supporting_evidence={
                "evidence_threshold": evidence_threshold,
                "high_quality_votes": len(high_quality_votes),
                "total_votes": len(votes),
                "agreement_level": consensus_result.agreement_level
            },
            dissenting_opinions=[],
            resolution_method="evidence_review"
        )

    def _human_intervention(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                          context: Dict[str, Any], step: EscalationStep) -> ArbitrationDecision:
        """Escalate to human decision maker"""
        logger.info("Escalating to human intervention")
        
        # Create comprehensive briefing for human
        briefing = {
            "conflict_id": conflict_analysis.conflict_id,
            "escalation_timestamp": datetime.now().isoformat(),
            "context": context,
            "conflict_analysis": {
                "severity": conflict_analysis.severity.value,
                "patterns": [p.value for p in conflict_analysis.patterns_detected],
                "root_causes": conflict_analysis.root_causes,
                "recommendations": conflict_analysis.resolution_recommendations
            },
            "votes": [
                {
                    "agent": vote.agent_id,
                    "vote": vote.vote,
                    "confidence": vote.confidence.value,
                    "reasoning": vote.reasoning,
                    "evidence_quality": vote.evidence_quality
                }
                for vote in votes
            ],
            "required_action": "Human decision required - automated arbitration insufficient"
        }
        
        # Store briefing for human review
        briefing_file = self.knowledge_base_path / "arbitration" / "human_escalations" / f"{conflict_analysis.conflict_id}.json"
        briefing_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(briefing_file, 'w') as f:
            json.dump(briefing, f, indent=2)
        
        return ArbitrationDecision(
            decision_id=f"human-{int(time.time())}",
            final_decision="PENDING_HUMAN_REVIEW",
            confidence_score=0.0,
            arbitration_type=ArbitrationType.HUMAN_ESCALATION,
            reasoning="Conflict escalated to human decision maker",
            supporting_evidence=briefing,
            dissenting_opinions=[],
            resolution_method="human_intervention"
        )

    def _executive_decision(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                          context: Dict[str, Any], step: EscalationStep) -> ArbitrationDecision:
        """Final executive decision (ultimate fallback)"""
        logger.info("Making executive decision - final escalation level")
        
        # Executive decision defaults to safe/conservative choice
        default_decision = context.get("safe_default", False)
        
        return ArbitrationDecision(
            decision_id=f"executive-{int(time.time())}",
            final_decision=default_decision,
            confidence_score=1.0,  # Executive decision is final
            arbitration_type=ArbitrationType.HUMAN_ESCALATION,
            reasoning="Executive decision - final escalation level reached",
            supporting_evidence={
                "escalation_exhausted": True,
                "safe_default_used": True,
                "context": context
            },
            dissenting_opinions=[],
            resolution_method="executive_decision"
        )

    # Helper methods
    def _initialize_strategies(self) -> Dict[EscalationStrategy, List[EscalationStep]]:
        """Initialize escalation strategy definitions"""
        return {
            EscalationStrategy.CONSERVATIVE: [
                EscalationStep(
                    level=EscalationLevel.WEIGHTED_CONSENSUS,
                    method_name="weighted_consensus",
                    timeout_minutes=10,
                    required_confidence=0.8,
                    success_criteria={"min_agreement": 0.8},
                    fallback_level=EscalationLevel.EXPERT_PANEL
                ),
                EscalationStep(
                    level=EscalationLevel.EXPERT_PANEL,
                    method_name="expert_panel",
                    timeout_minutes=30,
                    required_confidence=0.85,
                    success_criteria={"expert_consensus": 0.8},
                    fallback_level=EscalationLevel.HUMAN_INTERVENTION
                ),
                EscalationStep(
                    level=EscalationLevel.HUMAN_INTERVENTION,
                    method_name="human_intervention",
                    timeout_minutes=1440,  # 24 hours
                    required_confidence=0.95,
                    success_criteria={},
                    fallback_level=None
                )
            ],
            
            EscalationStrategy.BALANCED: [
                EscalationStep(
                    level=EscalationLevel.AUTOMATED_RESOLUTION,
                    method_name="automated_resolution",
                    timeout_minutes=5,
                    required_confidence=0.6,
                    success_criteria={},
                    fallback_level=EscalationLevel.WEIGHTED_CONSENSUS
                ),
                EscalationStep(
                    level=EscalationLevel.WEIGHTED_CONSENSUS,
                    method_name="weighted_consensus",
                    timeout_minutes=10,
                    required_confidence=0.7,
                    success_criteria={},
                    fallback_level=EscalationLevel.ARBITRATOR_AGENT
                ),
                EscalationStep(
                    level=EscalationLevel.ARBITRATOR_AGENT,
                    method_name="arbitrator_agent",
                    timeout_minutes=45,
                    required_confidence=0.75,
                    success_criteria={},
                    fallback_level=EscalationLevel.HUMAN_INTERVENTION
                ),
                EscalationStep(
                    level=EscalationLevel.HUMAN_INTERVENTION,
                    method_name="human_intervention",
                    timeout_minutes=1440,
                    required_confidence=0.9,
                    success_criteria={},
                    fallback_level=None
                )
            ],
            
            EscalationStrategy.AGGRESSIVE: [
                EscalationStep(
                    level=EscalationLevel.AUTOMATED_RESOLUTION,
                    method_name="automated_resolution",
                    timeout_minutes=10,
                    required_confidence=0.5,
                    success_criteria={},
                    fallback_level=EscalationLevel.WEIGHTED_CONSENSUS
                ),
                EscalationStep(
                    level=EscalationLevel.WEIGHTED_CONSENSUS,
                    method_name="weighted_consensus",
                    timeout_minutes=15,
                    required_confidence=0.6,
                    success_criteria={},
                    fallback_level=EscalationLevel.ARBITRATOR_AGENT
                ),
                EscalationStep(
                    level=EscalationLevel.ARBITRATOR_AGENT,
                    method_name="arbitrator_agent",
                    timeout_minutes=60,
                    required_confidence=0.65,
                    success_criteria={},
                    fallback_level=EscalationLevel.HUMAN_INTERVENTION
                ),
                EscalationStep(
                    level=EscalationLevel.HUMAN_INTERVENTION,
                    method_name="human_intervention",
                    timeout_minutes=480,  # 8 hours
                    required_confidence=0.85,
                    success_criteria={},
                    fallback_level=None
                )
            ],
            
            EscalationStrategy.SECURITY_FOCUSED: [
                EscalationStep(
                    level=EscalationLevel.EXPERT_PANEL,
                    method_name="expert_panel",
                    timeout_minutes=20,
                    required_confidence=0.9,
                    success_criteria={"security_expert_required": True},
                    fallback_level=EscalationLevel.EVIDENCE_REVIEW
                ),
                EscalationStep(
                    level=EscalationLevel.EVIDENCE_REVIEW,
                    method_name="evidence_review",
                    timeout_minutes=30,
                    required_confidence=0.9,
                    success_criteria={"evidence_threshold": 0.8},
                    fallback_level=EscalationLevel.HUMAN_INTERVENTION
                ),
                EscalationStep(
                    level=EscalationLevel.HUMAN_INTERVENTION,
                    method_name="human_intervention",
                    timeout_minutes=720,  # 12 hours
                    required_confidence=0.95,
                    success_criteria={},
                    fallback_level=None
                )
            ],
            
            EscalationStrategy.EFFICIENCY_FOCUSED: [
                EscalationStep(
                    level=EscalationLevel.AUTOMATED_RESOLUTION,
                    method_name="automated_resolution",
                    timeout_minutes=3,
                    required_confidence=0.55,
                    success_criteria={},
                    fallback_level=EscalationLevel.WEIGHTED_CONSENSUS
                ),
                EscalationStep(
                    level=EscalationLevel.WEIGHTED_CONSENSUS,
                    method_name="weighted_consensus",
                    timeout_minutes=8,
                    required_confidence=0.65,
                    success_criteria={},
                    fallback_level=EscalationLevel.ARBITRATOR_AGENT
                ),
                EscalationStep(
                    level=EscalationLevel.ARBITRATOR_AGENT,
                    method_name="arbitrator_agent",
                    timeout_minutes=30,
                    required_confidence=0.7,
                    success_criteria={},
                    fallback_level=EscalationLevel.HUMAN_INTERVENTION
                ),
                EscalationStep(
                    level=EscalationLevel.HUMAN_INTERVENTION,
                    method_name="human_intervention",
                    timeout_minutes=240,  # 4 hours
                    required_confidence=0.85,
                    success_criteria={},
                    fallback_level=None
                )
            ]
        }

    def _evaluate_step_success(self, decision: ArbitrationDecision, step: EscalationStep,
                             conflict_analysis: ConflictAnalysis, context: Dict[str, Any]) -> bool:
        """Evaluate if an escalation step was successful"""
        if not decision:
            return False
        
        # Check confidence threshold
        if decision.confidence_score < step.required_confidence:
            return False
        
        # Check success criteria
        if step.success_criteria:
            # Custom success criteria evaluation would go here
            pass
        
        # Special handling for human intervention
        if step.level == EscalationLevel.HUMAN_INTERVENTION:
            return decision.final_decision != "PENDING_HUMAN_REVIEW"
        
        return True

    def _determine_escalation_trigger(self, decision: ArbitrationDecision, step: EscalationStep,
                                    success: bool) -> EscalationTrigger:
        """Determine what triggered the need for escalation"""
        if not success:
            if decision and decision.confidence_score < step.required_confidence:
                return EscalationTrigger.CONFIDENCE_TOO_LOW
            else:
                return EscalationTrigger.METHOD_FAILED
        
        return EscalationTrigger.SEVERITY_THRESHOLD

    def _is_final_resolution(self, decision: ArbitrationDecision, step: EscalationStep,
                           context: Dict[str, Any]) -> bool:
        """Check if this decision is final and satisfactory"""
        # Human intervention decisions are always final (when not pending)
        if step.level == EscalationLevel.HUMAN_INTERVENTION:
            return decision.final_decision != "PENDING_HUMAN_REVIEW"
        
        # Executive decisions are always final
        if step.level == EscalationLevel.EXECUTIVE_DECISION:
            return True
        
        # For other levels, check if confidence meets context requirements
        required_confidence = 0.8 if context.get("security_critical", False) else 0.7
        return decision.confidence_score >= required_confidence

    def _select_best_decision(self, results: List[EscalationResult]) -> Optional[ArbitrationDecision]:
        """Select the best decision from escalation results"""
        successful_results = [r for r in results if r.success and r.decision]
        
        if not successful_results:
            return None
        
        # Return decision with highest confidence
        best_result = max(successful_results, key=lambda r: r.decision.confidence_score)
        return best_result.decision

    def _create_exhaustion_fallback(self, votes: List[AgentVote], context: Dict[str, Any],
                                  path_id: str) -> ArbitrationDecision:
        """Create fallback decision when escalation path is exhausted"""
        # Use safe default or simple majority as final fallback
        safe_default = context.get("safe_default", False)
        
        return ArbitrationDecision(
            decision_id=f"exhausted-{int(time.time())}",
            final_decision=safe_default,
            confidence_score=0.5,
            arbitration_type=ArbitrationType.WEIGHTED_RESOLUTION,
            reasoning="Escalation path exhausted - using safe default",
            supporting_evidence={
                "path_id": path_id,
                "safe_default": safe_default,
                "exhaustion_reason": "All escalation methods attempted"
            },
            dissenting_opinions=[],
            resolution_method="exhaustion_fallback"
        )

    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numerical score"""
        confidence_map = {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 1.0
        }
        return confidence_map.get(confidence, 0.5)

    def _parse_vote_value(self, vote_string: str):
        """Parse vote value from string representation"""
        try:
            if vote_string.lower() == 'true':
                return True
            elif vote_string.lower() == 'false':
                return False
            elif vote_string.replace('.', '').replace('-', '').isdigit():
                return float(vote_string)
            else:
                return vote_string
        except:
            return vote_string

    def _update_escalation_metrics(self, level: EscalationLevel, success: bool, execution_time: float):
        """Update escalation metrics"""
        if success:
            self.metrics["successful_resolutions_by_level"][level.value] += 1
        
        # Update average execution time
        current_avg = self.metrics["average_resolution_time_by_level"][level.value]
        total_attempts = self.metrics["successful_resolutions_by_level"][level.value]
        
        if total_attempts > 0:
            self.metrics["average_resolution_time_by_level"][level.value] = (
                (current_avg * (total_attempts - 1) + execution_time) / total_attempts
            )

    def get_escalation_metrics(self) -> Dict[str, Any]:
        """Get escalation engine performance metrics"""
        return {
            **self.metrics,
            "strategy_definitions": {
                strategy.value: len(steps) 
                for strategy, steps in self.strategies.items()
            },
            "configuration": self.config
        }


def main():
    """Demonstration of escalation engine functionality"""
    print("=== RIF Escalation Engine Demo ===\n")
    
    # Initialize system
    from consensus_architecture import ConsensusArchitecture
    from .conflict_detector import ConflictDetector
    
    consensus = ConsensusArchitecture()
    detector = ConflictDetector(consensus.agent_expertise)
    engine = EscalationEngine(consensus)
    
    # Example escalation scenario
    print("Example: Security-Critical Escalation")
    votes = [
        consensus.create_vote("rif-implementer", True, ConfidenceLevel.MEDIUM, "Implementation complete", 0.6),
        consensus.create_vote("rif-security", False, ConfidenceLevel.VERY_HIGH, "Security vulnerability found", 0.95),
        consensus.create_vote("rif-validator", True, ConfidenceLevel.LOW, "Basic tests pass", 0.4)
    ]
    
    context = {"domain": "security", "security_critical": True, "risk_level": "high"}
    
    # Analyze conflicts
    conflict_analysis = detector.analyze_conflicts(votes, context)
    print(f"Conflict Analysis: {conflict_analysis.severity.value} severity")
    print(f"Patterns: {[p.value for p in conflict_analysis.patterns_detected]}")
    
    # Create escalation path
    path = engine.create_escalation_path(conflict_analysis, context)
    print(f"\nEscalation Path: {path.strategy.value} strategy with {len(path.steps)} steps")
    for i, step in enumerate(path.steps):
        print(f"  {i+1}. {step.level.value} (timeout: {step.timeout_minutes}min, confidence: {step.required_confidence})")
    
    # Execute escalation path
    final_decision, results = engine.execute_escalation_path(path, votes, conflict_analysis, context)
    
    print(f"\nFinal Decision: {final_decision.final_decision}")
    print(f"Confidence: {final_decision.confidence_score:.2f}")
    print(f"Resolution Method: {final_decision.resolution_method}")
    print(f"Arbitration Type: {final_decision.arbitration_type.value}")
    
    print(f"\nEscalation Results ({len(results)} steps executed):")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.step.level.value}: {'SUCCESS' if result.success else 'FAILED'} "
              f"({result.execution_time_seconds:.2f}s)")
    
    # Show metrics
    print("\n=== Escalation Engine Metrics ===")
    metrics = engine.get_escalation_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict) and len(value) <= 10:
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif not isinstance(value, dict):
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()