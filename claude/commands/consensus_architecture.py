#!/usr/bin/env python3
"""
RIF Consensus Architecture - Issue #58
Comprehensive consensus system with voting mechanisms, threshold configurations, and arbitration rules.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VotingMechanism(Enum):
    """Supported voting mechanism types"""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_VOTING = "weighted_voting" 
    UNANIMOUS = "unanimous"
    VETO_POWER = "veto_power"
    SUPERMAJORITY = "supermajority"

class RiskLevel(Enum):
    """Risk assessment levels for decision making"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConfidenceLevel(Enum):
    """Agent confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class AgentVote:
    """Individual agent vote with metadata"""
    agent_id: str
    vote: Union[bool, float, str]
    confidence: ConfidenceLevel
    reasoning: str
    timestamp: datetime
    evidence_quality: float = 0.0
    expertise_score: float = 1.0

@dataclass
class VotingConfig:
    """Configuration for voting mechanisms"""
    mechanism: VotingMechanism
    threshold: float
    weights: Dict[str, float]
    use_cases: List[str]
    timeout_minutes: int = 30
    minimum_votes: int = 1
    allow_abstentions: bool = True

@dataclass
class ArbitrationRule:
    """Rules for handling voting conflicts and disagreements"""
    disagreement_threshold: float
    escalation_path: List[str]
    timeout_action: str
    require_unanimous_critical: bool = True
    max_escalation_levels: int = 3

@dataclass
class ConsensusResult:
    """Result of consensus calculation"""
    decision: Union[bool, float, str]
    confidence_score: float
    vote_count: int
    agreement_level: float
    mechanism_used: VotingMechanism
    arbitration_triggered: bool = False
    evidence_summary: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ConsensusArchitecture:
    """
    Core consensus architecture implementing multiple voting mechanisms
    with configurable thresholds and arbitration rules.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize consensus architecture with configuration"""
        self.config_path = config_path or "/Users/cal/DEV/RIF/config/consensus-architecture.yaml"
        self.voting_configs = self._load_voting_configs()
        self.arbitration_rules = self._load_arbitration_rules()
        self.agent_expertise = self._load_agent_expertise()
        self.active_votes: Dict[str, List[AgentVote]] = {}
        
        # Performance tracking
        self.consensus_metrics = {
            "total_decisions": 0,
            "arbitration_triggered": 0,
            "avg_consensus_time": 0.0,
            "mechanism_usage": {mechanism.value: 0 for mechanism in VotingMechanism}
        }

    def _load_voting_configs(self) -> Dict[str, VotingConfig]:
        """Load voting mechanism configurations"""
        default_configs = {
            "simple_majority": VotingConfig(
                mechanism=VotingMechanism.SIMPLE_MAJORITY,
                threshold=0.5,
                weights={},
                use_cases=["low_risk_decisions", "routine_approvals"],
                timeout_minutes=15,
                minimum_votes=2
            ),
            "weighted_voting": VotingConfig(
                mechanism=VotingMechanism.WEIGHTED_VOTING,
                threshold=0.7,
                weights={
                    "rif-validator": 1.5,
                    "rif-security": 2.0,
                    "rif-implementer": 1.0,
                    "rif-architect": 1.3,
                    "rif-analyst": 1.1
                },
                use_cases=["medium_risk_decisions", "architecture_changes"],
                timeout_minutes=30,
                minimum_votes=3
            ),
            "unanimous": VotingConfig(
                mechanism=VotingMechanism.UNANIMOUS,
                threshold=1.0,
                weights={},
                use_cases=["security_critical", "breaking_changes"],
                timeout_minutes=45,
                minimum_votes=2
            ),
            "veto_power": VotingConfig(
                mechanism=VotingMechanism.VETO_POWER,
                threshold=0.6,
                weights={
                    "rif-security": 10.0,  # Veto power
                    "rif-validator": 5.0   # Strong influence
                },
                use_cases=["compliance", "security", "quality_gates"],
                timeout_minutes=20,
                minimum_votes=2
            ),
            "supermajority": VotingConfig(
                mechanism=VotingMechanism.SUPERMAJORITY,
                threshold=0.8,
                weights={},
                use_cases=["high_impact_decisions", "policy_changes"],
                timeout_minutes=60,
                minimum_votes=4
            )
        }
        
        return default_configs

    def _load_arbitration_rules(self) -> Dict[str, ArbitrationRule]:
        """Load arbitration rules for conflict resolution"""
        default_rules = {
            "default": ArbitrationRule(
                disagreement_threshold=0.3,
                escalation_path=[
                    "try_weighted_voting",
                    "spawn_arbitrator",
                    "escalate_to_human"
                ],
                timeout_action="escalate_to_human",
                require_unanimous_critical=True,
                max_escalation_levels=3
            ),
            "security": ArbitrationRule(
                disagreement_threshold=0.1,
                escalation_path=[
                    "security_expert_review",
                    "unanimous_required",
                    "escalate_to_human"
                ],
                timeout_action="reject_by_default",
                require_unanimous_critical=True,
                max_escalation_levels=2
            ),
            "performance": ArbitrationRule(
                disagreement_threshold=0.4,
                escalation_path=[
                    "performance_benchmark",
                    "expert_review",
                    "escalate_to_human"
                ],
                timeout_action="accept_with_monitoring",
                require_unanimous_critical=False,
                max_escalation_levels=3
            )
        }
        
        return default_rules

    def _load_agent_expertise(self) -> Dict[str, Dict[str, float]]:
        """Load agent expertise scores for different domains"""
        return {
            "rif-security": {
                "security": 0.95,
                "compliance": 0.90,
                "vulnerability": 0.95,
                "general": 0.70
            },
            "rif-validator": {
                "testing": 0.90,
                "quality": 0.95,
                "validation": 0.95,
                "general": 0.75
            },
            "rif-implementer": {
                "coding": 0.85,
                "integration": 0.80,
                "debugging": 0.85,
                "general": 0.70
            },
            "rif-architect": {
                "design": 0.95,
                "scalability": 0.90,
                "patterns": 0.90,
                "general": 0.80
            },
            "rif-analyst": {
                "requirements": 0.90,
                "analysis": 0.95,
                "planning": 0.85,
                "general": 0.75
            }
        }

    def select_voting_mechanism(self, context: Dict[str, Any]) -> VotingConfig:
        """
        Select appropriate voting mechanism based on context and risk level
        
        Args:
            context: Decision context including risk level, domain, etc.
            
        Returns:
            VotingConfig: Selected voting configuration
        """
        risk_level = context.get("risk_level", RiskLevel.MEDIUM.value)
        domain = context.get("domain", "general")
        security_critical = context.get("security_critical", False)
        
        # Security-critical decisions always use veto power or unanimous
        if security_critical or domain == "security":
            return self.voting_configs["veto_power"] if risk_level != "critical" else self.voting_configs["unanimous"]
        
        # Risk-based selection
        if risk_level == RiskLevel.LOW.value:
            return self.voting_configs["simple_majority"]
        elif risk_level == RiskLevel.MEDIUM.value:
            return self.voting_configs["weighted_voting"]
        elif risk_level == RiskLevel.HIGH.value:
            return self.voting_configs["supermajority"]
        else:  # CRITICAL
            return self.voting_configs["unanimous"]

    def calculate_consensus(self, votes: List[AgentVote], config: VotingConfig, context: Dict[str, Any]) -> ConsensusResult:
        """
        Calculate consensus based on votes and configuration
        
        Args:
            votes: List of agent votes
            config: Voting configuration to use
            context: Decision context
            
        Returns:
            ConsensusResult: Result of consensus calculation
        """
        start_time = time.time()
        
        if not votes:
            return ConsensusResult(
                decision=False,
                confidence_score=0.0,
                vote_count=0,
                agreement_level=0.0,
                mechanism_used=config.mechanism
            )

        # Calculate weighted consensus based on mechanism
        if config.mechanism == VotingMechanism.SIMPLE_MAJORITY:
            result = self._calculate_simple_majority(votes, config)
        elif config.mechanism == VotingMechanism.WEIGHTED_VOTING:
            result = self._calculate_weighted_voting(votes, config)
        elif config.mechanism == VotingMechanism.UNANIMOUS:
            result = self._calculate_unanimous(votes, config)
        elif config.mechanism == VotingMechanism.VETO_POWER:
            result = self._calculate_veto_power(votes, config)
        elif config.mechanism == VotingMechanism.SUPERMAJORITY:
            result = self._calculate_supermajority(votes, config)
        else:
            raise ValueError(f"Unsupported voting mechanism: {config.mechanism}")

        # Check if arbitration is needed
        if result.agreement_level < (1.0 - self._get_arbitration_rule(context).disagreement_threshold):
            result.arbitration_triggered = True
            result = self._handle_arbitration(result, votes, config, context)

        # Update metrics
        self.consensus_metrics["total_decisions"] += 1
        self.consensus_metrics["mechanism_usage"][config.mechanism.value] += 1
        if result.arbitration_triggered:
            self.consensus_metrics["arbitration_triggered"] += 1
        
        # Update average consensus time
        consensus_time = time.time() - start_time
        total_decisions = self.consensus_metrics["total_decisions"]
        current_avg = self.consensus_metrics["avg_consensus_time"]
        self.consensus_metrics["avg_consensus_time"] = ((current_avg * (total_decisions - 1)) + consensus_time) / total_decisions

        return result

    def _calculate_simple_majority(self, votes: List[AgentVote], config: VotingConfig) -> ConsensusResult:
        """Calculate simple majority consensus"""
        positive_votes = sum(1 for vote in votes if vote.vote is True)
        total_votes = len(votes)
        
        agreement_level = positive_votes / total_votes if total_votes > 0 else 0.0
        decision = agreement_level >= config.threshold
        
        # Calculate confidence based on vote confidence and evidence quality
        avg_confidence = sum(self._confidence_to_score(vote.confidence) for vote in votes) / len(votes)
        avg_evidence = sum(vote.evidence_quality for vote in votes) / len(votes)
        confidence_score = (avg_confidence * 0.7 + avg_evidence * 0.3) * agreement_level

        return ConsensusResult(
            decision=decision,
            confidence_score=confidence_score,
            vote_count=total_votes,
            agreement_level=agreement_level,
            mechanism_used=config.mechanism
        )

    def _calculate_weighted_voting(self, votes: List[AgentVote], config: VotingConfig) -> ConsensusResult:
        """Calculate weighted voting consensus"""
        total_weight = 0.0
        weighted_positive = 0.0
        
        for vote in votes:
            weight = config.weights.get(vote.agent_id, 1.0)
            total_weight += weight
            if vote.vote is True:
                weighted_positive += weight
        
        if total_weight == 0:
            return ConsensusResult(
                decision=False,
                confidence_score=0.0,
                vote_count=len(votes),
                agreement_level=0.0,
                mechanism_used=config.mechanism
            )
        
        agreement_level = weighted_positive / total_weight
        decision = agreement_level >= config.threshold
        
        # Weighted confidence calculation
        weighted_confidence = sum(
            config.weights.get(vote.agent_id, 1.0) * 
            self._confidence_to_score(vote.confidence) * 
            vote.evidence_quality
            for vote in votes
        ) / total_weight if total_weight > 0 else 0.0
        
        return ConsensusResult(
            decision=decision,
            confidence_score=weighted_confidence * agreement_level,
            vote_count=len(votes),
            agreement_level=agreement_level,
            mechanism_used=config.mechanism
        )

    def _calculate_unanimous(self, votes: List[AgentVote], config: VotingConfig) -> ConsensusResult:
        """Calculate unanimous consensus - all must agree"""
        all_positive = all(vote.vote is True for vote in votes)
        agreement_level = 1.0 if all_positive else 0.0
        
        avg_confidence = sum(self._confidence_to_score(vote.confidence) for vote in votes) / len(votes) if votes else 0.0
        avg_evidence = sum(vote.evidence_quality for vote in votes) / len(votes) if votes else 0.0
        
        return ConsensusResult(
            decision=all_positive,
            confidence_score=(avg_confidence * 0.7 + avg_evidence * 0.3) if all_positive else 0.0,
            vote_count=len(votes),
            agreement_level=agreement_level,
            mechanism_used=config.mechanism
        )

    def _calculate_veto_power(self, votes: List[AgentVote], config: VotingConfig) -> ConsensusResult:
        """Calculate veto power consensus - certain agents can veto"""
        # Check for vetoes (negative votes from high-weight agents)
        for vote in votes:
            weight = config.weights.get(vote.agent_id, 1.0)
            if weight >= 5.0 and vote.vote is False:  # Veto power threshold
                return ConsensusResult(
                    decision=False,
                    confidence_score=0.0,
                    vote_count=len(votes),
                    agreement_level=0.0,
                    mechanism_used=config.mechanism,
                    evidence_summary={"veto_by": vote.agent_id, "reason": vote.reasoning}
                )
        
        # If no vetoes, calculate weighted consensus
        return self._calculate_weighted_voting(votes, config)

    def _calculate_supermajority(self, votes: List[AgentVote], config: VotingConfig) -> ConsensusResult:
        """Calculate supermajority consensus (typically >2/3)"""
        positive_votes = sum(1 for vote in votes if vote.vote is True)
        total_votes = len(votes)
        
        agreement_level = positive_votes / total_votes if total_votes > 0 else 0.0
        decision = agreement_level >= config.threshold  # 0.8 by default
        
        avg_confidence = sum(self._confidence_to_score(vote.confidence) for vote in votes) / len(votes) if votes else 0.0
        avg_evidence = sum(vote.evidence_quality for vote in votes) / len(votes) if votes else 0.0
        confidence_score = (avg_confidence * 0.7 + avg_evidence * 0.3) * agreement_level

        return ConsensusResult(
            decision=decision,
            confidence_score=confidence_score,
            vote_count=total_votes,
            agreement_level=agreement_level,
            mechanism_used=config.mechanism
        )

    def _handle_arbitration(self, result: ConsensusResult, votes: List[AgentVote], 
                           config: VotingConfig, context: Dict[str, Any]) -> ConsensusResult:
        """Handle arbitration when consensus is not clear"""
        arbitration_rule = self._get_arbitration_rule(context)
        
        logger.info(f"Arbitration triggered for {config.mechanism.value} vote")
        logger.info(f"Agreement level: {result.agreement_level:.2f}, threshold: {1.0 - arbitration_rule.disagreement_threshold:.2f}")
        
        # Try escalation path
        for escalation_step in arbitration_rule.escalation_path:
            if escalation_step == "try_weighted_voting" and config.mechanism != VotingMechanism.WEIGHTED_VOTING:
                # Try weighted voting as fallback
                weighted_config = self.voting_configs["weighted_voting"]
                fallback_result = self._calculate_weighted_voting(votes, weighted_config)
                if fallback_result.agreement_level >= (1.0 - arbitration_rule.disagreement_threshold):
                    fallback_result.arbitration_triggered = True
                    fallback_result.evidence_summary = {"escalation": "weighted_voting_fallback"}
                    return fallback_result
                    
            elif escalation_step == "spawn_arbitrator":
                # Would spawn arbitrator agent in real implementation
                result.evidence_summary = {"escalation": "arbitrator_required"}
                logger.info("Arbitrator agent spawn required")
                
            elif escalation_step == "escalate_to_human":
                result.evidence_summary = {"escalation": "human_review_required"}
                logger.info("Human review escalation required")
                break
        
        return result

    def _get_arbitration_rule(self, context: Dict[str, Any]) -> ArbitrationRule:
        """Get appropriate arbitration rule based on context"""
        domain = context.get("domain", "default")
        return self.arbitration_rules.get(domain, self.arbitration_rules["default"])

    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numerical score"""
        confidence_map = {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 1.0
        }
        return confidence_map.get(confidence, 0.5)

    def create_vote(self, agent_id: str, vote: Union[bool, float, str], 
                   confidence: ConfidenceLevel, reasoning: str,
                   evidence_quality: float = 0.0) -> AgentVote:
        """
        Create a standardized agent vote
        
        Args:
            agent_id: ID of the voting agent
            vote: The vote value (bool, float, or str)
            confidence: Agent's confidence level
            reasoning: Explanation for the vote
            evidence_quality: Quality score of supporting evidence
            
        Returns:
            AgentVote: Formatted vote object
        """
        expertise_score = self.agent_expertise.get(agent_id, {}).get("general", 1.0)
        
        return AgentVote(
            agent_id=agent_id,
            vote=vote,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            evidence_quality=max(0.0, min(1.0, evidence_quality)),
            expertise_score=expertise_score
        )

    def run_consensus_vote(self, decision_id: str, context: Dict[str, Any], 
                          timeout_minutes: Optional[int] = None) -> Tuple[ConsensusResult, List[AgentVote]]:
        """
        Run a complete consensus voting process
        
        Args:
            decision_id: Unique identifier for this decision
            context: Decision context and metadata
            timeout_minutes: Override default timeout
            
        Returns:
            Tuple[ConsensusResult, List[AgentVote]]: Consensus result and all votes
        """
        # Select appropriate voting mechanism
        voting_config = self.select_voting_mechanism(context)
        
        if timeout_minutes:
            voting_config.timeout_minutes = timeout_minutes
        
        logger.info(f"Starting consensus vote {decision_id} using {voting_config.mechanism.value}")
        
        # Initialize vote collection
        self.active_votes[decision_id] = []
        
        # In real implementation, this would notify agents and collect votes
        # For now, return empty result to demonstrate structure
        votes = self.active_votes.get(decision_id, [])
        
        if len(votes) < voting_config.minimum_votes:
            logger.warning(f"Insufficient votes for {decision_id}: {len(votes)} < {voting_config.minimum_votes}")
            return ConsensusResult(
                decision=False,
                confidence_score=0.0,
                vote_count=len(votes),
                agreement_level=0.0,
                mechanism_used=voting_config.mechanism,
                evidence_summary={"error": "insufficient_votes"}
            ), votes
        
        # Calculate consensus
        result = self.calculate_consensus(votes, voting_config, context)
        
        # Clean up
        if decision_id in self.active_votes:
            del self.active_votes[decision_id]
        
        return result, votes

    def add_vote_to_active_decision(self, decision_id: str, vote: AgentVote) -> bool:
        """
        Add a vote to an active decision process
        
        Args:
            decision_id: Decision identifier
            vote: Agent vote to add
            
        Returns:
            bool: Success status
        """
        if decision_id not in self.active_votes:
            logger.error(f"No active vote found for decision {decision_id}")
            return False
            
        # Check for duplicate votes from same agent
        for existing_vote in self.active_votes[decision_id]:
            if existing_vote.agent_id == vote.agent_id:
                logger.warning(f"Duplicate vote from {vote.agent_id} for {decision_id}, replacing")
                existing_vote.vote = vote.vote
                existing_vote.confidence = vote.confidence
                existing_vote.reasoning = vote.reasoning
                existing_vote.timestamp = vote.timestamp
                existing_vote.evidence_quality = vote.evidence_quality
                return True
        
        self.active_votes[decision_id].append(vote)
        logger.info(f"Added vote from {vote.agent_id} for decision {decision_id}")
        return True

    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get current consensus system metrics"""
        return {
            **self.consensus_metrics,
            "active_votes": len(self.active_votes),
            "voting_configs": len(self.voting_configs),
            "arbitration_rules": len(self.arbitration_rules)
        }

    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for persistence"""
        return {
            "voting_configs": {
                name: {
                    "mechanism": config.mechanism.value,
                    "threshold": config.threshold,
                    "weights": config.weights,
                    "use_cases": config.use_cases,
                    "timeout_minutes": config.timeout_minutes,
                    "minimum_votes": config.minimum_votes,
                    "allow_abstentions": config.allow_abstentions
                }
                for name, config in self.voting_configs.items()
            },
            "arbitration_rules": {
                name: {
                    "disagreement_threshold": rule.disagreement_threshold,
                    "escalation_path": rule.escalation_path,
                    "timeout_action": rule.timeout_action,
                    "require_unanimous_critical": rule.require_unanimous_critical,
                    "max_escalation_levels": rule.max_escalation_levels
                }
                for name, rule in self.arbitration_rules.items()
            },
            "agent_expertise": self.agent_expertise
        }


def main():
    """Demonstration and testing of consensus architecture"""
    consensus = ConsensusArchitecture()
    
    # Example 1: Simple majority vote
    print("\n=== Simple Majority Vote Example ===")
    votes = [
        consensus.create_vote("rif-implementer", True, ConfidenceLevel.HIGH, "Implementation looks good", 0.8),
        consensus.create_vote("rif-validator", True, ConfidenceLevel.MEDIUM, "Tests pass", 0.7),
        consensus.create_vote("rif-analyst", False, ConfidenceLevel.LOW, "Concerns about requirements", 0.3)
    ]
    
    config = consensus.voting_configs["simple_majority"]
    result = consensus.calculate_consensus(votes, config, {"risk_level": "low"})
    print(f"Decision: {result.decision}")
    print(f"Agreement: {result.agreement_level:.2f}")
    print(f"Confidence: {result.confidence_score:.2f}")
    
    # Example 2: Weighted voting
    print("\n=== Weighted Voting Example ===")
    config = consensus.voting_configs["weighted_voting"]
    result = consensus.calculate_consensus(votes, config, {"risk_level": "medium"})
    print(f"Decision: {result.decision}")
    print(f"Agreement: {result.agreement_level:.2f}")
    print(f"Confidence: {result.confidence_score:.2f}")
    
    # Example 3: Veto power demonstration
    print("\n=== Veto Power Example ===")
    security_veto = consensus.create_vote("rif-security", False, ConfidenceLevel.VERY_HIGH, "Security vulnerability detected", 0.95)
    votes_with_veto = votes + [security_veto]
    
    config = consensus.voting_configs["veto_power"]
    result = consensus.calculate_consensus(votes_with_veto, config, {"domain": "security"})
    print(f"Decision: {result.decision}")
    print(f"Agreement: {result.agreement_level:.2f}")
    print(f"Evidence: {result.evidence_summary}")
    
    # Print metrics
    print(f"\n=== Consensus Metrics ===")
    metrics = consensus.get_consensus_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()