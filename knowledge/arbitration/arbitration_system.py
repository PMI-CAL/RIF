#!/usr/bin/env python3
"""
RIF Arbitration System - Core Implementation
Multi-level conflict resolution system for agent consensus disagreements.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import hashlib
from pathlib import Path

# Import consensus and voting components
import sys
sys.path.insert(0, '/Users/cal/DEV/RIF/claude/commands')

from consensus_architecture import (
    ConsensusArchitecture, VotingMechanism, ConfidenceLevel, 
    AgentVote, VotingConfig, ConsensusResult
)
from voting_aggregator import (
    VotingAggregator, VoteConflict, ConflictType, AggregationReport
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbitrationStatus(Enum):
    """Status of arbitration process"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"
    HUMAN_REQUIRED = "human_required"

class ArbitrationType(Enum):
    """Types of arbitration approaches"""
    WEIGHTED_RESOLUTION = "weighted_resolution"
    EXPERT_PANEL = "expert_panel"
    ARBITRATOR_AGENT = "arbitrator_agent"
    HUMAN_ESCALATION = "human_escalation"
    EVIDENCE_BASED = "evidence_based"

@dataclass
class ArbitrationDecision:
    """Decision made through arbitration process"""
    decision_id: str
    final_decision: Union[bool, float, str]
    confidence_score: float
    arbitration_type: ArbitrationType
    reasoning: str
    supporting_evidence: Dict[str, Any]
    dissenting_opinions: List[str]
    resolution_method: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ArbitrationResult:
    """Complete result of arbitration process"""
    arbitration_id: str
    original_votes: List[AgentVote]
    conflicts_detected: List[VoteConflict]
    escalation_path: List[str]
    final_decision: ArbitrationDecision
    processing_time_seconds: float
    status: ArbitrationStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

class ArbitrationSystem:
    """
    Core arbitration system that resolves disagreements through escalation mechanisms
    """
    
    def __init__(self, consensus_architecture: Optional[ConsensusArchitecture] = None,
                 voting_aggregator: Optional[VotingAggregator] = None,
                 knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """Initialize arbitration system"""
        self.consensus = consensus_architecture or ConsensusArchitecture()
        self.aggregator = voting_aggregator or VotingAggregator(self.consensus)
        self.knowledge_base_path = Path(knowledge_base_path)
        
        # Arbitration thresholds and configurations
        self.arbitration_config = {
            "disagreement_threshold": 0.3,      # When to trigger arbitration
            "confidence_threshold": 0.5,        # Minimum confidence for resolution
            "escalation_timeout_minutes": 30,   # Timeout before escalation
            "max_escalation_levels": 3,         # Maximum escalation depth
            "expert_consensus_threshold": 0.8,  # Expert panel agreement threshold
            "evidence_quality_threshold": 0.7   # Minimum evidence quality
        }
        
        # Active arbitrations tracking
        self.active_arbitrations: Dict[str, ArbitrationResult] = {}
        self.completed_arbitrations: Dict[str, ArbitrationResult] = {}
        
        # Performance metrics
        self.metrics = {
            "total_arbitrations": 0,
            "successful_resolutions": 0,
            "escalated_to_human": 0,
            "average_resolution_time": 0.0,
            "resolution_method_distribution": {
                "weighted": 0,
                "expert_panel": 0,
                "arbitrator_agent": 0,
                "human": 0
            },
            "conflict_type_distribution": {}
        }

    def resolve_disagreement(self, votes: List[AgentVote], context: Dict[str, Any],
                           decision_id: Optional[str] = None) -> ArbitrationResult:
        """
        Main entry point for disagreement resolution
        
        Args:
            votes: List of conflicting agent votes
            context: Decision context and metadata
            decision_id: Optional specific decision ID
            
        Returns:
            ArbitrationResult: Complete arbitration process result
        """
        start_time = time.time()
        arbitration_id = self._generate_arbitration_id(votes, context)
        
        if decision_id is None:
            decision_id = f"arbitration-{arbitration_id[:8]}"
        
        logger.info(f"Starting arbitration for decision {decision_id} (ID: {arbitration_id})")
        
        # Step 1: Analyze conflicts
        conflicts = self._analyze_conflicts(votes, context)
        
        # Step 2: Check if arbitration is needed
        if not self._requires_arbitration(votes, conflicts, context):
            # Can be resolved through normal consensus
            result = self._resolve_through_consensus(votes, context, arbitration_id)
            result.processing_time_seconds = time.time() - start_time
            return result
        
        # Step 3: Initialize arbitration process
        arbitration_result = ArbitrationResult(
            arbitration_id=arbitration_id,
            original_votes=votes,
            conflicts_detected=conflicts,
            escalation_path=[],
            final_decision=None,
            processing_time_seconds=0.0,
            status=ArbitrationStatus.IN_PROGRESS,
            metadata={
                "decision_id": decision_id,
                "start_time": datetime.now().isoformat(),
                "context": context
            },
            audit_trail=[]
        )
        
        self.active_arbitrations[arbitration_id] = arbitration_result
        
        # Step 4: Attempt resolution through escalation path
        try:
            final_decision = self._execute_escalation_path(votes, conflicts, context, arbitration_result)
            arbitration_result.final_decision = final_decision
            arbitration_result.status = ArbitrationStatus.RESOLVED
            
            logger.info(f"Arbitration {arbitration_id} resolved: {final_decision.final_decision}")
            
        except Exception as e:
            logger.error(f"Arbitration {arbitration_id} failed: {str(e)}")
            arbitration_result.status = ArbitrationStatus.FAILED
            arbitration_result.metadata["error"] = str(e)
            
            # Create fallback decision
            arbitration_result.final_decision = self._create_fallback_decision(
                votes, context, arbitration_id, str(e)
            )
        
        # Step 5: Complete arbitration process
        arbitration_result.processing_time_seconds = time.time() - start_time
        self.completed_arbitrations[arbitration_id] = arbitration_result
        
        if arbitration_id in self.active_arbitrations:
            del self.active_arbitrations[arbitration_id]
        
        # Update metrics
        self._update_arbitration_metrics(arbitration_result)
        
        # Record decision for audit trail
        self._record_arbitration_decision(arbitration_result)
        
        return arbitration_result

    def _analyze_conflicts(self, votes: List[AgentVote], context: Dict[str, Any]) -> List[VoteConflict]:
        """Analyze votes to identify conflicts"""
        # Use voting aggregator's conflict detection capabilities
        temp_decision_id = f"temp-analysis-{int(time.time())}"
        
        try:
            # Create temporary vote collection for analysis
            voting_config = self.consensus.select_voting_mechanism(context)
            collection = self.aggregator.start_vote_collection(
                temp_decision_id, "Temporary Analysis", 
                self._determine_vote_type(votes), voting_config, context
            )
            
            # Add votes to collection
            for vote in votes:
                self.aggregator.cast_vote(temp_decision_id, vote)
            
            # Get conflict analysis
            conflicts = self.aggregator._detect_vote_conflicts(collection)
            
            # Clean up temporary collection
            if temp_decision_id in self.aggregator.active_collections:
                del self.aggregator.active_collections[temp_decision_id]
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error analyzing conflicts: {str(e)}")
            return []

    def _requires_arbitration(self, votes: List[AgentVote], conflicts: List[VoteConflict], 
                            context: Dict[str, Any]) -> bool:
        """Determine if arbitration is required"""
        if not conflicts:
            return False
        
        # Check severity of conflicts
        high_severity_conflicts = [c for c in conflicts if c.severity > 0.7]
        if high_severity_conflicts:
            return True
        
        # Check if it's a critical decision requiring arbitration
        if context.get("security_critical", False):
            return True
        
        # Check agreement level
        if len(votes) > 1:
            agreement_score = self._calculate_agreement_score(votes)
            if agreement_score < (1.0 - self.arbitration_config["disagreement_threshold"]):
                return True
        
        return False

    def _execute_escalation_path(self, votes: List[AgentVote], conflicts: List[VoteConflict],
                               context: Dict[str, Any], arbitration_result: ArbitrationResult) -> ArbitrationDecision:
        """Execute the escalation path for conflict resolution"""
        
        escalation_steps = self._determine_escalation_path(conflicts, context)
        arbitration_result.escalation_path = escalation_steps
        
        for step in escalation_steps:
            arbitration_result.audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": f"attempting_escalation_step",
                "step": step,
                "conflicts_remaining": len(conflicts)
            })
            
            try:
                if step == "weighted_resolution":
                    decision = self._attempt_weighted_resolution(votes, context, arbitration_result.arbitration_id)
                    
                elif step == "expert_panel_review":
                    decision = self._attempt_expert_panel_review(votes, conflicts, context, arbitration_result.arbitration_id)
                    
                elif step == "arbitrator_agent":
                    decision = self._spawn_arbitrator_agent(votes, conflicts, context, arbitration_result.arbitration_id)
                    
                elif step == "evidence_based_resolution":
                    decision = self._attempt_evidence_based_resolution(votes, conflicts, context, arbitration_result.arbitration_id)
                    
                elif step == "human_escalation":
                    decision = self._escalate_to_human(votes, conflicts, context, arbitration_result.arbitration_id)
                    arbitration_result.status = ArbitrationStatus.HUMAN_REQUIRED
                    
                else:
                    logger.warning(f"Unknown escalation step: {step}")
                    continue
                
                # Check if resolution is satisfactory
                if self._is_resolution_satisfactory(decision, conflicts, context):
                    arbitration_result.audit_trail.append({
                        "timestamp": datetime.now().isoformat(),
                        "action": "resolution_successful",
                        "step": step,
                        "confidence": decision.confidence_score
                    })
                    return decision
                    
            except Exception as e:
                logger.error(f"Escalation step {step} failed: {str(e)}")
                arbitration_result.audit_trail.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "escalation_step_failed",
                    "step": step,
                    "error": str(e)
                })
                continue
        
        # If all steps failed, create final fallback decision
        return self._create_fallback_decision(votes, context, arbitration_result.arbitration_id, "All escalation steps failed")

    def _attempt_weighted_resolution(self, votes: List[AgentVote], context: Dict[str, Any], 
                                   arbitration_id: str) -> ArbitrationDecision:
        """Attempt resolution using weighted voting"""
        logger.info(f"Attempting weighted resolution for arbitration {arbitration_id}")
        
        weighted_config = self.consensus.voting_configs["weighted_voting"]
        consensus_result = self.consensus.calculate_consensus(votes, weighted_config, context)
        
        # Enhance confidence based on weight distribution
        weight_distribution = self._analyze_weight_distribution(votes, weighted_config)
        adjusted_confidence = min(1.0, consensus_result.confidence_score * weight_distribution["balance_factor"])
        
        return ArbitrationDecision(
            decision_id=f"{arbitration_id}-weighted",
            final_decision=consensus_result.decision,
            confidence_score=adjusted_confidence,
            arbitration_type=ArbitrationType.WEIGHTED_RESOLUTION,
            reasoning=f"Weighted consensus with {consensus_result.agreement_level:.2f} agreement",
            supporting_evidence={
                "vote_weights": weighted_config.weights,
                "agreement_level": consensus_result.agreement_level,
                "weight_distribution": weight_distribution
            },
            dissenting_opinions=[
                f"{vote.agent_id}: {vote.reasoning}" 
                for vote in votes 
                if vote.vote != consensus_result.decision
            ],
            resolution_method="weighted_voting"
        )

    def _attempt_expert_panel_review(self, votes: List[AgentVote], conflicts: List[VoteConflict],
                                   context: Dict[str, Any], arbitration_id: str) -> ArbitrationDecision:
        """Attempt resolution through expert panel review"""
        logger.info(f"Attempting expert panel review for arbitration {arbitration_id}")
        
        domain = context.get("domain", "general")
        
        # Identify expert agents for this domain
        expert_votes = []
        for vote in votes:
            expertise_level = self.consensus.agent_expertise.get(vote.agent_id, {}).get(domain, 0.5)
            if expertise_level > 0.8:  # High expertise threshold
                expert_votes.append((vote, expertise_level))
        
        if not expert_votes:
            raise ValueError("No domain experts available for panel review")
        
        # Calculate expert-weighted consensus
        total_expertise = sum(expertise for _, expertise in expert_votes)
        expert_decision_weights = {}
        
        for vote, expertise in expert_votes:
            weight = expertise / total_expertise
            expert_decision_weights[vote.vote] = expert_decision_weights.get(vote.vote, 0) + weight
        
        # Find decision with highest expert weight
        final_decision = max(expert_decision_weights.keys(), key=lambda k: expert_decision_weights[k])
        confidence = expert_decision_weights[final_decision]
        
        return ArbitrationDecision(
            decision_id=f"{arbitration_id}-expert-panel",
            final_decision=final_decision,
            confidence_score=confidence,
            arbitration_type=ArbitrationType.EXPERT_PANEL,
            reasoning=f"Expert panel consensus from {len(expert_votes)} domain experts",
            supporting_evidence={
                "expert_votes": [(vote.agent_id, vote.vote, expertise) for vote, expertise in expert_votes],
                "expertise_weights": expert_decision_weights,
                "domain": domain
            },
            dissenting_opinions=[
                f"{vote.agent_id}: {vote.reasoning}"
                for vote, _ in expert_votes
                if vote.vote != final_decision
            ],
            resolution_method="expert_panel"
        )

    def _spawn_arbitrator_agent(self, votes: List[AgentVote], conflicts: List[VoteConflict],
                              context: Dict[str, Any], arbitration_id: str) -> ArbitrationDecision:
        """Spawn a specialized arbitrator agent for complex conflict resolution"""
        logger.info(f"Spawning arbitrator agent for arbitration {arbitration_id}")
        
        # In a full implementation, this would create and launch a specialized RIF agent
        # For now, we simulate sophisticated arbitration logic
        
        # Analyze evidence quality and reasoning
        evidence_scores = []
        reasoning_analysis = {}
        
        for vote in votes:
            evidence_score = vote.evidence_quality
            evidence_scores.append(evidence_score)
            reasoning_analysis[vote.agent_id] = {
                "vote": vote.vote,
                "confidence": vote.confidence.value,
                "reasoning": vote.reasoning,
                "evidence_quality": evidence_score
            }
        
        # Simulate arbitrator decision logic
        # Higher evidence quality votes get more weight
        weighted_decisions = {}
        total_evidence_weight = sum(evidence_scores)
        
        if total_evidence_weight > 0:
            for vote, evidence_score in zip(votes, evidence_scores):
                weight = evidence_score / total_evidence_weight
                weighted_decisions[vote.vote] = weighted_decisions.get(vote.vote, 0) + weight
        else:
            # Fallback to equal weighting
            for vote in votes:
                weighted_decisions[vote.vote] = weighted_decisions.get(vote.vote, 0) + (1.0 / len(votes))
        
        final_decision = max(weighted_decisions.keys(), key=lambda k: weighted_decisions[k])
        confidence = weighted_decisions[final_decision]
        
        # Adjust confidence based on evidence quality
        avg_evidence_quality = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.5
        adjusted_confidence = confidence * (0.5 + 0.5 * avg_evidence_quality)
        
        return ArbitrationDecision(
            decision_id=f"{arbitration_id}-arbitrator",
            final_decision=final_decision,
            confidence_score=adjusted_confidence,
            arbitration_type=ArbitrationType.ARBITRATOR_AGENT,
            reasoning=f"Arbitrator agent analysis based on evidence quality and reasoning depth",
            supporting_evidence={
                "reasoning_analysis": reasoning_analysis,
                "evidence_weighted_decisions": weighted_decisions,
                "average_evidence_quality": avg_evidence_quality,
                "conflict_summary": [conflict.description for conflict in conflicts]
            },
            dissenting_opinions=[
                f"{vote.agent_id}: {vote.reasoning}" 
                for vote in votes 
                if vote.vote != final_decision
            ],
            resolution_method="arbitrator_agent"
        )

    def _attempt_evidence_based_resolution(self, votes: List[AgentVote], conflicts: List[VoteConflict],
                                         context: Dict[str, Any], arbitration_id: str) -> ArbitrationDecision:
        """Attempt resolution based on evidence quality analysis"""
        logger.info(f"Attempting evidence-based resolution for arbitration {arbitration_id}")
        
        # Filter votes by evidence quality threshold
        high_quality_votes = [
            vote for vote in votes 
            if vote.evidence_quality >= self.arbitration_config["evidence_quality_threshold"]
        ]
        
        if not high_quality_votes:
            raise ValueError("No votes meet evidence quality threshold")
        
        # Use only high-quality evidence votes for decision
        voting_config = self.consensus.select_voting_mechanism(context)
        consensus_result = self.consensus.calculate_consensus(high_quality_votes, voting_config, context)
        
        return ArbitrationDecision(
            decision_id=f"{arbitration_id}-evidence",
            final_decision=consensus_result.decision,
            confidence_score=consensus_result.confidence_score,
            arbitration_type=ArbitrationType.EVIDENCE_BASED,
            reasoning=f"Evidence-based consensus from {len(high_quality_votes)} high-quality votes",
            supporting_evidence={
                "quality_threshold": self.arbitration_config["evidence_quality_threshold"],
                "high_quality_votes": len(high_quality_votes),
                "total_votes": len(votes),
                "evidence_scores": [vote.evidence_quality for vote in high_quality_votes]
            },
            dissenting_opinions=[
                f"{vote.agent_id}: {vote.reasoning}" 
                for vote in votes 
                if vote.vote != consensus_result.decision and vote.evidence_quality >= self.arbitration_config["evidence_quality_threshold"]
            ],
            resolution_method="evidence_based"
        )

    def _escalate_to_human(self, votes: List[AgentVote], conflicts: List[VoteConflict],
                         context: Dict[str, Any], arbitration_id: str) -> ArbitrationDecision:
        """Escalate decision to human intervention"""
        logger.info(f"Escalating arbitration {arbitration_id} to human intervention")
        
        # Create comprehensive summary for human review
        human_summary = {
            "arbitration_id": arbitration_id,
            "decision_context": context,
            "vote_summary": [
                {
                    "agent": vote.agent_id,
                    "vote": vote.vote,
                    "confidence": vote.confidence.value,
                    "reasoning": vote.reasoning,
                    "evidence_quality": vote.evidence_quality
                }
                for vote in votes
            ],
            "conflicts": [
                {
                    "type": conflict.conflict_type.value,
                    "description": conflict.description,
                    "severity": conflict.severity,
                    "suggestion": conflict.suggested_resolution
                }
                for conflict in conflicts
            ],
            "escalation_timestamp": datetime.now().isoformat()
        }
        
        # Store human escalation request
        escalation_file = self.knowledge_base_path / "arbitration" / "human_escalations" / f"{arbitration_id}.json"
        escalation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(escalation_file, 'w') as f:
            json.dump(human_summary, f, indent=2)
        
        logger.info(f"Human escalation request stored: {escalation_file}")
        
        # Create placeholder decision pending human review
        return ArbitrationDecision(
            decision_id=f"{arbitration_id}-human",
            final_decision="PENDING_HUMAN_REVIEW",
            confidence_score=0.0,
            arbitration_type=ArbitrationType.HUMAN_ESCALATION,
            reasoning="Conflict requires human intervention - automated arbitration insufficient",
            supporting_evidence=human_summary,
            dissenting_opinions=[f"{vote.agent_id}: {vote.reasoning}" for vote in votes],
            resolution_method="human_escalation"
        )

    def _determine_escalation_path(self, conflicts: List[VoteConflict], context: Dict[str, Any]) -> List[str]:
        """Determine appropriate escalation path based on conflicts and context"""
        escalation_steps = []
        
        # Check conflict types and severity
        conflict_types = {conflict.conflict_type for conflict in conflicts}
        max_severity = max(conflict.severity for conflict in conflicts) if conflicts else 0.0
        
        # Always try weighted resolution first for non-critical decisions
        if not context.get("security_critical", False):
            escalation_steps.append("weighted_resolution")
        
        # Add expert panel review for domain-specific decisions
        if context.get("domain") and context.get("domain") != "general":
            escalation_steps.append("expert_panel_review")
        
        # Add evidence-based resolution for quality-focused conflicts
        if ConflictType.LOW_CONFIDENCE in conflict_types or any(c.severity > 0.5 for c in conflicts):
            escalation_steps.append("evidence_based_resolution")
        
        # Add arbitrator agent for complex conflicts
        if max_severity > 0.6 or len(conflict_types) > 1:
            escalation_steps.append("arbitrator_agent")
        
        # Always have human escalation as final fallback
        escalation_steps.append("human_escalation")
        
        return escalation_steps

    def _is_resolution_satisfactory(self, decision: ArbitrationDecision, conflicts: List[VoteConflict],
                                  context: Dict[str, Any]) -> bool:
        """Check if arbitration decision is satisfactory"""
        # Check confidence threshold
        if decision.confidence_score < self.arbitration_config["confidence_threshold"]:
            return False
        
        # For security-critical decisions, require higher confidence
        if context.get("security_critical", False) and decision.confidence_score < 0.8:
            return False
        
        # Don't accept human escalation as satisfactory (it's pending)
        if decision.arbitration_type == ArbitrationType.HUMAN_ESCALATION:
            return decision.final_decision != "PENDING_HUMAN_REVIEW"
        
        return True

    def _resolve_through_consensus(self, votes: List[AgentVote], context: Dict[str, Any],
                                 arbitration_id: str) -> ArbitrationResult:
        """Resolve through normal consensus when arbitration is not needed"""
        voting_config = self.consensus.select_voting_mechanism(context)
        consensus_result = self.consensus.calculate_consensus(votes, voting_config, context)
        
        decision = ArbitrationDecision(
            decision_id=f"{arbitration_id}-consensus",
            final_decision=consensus_result.decision,
            confidence_score=consensus_result.confidence_score,
            arbitration_type=ArbitrationType.WEIGHTED_RESOLUTION,
            reasoning="Resolved through normal consensus - no arbitration required",
            supporting_evidence={
                "agreement_level": consensus_result.agreement_level,
                "mechanism": consensus_result.mechanism_used.value
            },
            dissenting_opinions=[],
            resolution_method="consensus"
        )
        
        return ArbitrationResult(
            arbitration_id=arbitration_id,
            original_votes=votes,
            conflicts_detected=[],
            escalation_path=["consensus"],
            final_decision=decision,
            processing_time_seconds=0.0,
            status=ArbitrationStatus.RESOLVED,
            audit_trail=[{
                "timestamp": datetime.now().isoformat(),
                "action": "resolved_through_consensus",
                "confidence": consensus_result.confidence_score
            }]
        )

    def _create_fallback_decision(self, votes: List[AgentVote], context: Dict[str, Any],
                                arbitration_id: str, reason: str) -> ArbitrationDecision:
        """Create a fallback decision when all other methods fail"""
        # Simple majority vote as ultimate fallback
        if votes:
            vote_counts = {}
            for vote in votes:
                vote_counts[vote.vote] = vote_counts.get(vote.vote, 0) + 1
            
            final_decision = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            confidence = vote_counts[final_decision] / len(votes)
        else:
            final_decision = False
            confidence = 0.0
        
        return ArbitrationDecision(
            decision_id=f"{arbitration_id}-fallback",
            final_decision=final_decision,
            confidence_score=confidence,
            arbitration_type=ArbitrationType.WEIGHTED_RESOLUTION,
            reasoning=f"Fallback decision due to: {reason}",
            supporting_evidence={"fallback_reason": reason},
            dissenting_opinions=[f"{vote.agent_id}: {vote.reasoning}" for vote in votes if vote.vote != final_decision],
            resolution_method="fallback"
        )

    # Helper methods
    def _generate_arbitration_id(self, votes: List[AgentVote], context: Dict[str, Any]) -> str:
        """Generate unique arbitration ID"""
        content = f"{len(votes)}-{context.get('domain', 'general')}-{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()

    def _determine_vote_type(self, votes: List[AgentVote]):
        """Determine vote type from votes list"""
        if not votes:
            return "boolean"
        
        first_vote = votes[0].vote
        if isinstance(first_vote, bool):
            return "boolean"
        elif isinstance(first_vote, (int, float)):
            return "numeric"
        else:
            return "categorical"

    def _calculate_agreement_score(self, votes: List[AgentVote]) -> float:
        """Calculate simple agreement score"""
        if len(votes) <= 1:
            return 1.0
        
        vote_counts = {}
        for vote in votes:
            vote_counts[vote.vote] = vote_counts.get(vote.vote, 0) + 1
        
        max_agreement = max(vote_counts.values())
        return max_agreement / len(votes)

    def _analyze_weight_distribution(self, votes: List[AgentVote], config: VotingConfig) -> Dict[str, float]:
        """Analyze weight distribution for weighted voting"""
        total_weight = sum(config.weights.get(vote.agent_id, 1.0) for vote in votes)
        max_weight = max(config.weights.get(vote.agent_id, 1.0) for vote in votes)
        
        balance_factor = 1.0 - (max_weight / total_weight) if total_weight > 0 else 0.0
        
        return {
            "total_weight": total_weight,
            "max_individual_weight": max_weight,
            "balance_factor": balance_factor,
            "weight_distribution": {vote.agent_id: config.weights.get(vote.agent_id, 1.0) for vote in votes}
        }

    def _update_arbitration_metrics(self, result: ArbitrationResult):
        """Update arbitration system metrics"""
        self.metrics["total_arbitrations"] += 1
        
        if result.status == ArbitrationStatus.RESOLVED:
            self.metrics["successful_resolutions"] += 1
        elif result.status == ArbitrationStatus.HUMAN_REQUIRED:
            self.metrics["escalated_to_human"] += 1
        
        # Update resolution method distribution
        if result.final_decision:
            method = result.final_decision.resolution_method
            if method in ["weighted_voting", "consensus"]:
                self.metrics["resolution_method_distribution"]["weighted"] += 1
            elif method == "expert_panel":
                self.metrics["resolution_method_distribution"]["expert_panel"] += 1
            elif method == "arbitrator_agent":
                self.metrics["resolution_method_distribution"]["arbitrator_agent"] += 1
            elif method == "human_escalation":
                self.metrics["resolution_method_distribution"]["human"] += 1
        
        # Update average resolution time
        total_arbitrations = self.metrics["total_arbitrations"]
        current_avg = self.metrics["average_resolution_time"]
        self.metrics["average_resolution_time"] = (
            (current_avg * (total_arbitrations - 1) + result.processing_time_seconds) / total_arbitrations
        )

    def _record_arbitration_decision(self, result: ArbitrationResult):
        """Record arbitration decision for audit trail"""
        try:
            decisions_dir = self.knowledge_base_path / "arbitration" / "decisions"
            decisions_dir.mkdir(parents=True, exist_ok=True)
            
            decision_file = decisions_dir / f"{result.arbitration_id}.json"
            
            with open(decision_file, 'w') as f:
                json.dump({
                    "arbitration_result": asdict(result),
                    "timestamp": datetime.now().isoformat(),
                    "system_version": "1.0.0"
                }, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to record arbitration decision: {str(e)}")

    # Public interface methods
    def get_active_arbitrations(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of active arbitrations"""
        return {
            arb_id: {
                "status": result.status.value,
                "conflicts_count": len(result.conflicts_detected),
                "escalation_level": len(result.escalation_path),
                "processing_time": result.processing_time_seconds
            }
            for arb_id, result in self.active_arbitrations.items()
        }

    def get_arbitration_metrics(self) -> Dict[str, Any]:
        """Get arbitration system metrics"""
        return {
            **self.metrics,
            "active_arbitrations": len(self.active_arbitrations),
            "completed_arbitrations": len(self.completed_arbitrations)
        }

    def get_arbitration_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent arbitration history"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_arbitrations = []
        for result in self.completed_arbitrations.values():
            if result.final_decision and result.final_decision.timestamp > cutoff_date:
                recent_arbitrations.append({
                    "arbitration_id": result.arbitration_id,
                    "final_decision": result.final_decision.final_decision,
                    "confidence": result.final_decision.confidence_score,
                    "resolution_method": result.final_decision.resolution_method,
                    "processing_time": result.processing_time_seconds,
                    "status": result.status.value
                })
        
        return recent_arbitrations


def main():
    """Demonstration of arbitration system functionality"""
    print("=== RIF Arbitration System Demo ===\n")
    
    # Initialize system
    consensus = ConsensusArchitecture()
    aggregator = VotingAggregator(consensus)
    arbitration = ArbitrationSystem(consensus, aggregator)
    
    # Example 1: Split decision requiring arbitration
    print("Example 1: Split Decision Arbitration")
    votes = [
        consensus.create_vote("rif-implementer", True, ConfidenceLevel.HIGH, "Implementation is solid", 0.8),
        consensus.create_vote("rif-validator", False, ConfidenceLevel.HIGH, "Tests are insufficient", 0.9),
        consensus.create_vote("rif-security", True, ConfidenceLevel.MEDIUM, "No major security issues", 0.6)
    ]
    
    context = {"domain": "implementation", "risk_level": "medium"}
    result = arbitration.resolve_disagreement(votes, context)
    
    print(f"Arbitration ID: {result.arbitration_id}")
    print(f"Status: {result.status.value}")
    print(f"Final Decision: {result.final_decision.final_decision}")
    print(f"Confidence: {result.final_decision.confidence_score:.2f}")
    print(f"Resolution Method: {result.final_decision.resolution_method}")
    print(f"Escalation Path: {' → '.join(result.escalation_path)}")
    print(f"Processing Time: {result.processing_time_seconds:.2f}s")
    print()
    
    # Example 2: Security-critical decision
    print("Example 2: Security-Critical Decision")
    security_votes = [
        consensus.create_vote("rif-security", False, ConfidenceLevel.VERY_HIGH, "Critical vulnerability detected", 0.95),
        consensus.create_vote("rif-implementer", True, ConfidenceLevel.MEDIUM, "Fix looks good", 0.7),
        consensus.create_vote("rif-validator", False, ConfidenceLevel.HIGH, "Security tests failing", 0.85)
    ]
    
    security_context = {"domain": "security", "security_critical": True, "risk_level": "critical"}
    security_result = arbitration.resolve_disagreement(security_votes, security_context)
    
    print(f"Arbitration ID: {security_result.arbitration_id}")
    print(f"Status: {security_result.status.value}")
    print(f"Final Decision: {security_result.final_decision.final_decision}")
    print(f"Confidence: {security_result.final_decision.confidence_score:.2f}")
    print(f"Arbitration Type: {security_result.final_decision.arbitration_type.value}")
    print(f"Processing Time: {security_result.processing_time_seconds:.2f}s")
    print()
    
    # Show system metrics
    print("=== Arbitration System Metrics ===")
    metrics = arbitration.get_arbitration_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()