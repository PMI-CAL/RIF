#!/usr/bin/env python3
"""
RIF Voting Aggregator - Issue #60
System to collect agent votes, calculate weighted consensus, and handle vote conflicts.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import statistics
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

# Import consensus architecture components
from consensus_architecture import (
    VotingMechanism, RiskLevel, ConfidenceLevel, AgentVote, 
    VotingConfig, ArbitrationRule, ConsensusResult, ConsensusArchitecture
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoteType(Enum):
    """Types of votes that can be cast"""
    BOOLEAN = "boolean"           # True/False votes
    NUMERIC = "numeric"           # Numeric scores (0.0 - 1.0)
    CATEGORICAL = "categorical"   # Category selection
    RANKING = "ranking"           # Ranked preferences
    WEIGHTED_SCORE = "weighted_score"  # Score with confidence weighting

class ConflictType(Enum):
    """Types of voting conflicts"""
    SPLIT_DECISION = "split_decision"       # Even split between options
    OUTLIER_DETECTED = "outlier_detected"   # One vote very different from others
    LOW_CONFIDENCE = "low_confidence"       # All votes have low confidence
    MISSING_EXPERTISE = "missing_expertise" # Key expert agents haven't voted
    TIMEOUT_PARTIAL = "timeout_partial"     # Some agents didn't vote in time

@dataclass
class VoteConflict:
    """Details about a voting conflict"""
    conflict_type: ConflictType
    description: str
    affected_agents: List[str]
    severity: float  # 0.0 - 1.0
    suggested_resolution: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VoteCollection:
    """Collection of votes for a specific decision"""
    decision_id: str
    decision_title: str
    vote_type: VoteType
    voting_config: VotingConfig
    votes: List[AgentVote] = field(default_factory=list)
    conflicts: List[VoteConflict] = field(default_factory=list)
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    collection_start: datetime = field(default_factory=datetime.now)
    collection_end: Optional[datetime] = None
    is_complete: bool = False

@dataclass
class AggregationReport:
    """Detailed report of vote aggregation process"""
    decision_id: str
    consensus_result: ConsensusResult
    vote_summary: Dict[str, Any]
    conflict_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time_seconds: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class VotingAggregator:
    """
    Core voting aggregator that collects votes from multiple agents and calculates consensus
    """
    
    def __init__(self, consensus_architecture: Optional[ConsensusArchitecture] = None):
        """Initialize voting aggregator"""
        self.consensus = consensus_architecture or ConsensusArchitecture()
        self.active_collections: Dict[str, VoteCollection] = {}
        self.completed_collections: Dict[str, VoteCollection] = {}
        self.aggregation_history: List[AggregationReport] = []
        
        # Conflict detection thresholds
        self.conflict_thresholds = {
            "split_decision": 0.4,      # Difference threshold for split decisions
            "outlier_detection": 2.0,   # Standard deviations for outlier detection
            "low_confidence": 0.3,      # Average confidence threshold
            "expertise_coverage": 0.7,  # Required expertise coverage ratio
        }
        
        # Aggregation metrics
        self.metrics = {
            "total_votes_processed": 0,
            "total_decisions": 0,
            "conflict_resolution_success_rate": 0.0,
            "average_processing_time": 0.0,
            "vote_type_distribution": defaultdict(int),
            "agent_participation_rates": defaultdict(float)
        }

    def start_vote_collection(self, decision_id: str, decision_title: str, 
                             vote_type: VoteType, voting_config: VotingConfig,
                             context: Dict[str, Any] = None,
                             deadline_minutes: Optional[int] = None) -> VoteCollection:
        """
        Start collecting votes for a decision
        
        Args:
            decision_id: Unique identifier for the decision
            decision_title: Human-readable title for the decision
            vote_type: Type of votes to collect
            voting_config: Configuration for the voting mechanism
            context: Additional context for the decision
            deadline_minutes: Optional deadline in minutes from now
            
        Returns:
            VoteCollection: The initialized vote collection
        """
        if decision_id in self.active_collections:
            raise ValueError(f"Vote collection already active for decision {decision_id}")
        
        deadline = None
        if deadline_minutes:
            deadline = datetime.now() + timedelta(minutes=deadline_minutes)
        
        collection = VoteCollection(
            decision_id=decision_id,
            decision_title=decision_title,
            vote_type=vote_type,
            voting_config=voting_config,
            context=context or {},
            deadline=deadline
        )
        
        self.active_collections[decision_id] = collection
        logger.info(f"Started vote collection for decision '{decision_title}' (ID: {decision_id})")
        
        return collection

    def cast_vote(self, decision_id: str, agent_vote: AgentVote) -> bool:
        """
        Cast a vote for a specific decision
        
        Args:
            decision_id: ID of the decision to vote on
            agent_vote: The agent's vote
            
        Returns:
            bool: Success status
        """
        if decision_id not in self.active_collections:
            logger.error(f"No active vote collection for decision {decision_id}")
            return False
        
        collection = self.active_collections[decision_id]
        
        # Check if voting deadline has passed
        if collection.deadline and datetime.now() > collection.deadline:
            logger.warning(f"Vote from {agent_vote.agent_id} rejected - deadline passed for {decision_id}")
            return False
        
        # Check for duplicate votes from same agent
        for i, existing_vote in enumerate(collection.votes):
            if existing_vote.agent_id == agent_vote.agent_id:
                logger.info(f"Replacing existing vote from {agent_vote.agent_id} for {decision_id}")
                collection.votes[i] = agent_vote
                return True
        
        # Add new vote
        collection.votes.append(agent_vote)
        self.metrics["total_votes_processed"] += 1
        self.metrics["vote_type_distribution"][collection.vote_type.value] += 1
        
        logger.info(f"Vote cast by {agent_vote.agent_id} for decision {decision_id}")
        
        # Check if we have enough votes to proceed
        if len(collection.votes) >= collection.voting_config.minimum_votes:
            logger.info(f"Minimum votes reached for {decision_id}: {len(collection.votes)}")
        
        return True

    def aggregate_votes(self, decision_id: str, force_completion: bool = False) -> AggregationReport:
        """
        Aggregate votes for a decision and calculate consensus
        
        Args:
            decision_id: ID of the decision to aggregate
            force_completion: Force aggregation even if conditions aren't met
            
        Returns:
            AggregationReport: Detailed report of aggregation process
        """
        start_time = time.time()
        
        if decision_id not in self.active_collections:
            raise ValueError(f"No active vote collection for decision {decision_id}")
        
        collection = self.active_collections[decision_id]
        
        # Check if we can aggregate
        if not force_completion:
            if len(collection.votes) < collection.voting_config.minimum_votes:
                raise ValueError(f"Insufficient votes: {len(collection.votes)} < {collection.voting_config.minimum_votes}")
            
            if collection.deadline and datetime.now() < collection.deadline:
                logger.warning(f"Aggregating before deadline for {decision_id}")
        
        # Detect conflicts
        conflicts = self._detect_vote_conflicts(collection)
        collection.conflicts = conflicts
        
        # Calculate consensus
        consensus_result = self.consensus.calculate_consensus(
            collection.votes, 
            collection.voting_config, 
            collection.context
        )
        
        # Generate quality metrics
        quality_metrics = self._calculate_quality_metrics(collection, consensus_result)
        
        # Generate vote summary
        vote_summary = self._generate_vote_summary(collection)
        
        # Analyze conflicts
        conflict_analysis = self._analyze_conflicts(conflicts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(collection, consensus_result, conflicts)
        
        # Create aggregation report
        processing_time = time.time() - start_time
        report = AggregationReport(
            decision_id=decision_id,
            consensus_result=consensus_result,
            vote_summary=vote_summary,
            conflict_analysis=conflict_analysis,
            quality_metrics=quality_metrics,
            processing_time_seconds=processing_time,
            recommendations=recommendations,
            metadata={
                "vote_count": len(collection.votes),
                "conflicts_detected": len(conflicts),
                "voting_mechanism": collection.voting_config.mechanism.value,
                "collection_duration": (datetime.now() - collection.collection_start).total_seconds()
            }
        )
        
        # Mark collection as complete
        collection.is_complete = True
        collection.collection_end = datetime.now()
        
        # Move to completed collections
        self.completed_collections[decision_id] = collection
        del self.active_collections[decision_id]
        
        # Update metrics
        self.metrics["total_decisions"] += 1
        self._update_processing_metrics(processing_time)
        self._update_participation_metrics(collection)
        
        # Store in history
        self.aggregation_history.append(report)
        
        logger.info(f"Vote aggregation completed for {decision_id}: {consensus_result.decision} "
                   f"(confidence: {consensus_result.confidence_score:.2f})")
        
        return report

    def _detect_vote_conflicts(self, collection: VoteCollection) -> List[VoteConflict]:
        """Detect various types of voting conflicts"""
        conflicts = []
        votes = collection.votes
        
        if not votes:
            return conflicts
        
        # Split decision detection
        if collection.vote_type == VoteType.BOOLEAN:
            conflicts.extend(self._detect_split_decision(collection))
        
        # Outlier detection for numeric votes
        if collection.vote_type in [VoteType.NUMERIC, VoteType.WEIGHTED_SCORE]:
            conflicts.extend(self._detect_outliers(collection))
        
        # Low confidence detection
        conflicts.extend(self._detect_low_confidence(collection))
        
        # Missing expertise detection
        conflicts.extend(self._detect_missing_expertise(collection))
        
        # Timeout partial votes
        if collection.deadline and datetime.now() > collection.deadline:
            conflicts.extend(self._detect_timeout_partial(collection))
        
        return conflicts

    def _detect_split_decision(self, collection: VoteCollection) -> List[VoteConflict]:
        """Detect split decisions in boolean votes"""
        conflicts = []
        votes = collection.votes
        
        if collection.vote_type != VoteType.BOOLEAN:
            return conflicts
        
        true_votes = sum(1 for vote in votes if vote.vote is True)
        false_votes = sum(1 for vote in votes if vote.vote is False)
        total_votes = len(votes)
        
        if total_votes > 1:
            split_ratio = abs(true_votes - false_votes) / total_votes
            
            if split_ratio <= self.conflict_thresholds["split_decision"]:
                conflicts.append(VoteConflict(
                    conflict_type=ConflictType.SPLIT_DECISION,
                    description=f"Votes are split: {true_votes} True vs {false_votes} False",
                    affected_agents=[vote.agent_id for vote in votes],
                    severity=1.0 - split_ratio,  # Higher severity for closer splits
                    suggested_resolution="Consider additional evidence gathering or expert consultation",
                    metadata={"true_votes": true_votes, "false_votes": false_votes}
                ))
        
        return conflicts

    def _detect_outliers(self, collection: VoteCollection) -> List[VoteConflict]:
        """Detect outlier votes in numeric data"""
        conflicts = []
        votes = collection.votes
        
        if len(votes) < 3:  # Need at least 3 votes for outlier detection
            return conflicts
        
        numeric_votes = []
        for vote in votes:
            if isinstance(vote.vote, (int, float)):
                numeric_votes.append((vote.agent_id, float(vote.vote)))
        
        if len(numeric_votes) < 3:
            return conflicts
        
        values = [vote[1] for vote in numeric_votes]
        mean_val = statistics.mean(values)
        
        if len(values) > 1:
            stdev = statistics.stdev(values)
            threshold = self.conflict_thresholds["outlier_detection"]
            
            outliers = []
            for agent_id, value in numeric_votes:
                if abs(value - mean_val) > threshold * stdev:
                    outliers.append(agent_id)
            
            if outliers:
                conflicts.append(VoteConflict(
                    conflict_type=ConflictType.OUTLIER_DETECTED,
                    description=f"Outlier votes detected from agents: {', '.join(outliers)}",
                    affected_agents=outliers,
                    severity=len(outliers) / len(numeric_votes),
                    suggested_resolution="Review outlier reasoning and consider additional validation",
                    metadata={"mean": mean_val, "stdev": stdev, "outlier_values": 
                             {agent: val for agent, val in numeric_votes if agent in outliers}}
                ))
        
        return conflicts

    def _detect_low_confidence(self, collection: VoteCollection) -> List[VoteConflict]:
        """Detect when all votes have low confidence"""
        conflicts = []
        votes = collection.votes
        
        if not votes:
            return conflicts
        
        confidence_scores = [self.consensus._confidence_to_score(vote.confidence) for vote in votes]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        if avg_confidence < self.conflict_thresholds["low_confidence"]:
            conflicts.append(VoteConflict(
                conflict_type=ConflictType.LOW_CONFIDENCE,
                description=f"Low average confidence: {avg_confidence:.2f}",
                affected_agents=[vote.agent_id for vote in votes],
                severity=1.0 - avg_confidence,
                suggested_resolution="Consider gathering more evidence before making final decision",
                metadata={"average_confidence": avg_confidence, "individual_confidences": confidence_scores}
            ))
        
        return conflicts

    def _detect_missing_expertise(self, collection: VoteCollection) -> List[VoteConflict]:
        """Detect when key expert agents haven't voted"""
        conflicts = []
        
        # Get domain from context
        domain = collection.context.get("domain", "general")
        
        # Find expert agents for this domain
        expert_agents = []
        for agent_id, expertise in self.consensus.agent_expertise.items():
            if expertise.get(domain, 0) > 0.8:  # High expertise threshold
                expert_agents.append(agent_id)
        
        # Check which experts have voted
        voted_agents = {vote.agent_id for vote in collection.votes}
        missing_experts = [agent for agent in expert_agents if agent not in voted_agents]
        
        if missing_experts:
            coverage_ratio = len([a for a in expert_agents if a in voted_agents]) / len(expert_agents) if expert_agents else 1.0
            
            if coverage_ratio < self.conflict_thresholds["expertise_coverage"]:
                conflicts.append(VoteConflict(
                    conflict_type=ConflictType.MISSING_EXPERTISE,
                    description=f"Missing votes from domain experts: {', '.join(missing_experts)}",
                    affected_agents=missing_experts,
                    severity=1.0 - coverage_ratio,
                    suggested_resolution="Wait for expert agents to vote or request their input",
                    metadata={"domain": domain, "missing_experts": missing_experts, "coverage": coverage_ratio}
                ))
        
        return conflicts

    def _detect_timeout_partial(self, collection: VoteCollection) -> List[VoteConflict]:
        """Detect when voting deadline passed with incomplete participation"""
        conflicts = []
        
        if not collection.deadline:
            return conflicts
        
        expected_agents = collection.context.get("expected_agents", [])
        if not expected_agents:
            return conflicts
        
        voted_agents = {vote.agent_id for vote in collection.votes}
        missing_agents = [agent for agent in expected_agents if agent not in voted_agents]
        
        if missing_agents:
            participation_rate = len(voted_agents) / len(expected_agents)
            
            conflicts.append(VoteConflict(
                conflict_type=ConflictType.TIMEOUT_PARTIAL,
                description=f"Voting deadline passed with {len(missing_agents)} agents not participating",
                affected_agents=missing_agents,
                severity=1.0 - participation_rate,
                suggested_resolution="Consider extending deadline or proceeding with available votes",
                metadata={"missing_agents": missing_agents, "participation_rate": participation_rate}
            ))
        
        return conflicts

    def _calculate_quality_metrics(self, collection: VoteCollection, 
                                  consensus_result: ConsensusResult) -> Dict[str, float]:
        """Calculate quality metrics for the vote aggregation"""
        votes = collection.votes
        
        metrics = {
            "participation_rate": 0.0,
            "confidence_consistency": 0.0,
            "expertise_alignment": 0.0,
            "temporal_consistency": 0.0,
            "evidence_quality": 0.0
        }
        
        if not votes:
            return metrics
        
        # Participation rate
        expected_agents = collection.context.get("expected_agents", [])
        if expected_agents:
            metrics["participation_rate"] = len(votes) / len(expected_agents)
        else:
            metrics["participation_rate"] = 1.0  # All expected agents voted
        
        # Confidence consistency (lower variance = higher consistency)
        confidence_scores = [self.consensus._confidence_to_score(vote.confidence) for vote in votes]
        if len(confidence_scores) > 1:
            confidence_variance = statistics.variance(confidence_scores)
            metrics["confidence_consistency"] = max(0.0, 1.0 - confidence_variance)
        else:
            metrics["confidence_consistency"] = 1.0
        
        # Expertise alignment (how well expertise matches decision importance)
        domain = collection.context.get("domain", "general")
        expertise_scores = []
        for vote in votes:
            expertise = self.consensus.agent_expertise.get(vote.agent_id, {}).get(domain, 0.5)
            expertise_scores.append(expertise)
        
        if expertise_scores:
            metrics["expertise_alignment"] = sum(expertise_scores) / len(expertise_scores)
        
        # Temporal consistency (how close in time were the votes)
        if len(votes) > 1:
            timestamps = [vote.timestamp for vote in votes]
            time_diffs = [(t - min(timestamps)).total_seconds() for t in timestamps]
            max_diff = max(time_diffs) if time_diffs else 0
            # Normalize to 0-1 where 1 hour spread = 0.5 consistency
            metrics["temporal_consistency"] = max(0.0, 1.0 - (max_diff / 7200))  # 2 hours = full penalty
        else:
            metrics["temporal_consistency"] = 1.0
        
        # Evidence quality
        evidence_scores = [vote.evidence_quality for vote in votes if vote.evidence_quality > 0]
        if evidence_scores:
            metrics["evidence_quality"] = sum(evidence_scores) / len(evidence_scores)
        
        return metrics

    def _generate_vote_summary(self, collection: VoteCollection) -> Dict[str, Any]:
        """Generate summary of votes"""
        votes = collection.votes
        
        summary = {
            "total_votes": len(votes),
            "vote_type": collection.vote_type.value,
            "voting_mechanism": collection.voting_config.mechanism.value,
            "collection_duration_seconds": 0.0,
            "vote_distribution": {},
            "agent_breakdown": {}
        }
        
        if collection.collection_end:
            summary["collection_duration_seconds"] = (
                collection.collection_end - collection.collection_start
            ).total_seconds()
        
        # Vote distribution analysis
        if collection.vote_type == VoteType.BOOLEAN:
            true_count = sum(1 for v in votes if v.vote is True)
            false_count = len(votes) - true_count
            summary["vote_distribution"] = {"true": true_count, "false": false_count}
            
        elif collection.vote_type in [VoteType.NUMERIC, VoteType.WEIGHTED_SCORE]:
            numeric_votes = [float(v.vote) for v in votes if isinstance(v.vote, (int, float))]
            if numeric_votes:
                summary["vote_distribution"] = {
                    "min": min(numeric_votes),
                    "max": max(numeric_votes),
                    "mean": statistics.mean(numeric_votes),
                    "median": statistics.median(numeric_votes)
                }
                if len(numeric_votes) > 1:
                    summary["vote_distribution"]["stdev"] = statistics.stdev(numeric_votes)
        
        # Agent breakdown
        for vote in votes:
            summary["agent_breakdown"][vote.agent_id] = {
                "vote": vote.vote,
                "confidence": vote.confidence.value,
                "evidence_quality": vote.evidence_quality,
                "timestamp": vote.timestamp.isoformat()
            }
        
        return summary

    def _analyze_conflicts(self, conflicts: List[VoteConflict]) -> Dict[str, Any]:
        """Analyze detected conflicts"""
        if not conflicts:
            return {"total_conflicts": 0, "severity_distribution": {}, "resolution_needed": False}
        
        analysis = {
            "total_conflicts": len(conflicts),
            "conflict_types": {},
            "severity_distribution": {"low": 0, "medium": 0, "high": 0},
            "resolution_needed": any(c.severity > 0.7 for c in conflicts),
            "highest_severity": max(c.severity for c in conflicts),
            "affected_agents": list(set(agent for c in conflicts for agent in c.affected_agents))
        }
        
        # Count conflict types
        for conflict in conflicts:
            conflict_type = conflict.conflict_type.value
            analysis["conflict_types"][conflict_type] = analysis["conflict_types"].get(conflict_type, 0) + 1
        
        # Categorize by severity
        for conflict in conflicts:
            if conflict.severity < 0.4:
                analysis["severity_distribution"]["low"] += 1
            elif conflict.severity < 0.7:
                analysis["severity_distribution"]["medium"] += 1
            else:
                analysis["severity_distribution"]["high"] += 1
        
        return analysis

    def _generate_recommendations(self, collection: VoteCollection, 
                                 consensus_result: ConsensusResult,
                                 conflicts: List[VoteConflict]) -> List[str]:
        """Generate recommendations based on voting results"""
        recommendations = []
        
        # Low confidence recommendations
        if consensus_result.confidence_score < 0.5:
            recommendations.append("Consider gathering additional evidence before finalizing decision")
        
        # Low agreement recommendations
        if consensus_result.agreement_level < 0.7:
            recommendations.append("Low agreement detected - consider discussion or arbitration")
        
        # Conflict-specific recommendations
        for conflict in conflicts:
            if conflict.severity > 0.7:
                recommendations.append(f"Address {conflict.conflict_type.value}: {conflict.suggested_resolution}")
        
        # Participation recommendations
        if len(collection.votes) < collection.voting_config.minimum_votes * 1.5:
            recommendations.append("Consider involving more agents for better decision coverage")
        
        # Timeout recommendations
        if collection.deadline and datetime.now() > collection.deadline:
            recommendations.append("Future votes should allow more time for agent participation")
        
        return recommendations

    def _update_processing_metrics(self, processing_time: float):
        """Update processing time metrics"""
        total_decisions = self.metrics["total_decisions"]
        current_avg = self.metrics["average_processing_time"]
        
        self.metrics["average_processing_time"] = (
            (current_avg * (total_decisions - 1) + processing_time) / total_decisions
        )

    def _update_participation_metrics(self, collection: VoteCollection):
        """Update agent participation rate metrics"""
        for vote in collection.votes:
            agent_id = vote.agent_id
            if agent_id not in self.metrics["agent_participation_rates"]:
                self.metrics["agent_participation_rates"][agent_id] = 0.0
            
            # Simple participation tracking - could be enhanced with more sophisticated metrics
            self.metrics["agent_participation_rates"][agent_id] += 1

    def get_active_collections(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all active vote collections"""
        return {
            decision_id: {
                "title": collection.decision_title,
                "vote_type": collection.vote_type.value,
                "votes_collected": len(collection.votes),
                "minimum_required": collection.voting_config.minimum_votes,
                "deadline": collection.deadline.isoformat() if collection.deadline else None,
                "time_remaining": (collection.deadline - datetime.now()).total_seconds() 
                                 if collection.deadline else None
            }
            for decision_id, collection in self.active_collections.items()
        }

    def get_aggregator_metrics(self) -> Dict[str, Any]:
        """Get current aggregator performance metrics"""
        return {
            **self.metrics,
            "active_collections": len(self.active_collections),
            "completed_collections": len(self.completed_collections),
            "recent_aggregations": len([r for r in self.aggregation_history 
                                      if (datetime.now() - datetime.fromisoformat(r.metadata.get("timestamp", "2000-01-01"))).days < 7])
        }

    def export_aggregation_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Export recent aggregation history for analysis"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_reports = []
        for report in self.aggregation_history:
            # Note: This would need proper timestamp handling in real implementation
            recent_reports.append({
                "decision_id": report.decision_id,
                "consensus_decision": report.consensus_result.decision,
                "confidence_score": report.consensus_result.confidence_score,
                "agreement_level": report.consensus_result.agreement_level,
                "vote_count": report.metadata.get("vote_count", 0),
                "conflicts_detected": report.metadata.get("conflicts_detected", 0),
                "processing_time": report.processing_time_seconds,
                "quality_score": sum(report.quality_metrics.values()) / len(report.quality_metrics) 
                                if report.quality_metrics else 0.0
            })
        
        return recent_reports


def main():
    """Demonstration of voting aggregator functionality"""
    
    # Initialize components
    consensus = ConsensusArchitecture()
    aggregator = VotingAggregator(consensus)
    
    print("=== RIF Voting Aggregator Demo ===\n")
    
    # Example 1: Boolean vote with conflict detection
    print("Example 1: Boolean Vote - Should we implement feature X?")
    decision_id = "feature-x-approval"
    
    voting_config = consensus.voting_configs["weighted_voting"]
    collection = aggregator.start_vote_collection(
        decision_id=decision_id,
        decision_title="Implement Feature X",
        vote_type=VoteType.BOOLEAN,
        voting_config=voting_config,
        context={"domain": "general", "expected_agents": ["rif-implementer", "rif-validator", "rif-security"]},
        deadline_minutes=30
    )
    
    # Cast votes with split decision
    votes = [
        consensus.create_vote("rif-implementer", True, ConfidenceLevel.HIGH, "Feature is well-designed", 0.8),
        consensus.create_vote("rif-validator", False, ConfidenceLevel.MEDIUM, "Tests need more work", 0.6),
        consensus.create_vote("rif-security", True, ConfidenceLevel.LOW, "No major security issues", 0.4)
    ]
    
    for vote in votes:
        aggregator.cast_vote(decision_id, vote)
    
    # Aggregate votes
    report = aggregator.aggregate_votes(decision_id, force_completion=True)
    
    print(f"Decision: {report.consensus_result.decision}")
    print(f"Confidence: {report.consensus_result.confidence_score:.2f}")
    print(f"Agreement: {report.consensus_result.agreement_level:.2f}")
    print(f"Conflicts detected: {len(report.conflict_analysis.get('conflict_types', {}))}")
    print(f"Recommendations: {len(report.recommendations)}")
    for rec in report.recommendations:
        print(f"  - {rec}")
    print()
    
    # Example 2: Numeric vote for quality assessment
    print("Example 2: Numeric Vote - Code Quality Score (0.0-1.0)")
    decision_id = "code-quality-assessment"
    
    voting_config = consensus.voting_configs["simple_majority"]
    aggregator.start_vote_collection(
        decision_id=decision_id,
        decision_title="Code Quality Assessment",
        vote_type=VoteType.NUMERIC,
        voting_config=voting_config,
        context={"domain": "quality"}
    )
    
    # Cast numeric votes with outlier
    quality_votes = [
        consensus.create_vote("rif-validator", 0.85, ConfidenceLevel.HIGH, "Good test coverage", 0.9),
        consensus.create_vote("rif-implementer", 0.80, ConfidenceLevel.MEDIUM, "Code is clean", 0.7),
        consensus.create_vote("rif-security", 0.40, ConfidenceLevel.VERY_HIGH, "Security concerns", 0.95)  # Outlier
    ]
    
    for vote in quality_votes:
        aggregator.cast_vote(decision_id, vote)
    
    report = aggregator.aggregate_votes(decision_id, force_completion=True)
    
    print(f"Average Score: {report.consensus_result.decision}")
    print(f"Confidence: {report.consensus_result.confidence_score:.2f}")
    print(f"Quality Metrics: {json.dumps(report.quality_metrics, indent=2)}")
    print(f"Vote Summary: {json.dumps(report.vote_summary['vote_distribution'], indent=2)}")
    
    if report.conflict_analysis['total_conflicts'] > 0:
        print(f"Conflicts: {report.conflict_analysis}")
    print()
    
    # Show aggregator metrics
    print("=== Aggregator Metrics ===")
    metrics = aggregator.get_aggregator_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()