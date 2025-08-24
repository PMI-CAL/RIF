#!/usr/bin/env python3
"""
RIF Conflict Detection System
Advanced conflict detection and analysis for agent consensus disagreements.
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import Counter

# Import consensus components
import sys
sys.path.insert(0, '/Users/cal/DEV/RIF/claude/commands')

from consensus_architecture import AgentVote, ConfidenceLevel
from voting_aggregator import VoteConflict, ConflictType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConflictPattern(Enum):
    """Types of conflict patterns"""
    POLARIZED = "polarized"              # Clear split between options
    SCATTERED = "scattered"              # Many different opinions
    UNCERTAIN = "uncertain"              # Low confidence across board
    EXPERTISE_GAP = "expertise_gap"      # Missing domain experts
    TEMPORAL_DRIFT = "temporal_drift"    # Opinions changed over time
    EVIDENCE_QUALITY = "evidence_quality" # Poor evidence supporting votes

@dataclass
class ConflictAnalysis:
    """Comprehensive analysis of voting conflicts"""
    conflict_id: str
    primary_conflict_type: ConflictType
    severity: ConflictSeverity
    confidence_impact: float
    patterns_detected: List[ConflictPattern]
    affected_agents: List[str]
    root_causes: List[str]
    resolution_recommendations: List[str]
    metrics: Dict[str, float]
    evidence_summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConflictMetrics:
    """Detailed metrics for conflict analysis"""
    vote_distribution_entropy: float
    confidence_variance: float
    evidence_quality_range: float
    expertise_coverage: float
    temporal_consistency: float
    reasoning_similarity: float
    outlier_count: int
    consensus_distance: float

class ConflictDetector:
    """
    Advanced conflict detection system that identifies and analyzes voting disagreements
    """
    
    def __init__(self, agent_expertise: Dict[str, Dict[str, float]] = None):
        """Initialize conflict detector"""
        self.agent_expertise = agent_expertise or {}
        
        # Detection thresholds
        self.thresholds = {
            "split_decision": 0.4,          # Vote distribution threshold
            "low_confidence": 0.4,          # Average confidence threshold
            "outlier_detection": 2.0,       # Standard deviations for outliers
            "expertise_coverage": 0.7,      # Required expertise coverage
            "evidence_quality": 0.6,        # Minimum evidence quality
            "temporal_consistency": 0.8,    # Time-based consistency threshold
            "reasoning_similarity": 0.3     # Reasoning similarity threshold
        }
        
        # Pattern detection settings
        self.pattern_weights = {
            ConflictPattern.POLARIZED: 1.0,
            ConflictPattern.SCATTERED: 0.8,
            ConflictPattern.UNCERTAIN: 0.9,
            ConflictPattern.EXPERTISE_GAP: 1.2,
            ConflictPattern.TEMPORAL_DRIFT: 0.7,
            ConflictPattern.EVIDENCE_QUALITY: 1.1
        }
        
        # Performance tracking
        self.detection_metrics = {
            "total_analyses": 0,
            "conflicts_detected": 0,
            "pattern_distribution": {pattern.value: 0 for pattern in ConflictPattern},
            "severity_distribution": {severity.value: 0 for severity in ConflictSeverity},
            "average_analysis_time": 0.0
        }

    def analyze_conflicts(self, votes: List[AgentVote], context: Dict[str, Any],
                         existing_conflicts: List[VoteConflict] = None) -> ConflictAnalysis:
        """
        Perform comprehensive conflict analysis on votes
        
        Args:
            votes: List of agent votes to analyze
            context: Decision context and metadata
            existing_conflicts: Previously detected basic conflicts
            
        Returns:
            ConflictAnalysis: Detailed conflict analysis
        """
        start_time = datetime.now()
        
        if not votes:
            return self._create_empty_analysis()
        
        conflict_id = f"conflict-{int(start_time.timestamp())}-{len(votes)}"
        
        # Step 1: Calculate conflict metrics
        metrics = self._calculate_conflict_metrics(votes, context)
        
        # Step 2: Identify conflict patterns
        patterns = self._detect_conflict_patterns(votes, metrics, context)
        
        # Step 3: Determine primary conflict type and severity
        primary_conflict, severity = self._classify_conflict(votes, patterns, metrics, existing_conflicts)
        
        # Step 4: Analyze root causes
        root_causes = self._identify_root_causes(votes, patterns, metrics, context)
        
        # Step 5: Generate resolution recommendations
        recommendations = self._generate_resolution_recommendations(patterns, metrics, context)
        
        # Step 6: Calculate confidence impact
        confidence_impact = self._calculate_confidence_impact(votes, patterns, metrics)
        
        # Step 7: Create evidence summary
        evidence_summary = self._create_evidence_summary(votes, metrics, patterns)
        
        # Create comprehensive analysis
        analysis = ConflictAnalysis(
            conflict_id=conflict_id,
            primary_conflict_type=primary_conflict,
            severity=severity,
            confidence_impact=confidence_impact,
            patterns_detected=patterns,
            affected_agents=[vote.agent_id for vote in votes],
            root_causes=root_causes,
            resolution_recommendations=recommendations,
            metrics=self._metrics_to_dict(metrics),
            evidence_summary=evidence_summary
        )
        
        # Update detection metrics
        self._update_detection_metrics(analysis, (datetime.now() - start_time).total_seconds())
        
        logger.info(f"Conflict analysis completed: {conflict_id} - {severity.value} severity")
        
        return analysis

    def _calculate_conflict_metrics(self, votes: List[AgentVote], context: Dict[str, Any]) -> ConflictMetrics:
        """Calculate detailed metrics for conflict analysis"""
        
        # Vote distribution entropy
        vote_values = [vote.vote for vote in votes]
        vote_distribution_entropy = self._calculate_entropy(vote_values)
        
        # Confidence variance
        confidence_scores = [self._confidence_to_score(vote.confidence) for vote in votes]
        confidence_variance = statistics.variance(confidence_scores) if len(confidence_scores) > 1 else 0.0
        
        # Evidence quality range
        evidence_qualities = [vote.evidence_quality for vote in votes]
        evidence_quality_range = max(evidence_qualities) - min(evidence_qualities) if evidence_qualities else 0.0
        
        # Expertise coverage
        domain = context.get("domain", "general")
        expertise_coverage = self._calculate_expertise_coverage(votes, domain)
        
        # Temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(votes)
        
        # Reasoning similarity
        reasoning_similarity = self._calculate_reasoning_similarity(votes)
        
        # Outlier count
        outlier_count = self._count_outliers(votes)
        
        # Consensus distance (how far from consensus are we)
        consensus_distance = self._calculate_consensus_distance(votes)
        
        return ConflictMetrics(
            vote_distribution_entropy=vote_distribution_entropy,
            confidence_variance=confidence_variance,
            evidence_quality_range=evidence_quality_range,
            expertise_coverage=expertise_coverage,
            temporal_consistency=temporal_consistency,
            reasoning_similarity=reasoning_similarity,
            outlier_count=outlier_count,
            consensus_distance=consensus_distance
        )

    def _detect_conflict_patterns(self, votes: List[AgentVote], metrics: ConflictMetrics, 
                                context: Dict[str, Any]) -> List[ConflictPattern]:
        """Detect specific conflict patterns in the votes"""
        patterns = []
        
        # Polarized pattern - clear split between two main options
        if self._is_polarized_conflict(votes, metrics):
            patterns.append(ConflictPattern.POLARIZED)
        
        # Scattered pattern - many different opinions
        if self._is_scattered_conflict(votes, metrics):
            patterns.append(ConflictPattern.SCATTERED)
        
        # Uncertain pattern - low confidence across the board
        if metrics.confidence_variance < 0.1 and max(self._confidence_to_score(vote.confidence) for vote in votes) < self.thresholds["low_confidence"]:
            patterns.append(ConflictPattern.UNCERTAIN)
        
        # Expertise gap pattern - missing domain experts
        if metrics.expertise_coverage < self.thresholds["expertise_coverage"]:
            patterns.append(ConflictPattern.EXPERTISE_GAP)
        
        # Temporal drift pattern - opinions changed over time
        if metrics.temporal_consistency < self.thresholds["temporal_consistency"]:
            patterns.append(ConflictPattern.TEMPORAL_DRIFT)
        
        # Evidence quality pattern - poor evidence supporting votes
        if min(vote.evidence_quality for vote in votes) < self.thresholds["evidence_quality"]:
            patterns.append(ConflictPattern.EVIDENCE_QUALITY)
        
        return patterns

    def _classify_conflict(self, votes: List[AgentVote], patterns: List[ConflictPattern],
                          metrics: ConflictMetrics, existing_conflicts: List[VoteConflict] = None) -> Tuple[ConflictType, ConflictSeverity]:
        """Classify the primary conflict type and determine severity"""
        
        # Determine primary conflict type
        if existing_conflicts:
            # Use the most severe existing conflict as primary
            primary_conflict = max(existing_conflicts, key=lambda c: c.severity).conflict_type
        else:
            # Infer primary conflict type from patterns
            if ConflictPattern.POLARIZED in patterns:
                primary_conflict = ConflictType.SPLIT_DECISION
            elif ConflictPattern.UNCERTAIN in patterns:
                primary_conflict = ConflictType.LOW_CONFIDENCE
            elif ConflictPattern.EXPERTISE_GAP in patterns:
                primary_conflict = ConflictType.MISSING_EXPERTISE
            elif ConflictPattern.SCATTERED in patterns and metrics.outlier_count > 0:
                primary_conflict = ConflictType.OUTLIER_DETECTED
            else:
                primary_conflict = ConflictType.SPLIT_DECISION  # Default
        
        # Calculate severity based on multiple factors
        severity_score = 0.0
        
        # Base severity from metrics
        severity_score += metrics.vote_distribution_entropy * 0.3
        severity_score += metrics.confidence_variance * 0.2
        severity_score += metrics.consensus_distance * 0.2
        severity_score += (1.0 - metrics.expertise_coverage) * 0.15
        severity_score += metrics.evidence_quality_range * 0.15
        
        # Pattern-based severity adjustments
        for pattern in patterns:
            severity_score += self.pattern_weights.get(pattern, 0.0) * 0.1
        
        # Normalize to 0-1 range
        severity_score = min(1.0, max(0.0, severity_score))
        
        # Convert to severity enum
        if severity_score < 0.3:
            severity = ConflictSeverity.LOW
        elif severity_score < 0.6:
            severity = ConflictSeverity.MEDIUM
        elif severity_score < 0.8:
            severity = ConflictSeverity.HIGH
        else:
            severity = ConflictSeverity.CRITICAL
        
        return primary_conflict, severity

    def _identify_root_causes(self, votes: List[AgentVote], patterns: List[ConflictPattern],
                            metrics: ConflictMetrics, context: Dict[str, Any]) -> List[str]:
        """Identify root causes of the conflict"""
        root_causes = []
        
        # Pattern-based root causes
        if ConflictPattern.POLARIZED in patterns:
            root_causes.append("Fundamental disagreement between agent assessments")
        
        if ConflictPattern.UNCERTAIN in patterns:
            root_causes.append("Insufficient information for confident decision making")
        
        if ConflictPattern.EXPERTISE_GAP in patterns:
            root_causes.append(f"Missing domain expertise for {context.get('domain', 'general')} decisions")
        
        if ConflictPattern.EVIDENCE_QUALITY in patterns:
            root_causes.append("Poor quality or inconsistent evidence supporting votes")
        
        if ConflictPattern.TEMPORAL_DRIFT in patterns:
            root_causes.append("Changing conditions or information over time")
        
        # Metric-based root causes
        if metrics.confidence_variance > 0.3:
            root_causes.append("High variance in agent confidence levels")
        
        if metrics.outlier_count > 0:
            root_causes.append(f"Outlier opinions from {metrics.outlier_count} agents")
        
        if metrics.reasoning_similarity < self.thresholds["reasoning_similarity"]:
            root_causes.append("Agents using different reasoning approaches")
        
        # Context-based root causes
        if context.get("security_critical", False):
            root_causes.append("Security-critical decision requiring heightened scrutiny")
        
        if context.get("risk_level") == "high":
            root_causes.append("High-risk decision with naturally higher disagreement threshold")
        
        return root_causes if root_causes else ["Unknown conflict source - requires deeper analysis"]

    def _generate_resolution_recommendations(self, patterns: List[ConflictPattern], 
                                           metrics: ConflictMetrics, context: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for resolving the conflict"""
        recommendations = []
        
        # Pattern-based recommendations
        if ConflictPattern.POLARIZED in patterns:
            recommendations.append("Consider evidence-based arbitration to break the tie")
            recommendations.append("Seek additional expert opinion for tie-breaking")
        
        if ConflictPattern.UNCERTAIN in patterns:
            recommendations.append("Gather additional evidence before making final decision")
            recommendations.append("Consider postponing decision until more information is available")
        
        if ConflictPattern.EXPERTISE_GAP in patterns:
            recommendations.append(f"Involve domain experts for {context.get('domain', 'general')} expertise")
            recommendations.append("Wait for expert agent consultation before proceeding")
        
        if ConflictPattern.EVIDENCE_QUALITY in patterns:
            recommendations.append("Require higher evidence standards before accepting votes")
            recommendations.append("Re-evaluate evidence quality and request supporting documentation")
        
        if ConflictPattern.TEMPORAL_DRIFT in patterns:
            recommendations.append("Re-synchronize agent assessments with current conditions")
            recommendations.append("Consider time-boxed re-voting to address temporal inconsistencies")
        
        # Metric-based recommendations
        if metrics.outlier_count > 0:
            recommendations.append("Investigate outlier reasoning for potential valuable insights")
            recommendations.append("Consider weighted voting to minimize outlier impact")
        
        if metrics.confidence_variance > 0.4:
            recommendations.append("Focus on building consensus among high-confidence agents")
            recommendations.append("Investigate sources of confidence disparities")
        
        # Context-based recommendations
        if context.get("security_critical", False):
            recommendations.append("Escalate to security expert for final validation")
            recommendations.append("Require unanimous agreement for security-critical decisions")
        
        # Default recommendations if none specific
        if not recommendations:
            recommendations.append("Consider weighted voting resolution")
            recommendations.append("Escalate to arbitrator agent if patterns persist")
        
        return recommendations

    def _calculate_confidence_impact(self, votes: List[AgentVote], patterns: List[ConflictPattern],
                                   metrics: ConflictMetrics) -> float:
        """Calculate how much the conflict impacts overall decision confidence"""
        base_confidence = sum(self._confidence_to_score(vote.confidence) for vote in votes) / len(votes)
        
        # Pattern-based confidence reduction
        confidence_reduction = 0.0
        
        if ConflictPattern.POLARIZED in patterns:
            confidence_reduction += 0.3
        
        if ConflictPattern.UNCERTAIN in patterns:
            confidence_reduction += 0.4
        
        if ConflictPattern.EXPERTISE_GAP in patterns:
            confidence_reduction += 0.2
        
        if ConflictPattern.EVIDENCE_QUALITY in patterns:
            confidence_reduction += 0.25
        
        # Metric-based adjustments
        confidence_reduction += metrics.confidence_variance * 0.3
        confidence_reduction += metrics.consensus_distance * 0.2
        confidence_reduction += (metrics.outlier_count / len(votes)) * 0.1
        
        # Calculate final confidence impact (how much confidence is reduced)
        confidence_impact = min(1.0, confidence_reduction)
        
        return confidence_impact

    def _create_evidence_summary(self, votes: List[AgentVote], metrics: ConflictMetrics,
                               patterns: List[ConflictPattern]) -> Dict[str, Any]:
        """Create comprehensive evidence summary"""
        return {
            "vote_summary": {
                "total_votes": len(votes),
                "unique_positions": len(set(vote.vote for vote in votes)),
                "confidence_distribution": {
                    level.value: sum(1 for vote in votes if vote.confidence == level)
                    for level in ConfidenceLevel
                },
                "evidence_quality_stats": {
                    "min": min(vote.evidence_quality for vote in votes),
                    "max": max(vote.evidence_quality for vote in votes),
                    "avg": sum(vote.evidence_quality for vote in votes) / len(votes)
                }
            },
            "conflict_characteristics": {
                "patterns_detected": [p.value for p in patterns],
                "entropy": metrics.vote_distribution_entropy,
                "consensus_distance": metrics.consensus_distance,
                "outlier_impact": metrics.outlier_count / len(votes) if votes else 0
            },
            "agent_analysis": [
                {
                    "agent_id": vote.agent_id,
                    "vote": vote.vote,
                    "confidence": vote.confidence.value,
                    "evidence_quality": vote.evidence_quality,
                    "reasoning_length": len(vote.reasoning),
                    "is_outlier": self._is_vote_outlier(vote, votes)
                }
                for vote in votes
            ]
        }

    # Helper methods for pattern detection
    def _is_polarized_conflict(self, votes: List[AgentVote], metrics: ConflictMetrics) -> bool:
        """Check if conflict shows polarization pattern"""
        vote_counts = Counter(vote.vote for vote in votes)
        
        if len(vote_counts) != 2:  # Not polarized if more than 2 positions
            return False
        
        values = list(vote_counts.values())
        total_votes = sum(values)
        
        # Polarized if both sides have significant representation
        min_side_ratio = min(values) / total_votes
        return 0.3 <= min_side_ratio <= 0.7  # Both sides have at least 30% representation

    def _is_scattered_conflict(self, votes: List[AgentVote], metrics: ConflictMetrics) -> bool:
        """Check if conflict shows scattered pattern"""
        unique_positions = len(set(vote.vote for vote in votes))
        
        # Scattered if many different positions relative to vote count
        scatter_ratio = unique_positions / len(votes)
        return scatter_ratio > 0.5 and metrics.vote_distribution_entropy > 1.5

    # Helper methods for calculations
    def _calculate_entropy(self, values: List[Any]) -> float:
        """Calculate Shannon entropy of value distribution"""
        if not values:
            return 0.0
        
        value_counts = Counter(values)
        total_count = len(values)
        
        entropy = 0.0
        for count in value_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy

    def _calculate_expertise_coverage(self, votes: List[AgentVote], domain: str) -> float:
        """Calculate expertise coverage for the domain"""
        if not votes:
            return 0.0
        
        expertise_scores = []
        for vote in votes:
            agent_expertise = self.agent_expertise.get(vote.agent_id, {})
            domain_expertise = agent_expertise.get(domain, 0.5)  # Default to medium expertise
            expertise_scores.append(domain_expertise)
        
        # Coverage is the average expertise level
        return sum(expertise_scores) / len(expertise_scores)

    def _calculate_temporal_consistency(self, votes: List[AgentVote]) -> float:
        """Calculate consistency of votes over time"""
        if len(votes) <= 1:
            return 1.0
        
        timestamps = [vote.timestamp for vote in votes]
        time_diffs = [(t - min(timestamps)).total_seconds() for t in timestamps]
        max_diff = max(time_diffs)
        
        # Consistency decreases as time span increases (normalize by 1 hour)
        consistency = max(0.0, 1.0 - (max_diff / 3600))
        return consistency

    def _calculate_reasoning_similarity(self, votes: List[AgentVote]) -> float:
        """Calculate similarity of reasoning approaches"""
        reasonings = [vote.reasoning.lower() for vote in votes]
        
        if len(reasonings) <= 1:
            return 1.0
        
        # Simple similarity based on common words (could be enhanced with NLP)
        all_words = set()
        for reasoning in reasonings:
            all_words.update(reasoning.split())
        
        if not all_words:
            return 0.0
        
        similarities = []
        for i in range(len(reasonings)):
            for j in range(i + 1, len(reasonings)):
                words_i = set(reasonings[i].split())
                words_j = set(reasonings[j].split())
                
                if not words_i or not words_j:
                    similarity = 0.0
                else:
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _count_outliers(self, votes: List[AgentVote]) -> int:
        """Count outlier votes using statistical methods"""
        if len(votes) < 3:
            return 0
        
        # For numeric votes, use statistical outlier detection
        numeric_votes = []
        for vote in votes:
            if isinstance(vote.vote, (int, float)):
                numeric_votes.append(float(vote.vote))
        
        if len(numeric_votes) < 3:
            return 0
        
        mean_val = statistics.mean(numeric_votes)
        if len(numeric_votes) > 1:
            stdev = statistics.stdev(numeric_votes)
            threshold = self.thresholds["outlier_detection"]
            
            outliers = [v for v in numeric_votes if abs(v - mean_val) > threshold * stdev]
            return len(outliers)
        
        return 0

    def _calculate_consensus_distance(self, votes: List[AgentVote]) -> float:
        """Calculate how far we are from consensus"""
        if not votes:
            return 1.0
        
        vote_counts = Counter(vote.vote for vote in votes)
        max_agreement = max(vote_counts.values())
        total_votes = len(votes)
        
        agreement_ratio = max_agreement / total_votes
        consensus_distance = 1.0 - agreement_ratio
        
        return consensus_distance

    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numerical score"""
        confidence_map = {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 1.0
        }
        return confidence_map.get(confidence, 0.5)

    def _is_vote_outlier(self, vote: AgentVote, all_votes: List[AgentVote]) -> bool:
        """Check if a specific vote is an outlier"""
        if len(all_votes) < 3:
            return False
        
        # Simple outlier check based on vote value
        vote_counts = Counter(v.vote for v in all_votes)
        return vote_counts[vote.vote] == 1 and len(vote_counts) > 2

    def _metrics_to_dict(self, metrics: ConflictMetrics) -> Dict[str, float]:
        """Convert metrics object to dictionary"""
        return {
            "vote_distribution_entropy": metrics.vote_distribution_entropy,
            "confidence_variance": metrics.confidence_variance,
            "evidence_quality_range": metrics.evidence_quality_range,
            "expertise_coverage": metrics.expertise_coverage,
            "temporal_consistency": metrics.temporal_consistency,
            "reasoning_similarity": metrics.reasoning_similarity,
            "outlier_count": float(metrics.outlier_count),
            "consensus_distance": metrics.consensus_distance
        }

    def _create_empty_analysis(self) -> ConflictAnalysis:
        """Create empty analysis for no-vote scenarios"""
        return ConflictAnalysis(
            conflict_id="no-conflict",
            primary_conflict_type=ConflictType.SPLIT_DECISION,
            severity=ConflictSeverity.LOW,
            confidence_impact=0.0,
            patterns_detected=[],
            affected_agents=[],
            root_causes=["No votes to analyze"],
            resolution_recommendations=["Collect votes before analysis"],
            metrics={},
            evidence_summary={}
        )

    def _update_detection_metrics(self, analysis: ConflictAnalysis, processing_time: float):
        """Update detection system metrics"""
        self.detection_metrics["total_analyses"] += 1
        
        if analysis.severity != ConflictSeverity.LOW:
            self.detection_metrics["conflicts_detected"] += 1
        
        # Update pattern distribution
        for pattern in analysis.patterns_detected:
            self.detection_metrics["pattern_distribution"][pattern.value] += 1
        
        # Update severity distribution
        self.detection_metrics["severity_distribution"][analysis.severity.value] += 1
        
        # Update average analysis time
        total_analyses = self.detection_metrics["total_analyses"]
        current_avg = self.detection_metrics["average_analysis_time"]
        self.detection_metrics["average_analysis_time"] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )

    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get conflict detection performance metrics"""
        return {
            **self.detection_metrics,
            "detection_rate": (
                self.detection_metrics["conflicts_detected"] / 
                max(1, self.detection_metrics["total_analyses"])
            ),
            "thresholds": self.thresholds,
            "pattern_weights": {k.value: v for k, v in self.pattern_weights.items()}
        }


def main():
    """Demonstration of conflict detection functionality"""
    print("=== RIF Conflict Detection System Demo ===\n")
    
    # Initialize system
    from consensus_architecture import ConsensusArchitecture
    consensus = ConsensusArchitecture()
    detector = ConflictDetector(consensus.agent_expertise)
    
    # Example 1: Polarized conflict
    print("Example 1: Polarized Conflict Detection")
    votes = [
        consensus.create_vote("rif-implementer", True, ConfidenceLevel.HIGH, "Implementation is ready", 0.8),
        consensus.create_vote("rif-validator", False, ConfidenceLevel.HIGH, "Tests are failing", 0.9),
        consensus.create_vote("rif-security", True, ConfidenceLevel.MEDIUM, "Security looks OK", 0.6),
        consensus.create_vote("rif-analyst", False, ConfidenceLevel.MEDIUM, "Requirements unclear", 0.5)
    ]
    
    context = {"domain": "implementation", "risk_level": "medium"}
    analysis = detector.analyze_conflicts(votes, context)
    
    print(f"Conflict ID: {analysis.conflict_id}")
    print(f"Primary Type: {analysis.primary_conflict_type.value}")
    print(f"Severity: {analysis.severity.value}")
    print(f"Patterns: {[p.value for p in analysis.patterns_detected]}")
    print(f"Confidence Impact: {analysis.confidence_impact:.2f}")
    print(f"Root Causes: {len(analysis.root_causes)}")
    for cause in analysis.root_causes:
        print(f"  - {cause}")
    print(f"Recommendations: {len(analysis.resolution_recommendations)}")
    for rec in analysis.resolution_recommendations[:3]:  # Show first 3
        print(f"  - {rec}")
    print()
    
    # Example 2: Low confidence conflict
    print("Example 2: Low Confidence Conflict")
    low_confidence_votes = [
        consensus.create_vote("agent1", True, ConfidenceLevel.LOW, "Uncertain about this", 0.3),
        consensus.create_vote("agent2", False, ConfidenceLevel.LOW, "Not sure either way", 0.2),
        consensus.create_vote("agent3", True, ConfidenceLevel.LOW, "Maybe okay", 0.4)
    ]
    
    uncertainty_context = {"domain": "general", "risk_level": "low"}
    uncertainty_analysis = detector.analyze_conflicts(low_confidence_votes, uncertainty_context)
    
    print(f"Conflict ID: {uncertainty_analysis.conflict_id}")
    print(f"Severity: {uncertainty_analysis.severity.value}")
    print(f"Patterns: {[p.value for p in uncertainty_analysis.patterns_detected]}")
    print(f"Key Metrics:")
    for key, value in uncertainty_analysis.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Show detector metrics
    print("=== Detection System Metrics ===")
    metrics = detector.get_detection_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()