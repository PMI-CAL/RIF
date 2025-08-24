#!/usr/bin/env python3
"""
RIF Weighted Voting Integration - Issue #62
Integration layer connecting VoteWeightCalculator with VotingAggregator.

This module provides seamless integration between the sophisticated vote weighting
algorithm and the existing voting aggregator system.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from claude.commands.voting_aggregator import VotingAggregator, VoteCollection, AggregationReport
    from claude.commands.consensus_architecture import AgentVote, VotingConfig, ConsensusResult
except ImportError:
    # Fallback for testing - define minimal classes
    from datetime import datetime
    from typing import Union
    from enum import Enum
    
    class ConfidenceLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        VERY_HIGH = "very_high"
    
    @dataclass
    class AgentVote:
        agent_id: str
        vote: Union[bool, float, str]
        confidence: ConfidenceLevel
        reasoning: str
        timestamp: datetime
        evidence_quality: float = 0.0
        expertise_score: float = 1.0
    
    @dataclass
    class VotingConfig:
        mechanism: str
        threshold: float
        weights: Dict[str, float]
        use_cases: List[str]
        timeout_minutes: int = 30
        minimum_votes: int = 1
        allow_abstentions: bool = True
    
    @dataclass 
    class ConsensusResult:
        decision: Union[bool, float, str]
        confidence_score: float
        vote_count: int
        agreement_level: float
        mechanism_used: str
        arbitration_triggered: bool = False
        evidence_summary: Dict[str, Any] = None
        timestamp: datetime = None
    
    @dataclass
    class AggregationReport:
        decision_id: str
        consensus_result: ConsensusResult
        vote_summary: Dict[str, Any]
        conflict_analysis: Dict[str, Any]
        quality_metrics: Dict[str, float]
        processing_time_seconds: float
        recommendations: List[str]
        metadata: Dict[str, Any] = field(default_factory=dict)

from knowledge.consensus.vote_weight_calculator import VoteWeightCalculator, WeightingStrategy
from knowledge.consensus.expertise_scorer import ExpertiseScorer
from knowledge.consensus.accuracy_tracker import AccuracyTracker, DecisionRecord, DecisionOutcome
from knowledge.consensus.confidence_adjuster import ConfidenceAdjuster, ConfidenceRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WeightedVotingConfig:
    """Configuration for weighted voting integration"""
    weighting_strategy: WeightingStrategy = WeightingStrategy.BALANCED
    enable_expertise_scoring: bool = True
    enable_accuracy_tracking: bool = True
    enable_confidence_adjustment: bool = True
    update_weights_on_outcome: bool = True
    weight_normalization_method: str = "ensemble"  # "individual" or "ensemble"
    min_weight_threshold: float = 0.1
    max_weight_threshold: float = 3.0


class WeightedVotingAggregator:
    """
    Enhanced voting aggregator with sophisticated vote weighting.
    
    This class extends the basic voting aggregator with the advanced
    vote weighting algorithm from Issue #62.
    """
    
    def __init__(self, consensus_architecture=None, 
                 weighting_config: WeightedVotingConfig = None):
        """Initialize weighted voting aggregator"""
        
        # Initialize base voting aggregator
        try:
            self.base_aggregator = VotingAggregator(consensus_architecture)
        except:
            logger.warning("Could not initialize base VotingAggregator - using mock")
            self.base_aggregator = None
        
        # Initialize weighting components
        self.weighting_config = weighting_config or WeightedVotingConfig()
        
        self.weight_calculator = VoteWeightCalculator(
            strategy=self.weighting_config.weighting_strategy
        )
        
        if self.weighting_config.enable_expertise_scoring:
            self.expertise_scorer = ExpertiseScorer()
        else:
            self.expertise_scorer = None
        
        if self.weighting_config.enable_accuracy_tracking:
            self.accuracy_tracker = AccuracyTracker()
        else:
            self.accuracy_tracker = None
            
        if self.weighting_config.enable_confidence_adjustment:
            self.confidence_adjuster = ConfidenceAdjuster()
        else:
            self.confidence_adjuster = None
        
        # Integration metrics
        self.integration_metrics = {
            'weighted_decisions': 0,
            'weight_adjustments_made': 0,
            'accuracy_improvements': 0,
            'confidence_calibrations': 0,
            'average_weight_variance': 0.0
        }
        
        logger.info("Weighted Voting Aggregator initialized with advanced vote weighting")
    
    def aggregate_votes_with_weighting(self, decision_id: str, 
                                     votes: List[AgentVote],
                                     voting_config: VotingConfig,
                                     context: Dict[str, Any]) -> AggregationReport:
        """
        Aggregate votes with sophisticated weighting algorithm.
        
        Args:
            decision_id: Unique identifier for the decision
            votes: List of agent votes
            voting_config: Voting configuration
            context: Decision context
            
        Returns:
            AggregationReport with weighted consensus results
        """
        try:
            start_time = datetime.now()
            
            # Calculate sophisticated weights for all voting agents
            agent_weights = self._calculate_sophisticated_weights(votes, context)
            
            # Create enhanced voting configuration with calculated weights
            enhanced_config = self._create_enhanced_voting_config(voting_config, agent_weights)
            
            # Perform weighted consensus calculation
            if self.base_aggregator:
                # Use existing aggregator with enhanced weights
                consensus_result = self.base_aggregator.consensus.calculate_consensus(
                    votes, enhanced_config, context
                )
            else:
                # Fallback to direct calculation
                consensus_result = self._calculate_weighted_consensus_direct(
                    votes, enhanced_config, context
                )
            
            # Generate comprehensive report
            report = self._generate_weighted_aggregation_report(
                decision_id, votes, consensus_result, agent_weights, context, start_time
            )
            
            # Update learning systems if outcome tracking enabled
            if self.weighting_config.update_weights_on_outcome:
                self._prepare_outcome_tracking(decision_id, votes, agent_weights, context)
            
            # Update metrics
            self._update_integration_metrics(agent_weights)
            
            logger.info(f"Weighted vote aggregation completed for {decision_id}: "
                       f"{consensus_result.decision} (weighted confidence: {consensus_result.confidence_score:.2f})")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in weighted vote aggregation for {decision_id}: {str(e)}")
            # Fallback to basic aggregation if available
            if self.base_aggregator:
                return self.base_aggregator.aggregate_votes(decision_id, force_completion=True)
            else:
                raise
    
    def update_decision_outcome(self, decision_id: str, outcome: bool, 
                              outcome_details: Dict[str, Any] = None) -> None:
        """
        Update learning systems with decision outcome.
        
        Args:
            decision_id: Decision identifier
            outcome: True if decision was correct/successful
            outcome_details: Additional outcome information
        """
        if not self.weighting_config.update_weights_on_outcome:
            return
        
        # Retrieve stored decision information
        decision_info = self._get_stored_decision_info(decision_id)
        if not decision_info:
            logger.warning(f"No stored information found for decision {decision_id}")
            return
        
        votes = decision_info['votes']
        context = decision_info['context']
        
        # Update accuracy tracking
        if self.accuracy_tracker:
            for vote in votes:
                decision_record = DecisionRecord(
                    decision_id=decision_id,
                    agent_id=vote.agent_id,
                    decision_timestamp=vote.timestamp,
                    decision_confidence=self._extract_confidence_from_vote(vote),
                    decision_content=vote.reasoning,
                    outcome=DecisionOutcome.SUCCESS if outcome else DecisionOutcome.FAILURE,
                    outcome_timestamp=datetime.now(),
                    context_category=context.get('domain', 'general'),
                    impact_level=context.get('impact_level', 'medium')
                )
                self.accuracy_tracker.record_decision_outcome(decision_record)
        
        # Update confidence calibration
        if self.confidence_adjuster:
            for vote in votes:
                confidence_record = ConfidenceRecord(
                    agent_id=vote.agent_id,
                    decision_id=decision_id,
                    stated_confidence=self._extract_confidence_from_vote(vote),
                    actual_outcome=1.0 if outcome else 0.0,
                    context_complexity=context.get('complexity', 'medium'),
                    domain=context.get('domain', 'general'),
                    timestamp=vote.timestamp
                )
                self.confidence_adjuster.record_confidence_outcome(confidence_record)
        
        # Update vote weight calculator accuracy records
        for vote in votes:
            self.weight_calculator.update_agent_accuracy(
                vote.agent_id, outcome, context
            )
        
        logger.info(f"Updated learning systems with outcome for decision {decision_id}: {outcome}")
    
    def get_agent_weight_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive weight profile for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with agent weight profile
        """
        profile = {
            'agent_id': agent_id,
            'weight_calculator_profile': {},
            'expertise_profile': {},
            'accuracy_profile': {},
            'confidence_profile': {}
        }
        
        # Get weight calculator metrics
        try:
            # Calculate weights for common contexts
            common_contexts = [
                {'domain': 'general', 'confidence': 0.8},
                {'domain': 'security', 'confidence': 0.8},
                {'domain': 'testing', 'confidence': 0.8},
                {'domain': 'architecture', 'confidence': 0.8}
            ]
            
            weight_samples = {}
            for context in common_contexts:
                weight = self.weight_calculator.calculate_weight(agent_id, context)
                weight_samples[context['domain']] = weight
            
            profile['weight_calculator_profile'] = {
                'domain_weights': weight_samples,
                'strategy': self.weight_calculator.strategy.value
            }
        except Exception as e:
            logger.debug(f"Error getting weight calculator profile for {agent_id}: {e}")
        
        # Get expertise profile
        if self.expertise_scorer:
            try:
                expertise_profile = self.expertise_scorer.get_agent_expertise_profile(agent_id)
                profile['expertise_profile'] = expertise_profile
            except Exception as e:
                logger.debug(f"Error getting expertise profile for {agent_id}: {e}")
        
        # Get accuracy profile
        if self.accuracy_tracker:
            try:
                accuracy_summary = self.accuracy_tracker.get_agent_performance_summary(agent_id)
                profile['accuracy_profile'] = accuracy_summary
            except Exception as e:
                logger.debug(f"Error getting accuracy profile for {agent_id}: {e}")
        
        # Get confidence profile
        if self.confidence_adjuster:
            try:
                confidence_patterns = self.confidence_adjuster.analyze_agent_confidence_patterns(agent_id)
                profile['confidence_profile'] = confidence_patterns
            except Exception as e:
                logger.debug(f"Error getting confidence profile for {agent_id}: {e}")
        
        return profile
    
    def _calculate_sophisticated_weights(self, votes: List[AgentVote], 
                                       context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sophisticated weights for all voting agents"""
        agent_weights = {}
        
        # Extract agent IDs and add ensemble context
        agent_ids = [vote.agent_id for vote in votes]
        enhanced_context = {
            **context,
            'ensemble_size': len(agent_ids),
            'decision_timestamp': datetime.now().isoformat()
        }
        
        # Calculate individual weights
        for vote in votes:
            # Enhanced context with vote-specific information
            vote_context = {
                **enhanced_context,
                'confidence': self._extract_confidence_from_vote(vote),
                'evidence_quality': getattr(vote, 'evidence_quality', 0.7),
                'agent_reasoning_length': len(vote.reasoning) if vote.reasoning else 0
            }
            
            # Calculate weight using sophisticated algorithm
            weight = self.weight_calculator.calculate_weight(vote.agent_id, vote_context)
            agent_weights[vote.agent_id] = weight
        
        # Apply ensemble-level normalization if configured
        if self.weighting_config.weight_normalization_method == "ensemble":
            agent_weights = self._normalize_ensemble_weights(agent_weights)
        
        return agent_weights
    
    def _create_enhanced_voting_config(self, base_config: VotingConfig, 
                                     agent_weights: Dict[str, float]) -> VotingConfig:
        """Create enhanced voting configuration with calculated weights"""
        # Create a new config with updated weights
        enhanced_config = VotingConfig(
            mechanism=base_config.mechanism,
            threshold=base_config.threshold,
            weights=agent_weights,  # Use calculated weights
            use_cases=base_config.use_cases,
            timeout_minutes=base_config.timeout_minutes,
            minimum_votes=base_config.minimum_votes,
            allow_abstentions=base_config.allow_abstentions
        )
        
        return enhanced_config
    
    def _calculate_weighted_consensus_direct(self, votes: List[AgentVote],
                                           config: VotingConfig,
                                           context: Dict[str, Any]) -> ConsensusResult:
        """Direct weighted consensus calculation (fallback)"""
        if not votes:
            return ConsensusResult(
                decision=False,
                confidence_score=0.0,
                vote_count=0,
                agreement_level=0.0,
                mechanism_used=config.mechanism
            )
        
        # Simple weighted majority for boolean votes
        total_weight = 0.0
        weighted_positive = 0.0
        
        for vote in votes:
            weight = config.weights.get(vote.agent_id, 1.0)
            total_weight += weight
            
            if vote.vote is True:
                weighted_positive += weight
        
        if total_weight == 0:
            agreement_level = 0.0
            decision = False
        else:
            agreement_level = weighted_positive / total_weight
            decision = agreement_level >= config.threshold
        
        # Calculate weighted confidence
        weighted_confidence = 0.0
        if total_weight > 0:
            for vote in votes:
                weight = config.weights.get(vote.agent_id, 1.0)
                vote_confidence = self._extract_confidence_from_vote(vote)
                weighted_confidence += (weight * vote_confidence) / total_weight
        
        return ConsensusResult(
            decision=decision,
            confidence_score=weighted_confidence * agreement_level,
            vote_count=len(votes),
            agreement_level=agreement_level,
            mechanism_used=config.mechanism
        )
    
    def _generate_weighted_aggregation_report(self, decision_id: str,
                                            votes: List[AgentVote],
                                            consensus_result: ConsensusResult,
                                            agent_weights: Dict[str, float],
                                            context: Dict[str, Any],
                                            start_time: datetime) -> AggregationReport:
        """Generate comprehensive aggregation report with weighting details"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate weight statistics
        weight_values = list(agent_weights.values())
        weight_stats = {
            'mean_weight': sum(weight_values) / len(weight_values) if weight_values else 0.0,
            'min_weight': min(weight_values) if weight_values else 0.0,
            'max_weight': max(weight_values) if weight_values else 0.0,
            'weight_variance': sum((w - sum(weight_values)/len(weight_values))**2 for w in weight_values) / len(weight_values) if len(weight_values) > 1 else 0.0
        }
        
        # Generate vote summary with weights
        vote_summary = {
            'total_votes': len(votes),
            'weighted_vote_distribution': {},
            'agent_weight_breakdown': {
                vote.agent_id: {
                    'vote': vote.vote,
                    'weight': agent_weights.get(vote.agent_id, 1.0),
                    'confidence': self._extract_confidence_from_vote(vote),
                    'reasoning': vote.reasoning
                }
                for vote in votes
            },
            'weight_statistics': weight_stats
        }
        
        # Quality metrics including weighting quality
        quality_metrics = {
            'weight_distribution_quality': 1.0 - min(1.0, weight_stats['weight_variance']),
            'ensemble_balance': self._calculate_ensemble_balance(agent_weights),
            'weighting_effectiveness': self._estimate_weighting_effectiveness(votes, agent_weights),
            'confidence_weight_alignment': self._calculate_confidence_weight_alignment(votes, agent_weights)
        }
        
        # Enhanced recommendations
        recommendations = []
        
        if weight_stats['weight_variance'] > 2.0:
            recommendations.append("High weight variance detected - consider reviewing agent expertise levels")
        
        if weight_stats['max_weight'] / weight_stats['min_weight'] > 5.0:
            recommendations.append("Large weight disparity - ensure all agents have opportunity for input")
        
        if quality_metrics['confidence_weight_alignment'] < 0.6:
            recommendations.append("Confidence and weight alignment could be improved - review calibration")
        
        # Create enhanced aggregation report
        report = AggregationReport(
            decision_id=decision_id,
            consensus_result=consensus_result,
            vote_summary=vote_summary,
            conflict_analysis={'total_conflicts': 0},  # Would be calculated by base aggregator
            quality_metrics=quality_metrics,
            processing_time_seconds=processing_time,
            recommendations=recommendations,
            metadata={
                'weighting_strategy': self.weighting_config.weighting_strategy.value,
                'total_weight': sum(weight_values),
                'agents_weighted': len(agent_weights),
                'context_domain': context.get('domain', 'general'),
                'enhancement_features': {
                    'expertise_scoring': self.weighting_config.enable_expertise_scoring,
                    'accuracy_tracking': self.weighting_config.enable_accuracy_tracking,
                    'confidence_adjustment': self.weighting_config.enable_confidence_adjustment
                }
            }
        )
        
        return report
    
    def _normalize_ensemble_weights(self, agent_weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights across the ensemble"""
        if not agent_weights:
            return {}
        
        # Apply bounds
        normalized = {}
        for agent, weight in agent_weights.items():
            normalized[agent] = max(
                self.weighting_config.min_weight_threshold,
                min(self.weighting_config.max_weight_threshold, weight)
            )
        
        return normalized
    
    def _extract_confidence_from_vote(self, vote: AgentVote) -> float:
        """Extract confidence value from vote object"""
        if hasattr(vote, 'confidence'):
            confidence = vote.confidence
            if hasattr(confidence, 'value'):  # Enum type
                confidence_map = {
                    'LOW': 0.25,
                    'MEDIUM': 0.5,
                    'HIGH': 0.75,
                    'VERY_HIGH': 1.0
                }
                return confidence_map.get(confidence.value, 0.5)
            elif isinstance(confidence, (int, float)):
                return max(0.0, min(1.0, float(confidence)))
        
        return 0.7  # Default confidence
    
    def _prepare_outcome_tracking(self, decision_id: str, votes: List[AgentVote],
                                agent_weights: Dict[str, float], context: Dict[str, Any]) -> None:
        """Prepare data structures for outcome tracking"""
        # Store decision information for later outcome updates
        decision_info = {
            'decision_id': decision_id,
            'votes': votes,
            'agent_weights': agent_weights,
            'context': context,
            'timestamp': datetime.now()
        }
        
        # Store in a simple in-memory cache (in production, use persistent storage)
        if not hasattr(self, '_decision_cache'):
            self._decision_cache = {}
        
        self._decision_cache[decision_id] = decision_info
    
    def _get_stored_decision_info(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored decision information"""
        if hasattr(self, '_decision_cache'):
            return self._decision_cache.get(decision_id)
        return None
    
    def _calculate_ensemble_balance(self, agent_weights: Dict[str, float]) -> float:
        """Calculate how balanced the ensemble weights are"""
        if not agent_weights:
            return 0.0
        
        weights = list(agent_weights.values())
        if len(weights) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (lower = more balanced)
        mean_weight = sum(weights) / len(weights)
        variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
        std_dev = variance ** 0.5
        
        if mean_weight == 0:
            return 0.0
        
        cv = std_dev / mean_weight
        balance = 1.0 / (1.0 + cv)  # Convert to 0-1 scale where 1 = perfectly balanced
        
        return balance
    
    def _estimate_weighting_effectiveness(self, votes: List[AgentVote],
                                        agent_weights: Dict[str, float]) -> float:
        """Estimate how effective the weighting is likely to be"""
        if not votes or not agent_weights:
            return 0.5
        
        # Simple heuristic: higher confidence votes should generally have higher weights
        confidence_weight_correlation = 0.0
        if len(votes) > 1:
            confidences = [self._extract_confidence_from_vote(vote) for vote in votes]
            weights = [agent_weights.get(vote.agent_id, 1.0) for vote in votes]
            
            # Calculate correlation coefficient (simplified)
            mean_conf = sum(confidences) / len(confidences)
            mean_weight = sum(weights) / len(weights)
            
            numerator = sum((c - mean_conf) * (w - mean_weight) for c, w in zip(confidences, weights))
            denom_conf = sum((c - mean_conf) ** 2 for c in confidences) ** 0.5
            denom_weight = sum((w - mean_weight) ** 2 for w in weights) ** 0.5
            
            if denom_conf > 0 and denom_weight > 0:
                correlation = numerator / (denom_conf * denom_weight)
                confidence_weight_correlation = max(0.0, correlation)  # Only positive correlation is good
        
        return confidence_weight_correlation
    
    def _calculate_confidence_weight_alignment(self, votes: List[AgentVote],
                                             agent_weights: Dict[str, float]) -> float:
        """Calculate how well confidence levels align with weights"""
        if not votes:
            return 0.5
        
        alignments = []
        for vote in votes:
            confidence = self._extract_confidence_from_vote(vote)
            weight = agent_weights.get(vote.agent_id, 1.0)
            
            # Normalize weight to 0-1 scale for comparison
            normalized_weight = (weight - 0.1) / (3.0 - 0.1)  # Assuming 0.1-3.0 range
            normalized_weight = max(0.0, min(1.0, normalized_weight))
            
            # Calculate alignment (1.0 - difference)
            alignment = 1.0 - abs(confidence - normalized_weight)
            alignments.append(alignment)
        
        return sum(alignments) / len(alignments) if alignments else 0.5
    
    def _update_integration_metrics(self, agent_weights: Dict[str, float]) -> None:
        """Update integration performance metrics"""
        self.integration_metrics['weighted_decisions'] += 1
        
        if agent_weights:
            weight_values = list(agent_weights.values())
            current_variance = sum((w - sum(weight_values)/len(weight_values))**2 for w in weight_values) / len(weight_values) if len(weight_values) > 1 else 0.0
            
            # Update running average of weight variance
            total_decisions = self.integration_metrics['weighted_decisions']
            current_avg_variance = self.integration_metrics['average_weight_variance']
            
            self.integration_metrics['average_weight_variance'] = (
                (current_avg_variance * (total_decisions - 1) + current_variance) / total_decisions
            )
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        base_metrics = {}
        if self.base_aggregator:
            base_metrics = self.base_aggregator.get_aggregator_metrics()
        
        component_metrics = {
            'weight_calculator': self.weight_calculator.get_calculation_metrics(),
            'expertise_scorer': self.expertise_scorer.get_scoring_metrics() if self.expertise_scorer else {},
            'accuracy_tracker': self.accuracy_tracker.get_tracker_metrics() if self.accuracy_tracker else {},
            'confidence_adjuster': self.confidence_adjuster.get_adjuster_metrics() if self.confidence_adjuster else {}
        }
        
        return {
            'integration_metrics': self.integration_metrics,
            'base_aggregator_metrics': base_metrics,
            'component_metrics': component_metrics,
            'configuration': {
                'weighting_strategy': self.weighting_config.weighting_strategy.value,
                'expertise_scoring_enabled': self.weighting_config.enable_expertise_scoring,
                'accuracy_tracking_enabled': self.weighting_config.enable_accuracy_tracking,
                'confidence_adjustment_enabled': self.weighting_config.enable_confidence_adjustment
            }
        }


def main():
    """Demonstration of weighted voting integration"""
    print("=== RIF Weighted Voting Integration Demo ===\n")
    
    # Initialize weighted voting aggregator
    config = WeightedVotingConfig(
        weighting_strategy=WeightingStrategy.BALANCED,
        enable_expertise_scoring=True,
        enable_accuracy_tracking=True,
        enable_confidence_adjustment=True
    )
    
    weighted_aggregator = WeightedVotingAggregator(weighting_config=config)
    
    print(f"Initialized weighted voting aggregator with strategy: {config.weighting_strategy.value}")
    print(f"Features enabled: expertise={config.enable_expertise_scoring}, "
          f"accuracy={config.enable_accuracy_tracking}, "
          f"confidence={config.enable_confidence_adjustment}")
    
    print("\n=== Integration Metrics ===")
    metrics = weighted_aggregator.get_integration_metrics()
    
    for category, data in metrics.items():
        if isinstance(data, dict) and data:
            print(f"\n{category.replace('_', ' ').title()}:")
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"  {key}: {json.dumps(value, indent=4)}")
                else:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()