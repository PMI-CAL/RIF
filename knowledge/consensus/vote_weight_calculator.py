#!/usr/bin/env python3
"""
RIF Vote Weight Calculator - Issue #62
Advanced vote weighting algorithm for agent consensus decisions.

This module implements sophisticated vote weighting that considers:
- Agent expertise in specific domains
- Historical accuracy tracking
- Confidence-based adjustments
- Dynamic weight calibration
- Consensus quality optimization
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from pathlib import Path

# Import consensus architecture components
try:
    from claude.commands.consensus_architecture import (
        AgentVote, ConfidenceLevel, VotingConfig, ConsensusResult
    )
except ImportError:
    # Fallback imports for testing
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExpertiseProfile:
    """Agent expertise profile across multiple domains"""
    domain_expertise: Dict[str, float] = field(default_factory=dict)
    cross_domain_bonus: float = 1.0
    expertise_decay_rate: float = 0.95  # Monthly decay
    last_update: datetime = field(default_factory=datetime.now)
    specialization_areas: List[str] = field(default_factory=list)


@dataclass
class AccuracyRecord:
    """Historical accuracy tracking for an agent"""
    decision_outcomes: List[bool] = field(default_factory=list)
    success_rate: float = 1.0
    context_specific_accuracy: Dict[str, float] = field(default_factory=dict)
    decision_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    performance_trend: float = 0.0  # Positive = improving


@dataclass
class WeightCalculationResult:
    """Detailed result of weight calculation"""
    final_weight: float
    base_weight: float
    expertise_factor: float
    accuracy_factor: float
    confidence_factor: float
    normalization_applied: bool
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)


class WeightingStrategy(Enum):
    """Different weighting strategies available"""
    EXPERTISE_FOCUSED = "expertise_focused"
    ACCURACY_FOCUSED = "accuracy_focused"
    BALANCED = "balanced"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE = "adaptive"


class VoteWeightCalculator:
    """
    Advanced vote weight calculator implementing multi-factor scoring algorithm.
    
    This system calculates voting weights based on:
    1. Agent expertise in relevant domains
    2. Historical accuracy of past decisions
    3. Confidence calibration and reliability
    4. Dynamic adjustment based on context
    5. Ensemble balancing and normalization
    """
    
    def __init__(self, knowledge_system=None, strategy: WeightingStrategy = WeightingStrategy.BALANCED):
        """Initialize the vote weight calculator"""
        self.knowledge_system = knowledge_system
        self.strategy = strategy
        
        # Agent tracking data
        self.expertise_profiles: Dict[str, ExpertiseProfile] = {}
        self.accuracy_records: Dict[str, AccuracyRecord] = {}
        
        # Configuration parameters
        self.weight_bounds = (0.1, 3.0)  # Min/max weight limits
        self.base_weight = 1.0
        
        # Factor weights by strategy
        self.strategy_weights = {
            WeightingStrategy.EXPERTISE_FOCUSED: {
                'expertise': 0.5,
                'accuracy': 0.25,
                'confidence': 0.25
            },
            WeightingStrategy.ACCURACY_FOCUSED: {
                'expertise': 0.25,
                'accuracy': 0.5,
                'confidence': 0.25
            },
            WeightingStrategy.BALANCED: {
                'expertise': 0.35,
                'accuracy': 0.35,
                'confidence': 0.30
            },
            WeightingStrategy.CONFIDENCE_WEIGHTED: {
                'expertise': 0.25,
                'accuracy': 0.25,
                'confidence': 0.5
            },
            WeightingStrategy.ADAPTIVE: {
                'expertise': 0.4,
                'accuracy': 0.4,
                'confidence': 0.2
            }
        }
        
        # Performance tracking
        self.calculation_metrics = {
            'total_calculations': 0,
            'average_calculation_time': 0.0,
            'weight_distribution_stats': {},
            'strategy_usage': {s.value: 0 for s in WeightingStrategy}
        }
        
        # Load existing data
        self._load_historical_data()
        
        logger.info(f"Vote Weight Calculator initialized with {strategy.value} strategy")
    
    def calculate_weight(self, agent: str, context: Dict[str, Any]) -> float:
        """
        Calculate voting weight for an agent in a specific context.
        
        This implements the core requirement from Issue #62:
        "Algorithm correctly calculates weights based on all factors"
        
        Args:
            agent: Agent identifier
            context: Decision context including domain, confidence, etc.
            
        Returns:
            Calculated voting weight (normalized between bounds)
        """
        start_time = time.time()
        
        try:
            # Get detailed calculation result
            result = self.calculate_detailed_weight(agent, context)
            
            # Update metrics
            self._update_calculation_metrics(time.time() - start_time, result.final_weight)
            
            return result.final_weight
            
        except Exception as e:
            logger.error(f"Error calculating weight for {agent}: {str(e)}")
            return self.base_weight
    
    def calculate_detailed_weight(self, agent: str, context: Dict[str, Any]) -> WeightCalculationResult:
        """
        Calculate detailed voting weight with full breakdown.
        
        Args:
            agent: Agent identifier
            context: Decision context
            
        Returns:
            WeightCalculationResult with detailed factor breakdown
        """
        # Initialize base weight
        base_weight = self.base_weight
        
        # Calculate individual factors
        expertise_factor = self._calculate_expertise_factor(agent, context)
        accuracy_factor = self._calculate_accuracy_factor(agent, context)
        confidence_factor = self._calculate_confidence_factor(context)
        
        # Apply strategy-specific weighting
        weights = self.strategy_weights[self.strategy]
        
        # Calculate composite weight
        raw_weight = base_weight * (
            (expertise_factor ** weights['expertise']) *
            (accuracy_factor ** weights['accuracy']) *
            (confidence_factor ** weights['confidence'])
        )
        
        # Apply normalization and bounds
        final_weight, normalization_applied = self._normalize_weight(raw_weight)
        
        # Create detailed result
        result = WeightCalculationResult(
            final_weight=final_weight,
            base_weight=base_weight,
            expertise_factor=expertise_factor,
            accuracy_factor=accuracy_factor,
            confidence_factor=confidence_factor,
            normalization_applied=normalization_applied,
            calculation_metadata={
                'agent_id': agent,
                'context_domain': context.get('domain', 'general'),
                'strategy': self.strategy.value,
                'raw_weight': raw_weight,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Log calculation details
        logger.debug(f"Weight calculation for {agent}: {final_weight:.3f} "
                    f"(expertise: {expertise_factor:.2f}, accuracy: {accuracy_factor:.2f}, "
                    f"confidence: {confidence_factor:.2f})")
        
        return result
    
    def calculate_ensemble_weights(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate normalized weights for an entire ensemble of agents.
        
        Args:
            agents: List of agent identifiers
            context: Decision context
            
        Returns:
            Dictionary mapping agent IDs to normalized weights
        """
        if not agents:
            return {}
        
        # Calculate individual weights
        individual_weights = {}
        for agent in agents:
            individual_weights[agent] = self.calculate_weight(agent, context)
        
        # Apply ensemble normalization
        ensemble_weights = self._normalize_ensemble_weights(individual_weights, context)
        
        logger.info(f"Calculated ensemble weights for {len(agents)} agents: "
                   f"mean={statistics.mean(ensemble_weights.values()):.2f}, "
                   f"std={statistics.stdev(ensemble_weights.values()):.2f}")
        
        return ensemble_weights
    
    def update_agent_accuracy(self, agent: str, decision_outcome: bool, 
                            context: Dict[str, Any] = None) -> None:
        """
        Update agent's historical accuracy based on decision outcome.
        
        Args:
            agent: Agent identifier
            decision_outcome: True if decision was correct/successful
            context: Optional context for domain-specific tracking
        """
        if agent not in self.accuracy_records:
            self.accuracy_records[agent] = AccuracyRecord()
        
        record = self.accuracy_records[agent]
        
        # Add new outcome to history
        record.decision_outcomes.append(decision_outcome)
        record.decision_count += 1
        
        # Maintain rolling window (keep last 100 decisions)
        if len(record.decision_outcomes) > 100:
            record.decision_outcomes = record.decision_outcomes[-100:]
        
        # Recalculate success rate with recency weighting
        record.success_rate = self._calculate_weighted_success_rate(record.decision_outcomes)
        
        # Update context-specific accuracy if context provided
        if context and 'domain' in context:
            domain = context['domain']
            if domain not in record.context_specific_accuracy:
                record.context_specific_accuracy[domain] = record.success_rate
            else:
                # Blend with existing accuracy (0.7 weight to new outcome)
                current_accuracy = record.context_specific_accuracy[domain]
                outcome_weight = 1.0 if decision_outcome else 0.0
                record.context_specific_accuracy[domain] = (
                    0.3 * current_accuracy + 0.7 * outcome_weight
                )
        
        # Calculate performance trend
        if len(record.decision_outcomes) >= 10:
            recent_10 = record.decision_outcomes[-10:]
            earlier_10 = record.decision_outcomes[-20:-10] if len(record.decision_outcomes) >= 20 else []
            
            if earlier_10:
                recent_rate = sum(recent_10) / len(recent_10)
                earlier_rate = sum(earlier_10) / len(earlier_10)
                record.performance_trend = recent_rate - earlier_rate
        
        record.last_updated = datetime.now()
        
        logger.debug(f"Updated accuracy for {agent}: {record.success_rate:.2f} "
                    f"({record.decision_count} decisions)")
    
    def update_agent_expertise(self, agent: str, domain: str, 
                             expertise_level: float) -> None:
        """
        Update agent's expertise level in a specific domain.
        
        Args:
            agent: Agent identifier
            domain: Domain name (e.g., 'security', 'testing', 'architecture')
            expertise_level: Expertise level (0.0 - 1.0)
        """
        if agent not in self.expertise_profiles:
            self.expertise_profiles[agent] = ExpertiseProfile()
        
        profile = self.expertise_profiles[agent]
        profile.domain_expertise[domain] = max(0.0, min(1.0, expertise_level))
        profile.last_update = datetime.now()
        
        # Update specialization areas (domains with >0.8 expertise)
        profile.specialization_areas = [
            d for d, e in profile.domain_expertise.items() if e > 0.8
        ]
        
        # Calculate cross-domain bonus (reward for multi-domain expertise)
        num_domains = len([e for e in profile.domain_expertise.values() if e > 0.5])
        profile.cross_domain_bonus = 1.0 + (num_domains - 1) * 0.1
        
        logger.info(f"Updated {agent} expertise in {domain}: {expertise_level:.2f}")
    
    def _calculate_expertise_factor(self, agent: str, context: Dict[str, Any]) -> float:
        """Calculate expertise factor for weight calculation"""
        if agent not in self.expertise_profiles:
            return 1.0
        
        profile = self.expertise_profiles[agent]
        domain = context.get('domain', 'general')
        
        # Get base expertise for the domain
        base_expertise = profile.domain_expertise.get(domain, 0.5)
        
        # Apply expertise decay based on time since last update
        time_decay = self._calculate_time_decay(profile.last_update, profile.expertise_decay_rate)
        current_expertise = base_expertise * time_decay
        
        # Apply cross-domain bonus
        expertise_with_bonus = current_expertise * profile.cross_domain_bonus
        
        # Convert to factor (0.3 - 2.0 range)
        expertise_factor = 0.3 + (expertise_with_bonus * 1.7)
        
        return min(2.0, max(0.3, expertise_factor))
    
    def _calculate_accuracy_factor(self, agent: str, context: Dict[str, Any]) -> float:
        """Calculate accuracy factor based on historical performance"""
        if agent not in self.accuracy_records:
            return 1.0  # Neutral factor for new agents
        
        record = self.accuracy_records[agent]
        domain = context.get('domain', 'general')
        
        # Use domain-specific accuracy if available, otherwise general
        if domain in record.context_specific_accuracy:
            accuracy = record.context_specific_accuracy[domain]
        else:
            accuracy = record.success_rate
        
        # Apply performance trend adjustment
        trend_adjustment = 1.0 + (record.performance_trend * 0.2)
        adjusted_accuracy = accuracy * trend_adjustment
        
        # Convert to factor (0.4 - 1.6 range)
        accuracy_factor = 0.4 + (adjusted_accuracy * 1.2)
        
        return min(1.6, max(0.4, accuracy_factor))
    
    def _calculate_confidence_factor(self, context: Dict[str, Any]) -> float:
        """Calculate confidence factor from context"""
        confidence = context.get('confidence', 1.0)
        
        # Handle different confidence input types
        if isinstance(confidence, ConfidenceLevel):
            confidence_map = {
                ConfidenceLevel.LOW: 0.25,
                ConfidenceLevel.MEDIUM: 0.5,
                ConfidenceLevel.HIGH: 0.75,
                ConfidenceLevel.VERY_HIGH: 1.0
            }
            confidence_score = confidence_map.get(confidence, 0.5)
        elif isinstance(confidence, str):
            confidence_map = {
                'low': 0.25,
                'medium': 0.5,
                'high': 0.75,
                'very_high': 1.0
            }
            confidence_score = confidence_map.get(confidence.lower(), 0.5)
        else:
            # Assume numeric confidence (0.0 - 1.0)
            confidence_score = max(0.0, min(1.0, float(confidence)))
        
        # Convert to factor (0.6 - 1.4 range)
        confidence_factor = 0.6 + (confidence_score * 0.8)
        
        return confidence_factor
    
    def _normalize_weight(self, raw_weight: float) -> Tuple[float, bool]:
        """Normalize weight within configured bounds"""
        min_weight, max_weight = self.weight_bounds
        
        if raw_weight < min_weight:
            return min_weight, True
        elif raw_weight > max_weight:
            return max_weight, True
        else:
            return raw_weight, False
    
    def _normalize_ensemble_weights(self, weights: Dict[str, float], 
                                  context: Dict[str, Any]) -> Dict[str, float]:
        """Apply ensemble-level normalization to prevent extreme weight concentrations"""
        if not weights:
            return {}
        
        # Apply outlier protection
        weight_values = list(weights.values())
        mean_weight = statistics.mean(weight_values)
        std_weight = statistics.stdev(weight_values) if len(weight_values) > 1 else 0
        
        # Limit weights that are more than 2 standard deviations from mean
        outlier_threshold = 2.0
        normalized_weights = {}
        
        for agent, weight in weights.items():
            if std_weight > 0:
                z_score = abs(weight - mean_weight) / std_weight
                if z_score > outlier_threshold:
                    # Clamp to less extreme value
                    direction = 1 if weight > mean_weight else -1
                    clamped_weight = mean_weight + (direction * outlier_threshold * std_weight)
                    normalized_weights[agent] = clamped_weight
                    logger.debug(f"Clamped outlier weight for {agent}: {weight:.3f} -> {clamped_weight:.3f}")
                else:
                    normalized_weights[agent] = weight
            else:
                normalized_weights[agent] = weight
        
        # Apply temporal smoothing if requested
        if context.get('temporal_smoothing', False):
            normalized_weights = self._apply_temporal_smoothing(normalized_weights)
        
        return normalized_weights
    
    def _apply_temporal_smoothing(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing to reduce weight volatility"""
        # This would blend with previous weights in a real implementation
        # For now, return unchanged
        return weights
    
    def _calculate_weighted_success_rate(self, outcomes: List[bool]) -> float:
        """Calculate success rate with recency weighting"""
        if not outcomes:
            return 1.0
        
        # Apply exponential decay weighting (more recent = higher weight)
        weights = [0.95 ** (len(outcomes) - i - 1) for i in range(len(outcomes))]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return sum(outcomes) / len(outcomes)
        
        weighted_success = sum(w * outcome for w, outcome in zip(weights, outcomes))
        return weighted_success / total_weight
    
    def _calculate_time_decay(self, last_update: datetime, decay_rate: float) -> float:
        """Calculate time-based decay factor"""
        months_since_update = (datetime.now() - last_update).days / 30.0
        return decay_rate ** months_since_update
    
    def _update_calculation_metrics(self, calculation_time: float, weight: float) -> None:
        """Update performance metrics"""
        self.calculation_metrics['total_calculations'] += 1
        
        # Update average calculation time
        total_calcs = self.calculation_metrics['total_calculations']
        current_avg = self.calculation_metrics['average_calculation_time']
        self.calculation_metrics['average_calculation_time'] = (
            (current_avg * (total_calcs - 1) + calculation_time) / total_calcs
        )
        
        # Track weight distribution
        weight_bucket = f"{weight:.1f}"
        if weight_bucket not in self.calculation_metrics['weight_distribution_stats']:
            self.calculation_metrics['weight_distribution_stats'][weight_bucket] = 0
        self.calculation_metrics['weight_distribution_stats'][weight_bucket] += 1
        
        # Update strategy usage
        self.calculation_metrics['strategy_usage'][self.strategy.value] += 1
    
    def _load_historical_data(self) -> None:
        """Load historical accuracy and expertise data"""
        try:
            # Load from knowledge system if available
            if hasattr(self.knowledge_system, 'load_agent_histories'):
                data = self.knowledge_system.load_agent_histories()
                if 'expertise_profiles' in data:
                    self.expertise_profiles.update(data['expertise_profiles'])
                if 'accuracy_records' in data:
                    self.accuracy_records.update(data['accuracy_records'])
        except Exception as e:
            logger.debug(f"Could not load historical data: {str(e)}")
        
        # Initialize with default agent profiles if none loaded
        if not self.expertise_profiles:
            self._initialize_default_profiles()
    
    def _initialize_default_profiles(self) -> None:
        """Initialize default agent expertise profiles"""
        default_agents = {
            'rif-security': {
                'security': 0.95,
                'compliance': 0.90,
                'vulnerability': 0.95,
                'general': 0.70
            },
            'rif-validator': {
                'testing': 0.90,
                'quality': 0.95,
                'validation': 0.95,
                'general': 0.75
            },
            'rif-implementer': {
                'coding': 0.85,
                'integration': 0.80,
                'debugging': 0.85,
                'general': 0.70
            },
            'rif-architect': {
                'design': 0.95,
                'scalability': 0.90,
                'patterns': 0.90,
                'general': 0.80
            },
            'rif-analyst': {
                'requirements': 0.90,
                'analysis': 0.95,
                'planning': 0.85,
                'general': 0.75
            }
        }
        
        for agent, expertise_map in default_agents.items():
            profile = ExpertiseProfile(domain_expertise=expertise_map)
            self.expertise_profiles[agent] = profile
            
            # Initialize accuracy record
            self.accuracy_records[agent] = AccuracyRecord(success_rate=0.8)
        
        logger.info("Initialized default agent expertise profiles")
    
    def get_calculation_metrics(self) -> Dict[str, Any]:
        """Get current calculation performance metrics"""
        return {
            **self.calculation_metrics,
            'active_agents': len(self.expertise_profiles),
            'tracked_accuracy_records': len(self.accuracy_records),
            'strategy': self.strategy.value,
            'weight_bounds': self.weight_bounds
        }
    
    def export_agent_data(self) -> Dict[str, Any]:
        """Export agent expertise and accuracy data for persistence"""
        return {
            'expertise_profiles': {
                agent: {
                    'domain_expertise': profile.domain_expertise,
                    'cross_domain_bonus': profile.cross_domain_bonus,
                    'specialization_areas': profile.specialization_areas,
                    'last_update': profile.last_update.isoformat()
                }
                for agent, profile in self.expertise_profiles.items()
            },
            'accuracy_records': {
                agent: {
                    'success_rate': record.success_rate,
                    'context_specific_accuracy': record.context_specific_accuracy,
                    'decision_count': record.decision_count,
                    'performance_trend': record.performance_trend,
                    'last_updated': record.last_updated.isoformat()
                }
                for agent, record in self.accuracy_records.items()
            },
            'calculation_metrics': self.calculation_metrics,
            'export_timestamp': datetime.now().isoformat()
        }


def main():
    """Demonstration of vote weight calculator functionality"""
    print("=== RIF Vote Weight Calculator Demo ===\n")
    
    # Initialize calculator
    calculator = VoteWeightCalculator(strategy=WeightingStrategy.BALANCED)
    
    # Example 1: Basic weight calculation
    print("Example 1: Basic Weight Calculation")
    context = {
        'domain': 'security',
        'confidence': 'high',
        'decision_type': 'approval'
    }
    
    agents = ['rif-security', 'rif-validator', 'rif-implementer']
    
    for agent in agents:
        result = calculator.calculate_detailed_weight(agent, context)
        print(f"{agent}: {result.final_weight:.3f} "
              f"(expertise: {result.expertise_factor:.2f}, "
              f"accuracy: {result.accuracy_factor:.2f}, "
              f"confidence: {result.confidence_factor:.2f})")
    
    print()
    
    # Example 2: Ensemble weight calculation
    print("Example 2: Ensemble Weight Calculation")
    ensemble_weights = calculator.calculate_ensemble_weights(agents, context)
    
    total_weight = sum(ensemble_weights.values())
    for agent, weight in ensemble_weights.items():
        percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
        print(f"{agent}: {weight:.3f} ({percentage:.1f}% of total voting power)")
    
    print()
    
    # Example 3: Accuracy update simulation
    print("Example 3: Accuracy Update Simulation")
    
    # Simulate some decision outcomes
    outcomes = [
        ('rif-security', True),
        ('rif-security', True),
        ('rif-security', False),
        ('rif-validator', True),
        ('rif-validator', True),
        ('rif-implementer', False),
        ('rif-implementer', True)
    ]
    
    for agent, outcome in outcomes:
        calculator.update_agent_accuracy(agent, outcome, context)
    
    # Recalculate weights after accuracy updates
    print("Weights after accuracy updates:")
    for agent in agents:
        weight = calculator.calculate_weight(agent, context)
        accuracy_record = calculator.accuracy_records.get(agent)
        accuracy = accuracy_record.success_rate if accuracy_record else 1.0
        print(f"{agent}: {weight:.3f} (accuracy: {accuracy:.2f})")
    
    print()
    
    # Example 4: Different strategies comparison
    print("Example 4: Strategy Comparison")
    strategies = [WeightingStrategy.EXPERTISE_FOCUSED, WeightingStrategy.ACCURACY_FOCUSED, 
                 WeightingStrategy.BALANCED]
    
    agent = 'rif-security'
    
    for strategy in strategies:
        calc = VoteWeightCalculator(strategy=strategy)
        # Copy our data to the new calculator
        calc.expertise_profiles = calculator.expertise_profiles
        calc.accuracy_records = calculator.accuracy_records
        
        weight = calc.calculate_weight(agent, context)
        print(f"{strategy.value}: {weight:.3f}")
    
    print()
    
    # Show metrics
    print("=== Calculator Metrics ===")
    metrics = calculator.get_calculation_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()