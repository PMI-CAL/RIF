#!/usr/bin/env python3
"""
RIF Accuracy Tracker - Historical Performance Analysis
Part of Issue #62 implementation for vote weighting algorithm.

This module provides sophisticated accuracy tracking that considers:
- Historical decision outcome tracking
- Context-specific performance analysis
- Performance trend identification
- Calibration quality assessment
- Predictive accuracy modeling
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionOutcome(Enum):
    """Types of decision outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    INCONCLUSIVE = "inconclusive"


class ContextCategory(Enum):
    """Categories for contextual accuracy tracking"""
    SECURITY_CRITICAL = "security_critical"
    PERFORMANCE_SENSITIVE = "performance_sensitive"
    HIGH_COMPLEXITY = "high_complexity"
    TIME_PRESSURE = "time_pressure"
    NOVEL_DOMAIN = "novel_domain"
    ROUTINE_TASK = "routine_task"


@dataclass
class DecisionRecord:
    """Record of a single decision and its outcome"""
    decision_id: str
    agent_id: str
    decision_timestamp: datetime
    decision_confidence: float
    decision_content: str
    outcome: DecisionOutcome
    outcome_timestamp: datetime
    context_category: Optional[str] = None
    impact_level: str = "medium"
    validation_score: float = 0.0  # How well the decision was validated
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for an agent"""
    agent_id: str
    overall_accuracy: float
    weighted_accuracy: float  # Weighted by impact and recency
    decision_count: int
    success_count: int
    confidence_calibration: float  # How well confidence matches outcomes
    context_specific_accuracy: Dict[str, float]
    temporal_trend: float  # Positive = improving, negative = declining
    prediction_reliability: float  # Consistency of accuracy over time
    last_updated: datetime


@dataclass
class CalibrationAnalysis:
    """Analysis of confidence calibration quality"""
    agent_id: str
    calibration_score: float  # 0.0 = poorly calibrated, 1.0 = perfectly calibrated
    overconfidence_bias: float  # Positive = overconfident, negative = underconfident
    reliability_score: float  # Consistency of calibration
    resolution_score: float  # Ability to discriminate between correct/incorrect
    brier_score: float  # Overall calibration quality metric
    calibration_curve_data: List[Tuple[float, float]]  # (confidence_bin, accuracy_in_bin)


class AccuracyTracker:
    """
    Advanced accuracy tracking system for RIF agents.
    
    This system tracks and analyzes agent decision accuracy including:
    - Historical success rate calculation with recency weighting
    - Context-specific performance analysis
    - Confidence calibration assessment
    - Performance trend identification
    - Predictive accuracy modeling
    - Outlier detection and handling
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize accuracy tracker"""
        self.knowledge_system = knowledge_system
        
        # Decision history storage
        self.decision_records: Dict[str, List[DecisionRecord]] = defaultdict(list)
        self.accuracy_cache: Dict[str, AccuracyMetrics] = {}
        self.calibration_cache: Dict[str, CalibrationAnalysis] = {}
        
        # Tracking parameters
        self.tracking_params = {
            'max_history_size': 200,  # Maximum decisions to track per agent
            'recency_decay_rate': 0.95,  # Monthly decay for recency weighting
            'min_decisions_for_trend': 10,  # Minimum decisions to calculate trend
            'calibration_bins': 10,  # Number of bins for calibration analysis
            'outlier_threshold': 2.0,  # Standard deviations for outlier detection
            'confidence_threshold': 0.1,  # Minimum confidence difference for analysis
            'temporal_window_days': 90  # Days for temporal trend analysis
        }
        
        # Impact level weights for weighted accuracy
        self.impact_weights = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'critical': 2.0
        }
        
        # Performance tracking
        self.tracker_metrics = {
            'total_decisions_tracked': 0,
            'total_agents_tracked': 0,
            'average_calibration_quality': 0.0,
            'context_coverage_stats': defaultdict(int),
            'accuracy_distribution': defaultdict(int)
        }
        
        # Load existing data
        self._load_historical_records()
        
        logger.info("Accuracy Tracker initialized")
    
    def record_decision_outcome(self, decision_record: DecisionRecord) -> None:
        """
        Record a decision outcome for accuracy tracking.
        
        Args:
            decision_record: Complete decision record with outcome
        """
        agent_id = decision_record.agent_id
        
        # Add to decision history
        self.decision_records[agent_id].append(decision_record)
        
        # Maintain history size limit
        if len(self.decision_records[agent_id]) > self.tracking_params['max_history_size']:
            self.decision_records[agent_id] = self.decision_records[agent_id][-self.tracking_params['max_history_size']:]
        
        # Invalidate cached metrics for this agent
        if agent_id in self.accuracy_cache:
            del self.accuracy_cache[agent_id]
        if agent_id in self.calibration_cache:
            del self.calibration_cache[agent_id]
        
        # Update tracking metrics
        self.tracker_metrics['total_decisions_tracked'] += 1
        if decision_record.context_category:
            self.tracker_metrics['context_coverage_stats'][decision_record.context_category] += 1
        
        logger.debug(f"Recorded decision outcome for {agent_id}: {decision_record.outcome.value}")
    
    def calculate_accuracy_metrics(self, agent_id: str) -> AccuracyMetrics:
        """
        Calculate comprehensive accuracy metrics for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            AccuracyMetrics with detailed accuracy analysis
        """
        # Check cache first
        if agent_id in self.accuracy_cache:
            cached_metrics = self.accuracy_cache[agent_id]
            # Return cached if recent (within 1 hour)
            if (datetime.now() - cached_metrics.last_updated).seconds < 3600:
                return cached_metrics
        
        decisions = self.decision_records.get(agent_id, [])
        
        if not decisions:
            return self._create_default_metrics(agent_id)
        
        # Calculate basic accuracy
        success_outcomes = [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS]
        success_count = sum(1 for d in decisions if d.outcome in success_outcomes)
        overall_accuracy = success_count / len(decisions)
        
        # Calculate weighted accuracy (considering impact and recency)
        weighted_accuracy = self._calculate_weighted_accuracy(decisions)
        
        # Calculate context-specific accuracy
        context_accuracy = self._calculate_context_specific_accuracy(decisions)
        
        # Calculate temporal trend
        temporal_trend = self._calculate_temporal_trend(decisions)
        
        # Calculate prediction reliability
        reliability = self._calculate_prediction_reliability(decisions)
        
        # Calculate confidence calibration
        calibration = self._calculate_confidence_calibration_simple(decisions)
        
        # Create metrics object
        metrics = AccuracyMetrics(
            agent_id=agent_id,
            overall_accuracy=overall_accuracy,
            weighted_accuracy=weighted_accuracy,
            decision_count=len(decisions),
            success_count=success_count,
            confidence_calibration=calibration,
            context_specific_accuracy=context_accuracy,
            temporal_trend=temporal_trend,
            prediction_reliability=reliability,
            last_updated=datetime.now()
        )
        
        # Cache the results
        self.accuracy_cache[agent_id] = metrics
        
        # Update tracker metrics
        self._update_tracker_metrics(metrics)
        
        logger.debug(f"Calculated accuracy metrics for {agent_id}: {overall_accuracy:.3f}")
        
        return metrics
    
    def calculate_accuracy_factor(self, agent_id: str, context: Dict[str, Any] = None) -> float:
        """
        Calculate accuracy factor for vote weighting.
        
        This is the main interface used by VoteWeightCalculator.
        
        Args:
            agent_id: Agent identifier
            context: Optional context for context-specific accuracy
            
        Returns:
            Accuracy factor (0.4 - 1.6 range for weight calculation)
        """
        metrics = self.calculate_accuracy_metrics(agent_id)
        
        # Start with base accuracy
        base_accuracy = metrics.weighted_accuracy
        
        # Apply context-specific adjustment if context provided
        if context and 'domain' in context:
            context_key = context['domain']
            if context_key in metrics.context_specific_accuracy:
                context_accuracy = metrics.context_specific_accuracy[context_key]
                # Blend base and context accuracy
                base_accuracy = base_accuracy * 0.6 + context_accuracy * 0.4
        
        # Apply temporal trend adjustment
        trend_adjustment = 1.0 + (metrics.temporal_trend * 0.2)
        adjusted_accuracy = base_accuracy * trend_adjustment
        
        # Apply reliability bonus/penalty
        reliability_adjustment = 1.0 + (metrics.prediction_reliability - 0.5) * 0.1
        final_accuracy = adjusted_accuracy * reliability_adjustment
        
        # Convert to factor (0.4 - 1.6 range)
        accuracy_factor = 0.4 + (final_accuracy * 1.2)
        
        return min(1.6, max(0.4, accuracy_factor))
    
    def analyze_confidence_calibration(self, agent_id: str) -> CalibrationAnalysis:
        """
        Perform detailed confidence calibration analysis.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            CalibrationAnalysis with detailed calibration metrics
        """
        # Check cache
        if agent_id in self.calibration_cache:
            return self.calibration_cache[agent_id]
        
        decisions = self.decision_records.get(agent_id, [])
        
        if len(decisions) < 5:  # Need minimum decisions for reliable calibration
            return self._create_default_calibration(agent_id)
        
        # Extract confidence and outcome data
        confidences = []
        outcomes = []
        
        for decision in decisions:
            if decision.outcome != DecisionOutcome.INCONCLUSIVE:
                confidences.append(decision.decision_confidence)
                outcomes.append(1.0 if decision.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS] else 0.0)
        
        if len(confidences) < 5:
            return self._create_default_calibration(agent_id)
        
        # Calculate calibration metrics
        calibration_score = self._calculate_calibration_score(confidences, outcomes)
        overconfidence_bias = self._calculate_overconfidence_bias(confidences, outcomes)
        reliability_score = self._calculate_reliability_score(confidences, outcomes)
        resolution_score = self._calculate_resolution_score(confidences, outcomes)
        brier_score = self._calculate_brier_score(confidences, outcomes)
        calibration_curve = self._calculate_calibration_curve(confidences, outcomes)
        
        analysis = CalibrationAnalysis(
            agent_id=agent_id,
            calibration_score=calibration_score,
            overconfidence_bias=overconfidence_bias,
            reliability_score=reliability_score,
            resolution_score=resolution_score,
            brier_score=brier_score,
            calibration_curve_data=calibration_curve
        )
        
        # Cache the results
        self.calibration_cache[agent_id] = analysis
        
        return analysis
    
    def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with performance summary
        """
        metrics = self.calculate_accuracy_metrics(agent_id)
        calibration = self.analyze_confidence_calibration(agent_id)
        
        # Calculate additional insights
        recent_decisions = self.decision_records.get(agent_id, [])[-10:]  # Last 10 decisions
        recent_accuracy = (
            sum(1 for d in recent_decisions if d.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS])
            / len(recent_decisions) if recent_decisions else 0.0
        )
        
        return {
            'agent_id': agent_id,
            'overall_performance': {
                'accuracy': metrics.overall_accuracy,
                'weighted_accuracy': metrics.weighted_accuracy,
                'recent_accuracy': recent_accuracy,
                'decision_count': metrics.decision_count,
                'trend': 'improving' if metrics.temporal_trend > 0 else 'declining' if metrics.temporal_trend < 0 else 'stable'
            },
            'calibration_quality': {
                'calibration_score': calibration.calibration_score,
                'overconfidence_bias': calibration.overconfidence_bias,
                'reliability': calibration.reliability_score,
                'brier_score': calibration.brier_score
            },
            'context_performance': metrics.context_specific_accuracy,
            'reliability_metrics': {
                'prediction_reliability': metrics.prediction_reliability,
                'confidence_calibration': metrics.confidence_calibration,
                'temporal_consistency': 1.0 - abs(metrics.temporal_trend)
            },
            'performance_insights': self._generate_performance_insights(metrics, calibration)
        }
    
    def _calculate_weighted_accuracy(self, decisions: List[DecisionRecord]) -> float:
        """Calculate weighted accuracy considering impact and recency"""
        if not decisions:
            return 0.5
        
        total_weight = 0.0
        weighted_success = 0.0
        current_time = datetime.now()
        decay_rate = self.tracking_params['recency_decay_rate']
        
        for decision in decisions:
            # Calculate recency weight
            months_old = (current_time - decision.decision_timestamp).days / 30.0
            recency_weight = decay_rate ** months_old
            
            # Get impact weight
            impact_weight = self.impact_weights.get(decision.impact_level, 1.0)
            
            # Combined weight
            combined_weight = recency_weight * impact_weight
            total_weight += combined_weight
            
            # Add success weight
            if decision.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS]:
                weighted_success += combined_weight
        
        return weighted_success / total_weight if total_weight > 0 else 0.5
    
    def _calculate_context_specific_accuracy(self, decisions: List[DecisionRecord]) -> Dict[str, float]:
        """Calculate accuracy for different context categories"""
        context_groups = defaultdict(list)
        
        # Group decisions by context
        for decision in decisions:
            if decision.context_category:
                context_groups[decision.context_category].append(decision)
        
        # Calculate accuracy for each context
        context_accuracy = {}
        success_outcomes = [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS]
        
        for context, context_decisions in context_groups.items():
            if len(context_decisions) >= 3:  # Need minimum decisions for reliable metric
                success_count = sum(1 for d in context_decisions if d.outcome in success_outcomes)
                context_accuracy[context] = success_count / len(context_decisions)
        
        return context_accuracy
    
    def _calculate_temporal_trend(self, decisions: List[DecisionRecord]) -> float:
        """Calculate temporal trend in accuracy"""
        if len(decisions) < self.tracking_params['min_decisions_for_trend']:
            return 0.0
        
        # Sort decisions by timestamp
        sorted_decisions = sorted(decisions, key=lambda d: d.decision_timestamp)
        
        # Create time series data
        success_outcomes = [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS]
        
        # Use sliding window to calculate trend
        window_size = min(10, len(sorted_decisions) // 2)
        if window_size < 5:
            return 0.0
        
        # Calculate accuracy in first and last windows
        first_window = sorted_decisions[:window_size]
        last_window = sorted_decisions[-window_size:]
        
        first_accuracy = sum(1 for d in first_window if d.outcome in success_outcomes) / len(first_window)
        last_accuracy = sum(1 for d in last_window if d.outcome in success_outcomes) / len(last_window)
        
        # Return normalized trend (-1 to 1)
        trend = last_accuracy - first_accuracy
        return max(-1.0, min(1.0, trend))
    
    def _calculate_prediction_reliability(self, decisions: List[DecisionRecord]) -> float:
        """Calculate consistency of prediction accuracy over time"""
        if len(decisions) < 10:
            return 0.5
        
        # Calculate accuracy in overlapping windows
        window_size = 5
        accuracies = []
        success_outcomes = [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS]
        
        for i in range(len(decisions) - window_size + 1):
            window = decisions[i:i + window_size]
            success_count = sum(1 for d in window if d.outcome in success_outcomes)
            accuracy = success_count / len(window)
            accuracies.append(accuracy)
        
        if len(accuracies) < 2:
            return 0.5
        
        # Reliability is inverse of variance (normalized)
        variance = statistics.variance(accuracies)
        reliability = 1.0 / (1.0 + variance * 10)  # Scale variance appropriately
        
        return min(1.0, max(0.0, reliability))
    
    def _calculate_confidence_calibration_simple(self, decisions: List[DecisionRecord]) -> float:
        """Simple confidence calibration calculation"""
        if len(decisions) < 5:
            return 0.5
        
        calibration_errors = []
        success_outcomes = [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS]
        
        for decision in decisions:
            if decision.outcome != DecisionOutcome.INCONCLUSIVE:
                actual_success = 1.0 if decision.outcome in success_outcomes else 0.0
                predicted_confidence = decision.decision_confidence
                error = abs(predicted_confidence - actual_success)
                calibration_errors.append(error)
        
        if not calibration_errors:
            return 0.5
        
        # Convert average error to calibration score
        avg_error = statistics.mean(calibration_errors)
        calibration_score = 1.0 - avg_error
        
        return max(0.0, min(1.0, calibration_score))
    
    def _calculate_calibration_score(self, confidences: List[float], outcomes: List[float]) -> float:
        """Calculate detailed calibration score using binning"""
        n_bins = self.tracking_params['calibration_bins']
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        calibration_errors = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = [(conf >= bin_lower and conf < bin_upper) for conf in confidences]
            
            if not any(in_bin):
                continue
            
            # Calculate average confidence and accuracy in bin
            bin_confidences = [conf for conf, in_b in zip(confidences, in_bin) if in_b]
            bin_outcomes = [out for out, in_b in zip(outcomes, in_bin) if in_b]
            
            if bin_confidences and bin_outcomes:
                avg_confidence = statistics.mean(bin_confidences)
                avg_accuracy = statistics.mean(bin_outcomes)
                bin_error = abs(avg_confidence - avg_accuracy)
                calibration_errors.append(bin_error)
        
        if not calibration_errors:
            return 0.5
        
        # Return calibration score (1.0 - average calibration error)
        avg_calibration_error = statistics.mean(calibration_errors)
        return 1.0 - avg_calibration_error
    
    def _calculate_overconfidence_bias(self, confidences: List[float], outcomes: List[float]) -> float:
        """Calculate overconfidence bias"""
        avg_confidence = statistics.mean(confidences)
        avg_accuracy = statistics.mean(outcomes)
        
        return avg_confidence - avg_accuracy
    
    def _calculate_reliability_score(self, confidences: List[float], outcomes: List[float]) -> float:
        """Calculate reliability score (consistency of calibration)"""
        # This is a simplified version - in practice would use more sophisticated binning
        calibration_score = self._calculate_calibration_score(confidences, outcomes)
        return calibration_score  # Simplified - return same as calibration
    
    def _calculate_resolution_score(self, confidences: List[float], outcomes: List[float]) -> float:
        """Calculate resolution (ability to discriminate)"""
        if len(set(outcomes)) < 2:  # All same outcome
            return 0.0
        
        # Separate confidences by outcome
        positive_confidences = [conf for conf, out in zip(confidences, outcomes) if out > 0.5]
        negative_confidences = [conf for conf, out in zip(confidences, outcomes) if out <= 0.5]
        
        if not positive_confidences or not negative_confidences:
            return 0.0
        
        # Good resolution means higher confidence for positive outcomes
        avg_positive_conf = statistics.mean(positive_confidences)
        avg_negative_conf = statistics.mean(negative_confidences)
        
        resolution = avg_positive_conf - avg_negative_conf
        return max(0.0, min(1.0, resolution))
    
    def _calculate_brier_score(self, confidences: List[float], outcomes: List[float]) -> float:
        """Calculate Brier score"""
        squared_errors = [(conf - out) ** 2 for conf, out in zip(confidences, outcomes)]
        return statistics.mean(squared_errors)
    
    def _calculate_calibration_curve(self, confidences: List[float], outcomes: List[float]) -> List[Tuple[float, float]]:
        """Calculate calibration curve data points"""
        n_bins = self.tracking_params['calibration_bins']
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        curve_points = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            bin_center = (bin_lower + bin_upper) / 2
            
            # Find predictions in this bin
            in_bin = [(conf >= bin_lower and conf < bin_upper) for conf in confidences]
            
            if any(in_bin):
                bin_outcomes = [out for out, in_b in zip(outcomes, in_bin) if in_b]
                if bin_outcomes:
                    avg_accuracy = statistics.mean(bin_outcomes)
                    curve_points.append((bin_center, avg_accuracy))
        
        return curve_points
    
    def _create_default_metrics(self, agent_id: str) -> AccuracyMetrics:
        """Create default metrics for agent with no history"""
        return AccuracyMetrics(
            agent_id=agent_id,
            overall_accuracy=0.5,
            weighted_accuracy=0.5,
            decision_count=0,
            success_count=0,
            confidence_calibration=0.5,
            context_specific_accuracy={},
            temporal_trend=0.0,
            prediction_reliability=0.5,
            last_updated=datetime.now()
        )
    
    def _create_default_calibration(self, agent_id: str) -> CalibrationAnalysis:
        """Create default calibration analysis for agent with insufficient data"""
        return CalibrationAnalysis(
            agent_id=agent_id,
            calibration_score=0.5,
            overconfidence_bias=0.0,
            reliability_score=0.5,
            resolution_score=0.5,
            brier_score=0.25,
            calibration_curve_data=[]
        )
    
    def _update_tracker_metrics(self, metrics: AccuracyMetrics) -> None:
        """Update tracker performance metrics"""
        # Update agent count
        self.tracker_metrics['total_agents_tracked'] = len(self.decision_records)
        
        # Update accuracy distribution
        accuracy_bucket = f"{metrics.overall_accuracy:.1f}"
        self.tracker_metrics['accuracy_distribution'][accuracy_bucket] += 1
        
        # Update average calibration (simple approximation)
        current_avg_calibration = self.tracker_metrics['average_calibration_quality']
        total_agents = len(self.accuracy_cache)
        if total_agents > 0:
            self.tracker_metrics['average_calibration_quality'] = (
                (current_avg_calibration * (total_agents - 1) + metrics.confidence_calibration) / total_agents
            )
    
    def _generate_performance_insights(self, metrics: AccuracyMetrics, 
                                     calibration: CalibrationAnalysis) -> List[str]:
        """Generate actionable performance insights"""
        insights = []
        
        # Accuracy insights
        if metrics.overall_accuracy < 0.6:
            insights.append("Overall accuracy is below optimal threshold - consider additional training")
        elif metrics.overall_accuracy > 0.85:
            insights.append("Excellent overall accuracy performance")
        
        # Trend insights
        if metrics.temporal_trend > 0.1:
            insights.append("Performance is improving over time")
        elif metrics.temporal_trend < -0.1:
            insights.append("Performance decline detected - review recent decisions")
        
        # Calibration insights
        if calibration.overconfidence_bias > 0.1:
            insights.append("Tendency toward overconfidence - consider more conservative estimates")
        elif calibration.overconfidence_bias < -0.1:
            insights.append("Tendency toward underconfidence - appropriate but consider decision impact")
        
        # Reliability insights
        if metrics.prediction_reliability < 0.6:
            insights.append("Inconsistent performance - focus on decision process standardization")
        
        return insights
    
    def _load_historical_records(self) -> None:
        """Load historical decision records"""
        # This would load from knowledge system or database
        # For now, initialize with empty records
        pass
    
    def get_tracker_metrics(self) -> Dict[str, Any]:
        """Get tracker performance metrics"""
        return {
            **self.tracker_metrics,
            'agents_with_sufficient_data': len([a for a in self.decision_records.values() if len(a) >= 5]),
            'average_decisions_per_agent': statistics.mean([len(records) for records in self.decision_records.values()]) if self.decision_records else 0.0,
            'cache_hit_rates': {
                'accuracy_cache_size': len(self.accuracy_cache),
                'calibration_cache_size': len(self.calibration_cache)
            }
        }
    
    def export_tracking_data(self) -> Dict[str, Any]:
        """Export tracking data for persistence"""
        return {
            'decision_records': {
                agent: [
                    {
                        'decision_id': record.decision_id,
                        'decision_timestamp': record.decision_timestamp.isoformat(),
                        'decision_confidence': record.decision_confidence,
                        'outcome': record.outcome.value,
                        'outcome_timestamp': record.outcome_timestamp.isoformat(),
                        'context_category': record.context_category,
                        'impact_level': record.impact_level,
                        'validation_score': record.validation_score
                    }
                    for record in records
                ]
                for agent, records in self.decision_records.items()
            },
            'tracker_metrics': dict(self.tracker_metrics),
            'export_timestamp': datetime.now().isoformat()
        }


def main():
    """Demonstration of accuracy tracking functionality"""
    print("=== RIF Accuracy Tracker Demo ===\n")
    
    # Initialize tracker
    tracker = AccuracyTracker()
    
    # Example 1: Record decision outcomes
    print("Example 1: Recording Decision Outcomes")
    
    # Simulate decision records
    sample_decisions = [
        DecisionRecord(
            decision_id="decision_1",
            agent_id="rif-security",
            decision_timestamp=datetime.now() - timedelta(days=10),
            decision_confidence=0.85,
            decision_content="Approve security implementation",
            outcome=DecisionOutcome.SUCCESS,
            outcome_timestamp=datetime.now() - timedelta(days=8),
            context_category="security_critical",
            impact_level="high"
        ),
        DecisionRecord(
            decision_id="decision_2", 
            agent_id="rif-security",
            decision_timestamp=datetime.now() - timedelta(days=5),
            decision_confidence=0.90,
            decision_content="Validate encryption approach",
            outcome=DecisionOutcome.FAILURE,
            outcome_timestamp=datetime.now() - timedelta(days=3),
            context_category="security_critical",
            impact_level="critical"
        ),
        DecisionRecord(
            decision_id="decision_3",
            agent_id="rif-security", 
            decision_timestamp=datetime.now() - timedelta(days=2),
            decision_confidence=0.70,
            decision_content="Assess vulnerability impact",
            outcome=DecisionOutcome.SUCCESS,
            outcome_timestamp=datetime.now() - timedelta(days=1),
            context_category="security_critical",
            impact_level="medium"
        )
    ]
    
    for decision in sample_decisions:
        tracker.record_decision_outcome(decision)
        print(f"Recorded {decision.outcome.value} for {decision.agent_id}")
    
    print()
    
    # Example 2: Calculate accuracy metrics
    print("Example 2: Accuracy Metrics Calculation")
    
    metrics = tracker.calculate_accuracy_metrics("rif-security")
    print(f"Overall accuracy: {metrics.overall_accuracy:.3f}")
    print(f"Weighted accuracy: {metrics.weighted_accuracy:.3f}")
    print(f"Decision count: {metrics.decision_count}")
    print(f"Temporal trend: {metrics.temporal_trend:.3f}")
    print(f"Reliability: {metrics.prediction_reliability:.3f}")
    print(f"Context accuracy: {metrics.context_specific_accuracy}")
    
    print()
    
    # Example 3: Confidence calibration analysis
    print("Example 3: Confidence Calibration Analysis")
    
    calibration = tracker.analyze_confidence_calibration("rif-security")
    print(f"Calibration score: {calibration.calibration_score:.3f}")
    print(f"Overconfidence bias: {calibration.overconfidence_bias:.3f}")
    print(f"Brier score: {calibration.brier_score:.3f}")
    print(f"Resolution: {calibration.resolution_score:.3f}")
    
    print()
    
    # Example 4: Accuracy factor for vote weighting
    print("Example 4: Accuracy Factor for Vote Weighting")
    
    context = {'domain': 'security_critical'}
    factor = tracker.calculate_accuracy_factor("rif-security", context)
    print(f"Accuracy factor: {factor:.3f}")
    
    print()
    
    # Example 5: Performance summary
    print("Example 5: Performance Summary")
    
    summary = tracker.get_agent_performance_summary("rif-security")
    print("Overall Performance:")
    for key, value in summary['overall_performance'].items():
        print(f"  {key}: {value}")
    
    print("\nPerformance Insights:")
    for insight in summary['performance_insights']:
        print(f"  - {insight}")
    
    print()
    
    # Show metrics
    print("=== Tracker Metrics ===")
    metrics = tracker.get_tracker_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()