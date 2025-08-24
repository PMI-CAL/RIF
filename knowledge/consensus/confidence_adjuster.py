#!/usr/bin/env python3
"""
RIF Confidence Adjuster - Confidence-Based Vote Weighting
Part of Issue #62 implementation for vote weighting algorithm.

This module provides sophisticated confidence adjustment that considers:
- Agent self-assessment accuracy calibration
- Overconfidence and underconfidence corrections
- Context-sensitive confidence scaling
- Temporal confidence consistency
- Ensemble confidence harmonization
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


class ConfidenceScale(Enum):
    """Different confidence scale types"""
    BINARY = "binary"          # High/Low confidence
    CATEGORICAL = "categorical"  # Very Low, Low, Medium, High, Very High
    CONTINUOUS = "continuous"   # 0.0 - 1.0 scale
    PERCENTAGE = "percentage"   # 0% - 100% scale


class ConfidenceBias(Enum):
    """Types of confidence bias"""
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    WELL_CALIBRATED = "well_calibrated"
    INCONSISTENT = "inconsistent"


@dataclass
class ConfidenceRecord:
    """Record of confidence assessment and actual outcome"""
    agent_id: str
    decision_id: str
    stated_confidence: float  # What the agent stated
    actual_outcome: float     # What actually happened (0.0 or 1.0)
    context_complexity: str   # "low", "medium", "high"
    domain: str
    timestamp: datetime
    decision_impact: str = "medium"
    calibration_error: float = 0.0  # Calculated after outcome known


@dataclass
class ConfidenceCalibration:
    """Calibration analysis for an agent's confidence assessments"""
    agent_id: str
    calibration_slope: float  # Linear relationship between confidence and accuracy
    calibration_intercept: float
    mean_calibration_error: float
    confidence_bias_type: ConfidenceBias
    confidence_consistency: float  # How consistent confidence levels are
    overconfidence_penalty: float  # Multiplier for overconfident agents
    underconfidence_boost: float   # Multiplier for underconfident agents
    context_sensitivity: Dict[str, float]  # How confidence varies by context
    last_updated: datetime


@dataclass
class ConfidenceAdjustment:
    """Result of confidence adjustment calculation"""
    original_confidence: float
    adjusted_confidence: float
    adjustment_factor: float
    bias_correction: float
    context_scaling: float
    temporal_consistency: float
    ensemble_harmonization: float
    adjustment_metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceAdjuster:
    """
    Advanced confidence adjustment system for RIF vote weighting.
    
    This system adjusts agent confidence levels to improve voting accuracy by:
    - Calibrating confidence based on historical accuracy
    - Correcting for systematic over/underconfidence bias
    - Scaling confidence based on decision context
    - Ensuring temporal consistency in confidence assessment
    - Harmonizing confidence across agent ensembles
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize confidence adjuster"""
        self.knowledge_system = knowledge_system
        
        # Confidence tracking data
        self.confidence_records: Dict[str, List[ConfidenceRecord]] = defaultdict(list)
        self.calibration_models: Dict[str, ConfidenceCalibration] = {}
        self.adjustment_cache: Dict[str, Dict[str, ConfidenceAdjustment]] = defaultdict(dict)
        
        # Adjustment parameters
        self.adjustment_params = {
            'max_records_per_agent': 100,
            'min_records_for_calibration': 10,
            'overconfidence_penalty_rate': 0.15,  # Max penalty for overconfidence
            'underconfidence_boost_rate': 0.10,   # Max boost for underconfidence
            'temporal_consistency_window': 30,     # Days for consistency analysis
            'context_scaling_factors': {
                'low_complexity': 1.0,
                'medium_complexity': 0.95,
                'high_complexity': 0.90
            },
            'ensemble_harmonization_strength': 0.1,  # How much to adjust for ensemble consistency
            'calibration_update_threshold': 0.05     # Minimum change to trigger recalibration
        }
        
        # Bias detection thresholds
        self.bias_thresholds = {
            'overconfident_threshold': 0.10,    # Mean error for overconfidence
            'underconfident_threshold': -0.10,  # Mean error for underconfidence
            'consistency_threshold': 0.15       # Std dev for consistency
        }
        
        # Performance tracking
        self.adjuster_metrics = {
            'total_adjustments_made': 0,
            'average_adjustment_magnitude': 0.0,
            'bias_correction_stats': {
                'overconfidence_corrections': 0,
                'underconfidence_corrections': 0,
                'well_calibrated_count': 0
            },
            'context_adjustment_frequency': defaultdict(int),
            'calibration_accuracy_improvement': 0.0
        }
        
        # Load existing calibration data
        self._load_calibration_history()
        
        logger.info("Confidence Adjuster initialized")
    
    def record_confidence_outcome(self, confidence_record: ConfidenceRecord) -> None:
        """
        Record a confidence assessment and its actual outcome.
        
        Args:
            confidence_record: Complete confidence record with outcome
        """
        agent_id = confidence_record.agent_id
        
        # Calculate calibration error
        confidence_record.calibration_error = (
            confidence_record.stated_confidence - confidence_record.actual_outcome
        )
        
        # Add to records
        self.confidence_records[agent_id].append(confidence_record)
        
        # Maintain record limit
        max_records = self.adjustment_params['max_records_per_agent']
        if len(self.confidence_records[agent_id]) > max_records:
            self.confidence_records[agent_id] = self.confidence_records[agent_id][-max_records:]
        
        # Invalidate calibration cache for this agent
        if agent_id in self.calibration_models:
            old_calibration = self.calibration_models[agent_id]
            # Only recalibrate if significant change
            if abs(confidence_record.calibration_error) > self.adjustment_params['calibration_update_threshold']:
                del self.calibration_models[agent_id]
        
        # Clear adjustment cache
        if agent_id in self.adjustment_cache:
            self.adjustment_cache[agent_id].clear()
        
        logger.debug(f"Recorded confidence outcome for {agent_id}: "
                    f"stated {confidence_record.stated_confidence:.2f}, "
                    f"actual {confidence_record.actual_outcome:.2f}")
    
    def calculate_confidence_factor(self, agent_id: str, context: Dict[str, Any]) -> float:
        """
        Calculate adjusted confidence factor for vote weighting.
        
        This is the main interface used by VoteWeightCalculator.
        
        Args:
            agent_id: Agent identifier
            context: Decision context including stated confidence
            
        Returns:
            Adjusted confidence factor (0.6 - 1.4 range for weight calculation)
        """
        # Extract confidence from context
        stated_confidence = self._extract_confidence_from_context(context)
        
        # Get detailed adjustment
        adjustment = self.adjust_confidence(agent_id, stated_confidence, context)
        
        # Convert to factor (0.6 - 1.4 range)
        confidence_factor = 0.6 + (adjustment.adjusted_confidence * 0.8)
        
        return min(1.4, max(0.6, confidence_factor))
    
    def adjust_confidence(self, agent_id: str, stated_confidence: float, 
                         context: Dict[str, Any]) -> ConfidenceAdjustment:
        """
        Perform comprehensive confidence adjustment.
        
        Args:
            agent_id: Agent identifier
            stated_confidence: Agent's stated confidence level
            context: Decision context
            
        Returns:
            ConfidenceAdjustment with detailed adjustment breakdown
        """
        try:
            # Check cache first
            cache_key = f"{stated_confidence:.2f}_{context.get('domain', 'general')}"
            if cache_key in self.adjustment_cache[agent_id]:
                return self.adjustment_cache[agent_id][cache_key]
            
            # Get or create calibration model
            calibration = self._get_agent_calibration(agent_id)
            
            # Apply bias correction
            bias_corrected_confidence = self._apply_bias_correction(
                stated_confidence, calibration
            )
            
            # Apply context scaling
            context_scaled_confidence = self._apply_context_scaling(
                bias_corrected_confidence, context
            )
            
            # Apply temporal consistency adjustment
            temporal_adjusted_confidence = self._apply_temporal_consistency(
                agent_id, context_scaled_confidence, context
            )
            
            # Apply ensemble harmonization (if ensemble context provided)
            final_confidence = self._apply_ensemble_harmonization(
                temporal_adjusted_confidence, context
            )
            
            # Calculate adjustment components
            adjustment_factor = final_confidence / stated_confidence if stated_confidence > 0 else 1.0
            bias_correction = bias_corrected_confidence - stated_confidence
            context_scaling = context_scaled_confidence - bias_corrected_confidence
            temporal_consistency = temporal_adjusted_confidence - context_scaled_confidence
            ensemble_harmonization = final_confidence - temporal_adjusted_confidence
            
            # Create adjustment result
            adjustment = ConfidenceAdjustment(
                original_confidence=stated_confidence,
                adjusted_confidence=final_confidence,
                adjustment_factor=adjustment_factor,
                bias_correction=bias_correction,
                context_scaling=context_scaling,
                temporal_consistency=temporal_consistency,
                ensemble_harmonization=ensemble_harmonization,
                adjustment_metadata={
                    'agent_id': agent_id,
                    'calibration_type': calibration.confidence_bias_type.value,
                    'context_domain': context.get('domain', 'general'),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Cache the result
            self.adjustment_cache[agent_id][cache_key] = adjustment
            
            # Update metrics
            self._update_adjustment_metrics(adjustment)
            
            logger.debug(f"Adjusted confidence for {agent_id}: "
                        f"{stated_confidence:.3f} -> {final_confidence:.3f}")
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error adjusting confidence for {agent_id}: {str(e)}")
            return self._create_fallback_adjustment(stated_confidence)
    
    def analyze_agent_confidence_patterns(self, agent_id: str) -> Dict[str, Any]:
        """
        Analyze confidence patterns for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with confidence pattern analysis
        """
        records = self.confidence_records.get(agent_id, [])
        calibration = self._get_agent_calibration(agent_id)
        
        if not records:
            return self._create_default_pattern_analysis(agent_id)
        
        # Basic statistics
        stated_confidences = [r.stated_confidence for r in records]
        calibration_errors = [r.calibration_error for r in records]
        
        analysis = {
            'agent_id': agent_id,
            'record_count': len(records),
            'confidence_statistics': {
                'mean_stated_confidence': statistics.mean(stated_confidences),
                'confidence_std_dev': statistics.stdev(stated_confidences) if len(stated_confidences) > 1 else 0.0,
                'min_confidence': min(stated_confidences),
                'max_confidence': max(stated_confidences)
            },
            'calibration_analysis': {
                'mean_calibration_error': statistics.mean(calibration_errors),
                'calibration_std_dev': statistics.stdev(calibration_errors) if len(calibration_errors) > 1 else 0.0,
                'bias_type': calibration.confidence_bias_type.value,
                'calibration_quality': 1.0 - abs(calibration.mean_calibration_error)
            },
            'context_patterns': self._analyze_context_patterns(records),
            'temporal_patterns': self._analyze_temporal_patterns(records),
            'recommendations': self._generate_confidence_recommendations(calibration, records)
        }
        
        return analysis
    
    def _extract_confidence_from_context(self, context: Dict[str, Any]) -> float:
        """Extract confidence value from context dictionary"""
        confidence = context.get('confidence', 1.0)
        
        # Handle different confidence input types
        if isinstance(confidence, str):
            confidence_map = {
                'very_low': 0.2,
                'low': 0.4,
                'medium': 0.6,
                'high': 0.8,
                'very_high': 1.0
            }
            return confidence_map.get(confidence.lower(), 0.6)
        elif isinstance(confidence, (int, float)):
            # Normalize to 0-1 range
            if confidence > 1.0:
                return confidence / 100.0  # Assume percentage
            else:
                return max(0.0, min(1.0, float(confidence)))
        else:
            return 0.6  # Default medium confidence
    
    def _get_agent_calibration(self, agent_id: str) -> ConfidenceCalibration:
        """Get or create calibration model for agent"""
        if agent_id in self.calibration_models:
            return self.calibration_models[agent_id]
        
        # Create new calibration model
        records = self.confidence_records.get(agent_id, [])
        calibration = self._calculate_agent_calibration(agent_id, records)
        self.calibration_models[agent_id] = calibration
        
        return calibration
    
    def _calculate_agent_calibration(self, agent_id: str, 
                                   records: List[ConfidenceRecord]) -> ConfidenceCalibration:
        """Calculate calibration model for an agent"""
        min_records = self.adjustment_params['min_records_for_calibration']
        
        if len(records) < min_records:
            return self._create_default_calibration(agent_id)
        
        # Extract confidence and outcome data
        confidences = [r.stated_confidence for r in records]
        outcomes = [r.actual_outcome for r in records]
        errors = [r.calibration_error for r in records]
        
        # Calculate calibration metrics
        mean_error = statistics.mean(errors)
        error_std = statistics.stdev(errors) if len(errors) > 1 else 0.0
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        # Determine bias type
        bias_type = self._determine_bias_type(mean_error, error_std)
        
        # Calculate calibration slope and intercept (simple linear regression)
        slope, intercept = self._calculate_calibration_line(confidences, outcomes)
        
        # Calculate context sensitivity
        context_sensitivity = self._calculate_context_sensitivity(records)
        
        # Calculate adjustment factors based on bias
        overconfidence_penalty, underconfidence_boost = self._calculate_adjustment_factors(
            bias_type, mean_error
        )
        
        calibration = ConfidenceCalibration(
            agent_id=agent_id,
            calibration_slope=slope,
            calibration_intercept=intercept,
            mean_calibration_error=mean_error,
            confidence_bias_type=bias_type,
            confidence_consistency=1.0 / (1.0 + error_std),  # Inverse of error variance
            overconfidence_penalty=overconfidence_penalty,
            underconfidence_boost=underconfidence_boost,
            context_sensitivity=context_sensitivity,
            last_updated=datetime.now()
        )
        
        return calibration
    
    def _apply_bias_correction(self, confidence: float, 
                             calibration: ConfidenceCalibration) -> float:
        """Apply bias correction to confidence"""
        if calibration.confidence_bias_type == ConfidenceBias.OVERCONFIDENT:
            # Reduce confidence for overconfident agents
            penalty = calibration.overconfidence_penalty
            corrected = confidence * (1.0 - penalty)
        elif calibration.confidence_bias_type == ConfidenceBias.UNDERCONFIDENT:
            # Boost confidence for underconfident agents
            boost = calibration.underconfidence_boost
            corrected = confidence * (1.0 + boost)
        else:
            # Well-calibrated or inconsistent - minimal adjustment
            corrected = confidence * (1.0 + calibration.mean_calibration_error * -0.1)
        
        return max(0.0, min(1.0, corrected))
    
    def _apply_context_scaling(self, confidence: float, context: Dict[str, Any]) -> float:
        """Apply context-based confidence scaling"""
        complexity = context.get('complexity', 'medium')
        domain = context.get('domain', 'general')
        
        # Get scaling factor for complexity
        scaling_factors = self.adjustment_params['context_scaling_factors']
        complexity_key = f"{complexity}_complexity"
        scaling_factor = scaling_factors.get(complexity_key, 0.95)
        
        # Apply domain-specific adjustments
        domain_adjustments = {
            'security': 0.95,    # Conservative for security
            'performance': 0.98, # Slightly conservative for performance
            'general': 1.0       # No adjustment for general decisions
        }
        domain_factor = domain_adjustments.get(domain, 1.0)
        
        # Combined scaling
        scaled_confidence = confidence * scaling_factor * domain_factor
        
        return max(0.0, min(1.0, scaled_confidence))
    
    def _apply_temporal_consistency(self, agent_id: str, confidence: float, 
                                  context: Dict[str, Any]) -> float:
        """Apply temporal consistency adjustment"""
        records = self.confidence_records.get(agent_id, [])
        
        if len(records) < 3:
            return confidence  # Need minimum history for consistency check
        
        # Get recent confidence levels
        window_days = self.adjustment_params['temporal_consistency_window']
        recent_date = datetime.now() - timedelta(days=window_days)
        recent_records = [r for r in records if r.timestamp >= recent_date]
        
        if len(recent_records) < 2:
            return confidence
        
        # Calculate consistency
        recent_confidences = [r.stated_confidence for r in recent_records]
        recent_mean = statistics.mean(recent_confidences)
        recent_std = statistics.stdev(recent_confidences) if len(recent_confidences) > 1 else 0.0
        
        # If current confidence is very different from recent pattern, adjust toward consistency
        if recent_std < 0.1:  # Very consistent recent pattern
            deviation = abs(confidence - recent_mean)
            if deviation > 0.2:  # Significant deviation
                # Move toward recent pattern
                adjustment_strength = 0.3
                adjusted = confidence * (1 - adjustment_strength) + recent_mean * adjustment_strength
                return max(0.0, min(1.0, adjusted))
        
        return confidence
    
    def _apply_ensemble_harmonization(self, confidence: float, 
                                    context: Dict[str, Any]) -> float:
        """Apply ensemble-level confidence harmonization"""
        # This would adjust confidence based on ensemble patterns
        # For now, return unchanged - would be implemented with ensemble context
        ensemble_confidences = context.get('ensemble_confidences', [])
        
        if len(ensemble_confidences) < 2:
            return confidence
        
        # Calculate ensemble statistics
        ensemble_mean = statistics.mean(ensemble_confidences)
        ensemble_std = statistics.stdev(ensemble_confidences)
        
        # If this confidence is an outlier, adjust toward ensemble
        if ensemble_std > 0:
            z_score = abs(confidence - ensemble_mean) / ensemble_std
            if z_score > 2.0:  # Outlier
                harmonization_strength = self.adjustment_params['ensemble_harmonization_strength']
                harmonized = confidence * (1 - harmonization_strength) + ensemble_mean * harmonization_strength
                return max(0.0, min(1.0, harmonized))
        
        return confidence
    
    def _determine_bias_type(self, mean_error: float, error_std: float) -> ConfidenceBias:
        """Determine confidence bias type from calibration errors"""
        overconfident_threshold = self.bias_thresholds['overconfident_threshold']
        underconfident_threshold = self.bias_thresholds['underconfident_threshold']
        consistency_threshold = self.bias_thresholds['consistency_threshold']
        
        if mean_error > overconfident_threshold:
            return ConfidenceBias.OVERCONFIDENT
        elif mean_error < underconfident_threshold:
            return ConfidenceBias.UNDERCONFIDENT
        elif error_std > consistency_threshold:
            return ConfidenceBias.INCONSISTENT
        else:
            return ConfidenceBias.WELL_CALIBRATED
    
    def _calculate_calibration_line(self, confidences: List[float], 
                                  outcomes: List[float]) -> Tuple[float, float]:
        """Calculate calibration line slope and intercept"""
        if len(confidences) < 2:
            return 1.0, 0.0
        
        # Simple linear regression
        n = len(confidences)
        x_mean = statistics.mean(confidences)
        y_mean = statistics.mean(outcomes)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(confidences, outcomes))
        denominator = sum((x - x_mean) ** 2 for x in confidences)
        
        if denominator == 0:
            return 1.0, y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return slope, intercept
    
    def _calculate_context_sensitivity(self, records: List[ConfidenceRecord]) -> Dict[str, float]:
        """Calculate how confidence varies by context"""
        context_groups = defaultdict(list)
        
        for record in records:
            context_key = f"{record.domain}_{record.context_complexity}"
            context_groups[context_key].append(record.stated_confidence)
        
        context_sensitivity = {}
        for context, confidences in context_groups.items():
            if len(confidences) >= 3:
                context_sensitivity[context] = statistics.mean(confidences)
        
        return context_sensitivity
    
    def _calculate_adjustment_factors(self, bias_type: ConfidenceBias, 
                                    mean_error: float) -> Tuple[float, float]:
        """Calculate overconfidence penalty and underconfidence boost"""
        max_penalty = self.adjustment_params['overconfidence_penalty_rate']
        max_boost = self.adjustment_params['underconfidence_boost_rate']
        
        if bias_type == ConfidenceBias.OVERCONFIDENT:
            penalty = min(max_penalty, abs(mean_error))
            return penalty, 0.0
        elif bias_type == ConfidenceBias.UNDERCONFIDENT:
            boost = min(max_boost, abs(mean_error))
            return 0.0, boost
        else:
            return 0.0, 0.0
    
    def _analyze_context_patterns(self, records: List[ConfidenceRecord]) -> Dict[str, Any]:
        """Analyze confidence patterns by context"""
        domain_patterns = defaultdict(list)
        complexity_patterns = defaultdict(list)
        
        for record in records:
            domain_patterns[record.domain].append(record.calibration_error)
            complexity_patterns[record.context_complexity].append(record.calibration_error)
        
        patterns = {
            'domain_calibration': {
                domain: statistics.mean(errors) 
                for domain, errors in domain_patterns.items() 
                if len(errors) >= 3
            },
            'complexity_calibration': {
                complexity: statistics.mean(errors)
                for complexity, errors in complexity_patterns.items()
                if len(errors) >= 3
            }
        }
        
        return patterns
    
    def _analyze_temporal_patterns(self, records: List[ConfidenceRecord]) -> Dict[str, Any]:
        """Analyze temporal confidence patterns"""
        if len(records) < 10:
            return {'insufficient_data': True}
        
        # Sort by timestamp
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        
        # Calculate trend in calibration error
        errors = [r.calibration_error for r in sorted_records]
        x_values = list(range(len(errors)))
        
        # Simple linear regression for trend
        slope, _ = self._calculate_calibration_line(x_values, errors)
        
        # Recent vs. historical performance
        midpoint = len(sorted_records) // 2
        recent_errors = errors[midpoint:]
        historical_errors = errors[:midpoint]
        
        recent_mean = statistics.mean(recent_errors)
        historical_mean = statistics.mean(historical_errors)
        
        return {
            'calibration_trend': slope,
            'recent_vs_historical': {
                'recent_error': recent_mean,
                'historical_error': historical_mean,
                'improvement': historical_mean - recent_mean
            },
            'trend_interpretation': 'improving' if slope < -0.01 else 'declining' if slope > 0.01 else 'stable'
        }
    
    def _generate_confidence_recommendations(self, calibration: ConfidenceCalibration, 
                                           records: List[ConfidenceRecord]) -> List[str]:
        """Generate recommendations for confidence improvement"""
        recommendations = []
        
        if calibration.confidence_bias_type == ConfidenceBias.OVERCONFIDENT:
            recommendations.append("Consider more conservative confidence assessments")
            recommendations.append("Seek additional validation before high-confidence decisions")
        elif calibration.confidence_bias_type == ConfidenceBias.UNDERCONFIDENT:
            recommendations.append("Trust your expertise more - your assessments are often correct")
            recommendations.append("Consider the impact of excessive caution on decision making")
        elif calibration.confidence_bias_type == ConfidenceBias.INCONSISTENT:
            recommendations.append("Focus on developing consistent confidence assessment process")
            recommendations.append("Consider decision context more systematically")
        
        if calibration.confidence_consistency < 0.7:
            recommendations.append("Work on consistency in confidence assessment approach")
        
        return recommendations
    
    def _create_default_calibration(self, agent_id: str) -> ConfidenceCalibration:
        """Create default calibration for agent with insufficient data"""
        return ConfidenceCalibration(
            agent_id=agent_id,
            calibration_slope=1.0,
            calibration_intercept=0.0,
            mean_calibration_error=0.0,
            confidence_bias_type=ConfidenceBias.WELL_CALIBRATED,
            confidence_consistency=0.8,
            overconfidence_penalty=0.0,
            underconfidence_boost=0.0,
            context_sensitivity={},
            last_updated=datetime.now()
        )
    
    def _create_fallback_adjustment(self, stated_confidence: float) -> ConfidenceAdjustment:
        """Create fallback adjustment when error occurs"""
        return ConfidenceAdjustment(
            original_confidence=stated_confidence,
            adjusted_confidence=stated_confidence,
            adjustment_factor=1.0,
            bias_correction=0.0,
            context_scaling=0.0,
            temporal_consistency=0.0,
            ensemble_harmonization=0.0
        )
    
    def _create_default_pattern_analysis(self, agent_id: str) -> Dict[str, Any]:
        """Create default pattern analysis for agent with no data"""
        return {
            'agent_id': agent_id,
            'record_count': 0,
            'confidence_statistics': {},
            'calibration_analysis': {'insufficient_data': True},
            'context_patterns': {},
            'temporal_patterns': {},
            'recommendations': ['Build confidence assessment history through more decisions']
        }
    
    def _update_adjustment_metrics(self, adjustment: ConfidenceAdjustment) -> None:
        """Update adjuster performance metrics"""
        self.adjuster_metrics['total_adjustments_made'] += 1
        
        # Update average adjustment magnitude
        adjustment_magnitude = abs(adjustment.adjustment_factor - 1.0)
        total_adjustments = self.adjuster_metrics['total_adjustments_made']
        current_avg_magnitude = self.adjuster_metrics['average_adjustment_magnitude']
        
        self.adjuster_metrics['average_adjustment_magnitude'] = (
            (current_avg_magnitude * (total_adjustments - 1) + adjustment_magnitude) / total_adjustments
        )
        
        # Update bias correction stats
        if adjustment.bias_correction > 0.05:
            self.adjuster_metrics['bias_correction_stats']['underconfidence_corrections'] += 1
        elif adjustment.bias_correction < -0.05:
            self.adjuster_metrics['bias_correction_stats']['overconfidence_corrections'] += 1
        else:
            self.adjuster_metrics['bias_correction_stats']['well_calibrated_count'] += 1
        
        # Update context adjustment frequency
        domain = adjustment.adjustment_metadata.get('context_domain', 'general')
        self.adjuster_metrics['context_adjustment_frequency'][domain] += 1
    
    def _load_calibration_history(self) -> None:
        """Load historical calibration data"""
        # This would load from knowledge system or database
        # For now, initialize with empty data
        pass
    
    def get_adjuster_metrics(self) -> Dict[str, Any]:
        """Get adjuster performance metrics"""
        return {
            **self.adjuster_metrics,
            'agents_with_calibration': len(self.calibration_models),
            'total_confidence_records': sum(len(records) for records in self.confidence_records.values()),
            'cache_efficiency': {
                'cached_agents': len(self.adjustment_cache),
                'average_cache_size': statistics.mean([len(cache) for cache in self.adjustment_cache.values()]) if self.adjustment_cache else 0.0
            }
        }
    
    def export_confidence_data(self) -> Dict[str, Any]:
        """Export confidence data for persistence"""
        return {
            'confidence_records': {
                agent: [
                    {
                        'decision_id': record.decision_id,
                        'stated_confidence': record.stated_confidence,
                        'actual_outcome': record.actual_outcome,
                        'context_complexity': record.context_complexity,
                        'domain': record.domain,
                        'timestamp': record.timestamp.isoformat(),
                        'calibration_error': record.calibration_error
                    }
                    for record in records
                ]
                for agent, records in self.confidence_records.items()
            },
            'calibration_models': {
                agent: {
                    'calibration_slope': calibration.calibration_slope,
                    'calibration_intercept': calibration.calibration_intercept,
                    'mean_calibration_error': calibration.mean_calibration_error,
                    'confidence_bias_type': calibration.confidence_bias_type.value,
                    'confidence_consistency': calibration.confidence_consistency,
                    'last_updated': calibration.last_updated.isoformat()
                }
                for agent, calibration in self.calibration_models.items()
            },
            'adjuster_metrics': dict(self.adjuster_metrics),
            'export_timestamp': datetime.now().isoformat()
        }


def main():
    """Demonstration of confidence adjustment functionality"""
    print("=== RIF Confidence Adjuster Demo ===\n")
    
    # Initialize adjuster
    adjuster = ConfidenceAdjuster()
    
    # Example 1: Record confidence outcomes
    print("Example 1: Recording Confidence Outcomes")
    
    sample_records = [
        ConfidenceRecord(
            agent_id="rif-implementer",
            decision_id="decision_1",
            stated_confidence=0.9,
            actual_outcome=1.0,  # Success
            context_complexity="medium",
            domain="implementation",
            timestamp=datetime.now() - timedelta(days=5)
        ),
        ConfidenceRecord(
            agent_id="rif-implementer",
            decision_id="decision_2",
            stated_confidence=0.85,
            actual_outcome=0.0,  # Failure
            context_complexity="high",
            domain="implementation",
            timestamp=datetime.now() - timedelta(days=3)
        ),
        ConfidenceRecord(
            agent_id="rif-implementer",
            decision_id="decision_3",
            stated_confidence=0.75,
            actual_outcome=1.0,  # Success
            context_complexity="medium",
            domain="implementation",
            timestamp=datetime.now() - timedelta(days=1)
        )
    ]
    
    for record in sample_records:
        adjuster.record_confidence_outcome(record)
        print(f"Recorded confidence outcome: {record.stated_confidence:.2f} -> {record.actual_outcome}")
    
    print()
    
    # Example 2: Confidence adjustment
    print("Example 2: Confidence Adjustment")
    
    context = {
        'confidence': 0.8,
        'domain': 'implementation',
        'complexity': 'high'
    }
    
    adjustment = adjuster.adjust_confidence("rif-implementer", 0.8, context)
    print(f"Original confidence: {adjustment.original_confidence:.3f}")
    print(f"Adjusted confidence: {adjustment.adjusted_confidence:.3f}")
    print(f"Adjustment factor: {adjustment.adjustment_factor:.3f}")
    print(f"Bias correction: {adjustment.bias_correction:.3f}")
    print(f"Context scaling: {adjustment.context_scaling:.3f}")
    
    print()
    
    # Example 3: Confidence factor for vote weighting
    print("Example 3: Confidence Factor for Vote Weighting")
    
    factor = adjuster.calculate_confidence_factor("rif-implementer", context)
    print(f"Confidence factor for vote weighting: {factor:.3f}")
    
    print()
    
    # Example 4: Agent confidence pattern analysis
    print("Example 4: Agent Confidence Pattern Analysis")
    
    analysis = adjuster.analyze_agent_confidence_patterns("rif-implementer")
    
    print(f"Record count: {analysis['record_count']}")
    if 'confidence_statistics' in analysis and analysis['confidence_statistics']:
        stats = analysis['confidence_statistics']
        print(f"Mean stated confidence: {stats['mean_stated_confidence']:.3f}")
        print(f"Confidence std dev: {stats['confidence_std_dev']:.3f}")
    
    if 'calibration_analysis' in analysis and 'bias_type' in analysis['calibration_analysis']:
        calib = analysis['calibration_analysis']
        print(f"Bias type: {calib['bias_type']}")
        print(f"Calibration quality: {calib.get('calibration_quality', 'N/A')}")
    
    print("\nRecommendations:")
    for rec in analysis.get('recommendations', []):
        print(f"  - {rec}")
    
    print()
    
    # Show metrics
    print("=== Adjuster Metrics ===")
    metrics = adjuster.get_adjuster_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()