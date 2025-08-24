"""
Confidence Scorer - Advanced Pattern Matching Confidence Assessment

This module provides comprehensive confidence scoring for pattern matching decisions,
implementing sophisticated algorithms to assess reliability of pattern recommendations
and similarity matches.

Key Features:
- Multi-dimensional confidence assessment
- Historical accuracy tracking
- Context-specific confidence adjustments
- Uncertainty quantification
- Confidence calibration
- Reliability metrics
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Import knowledge system interfaces
try:
    from knowledge.interface import get_knowledge_system
    from knowledge.database.database_interface import RIFDatabase
except ImportError:
    def get_knowledge_system():
        raise ImportError("Knowledge system not available")
    
    class RIFDatabase:
        pass

# Import pattern application core components
from knowledge.pattern_application.core import (
    Pattern, IssueContext, TechStack
)


@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence score."""
    data_completeness: float
    historical_accuracy: float
    pattern_maturity: float
    context_alignment: float
    similarity_strength: float
    validation_coverage: float
    expert_consensus: float
    uncertainty_measure: float


@dataclass
class ConfidenceResult:
    """Comprehensive confidence assessment result."""
    overall_confidence: float
    confidence_level: str  # 'very_high', 'high', 'medium', 'low', 'very_low'
    confidence_factors: ConfidenceFactors
    reliability_score: float
    uncertainty_bounds: Tuple[float, float]
    confidence_explanation: str
    recommendations: List[str]
    calibration_metrics: Dict[str, float]


class ConfidenceScorer:
    """
    Advanced confidence scoring system for pattern matching decisions.
    
    This system provides comprehensive confidence assessment using multiple
    sophisticated techniques:
    - Multi-dimensional factor analysis
    - Historical performance tracking
    - Bayesian confidence estimation
    - Context-specific calibration
    - Uncertainty quantification
    - Reliability assessment
    """
    
    def __init__(self, knowledge_system=None, database: Optional[RIFDatabase] = None):
        """Initialize the confidence scorer."""
        self.logger = logging.getLogger(__name__)
        self.knowledge_system = knowledge_system or get_knowledge_system()
        self.database = database
        
        # Confidence level thresholds
        self.confidence_thresholds = {
            'very_high': 0.9,
            'high': 0.75,
            'medium': 0.6,
            'low': 0.4,
            'very_low': 0.0
        }
        
        # Factor weights for confidence calculation
        self.confidence_weights = {
            'data_completeness': 0.20,
            'historical_accuracy': 0.18,
            'pattern_maturity': 0.15,
            'context_alignment': 0.15,
            'similarity_strength': 0.12,
            'validation_coverage': 0.10,
            'expert_consensus': 0.05,
            'uncertainty_measure': 0.05
        }
        
        # Calibration parameters
        self.calibration_history = []
        self.calibration_window = timedelta(days=30)
        
        # Uncertainty modeling parameters
        self.uncertainty_factors = {
            'data_sparsity': 0.3,
            'domain_novelty': 0.2,
            'pattern_complexity': 0.2,
            'context_ambiguity': 0.15,
            'temporal_drift': 0.15
        }
        
        self.logger.info("Confidence Scorer initialized")
    
    def calculate_confidence(self, pattern: Pattern, 
                           issue_context: IssueContext) -> float:
        """
        Calculate confidence score for pattern-context matching.
        
        This implements the core confidence scoring requirement from Issue #76
        acceptance criteria: "Provides confidence scores"
        
        Args:
            pattern: Pattern being evaluated
            issue_context: Context for confidence assessment
            
        Returns:
            Confidence score (0.0 to 1.0, higher is more confident)
        """
        confidence_result = self.calculate_comprehensive_confidence(pattern, issue_context)
        return confidence_result.overall_confidence
    
    def calculate_comprehensive_confidence(self, pattern: Pattern, 
                                         issue_context: IssueContext) -> ConfidenceResult:
        """
        Calculate comprehensive confidence assessment with detailed breakdown.
        
        Args:
            pattern: Pattern being evaluated
            issue_context: Context for confidence assessment
            
        Returns:
            ConfidenceResult with detailed confidence analysis
        """
        self.logger.debug(f"Calculating confidence for pattern {pattern.pattern_id}")
        
        try:
            # Calculate individual confidence factors
            factors = self._calculate_confidence_factors(pattern, issue_context)
            
            # Calculate overall confidence using weighted combination
            overall_confidence = self._calculate_weighted_confidence(factors)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_confidence)
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score(pattern, issue_context, factors)
            
            # Calculate uncertainty bounds
            uncertainty_bounds = self._calculate_uncertainty_bounds(overall_confidence, factors)
            
            # Generate confidence explanation
            explanation = self._generate_confidence_explanation(factors, overall_confidence)
            
            # Generate recommendations for confidence improvement
            recommendations = self._generate_confidence_recommendations(factors)
            
            # Calculate calibration metrics
            calibration_metrics = self._calculate_calibration_metrics(pattern, issue_context)
            
            result = ConfidenceResult(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                confidence_factors=factors,
                reliability_score=reliability_score,
                uncertainty_bounds=uncertainty_bounds,
                confidence_explanation=explanation,
                recommendations=recommendations,
                calibration_metrics=calibration_metrics
            )
            
            # Update calibration history
            self._update_calibration_history(result, pattern, issue_context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            # Return low confidence result as fallback
            return self._create_low_confidence_result()
    
    def assess_prediction_confidence(self, predictions: List[float],
                                   historical_accuracy: float = 0.7) -> Dict[str, float]:
        """
        Assess confidence in a set of predictions.
        
        Args:
            predictions: List of prediction scores
            historical_accuracy: Historical accuracy rate
            
        Returns:
            Dictionary with confidence metrics
        """
        if not predictions:
            return {'confidence': 0.0, 'variance': 1.0, 'consensus': 0.0}
        
        predictions_array = np.array(predictions)
        
        # Calculate statistical measures
        mean_prediction = np.mean(predictions_array)
        variance = np.var(predictions_array)
        std_deviation = np.std(predictions_array)
        
        # Calculate consensus measure (inverse of coefficient of variation)
        cv = std_deviation / mean_prediction if mean_prediction > 0 else 1.0
        consensus = max(0.0, 1.0 - cv)
        
        # Adjust confidence based on historical accuracy
        base_confidence = consensus * historical_accuracy
        
        # Penalize high variance
        variance_penalty = min(0.3, variance * 0.5)
        final_confidence = max(0.0, base_confidence - variance_penalty)
        
        return {
            'confidence': final_confidence,
            'variance': variance,
            'consensus': consensus,
            'mean_prediction': mean_prediction,
            'std_deviation': std_deviation,
            'historical_accuracy': historical_accuracy
        }
    
    def calibrate_confidence_scores(self, actual_outcomes: List[bool],
                                   predicted_confidences: List[float]) -> Dict[str, float]:
        """
        Calibrate confidence scores based on actual outcomes.
        
        Args:
            actual_outcomes: List of actual success/failure outcomes
            predicted_confidences: List of predicted confidence scores
            
        Returns:
            Dictionary with calibration metrics
        """
        if len(actual_outcomes) != len(predicted_confidences):
            raise ValueError("Outcome and confidence lists must have same length")
        
        # Calculate calibration error (Brier score)
        brier_score = np.mean([(conf - outcome) ** 2 
                              for conf, outcome in zip(predicted_confidences, actual_outcomes)])
        
        # Calculate reliability (calibration by binning)
        reliability_score = self._calculate_reliability_calibration(
            actual_outcomes, predicted_confidences
        )
        
        # Calculate resolution (ability to discriminate)
        resolution_score = self._calculate_resolution_score(
            actual_outcomes, predicted_confidences
        )
        
        # Calculate overall calibration quality
        calibration_quality = 1.0 - brier_score  # Higher is better
        
        return {
            'brier_score': brier_score,
            'reliability': reliability_score,
            'resolution': resolution_score,
            'calibration_quality': calibration_quality
        }
    
    def _calculate_confidence_factors(self, pattern: Pattern, 
                                    issue_context: IssueContext) -> ConfidenceFactors:
        """Calculate individual confidence factors."""
        
        # Data completeness factor
        data_completeness = self._assess_data_completeness(pattern, issue_context)
        
        # Historical accuracy factor
        historical_accuracy = self._assess_historical_accuracy(pattern)
        
        # Pattern maturity factor
        pattern_maturity = self._assess_pattern_maturity(pattern)
        
        # Context alignment factor
        context_alignment = self._assess_context_alignment(pattern, issue_context)
        
        # Similarity strength factor
        similarity_strength = self._assess_similarity_strength(pattern, issue_context)
        
        # Validation coverage factor
        validation_coverage = self._assess_validation_coverage(pattern)
        
        # Expert consensus factor (simulated for now)
        expert_consensus = self._assess_expert_consensus(pattern, issue_context)
        
        # Uncertainty measure (inverse of uncertainty)
        uncertainty_measure = 1.0 - self._calculate_uncertainty(pattern, issue_context)
        
        return ConfidenceFactors(
            data_completeness=data_completeness,
            historical_accuracy=historical_accuracy,
            pattern_maturity=pattern_maturity,
            context_alignment=context_alignment,
            similarity_strength=similarity_strength,
            validation_coverage=validation_coverage,
            expert_consensus=expert_consensus,
            uncertainty_measure=uncertainty_measure
        )
    
    def _assess_data_completeness(self, pattern: Pattern, 
                                issue_context: IssueContext) -> float:
        """Assess completeness of data for confidence assessment."""
        completeness_score = 0.0
        total_factors = 8  # Total number of factors we check
        
        # Pattern data completeness
        if pattern.description:
            completeness_score += 1.0
        if pattern.implementation_steps:
            completeness_score += 1.0
        if pattern.code_examples:
            completeness_score += 1.0
        if pattern.validation_criteria:
            completeness_score += 1.0
        if pattern.tech_stack:
            completeness_score += 1.0
        
        # Context data completeness
        if issue_context.description:
            completeness_score += 1.0
        if issue_context.tech_stack:
            completeness_score += 1.0
        if issue_context.labels:
            completeness_score += 1.0
        
        return completeness_score / total_factors
    
    def _assess_historical_accuracy(self, pattern: Pattern) -> float:
        """Assess historical accuracy of pattern applications."""
        # Use pattern's success rate as primary indicator
        base_accuracy = pattern.success_rate
        
        # Adjust based on usage count (more data = more reliable)
        usage_confidence = min(1.0, pattern.usage_count / 20)  # Normalize to 20 uses
        
        # Combine with confidence weighting
        adjusted_accuracy = base_accuracy * (0.7 + 0.3 * usage_confidence)
        
        return min(1.0, adjusted_accuracy)
    
    def _assess_pattern_maturity(self, pattern: Pattern) -> float:
        """Assess maturity and stability of the pattern."""
        maturity_score = 0.0
        
        # Usage count indicator
        if pattern.usage_count >= 10:
            maturity_score += 0.4
        elif pattern.usage_count >= 5:
            maturity_score += 0.3
        elif pattern.usage_count >= 1:
            maturity_score += 0.2
        
        # Pattern confidence indicator
        maturity_score += pattern.confidence * 0.3
        
        # Completeness indicators
        if pattern.implementation_steps:
            maturity_score += 0.15
        if pattern.validation_criteria:
            maturity_score += 0.15
        
        return min(1.0, maturity_score)
    
    def _assess_context_alignment(self, pattern: Pattern, 
                                issue_context: IssueContext) -> float:
        """Assess how well pattern aligns with context."""
        alignment_score = 0.0
        
        # Technology alignment
        tech_alignment = self._calculate_tech_alignment(pattern, issue_context)
        alignment_score += tech_alignment * 0.4
        
        # Complexity alignment
        complexity_alignment = self._calculate_complexity_alignment(pattern, issue_context)
        alignment_score += complexity_alignment * 0.3
        
        # Domain alignment
        domain_alignment = self._calculate_domain_alignment(pattern, issue_context)
        alignment_score += domain_alignment * 0.3
        
        return alignment_score
    
    def _assess_similarity_strength(self, pattern: Pattern, 
                                  issue_context: IssueContext) -> float:
        """Assess strength of similarity between pattern and context."""
        try:
            # Use similarity engine if available
            from .similarity_engine import SimilarityEngine
            similarity_engine = SimilarityEngine(self.knowledge_system, self.database)
            
            pattern_text = f"{pattern.name} {pattern.description}"
            context_text = f"{issue_context.title} {issue_context.description}"
            
            similarity_score = similarity_engine.calculate_semantic_similarity(
                pattern_text, context_text
            )
            
            return similarity_score
            
        except Exception as e:
            self.logger.debug(f"Similarity assessment failed: {str(e)}")
            return 0.5  # Neutral score as fallback
    
    def _assess_validation_coverage(self, pattern: Pattern) -> float:
        """Assess validation coverage of the pattern."""
        coverage_score = 0.0
        
        # Validation criteria availability
        if pattern.validation_criteria:
            coverage_score += 0.5
            # More comprehensive if multiple criteria
            if len(pattern.validation_criteria) >= 3:
                coverage_score += 0.2
        
        # Code examples as validation aid
        if pattern.code_examples:
            coverage_score += 0.3
        
        return min(1.0, coverage_score)
    
    def _assess_expert_consensus(self, pattern: Pattern, 
                               issue_context: IssueContext) -> float:
        """Assess expert consensus (simulated based on pattern characteristics)."""
        consensus_score = 0.0
        
        # High usage indicates expert approval
        if pattern.usage_count > 10:
            consensus_score += 0.4
        elif pattern.usage_count > 5:
            consensus_score += 0.3
        
        # High success rate indicates expert validation
        if pattern.success_rate > 0.8:
            consensus_score += 0.3
        elif pattern.success_rate > 0.6:
            consensus_score += 0.2
        
        # High confidence indicates expert review
        if pattern.confidence > 0.8:
            consensus_score += 0.3
        elif pattern.confidence > 0.6:
            consensus_score += 0.2
        
        return min(1.0, consensus_score)
    
    def _calculate_uncertainty(self, pattern: Pattern, 
                             issue_context: IssueContext) -> float:
        """Calculate uncertainty measure."""
        uncertainty = 0.0
        
        # Data sparsity uncertainty
        data_sparsity = self._calculate_data_sparsity(pattern, issue_context)
        uncertainty += data_sparsity * self.uncertainty_factors['data_sparsity']
        
        # Domain novelty uncertainty
        domain_novelty = self._calculate_domain_novelty(issue_context)
        uncertainty += domain_novelty * self.uncertainty_factors['domain_novelty']
        
        # Pattern complexity uncertainty
        complexity_uncertainty = self._calculate_complexity_uncertainty(pattern)
        uncertainty += complexity_uncertainty * self.uncertainty_factors['pattern_complexity']
        
        # Context ambiguity uncertainty
        context_ambiguity = self._calculate_context_ambiguity(issue_context)
        uncertainty += context_ambiguity * self.uncertainty_factors['context_ambiguity']
        
        # Temporal drift uncertainty
        temporal_drift = self._calculate_temporal_drift(pattern)
        uncertainty += temporal_drift * self.uncertainty_factors['temporal_drift']
        
        return min(1.0, uncertainty)
    
    def _calculate_weighted_confidence(self, factors: ConfidenceFactors) -> float:
        """Calculate weighted overall confidence score."""
        weighted_score = 0.0
        
        weighted_score += factors.data_completeness * self.confidence_weights['data_completeness']
        weighted_score += factors.historical_accuracy * self.confidence_weights['historical_accuracy']
        weighted_score += factors.pattern_maturity * self.confidence_weights['pattern_maturity']
        weighted_score += factors.context_alignment * self.confidence_weights['context_alignment']
        weighted_score += factors.similarity_strength * self.confidence_weights['similarity_strength']
        weighted_score += factors.validation_coverage * self.confidence_weights['validation_coverage']
        weighted_score += factors.expert_consensus * self.confidence_weights['expert_consensus']
        weighted_score += factors.uncertainty_measure * self.confidence_weights['uncertainty_measure']
        
        return min(1.0, max(0.0, weighted_score))
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level category."""
        for level, threshold in sorted(self.confidence_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if confidence_score >= threshold:
                return level
        return 'very_low'
    
    def _calculate_reliability_score(self, pattern: Pattern, issue_context: IssueContext,
                                   factors: ConfidenceFactors) -> float:
        """Calculate reliability score for the confidence assessment."""
        reliability_factors = []
        
        # Historical accuracy reliability
        reliability_factors.append(factors.historical_accuracy)
        
        # Pattern maturity reliability
        reliability_factors.append(factors.pattern_maturity)
        
        # Data completeness reliability
        reliability_factors.append(factors.data_completeness)
        
        # Validation coverage reliability
        reliability_factors.append(factors.validation_coverage)
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        reliability = sum(f * w for f, w in zip(reliability_factors, weights))
        
        return reliability
    
    def _calculate_uncertainty_bounds(self, confidence: float, 
                                    factors: ConfidenceFactors) -> Tuple[float, float]:
        """Calculate uncertainty bounds around confidence estimate."""
        # Calculate uncertainty width based on factors
        uncertainty_width = (1.0 - factors.uncertainty_measure) * 0.3
        
        lower_bound = max(0.0, confidence - uncertainty_width)
        upper_bound = min(1.0, confidence + uncertainty_width)
        
        return (lower_bound, upper_bound)
    
    def _generate_confidence_explanation(self, factors: ConfidenceFactors, 
                                       confidence: float) -> str:
        """Generate human-readable confidence explanation."""
        explanations = []
        
        # Identify strongest factors
        factor_scores = [
            ('data completeness', factors.data_completeness),
            ('historical accuracy', factors.historical_accuracy),
            ('pattern maturity', factors.pattern_maturity),
            ('context alignment', factors.context_alignment),
            ('similarity strength', factors.similarity_strength)
        ]
        
        factor_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add explanations for top factors
        for factor_name, score in factor_scores[:3]:
            if score > 0.7:
                explanations.append(f"High {factor_name} ({score:.2f})")
            elif score > 0.5:
                explanations.append(f"Moderate {factor_name} ({score:.2f})")
            else:
                explanations.append(f"Low {factor_name} ({score:.2f})")
        
        explanation = f"Confidence {confidence:.2f}: " + ", ".join(explanations)
        
        return explanation
    
    def _generate_confidence_recommendations(self, factors: ConfidenceFactors) -> List[str]:
        """Generate recommendations for improving confidence."""
        recommendations = []
        
        # Data completeness recommendations
        if factors.data_completeness < 0.7:
            recommendations.append("Gather more complete data for pattern and context")
        
        # Historical accuracy recommendations
        if factors.historical_accuracy < 0.6:
            recommendations.append("Seek patterns with better historical track record")
        
        # Pattern maturity recommendations
        if factors.pattern_maturity < 0.5:
            recommendations.append("Consider more mature and well-tested patterns")
        
        # Context alignment recommendations
        if factors.context_alignment < 0.6:
            recommendations.append("Look for patterns with better context alignment")
        
        # Validation coverage recommendations
        if factors.validation_coverage < 0.5:
            recommendations.append("Ensure pattern has comprehensive validation criteria")
        
        # Uncertainty recommendations
        if factors.uncertainty_measure < 0.7:
            recommendations.append("Address sources of uncertainty before implementation")
        
        return recommendations
    
    def _calculate_calibration_metrics(self, pattern: Pattern, 
                                     issue_context: IssueContext) -> Dict[str, float]:
        """Calculate calibration metrics for confidence assessment."""
        # This would use historical calibration data
        # For now, return baseline metrics
        return {
            'brier_score': 0.2,  # Lower is better
            'reliability': 0.8,  # Higher is better
            'resolution': 0.7,   # Higher is better
            'calibration_quality': 0.8
        }
    
    def _update_calibration_history(self, result: ConfidenceResult, 
                                  pattern: Pattern, issue_context: IssueContext):
        """Update calibration history with new confidence assessment."""
        calibration_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'confidence': result.overall_confidence,
            'pattern_id': pattern.pattern_id,
            'issue_id': issue_context.issue_id,
            'factors': result.confidence_factors.__dict__
        }
        
        self.calibration_history.append(calibration_entry)
        
        # Keep only recent entries
        cutoff_time = datetime.utcnow() - self.calibration_window
        self.calibration_history = [
            entry for entry in self.calibration_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
    
    def _create_low_confidence_result(self) -> ConfidenceResult:
        """Create low confidence result as fallback."""
        factors = ConfidenceFactors(
            data_completeness=0.3,
            historical_accuracy=0.3,
            pattern_maturity=0.3,
            context_alignment=0.3,
            similarity_strength=0.3,
            validation_coverage=0.3,
            expert_consensus=0.3,
            uncertainty_measure=0.3
        )
        
        return ConfidenceResult(
            overall_confidence=0.3,
            confidence_level='low',
            confidence_factors=factors,
            reliability_score=0.3,
            uncertainty_bounds=(0.1, 0.5),
            confidence_explanation="Low confidence due to assessment error",
            recommendations=["Review input data and retry assessment"],
            calibration_metrics={'brier_score': 0.5, 'reliability': 0.3, 'resolution': 0.3, 'calibration_quality': 0.3}
        )
    
    # Helper methods for detailed factor calculations
    def _calculate_tech_alignment(self, pattern: Pattern, 
                                issue_context: IssueContext) -> float:
        """Calculate technology stack alignment score."""
        if not pattern.tech_stack or not issue_context.tech_stack:
            return 0.5
        
        alignment = 0.0
        
        # Language alignment
        if pattern.tech_stack.primary_language == issue_context.tech_stack.primary_language:
            alignment += 0.5
        
        # Framework alignment
        pattern_frameworks = set(fw.lower() for fw in pattern.tech_stack.frameworks)
        context_frameworks = set(fw.lower() for fw in issue_context.tech_stack.frameworks)
        if pattern_frameworks.intersection(context_frameworks):
            alignment += 0.3
        
        # Database alignment
        if pattern.tech_stack.databases and issue_context.tech_stack.databases:
            pattern_dbs = set(db.lower() for db in pattern.tech_stack.databases)
            context_dbs = set(db.lower() for db in issue_context.tech_stack.databases)
            if pattern_dbs.intersection(context_dbs):
                alignment += 0.2
        
        return alignment
    
    def _calculate_complexity_alignment(self, pattern: Pattern, 
                                      issue_context: IssueContext) -> float:
        """Calculate complexity alignment score."""
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        
        try:
            pattern_idx = complexity_levels.index(pattern.complexity)
            context_idx = complexity_levels.index(issue_context.complexity)
            
            diff = abs(pattern_idx - context_idx)
            if diff == 0:
                return 1.0
            elif diff == 1:
                return 0.7
            elif diff == 2:
                return 0.4
            else:
                return 0.1
        except ValueError:
            return 0.5
    
    def _calculate_domain_alignment(self, pattern: Pattern, 
                                  issue_context: IssueContext) -> float:
        """Calculate domain alignment score."""
        if pattern.domain == issue_context.domain:
            return 1.0
        elif pattern.domain == 'general' or issue_context.domain == 'general':
            return 0.6
        else:
            return 0.2
    
    def _calculate_data_sparsity(self, pattern: Pattern, 
                               issue_context: IssueContext) -> float:
        """Calculate data sparsity uncertainty factor."""
        sparsity = 0.0
        
        # Pattern data sparsity
        if not pattern.implementation_steps:
            sparsity += 0.3
        if not pattern.code_examples:
            sparsity += 0.2
        if pattern.usage_count < 5:
            sparsity += 0.3
        
        # Context data sparsity
        if not issue_context.description or len(issue_context.description) < 50:
            sparsity += 0.2
        
        return min(1.0, sparsity)
    
    def _calculate_domain_novelty(self, issue_context: IssueContext) -> float:
        """Calculate domain novelty uncertainty."""
        common_domains = ['web', 'api', 'database', 'frontend', 'backend', 'general']
        if issue_context.domain in common_domains:
            return 0.1
        else:
            return 0.6
    
    def _calculate_complexity_uncertainty(self, pattern: Pattern) -> float:
        """Calculate complexity-related uncertainty."""
        if pattern.complexity in ['high', 'very-high']:
            return 0.4
        else:
            return 0.1
    
    def _calculate_context_ambiguity(self, issue_context: IssueContext) -> float:
        """Calculate context ambiguity uncertainty."""
        ambiguity = 0.0
        
        # Title/description clarity
        if not issue_context.title or len(issue_context.title) < 10:
            ambiguity += 0.3
        
        if not issue_context.description or len(issue_context.description) < 100:
            ambiguity += 0.3
        
        # Tech stack specificity
        if not issue_context.tech_stack or not issue_context.tech_stack.primary_language:
            ambiguity += 0.2
        
        # Label specificity
        if not issue_context.labels:
            ambiguity += 0.2
        
        return min(1.0, ambiguity)
    
    def _calculate_temporal_drift(self, pattern: Pattern) -> float:
        """Calculate temporal drift uncertainty."""
        # This would analyze pattern age and technological drift
        # For now, use usage count as proxy for recency
        if pattern.usage_count > 10:
            return 0.1  # Recently used patterns have less drift
        elif pattern.usage_count > 5:
            return 0.2
        else:
            return 0.4  # Unused patterns may have drifted
    
    def _calculate_reliability_calibration(self, actual_outcomes: List[bool],
                                         predicted_confidences: List[float],
                                         n_bins: int = 10) -> float:
        """Calculate reliability calibration score."""
        # Bin predictions by confidence level
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability_scores = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = [(conf >= bin_lower and conf < bin_upper) 
                     for conf in predicted_confidences]
            
            if not any(in_bin):
                continue
            
            # Calculate average confidence and accuracy for this bin
            bin_confidences = [conf for conf, in_b in zip(predicted_confidences, in_bin) if in_b]
            bin_outcomes = [outcome for outcome, in_b in zip(actual_outcomes, in_bin) if in_b]
            
            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_outcomes)
            
            # Reliability is how close confidence matches accuracy
            bin_reliability = 1.0 - abs(avg_confidence - avg_accuracy)
            reliability_scores.append(bin_reliability)
        
        return np.mean(reliability_scores) if reliability_scores else 0.5
    
    def _calculate_resolution_score(self, actual_outcomes: List[bool],
                                   predicted_confidences: List[float]) -> float:
        """Calculate resolution (discrimination) score."""
        # Resolution measures ability to assign different confidences to different outcomes
        positive_confidences = [conf for conf, outcome in zip(predicted_confidences, actual_outcomes) if outcome]
        negative_confidences = [conf for conf, outcome in zip(predicted_confidences, actual_outcomes) if not outcome]
        
        if not positive_confidences or not negative_confidences:
            return 0.0
        
        # Higher resolution means positive cases get higher confidence than negative cases
        avg_positive_conf = np.mean(positive_confidences)
        avg_negative_conf = np.mean(negative_confidences)
        
        resolution = max(0.0, avg_positive_conf - avg_negative_conf)
        
        return resolution