#!/usr/bin/env python3
"""
RIF Consensus Module - Issue #62 Vote Weighting System

This module provides sophisticated vote weighting capabilities for the RIF consensus system.

Components:
- VoteWeightCalculator: Multi-factor vote weight calculation
- ExpertiseScorer: Agent domain expertise evaluation  
- AccuracyTracker: Historical performance analysis
- ConfidenceAdjuster: Confidence calibration and bias correction
- WeightedVotingAggregator: Integration with existing consensus system

Features:
- Multi-dimensional weight calculation based on expertise, accuracy, and confidence
- Domain-specific expertise assessment with evidence tracking
- Historical accuracy tracking with trend analysis
- Confidence bias detection and correction
- Ensemble weight normalization and optimization
- Learning and adaptation based on decision outcomes
- Integration with existing VotingAggregator and ConsensusArchitecture
"""

from .vote_weight_calculator import (
    VoteWeightCalculator,
    WeightingStrategy,
    ExpertiseProfile,
    AccuracyRecord,
    WeightCalculationResult
)

from .expertise_scorer import (
    ExpertiseScorer,
    ExpertiseDomain,
    ExpertiseEvidence,
    ExpertiseAssessment
)

from .accuracy_tracker import (
    AccuracyTracker,
    DecisionRecord,
    DecisionOutcome,
    AccuracyMetrics,
    CalibrationAnalysis
)

from .confidence_adjuster import (
    ConfidenceAdjuster,
    ConfidenceRecord,
    ConfidenceCalibration,
    ConfidenceAdjustment,
    ConfidenceScale,
    ConfidenceBias
)

# Integration module available but not imported by default due to external dependencies
# from .weighted_voting_integration import (
#     WeightedVotingAggregator,
#     WeightedVotingConfig
# )

__version__ = "1.0.0"
__author__ = "RIF Development Team"
__description__ = "Advanced vote weighting system for RIF consensus decisions"

# Export main classes
__all__ = [
    # Core weight calculation
    "VoteWeightCalculator",
    "WeightingStrategy", 
    "WeightCalculationResult",
    
    # Expertise scoring
    "ExpertiseScorer",
    "ExpertiseDomain",
    "ExpertiseEvidence", 
    "ExpertiseAssessment",
    
    # Accuracy tracking
    "AccuracyTracker",
    "DecisionRecord",
    "DecisionOutcome",
    "AccuracyMetrics",
    "CalibrationAnalysis",
    
    # Confidence adjustment
    "ConfidenceAdjuster",
    "ConfidenceRecord",
    "ConfidenceCalibration", 
    "ConfidenceAdjustment",
    "ConfidenceScale",
    "ConfidenceBias",
    
    # Integration (available separately)
    # "WeightedVotingAggregator",
    # "WeightedVotingConfig",
    
    # Supporting types
    "ExpertiseProfile",
    "AccuracyRecord"
]