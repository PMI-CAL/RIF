"""
Pattern Matching System - Issue #76

This module provides sophisticated pattern matching functionality for the RIF system,
including similarity detection, pattern ranking, recommendation generation, and
confidence scoring.

Key Components:
- AdvancedPatternMatcher: Main pattern matching engine
- SimilarityEngine: Issue and pattern similarity detection
- PatternRanker: Multi-criteria pattern ranking system
- RecommendationGenerator: Context-aware recommendation system
- ConfidenceScorer: Comprehensive confidence evaluation
"""

from .advanced_matcher import AdvancedPatternMatcher
from .similarity_engine import SimilarityEngine
from .pattern_ranker import PatternRanker
from .recommendation_generator import RecommendationGenerator
from .confidence_scorer import ConfidenceScorer

__all__ = [
    'AdvancedPatternMatcher',
    'SimilarityEngine',
    'PatternRanker', 
    'RecommendationGenerator',
    'ConfidenceScorer'
]

__version__ = '1.0.0'