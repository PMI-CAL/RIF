"""
RIF Pattern Extraction Engine

This module provides comprehensive pattern extraction capabilities for the RIF framework,
mining successful patterns from completed issues, categorizing them, and calculating
success metrics for pattern application and learning.

The pattern extraction engine supports multiple extraction methods:
- AST-based code pattern analysis
- Regex-based workflow pattern identification  
- Statistical decision pattern analysis
- Multi-source data integration

Key Components:
    - PatternDiscoveryEngine: Core pattern identification and extraction
    - CodePatternExtractor: AST-based code pattern analysis
    - WorkflowPatternExtractor: State transition and workflow analysis
    - DecisionPatternExtractor: Decision pattern and trade-off analysis
    - SuccessMetricsCalculator: Multi-dimensional success scoring

Usage:
    from knowledge.pattern_extraction import PatternExtractionEngine
    
    engine = PatternExtractionEngine()
    patterns = engine.extract_patterns(completed_issue)
    
    for pattern in patterns:
        print(f"Pattern: {pattern.title}")
        print(f"Success Rate: {pattern.success_rate}")
        print(f"Applicability: {pattern.applicability_score}")
"""

from .discovery_engine import PatternDiscoveryEngine
from .code_extractor import CodePatternExtractor  
from .workflow_extractor import WorkflowPatternExtractor
from .decision_extractor import DecisionPatternExtractor
from .success_metrics import SuccessMetricsCalculator
from .cache import PatternExtractionCache, get_global_cache

__all__ = [
    'PatternDiscoveryEngine',
    'CodePatternExtractor', 
    'WorkflowPatternExtractor',
    'DecisionPatternExtractor',
    'SuccessMetricsCalculator',
    'PatternExtractionCache',
    'get_global_cache'
]

__version__ = '1.0.0'
__author__ = 'RIF Pattern Extraction Team'