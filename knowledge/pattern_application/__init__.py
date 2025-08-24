"""
Pattern Application Engine

This module implements the Pattern Application Engine for RIF, which applies
learned patterns to new issues with context adaptation and success tracking.

Key Components:
- PatternApplicationEngine: Main engine for pattern application
- ContextExtractor: Extracts context from GitHub issues
- PatternMatcher: Finds and ranks applicable patterns (abstraction layer)
- PatternAdapter: Adapts patterns to specific contexts
- PlanGenerator: Generates implementation plans from patterns
- SuccessTracker: Tracks application success and metrics

The system is designed to work with Issue #76 (Pattern Matching System) when
available, but includes abstraction layers for parallel development.
"""

from .core import (
    # Data Models
    Pattern, IssueContext, TechStack, IssueConstraints,
    AdaptationResult, ImplementationPlan, ImplementationTask,
    ApplicationRecord,
    
    # Enums
    PatternApplicationStatus, AdaptationStrategy,
    
    # Interfaces
    PatternMatchingInterface, PatternApplicationInterface,
    ContextExtractionInterface,
    
    # Utilities
    generate_id, load_pattern_from_json,
    
    # Exceptions
    PatternApplicationError, PatternNotFoundError,
    ContextExtractionError, PatternAdaptationError
)

from .engine import PatternApplicationEngine
from .context_extractor import ContextExtractor
from .pattern_matcher import BasicPatternMatcher, InterimPatternMatcher
from .pattern_adapter import PatternAdapter
from .plan_generator import PlanGenerator
from .success_tracker import SuccessTracker

__all__ = [
    # Core data models
    'Pattern', 'IssueContext', 'TechStack', 'IssueConstraints',
    'AdaptationResult', 'ImplementationPlan', 'ImplementationTask',
    'ApplicationRecord',
    
    # Enums
    'PatternApplicationStatus', 'AdaptationStrategy',
    
    # Interfaces
    'PatternMatchingInterface', 'PatternApplicationInterface',
    'ContextExtractionInterface',
    
    # Main engine and components
    'PatternApplicationEngine',
    'ContextExtractor',
    'BasicPatternMatcher', 'InterimPatternMatcher',
    'PatternAdapter',
    'PlanGenerator',
    'SuccessTracker',
    
    # Utilities
    'generate_id', 'load_pattern_from_json',
    
    # Exceptions
    'PatternApplicationError', 'PatternNotFoundError',
    'ContextExtractionError', 'PatternAdaptationError'
]