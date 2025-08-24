"""
Code relationship detection and analysis system.

This package provides tools for detecting and storing relationships between code entities,
including imports, function calls, inheritance, and other dependency patterns.
"""

from .relationship_types import (
    RelationshipType,
    CodeRelationship,
    RelationshipDetectionResult
)

from .base_analyzer import BaseRelationshipAnalyzer
from .relationship_detector import RelationshipDetector

__all__ = [
    'RelationshipType',
    'CodeRelationship', 
    'RelationshipDetectionResult',
    'BaseRelationshipAnalyzer',
    'RelationshipDetector'
]