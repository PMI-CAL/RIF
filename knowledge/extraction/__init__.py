"""
Entity extraction package for the RIF hybrid knowledge system.

This package provides AST-based entity extraction for multiple programming languages,
building on the tree-sitter parsing infrastructure.
"""

from .entity_extractor import EntityExtractor
from .entity_types import EntityType, CodeEntity, SourceLocation
from .language_extractors import (
    JavaScriptExtractor,
    PythonExtractor,
    GoExtractor,
    RustExtractor
)

__all__ = [
    'EntityExtractor',
    'EntityType',
    'CodeEntity', 
    'SourceLocation',
    'JavaScriptExtractor',
    'PythonExtractor',
    'GoExtractor',
    'RustExtractor'
]