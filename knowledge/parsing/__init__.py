"""
Tree-sitter parsing infrastructure for RIF hybrid knowledge system.

This package provides multi-language AST parsing with intelligent caching
for JavaScript, Python, Go, and Rust codebases.

Core Components:
- ParserManager: Singleton manager for thread-safe parser pool
- ASTCache: LRU cache for parsed ASTs with 100-file limit
- LanguageDetector: File extension-based language identification
- IncrementalParser: Optimized re-parsing for modified files

Performance Targets:
- Initial parsing: <2s for 10K LOC files
- Cache retrieval: <50ms average response time
- Incremental updates: <500ms for typical changes
- Memory usage: <200MB with full cache
"""

from .parser_manager import ParserManager
from .language_detector import LanguageDetector
from .ast_cache import ASTCache, ASTCacheEntry
from .exceptions import (
    ParsingError,
    LanguageNotSupportedError,
    GrammarNotFoundError,
    CacheError
)

# Version info
__version__ = "1.0.0"
__author__ = "RIF Framework"

# Package exports
__all__ = [
    "ParserManager",
    "LanguageDetector",
    "ASTCache",
    "ASTCacheEntry",
    "ParsingError", 
    "LanguageNotSupportedError",
    "GrammarNotFoundError",
    "CacheError"
]

# Singleton instance - accessed via ParserManager.get_instance()
_parser_manager_instance = None

def get_parser_manager():
    """
    Get the singleton ParserManager instance.
    
    Returns:
        ParserManager: The singleton parser manager instance
    """
    global _parser_manager_instance
    if _parser_manager_instance is None:
        _parser_manager_instance = ParserManager()
    return _parser_manager_instance