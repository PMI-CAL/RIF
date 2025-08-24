"""
Core ParserManager singleton for thread-safe AST parsing with multi-language support.

The ParserManager provides a centralized interface for parsing code files using
tree-sitter parsers with intelligent caching and language detection.
"""

import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict

try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from .exceptions import (
    ParsingError,
    LanguageNotSupportedError, 
    GrammarNotFoundError,
    MemoryLimitExceededError,
    ThreadSafetyError
)
from .language_detector import LanguageDetector
from .ast_cache import ASTCache


class ParserManager:
    """
    Singleton manager for thread-safe AST parsing with multi-language support.
    
    Features:
    - Thread-safe parser pool with language-specific parsers
    - Automatic language detection from file extensions
    - Memory usage monitoring with configurable limits
    - Performance metrics collection
    - Grammar compilation and caching
    
    Supported Languages:
    - JavaScript (.js, .jsx, .mjs, .cjs)
    - Python (.py, .pyx, .pyi) 
    - Go (.go)
    - Rust (.rs)
    """
    
    _instance = None
    _lock = threading.RLock()
    
    # Language configuration now managed by LanguageDetector
    
    def __new__(cls):
        """Singleton pattern implementation with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize parser manager if not already initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            # Check tree-sitter availability
            if not TREE_SITTER_AVAILABLE:
                raise ImportError(
                    "tree-sitter package not available. Install with: pip install tree-sitter"
                )
            
            # Initialize language detector
            self._language_detector = LanguageDetector()
            
            # Initialize AST cache (100 files, 200MB limit)
            self._ast_cache = ASTCache(max_entries=100, max_memory_mb=200)
            
            # Initialize core components
            self._parsers: Dict[str, tree_sitter.Parser] = {}
            self._languages: Dict[str, tree_sitter.Language] = {}
            
            # Performance monitoring
            self._parse_times: Dict[str, List[float]] = defaultdict(list)
            self._memory_usage: Dict[str, int] = {}
            self._parse_counts: Dict[str, int] = defaultdict(int)
            
            # Configuration
            self._memory_limit_mb = 200  # Memory limit in MB
            self._max_file_size = 10_000_000  # 10MB max file size
            
            # Thread safety - create locks for supported languages
            supported_langs = self._language_detector.get_supported_languages()
            self._parser_locks: Dict[str, threading.RLock] = {
                lang: threading.RLock() for lang in supported_langs
            }
            
            self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'ParserManager':
        """Get the singleton ParserManager instance."""
        return cls()
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Language identifier or None if not supported
        """
        return self._language_detector.detect_language(file_path)
    
    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return self._language_detector.is_supported(language)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language identifiers."""
        return self._language_detector.get_supported_languages()
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return self._language_detector.get_supported_extensions()
    
    def _get_parser(self, language: str) -> tree_sitter.Parser:
        """
        Get or create a parser for the specified language.
        
        Args:
            language: Language identifier
            
        Returns:
            Tree-sitter parser instance
            
        Raises:
            LanguageNotSupportedError: If language is not supported
            GrammarNotFoundError: If grammar cannot be loaded
        """
        if not self.is_language_supported(language):
            raise LanguageNotSupportedError(
                language, 
                supported_languages=self.get_supported_languages()
            )
        
        # Thread-safe parser access
        with self._parser_locks[language]:
            if language not in self._parsers:
                # Load language grammar
                grammar = self._load_language_grammar(language)
                
                # Create parser
                parser = tree_sitter.Parser(grammar)
                
                self._parsers[language] = parser
                
            return self._parsers[language]
    
    def _load_language_grammar(self, language: str) -> tree_sitter.Language:
        """
        Load tree-sitter language grammar.
        
        Args:
            language: Language identifier
            
        Returns:
            Tree-sitter Language object
            
        Raises:
            GrammarNotFoundError: If grammar cannot be loaded
        """
        if language not in self._languages:
            # Use LanguageDetector to load the grammar
            grammar = self._language_detector.get_language_grammar(language)
            self._languages[language] = grammar
        
        return self._languages[language]
    
    def _check_memory_usage(self, additional_mb: int = 0):
        """
        Check if memory usage is within limits.
        
        Args:
            additional_mb: Additional memory that will be allocated
            
        Raises:
            MemoryLimitExceededError: If memory limit would be exceeded
        """
        # Simplified memory tracking - will be enhanced in Phase 3
        total_usage = sum(self._memory_usage.values()) + additional_mb
        if total_usage > self._memory_limit_mb:
            raise MemoryLimitExceededError(total_usage, self._memory_limit_mb)
    
    def _validate_file_size(self, file_path: str):
        """
        Validate file size is within processing limits.
        
        Args:
            file_path: Path to file to validate
            
        Raises:
            ParsingError: If file is too large
        """
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self._max_file_size:
                raise ParsingError(
                    f"File size {file_size} bytes exceeds maximum {self._max_file_size} bytes",
                    file_path=file_path
                )
        except OSError as e:
            raise ParsingError(f"Cannot access file: {e}", file_path=file_path)
    
    def parse_file(self, file_path: str, language: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Parse a source code file and return AST information.
        
        Args:
            file_path: Path to the file to parse
            language: Language identifier (auto-detected if None)
            use_cache: Whether to use/populate cache
            
        Returns:
            Dictionary containing AST and metadata
            
        Raises:
            ParsingError: If parsing fails
            LanguageNotSupportedError: If language is not supported
            MemoryLimitExceededError: If memory limit exceeded
        """
        start_time = time.time()
        
        try:
            # Validate file
            self._validate_file_size(file_path)
            
            # Detect language if not provided
            if language is None:
                language = self.detect_language(file_path)
                if language is None:
                    raise LanguageNotSupportedError(
                        f"Unknown language for file: {file_path}",
                        file_path=file_path,
                        supported_languages=self.get_supported_languages()
                    )
            
            # Check cache first
            if use_cache:
                cached_result = self._ast_cache.get(file_path, language)
                if cached_result is not None:
                    # Cache hit - update metrics and return
                    cache_time = time.time() - start_time
                    self._update_metrics(language, cache_time)
                    cached_result['cache_hit'] = True
                    cached_result['parse_time'] = cache_time
                    return cached_result
            
            # Cache miss or cache disabled - parse file
            # Check memory limits
            self._check_memory_usage(additional_mb=50)  # Estimate 50MB for parsing
            
            # Get parser - now with real grammar support
            parser = self._get_parser(language)
            
            # Read file content
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            # Parse with thread safety
            with self._parser_locks[language]:
                tree = parser.parse(source_code)
            
            # Create result
            parse_result = {
                'file_path': file_path,
                'language': language,
                'tree': tree,
                'root_node': tree.root_node if tree else None,
                'source_size': len(source_code),
                'parse_time': time.time() - start_time,
                'timestamp': int(time.time()),
                'has_error': tree.root_node.has_error if tree and tree.root_node else False,
                'cache_hit': False
            }
            
            # Store in cache if enabled
            if use_cache:
                try:
                    self._ast_cache.put(file_path, language, tree, parse_result)
                except Exception as e:
                    # Cache failure shouldn't break parsing
                    print(f"Warning: Failed to cache result for {file_path}: {e}")
            
            # Update metrics
            self._update_metrics(language, parse_result['parse_time'])
            
            return parse_result
            
        except Exception as e:
            if isinstance(e, (ParsingError, LanguageNotSupportedError, MemoryLimitExceededError)):
                raise
            else:
                raise ParsingError(f"Unexpected error parsing {file_path}: {e}", 
                                 file_path=file_path, language=language, original_error=e)
    
    def _create_mock_parse_result(self, file_path: str, language: str) -> Dict[str, Any]:
        """
        Create a mock parse result for Phase 1 testing.
        
        This will be removed in Phase 2 when actual parsing is implemented.
        """
        parse_time = 0.1  # Mock parse time
        
        result = {
            'file_path': file_path,
            'language': language,
            'tree': None,  # Mock - will contain actual tree in Phase 2
            'source_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'parse_time': parse_time,
            'timestamp': int(time.time()),
            'mock': True  # Indicates this is a mock result
        }
        
        # Update metrics for mock results too
        self._update_metrics(language, parse_time)
        
        return result
    
    def _update_metrics(self, language: str, parse_time: float):
        """Update performance metrics."""
        self._parse_times[language].append(parse_time)
        self._parse_counts[language] += 1
        
        # Keep only last 100 parse times per language for moving averages
        if len(self._parse_times[language]) > 100:
            self._parse_times[language] = self._parse_times[language][-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics including cache statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = {
            'supported_languages': self.get_supported_languages(),
            'parse_counts': dict(self._parse_counts),
            'average_parse_times': {},
            'memory_usage_mb': dict(self._memory_usage),
            'total_memory_mb': sum(self._memory_usage.values()),
            'memory_limit_mb': self._memory_limit_mb
        }
        
        # Calculate average parse times
        for language, times in self._parse_times.items():
            if times:
                metrics['average_parse_times'][language] = sum(times) / len(times)
        
        # Add cache metrics
        cache_metrics = self._ast_cache.get_metrics()
        metrics['cache'] = cache_metrics
        
        return metrics
    
    def invalidate_cache(self, file_path: str, language: Optional[str] = None):
        """
        Invalidate cached entries for a file.
        
        Args:
            file_path: Path to file to invalidate
            language: Specific language to invalidate, or None for all
        """
        self._ast_cache.invalidate(file_path, language)
    
    def clear_cache(self):
        """Clear all cached entries."""
        self._ast_cache.clear()
    
    def cleanup_invalid_cache(self) -> int:
        """
        Remove invalid entries from cache.
        
        Returns:
            Number of entries removed
        """
        return self._ast_cache.cleanup_invalid()
    
    def get_cache_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed cache information.
        
        Returns:
            List of cache entry details
        """
        return self._ast_cache.get_cache_info()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        with self._lock:
            self._parse_times.clear()
            self._parse_counts.clear()
            self._memory_usage.clear()
    
    def shutdown(self):
        """Clean shutdown of parser manager."""
        with self._lock:
            # Clear parsers and languages
            self._parsers.clear()
            self._languages.clear()
            
            # Clear cache
            self.clear_cache()
            
            # Reset metrics
            self.reset_metrics()