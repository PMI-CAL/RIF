"""
Language detection and grammar loading for tree-sitter parsing.

This module provides functionality to detect programming languages from
file paths and load the corresponding tree-sitter grammars.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import tree_sitter
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_go
    import tree_sitter_rust
    LANGUAGES_AVAILABLE = True
except ImportError:
    LANGUAGES_AVAILABLE = False

from .exceptions import LanguageNotSupportedError, GrammarNotFoundError


class LanguageDetector:
    """
    Handles language detection and grammar loading for supported languages.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize language detector with configuration.
        
        Args:
            config_path: Path to languages.yaml config file
        """
        self._config = None
        self._extension_map = {}
        self._grammar_cache = {}
        self._parser_cache = {}
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "languages.yaml"
        
        self._load_config(config_path)
        self._build_extension_map()
    
    def _load_config(self, config_path: str):
        """Load language configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise GrammarNotFoundError(
                "configuration",
                grammar_path=str(config_path)
            )
        except yaml.YAMLError as e:
            raise GrammarNotFoundError(
                "configuration", 
                f"Invalid YAML configuration: {e}"
            )
    
    def _build_extension_map(self):
        """Build mapping from file extensions to language identifiers."""
        self._extension_map = {}
        
        if not self._config or 'languages' not in self._config:
            return
            
        for lang_id, lang_config in self._config['languages'].items():
            extensions = lang_config.get('extensions', [])
            for ext in extensions:
                self._extension_map[ext.lower()] = lang_id
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Language identifier or None if not supported
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        return self._extension_map.get(extension)
    
    def is_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self._config.get('languages', {})
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language identifiers."""
        return list(self._config.get('languages', {}).keys())
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self._extension_map.keys())
    
    def get_language_config(self, language: str) -> Dict[str, Any]:
        """
        Get configuration for a specific language.
        
        Args:
            language: Language identifier
            
        Returns:
            Language configuration dictionary
            
        Raises:
            LanguageNotSupportedError: If language is not supported
        """
        if not self.is_supported(language):
            raise LanguageNotSupportedError(
                language,
                supported_languages=self.get_supported_languages()
            )
        
        return self._config['languages'][language]
    
    def get_language_grammar(self, language: str):
        """
        Get tree-sitter Language object for the specified language.
        
        Args:
            language: Language identifier
            
        Returns:
            tree_sitter.Language object
            
        Raises:
            LanguageNotSupportedError: If language is not supported
            GrammarNotFoundError: If grammar cannot be loaded
        """
        if not LANGUAGES_AVAILABLE:
            raise GrammarNotFoundError(
                language,
                "tree-sitter language packages not available"
            )
        
        if not self.is_supported(language):
            raise LanguageNotSupportedError(
                language,
                supported_languages=self.get_supported_languages()
            )
        
        # Check cache first
        if language in self._grammar_cache:
            return self._grammar_cache[language]
        
        # Load grammar using individual language packages
        try:
            if language == 'javascript':
                grammar_capsule = tree_sitter_javascript.language()
            elif language == 'python':
                grammar_capsule = tree_sitter_python.language()
            elif language == 'go':
                grammar_capsule = tree_sitter_go.language()
            elif language == 'rust':
                grammar_capsule = tree_sitter_rust.language()
            else:
                raise GrammarNotFoundError(
                    language,
                    f"No grammar package found for language: {language}"
                )
            
            # Convert capsule to Language object
            try:
                grammar = tree_sitter.Language(grammar_capsule)
                
                # Verify version compatibility - be more lenient for newer versions
                if grammar.version < 13:
                    raise GrammarNotFoundError(
                        language,
                        f"Language version {grammar.version} is too old. "
                        f"Minimum supported version: 13"
                    )
                elif grammar.version > 15:
                    # Warn but allow newer versions
                    print(f"Warning: {language} grammar version {grammar.version} is newer than tested. Proceeding with caution.")
                    
            except Exception as version_error:
                # Handle tree-sitter version incompatibility
                if "Incompatible Language version" in str(version_error):
                    raise GrammarNotFoundError(
                        language,
                        f"Grammar version incompatible with current tree-sitter library. "
                        f"This may require updating tree-sitter or using a different grammar version. "
                        f"Error: {version_error}"
                    )
                else:
                    raise version_error
            
            self._grammar_cache[language] = grammar
            return grammar
            
        except Exception as e:
            raise GrammarNotFoundError(
                language,
                f"Failed to load grammar for {language}: {e}"
            )
    
    def get_language_parser(self, language: str):
        """
        Get tree-sitter Parser configured for the specified language.
        
        Args:
            language: Language identifier
            
        Returns:
            tree_sitter.Parser object
            
        Raises:
            LanguageNotSupportedError: If language is not supported
            GrammarNotFoundError: If grammar cannot be loaded
        """
        if not LANGUAGES_AVAILABLE:
            raise GrammarNotFoundError(
                language,
                "tree-sitter language packages not available"
            )
        
        # Check cache first
        if language in self._parser_cache:
            return self._parser_cache[language]
        
        # Create parser and set language
        try:
            grammar = self.get_language_grammar(language)
            parser = tree_sitter.Parser(grammar)
            
            self._parser_cache[language] = parser
            return parser
            
        except Exception as e:
            raise GrammarNotFoundError(
                language,
                f"Failed to create parser for {language}: {e}"
            )
    
    def get_performance_estimate(self, language: str) -> Dict[str, Any]:
        """
        Get performance estimates for a language.
        
        Args:
            language: Language identifier
            
        Returns:
            Dictionary with performance estimates
        """
        lang_config = self.get_language_config(language)
        performance = lang_config.get('performance', {})
        
        return {
            'expected_parse_time_ms': performance.get('expected_parse_time_ms', 100),
            'memory_estimate_mb': performance.get('memory_estimate_mb', 2.0),
            'language': language
        }
    
    def get_language_features(self, language: str) -> List[str]:
        """
        Get list of supported features for a language.
        
        Args:
            language: Language identifier
            
        Returns:
            List of feature names
        """
        lang_config = self.get_language_config(language)
        return lang_config.get('features', [])
    
    def clear_cache(self):
        """Clear grammar and parser caches."""
        self._grammar_cache.clear()
        self._parser_cache.clear()