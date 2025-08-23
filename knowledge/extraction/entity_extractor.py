"""
Main entity extractor that coordinates language-specific extractors and integrates with storage.
"""

import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..parsing.parser_manager import ParserManager
from ..parsing.exceptions import ParsingError, LanguageNotSupportedError

from .entity_types import CodeEntity, ExtractionResult
from .language_extractors import (
    JavaScriptExtractor,
    PythonExtractor,
    GoExtractor,
    RustExtractor
)


class EntityExtractor:
    """
    Main entity extractor that coordinates language-specific extractors.
    
    This class handles:
    - File parsing using ParserManager
    - Dispatching to appropriate language extractors
    - Batch processing and error handling
    - Integration with storage layer
    """
    
    def __init__(self):
        self.parser_manager = ParserManager.get_instance()
        self.logger = logging.getLogger(__name__)
        
        # Initialize language-specific extractors
        self._extractors = {
            'javascript': JavaScriptExtractor(),
            'python': PythonExtractor(),
            'go': GoExtractor(),
            'rust': RustExtractor(),
        }
        
        # Performance metrics
        self._extraction_metrics = {
            'files_processed': 0,
            'entities_extracted': 0,
            'extraction_time': 0.0,
            'errors': 0
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of languages supported for entity extraction."""
        return list(self._extractors.keys())
    
    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported for entity extraction."""
        return language in self._extractors
    
    def extract_from_file(self, file_path: str, language: Optional[str] = None) -> ExtractionResult:
        """
        Extract entities from a single file.
        
        Args:
            file_path: Path to the file to process
            language: Language identifier (auto-detected if None)
            
        Returns:
            ExtractionResult containing extracted entities or error information
        """
        start_time = time.time()
        
        try:
            # Parse file using ParserManager
            parse_result = self.parser_manager.parse_file(file_path, language, use_cache=True)
            
            detected_language = parse_result['language']
            
            # Check if we have an extractor for this language
            if not self.is_language_supported(detected_language):
                return ExtractionResult(
                    file_path=file_path,
                    language=detected_language,
                    entities=[],
                    extraction_time=time.time() - start_time,
                    success=False,
                    error_message=f"No entity extractor available for language: {detected_language}"
                )
            
            # Check if parsing was successful
            if parse_result.get('has_error', False) or not parse_result.get('tree'):
                return ExtractionResult(
                    file_path=file_path,
                    language=detected_language,
                    entities=[],
                    extraction_time=time.time() - start_time,
                    success=False,
                    error_message="File parsing failed or AST contains errors"
                )
            
            # Extract entities using language-specific extractor
            extractor = self._extractors[detected_language]
            
            # Read source code for extraction
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            entities = extractor.extract_entities(parse_result['tree'], file_path, source_code)
            
            # Update metrics
            extraction_time = time.time() - start_time
            self._update_metrics(len(entities), extraction_time, success=True)
            
            return ExtractionResult(
                file_path=file_path,
                language=detected_language,
                entities=entities,
                extraction_time=extraction_time,
                success=True
            )
            
        except (ParsingError, LanguageNotSupportedError) as e:
            extraction_time = time.time() - start_time
            self._update_metrics(0, extraction_time, success=False)
            
            return ExtractionResult(
                file_path=file_path,
                language=language or 'unknown',
                entities=[],
                extraction_time=extraction_time,
                success=False,
                error_message=str(e)
            )
        
        except Exception as e:
            extraction_time = time.time() - start_time
            self._update_metrics(0, extraction_time, success=False)
            
            self.logger.error(f"Unexpected error extracting entities from {file_path}: {e}")
            
            return ExtractionResult(
                file_path=file_path,
                language=language or 'unknown',
                entities=[],
                extraction_time=extraction_time,
                success=False,
                error_message=f"Unexpected error: {e}"
            )
    
    def extract_from_files(self, file_paths: List[str], 
                          batch_size: int = 50, 
                          continue_on_error: bool = True) -> List[ExtractionResult]:
        """
        Extract entities from multiple files with batch processing.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Number of files to process in each batch
            continue_on_error: Whether to continue processing if individual files fail
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        # Process files in batches to manage memory usage
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}: files {i+1}-{min(i+batch_size, len(file_paths))}")
            
            for file_path in batch:
                result = self.extract_from_file(file_path)
                results.append(result)
                
                if not result.success and not continue_on_error:
                    self.logger.error(f"Stopping processing due to error in {file_path}: {result.error_message}")
                    break
            
            # Optional: Cleanup caches between batches for memory management
            if len(results) % (batch_size * 2) == 0:
                self.parser_manager.cleanup_invalid_cache()
        
        return results
    
    def extract_from_directory(self, directory_path: str, 
                             extensions: Optional[List[str]] = None,
                             recursive: bool = True,
                             exclude_patterns: Optional[List[str]] = None) -> List[ExtractionResult]:
        """
        Extract entities from all supported files in a directory.
        
        Args:
            directory_path: Path to directory to scan
            extensions: File extensions to include (default: all supported)
            recursive: Whether to scan subdirectories
            exclude_patterns: Patterns to exclude (e.g., ['test', 'spec', 'node_modules'])
            
        Returns:
            List of ExtractionResult objects
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Get supported extensions if not specified
        if extensions is None:
            extensions = self.parser_manager.get_supported_extensions()
        
        # Find files to process
        file_paths = []
        
        pattern = '**/*' if recursive else '*'
        for ext in extensions:
            ext_pattern = f"{pattern}.{ext.lstrip('.')}"
            files = list(directory.glob(ext_pattern))
            file_paths.extend([str(f) for f in files if f.is_file()])
        
        # Apply exclusion patterns
        if exclude_patterns:
            filtered_paths = []
            for file_path in file_paths:
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern in file_path:
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_paths.append(file_path)
            file_paths = filtered_paths
        
        self.logger.info(f"Found {len(file_paths)} files to process in {directory_path}")
        
        return self.extract_from_files(file_paths)
    
    def get_extraction_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for entity extraction."""
        metrics = dict(self._extraction_metrics)
        
        # Add derived metrics
        if metrics['files_processed'] > 0:
            metrics['avg_extraction_time'] = metrics['extraction_time'] / metrics['files_processed']
            metrics['avg_entities_per_file'] = metrics['entities_extracted'] / metrics['files_processed']
            metrics['success_rate'] = 1.0 - (metrics['errors'] / metrics['files_processed'])
        else:
            metrics['avg_extraction_time'] = 0.0
            metrics['avg_entities_per_file'] = 0.0
            metrics['success_rate'] = 0.0
        
        # Add parser metrics
        parser_metrics = self.parser_manager.get_metrics()
        metrics['parser'] = parser_metrics
        
        # Add extractor-specific metrics
        extractor_info = {}
        for lang, extractor in self._extractors.items():
            extractor_info[lang] = {
                'supported_entity_types': [et.value for et in extractor.get_supported_entity_types()]
            }
        metrics['extractors'] = extractor_info
        
        return metrics
    
    def reset_metrics(self):
        """Reset extraction metrics."""
        self._extraction_metrics = {
            'files_processed': 0,
            'entities_extracted': 0,
            'extraction_time': 0.0,
            'errors': 0
        }
    
    def _update_metrics(self, entities_count: int, extraction_time: float, success: bool):
        """Update internal metrics."""
        self._extraction_metrics['files_processed'] += 1
        self._extraction_metrics['entities_extracted'] += entities_count
        self._extraction_metrics['extraction_time'] += extraction_time
        
        if not success:
            self._extraction_metrics['errors'] += 1
    
    def validate_extraction(self, result: ExtractionResult) -> Dict[str, Any]:
        """
        Validate extraction results for quality assurance.
        
        Args:
            result: ExtractionResult to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        if not result.success:
            validation['valid'] = False
            validation['errors'].append(f"Extraction failed: {result.error_message}")
            return validation
        
        # Basic statistics
        entity_counts = result.get_entity_counts()
        validation['stats'] = entity_counts
        
        # Validation checks
        if not result.entities:
            validation['warnings'].append("No entities extracted from file")
        
        # Check for duplicate entities
        entity_signatures = []
        duplicates = 0
        for entity in result.entities:
            signature = f"{entity.type.value}:{entity.name}:{entity.file_path}"
            if signature in entity_signatures:
                duplicates += 1
            else:
                entity_signatures.append(signature)
        
        if duplicates > 0:
            validation['warnings'].append(f"Found {duplicates} duplicate entities")
        
        # Check entity naming patterns
        unnamed_entities = [e for e in result.entities if not e.name.strip()]
        if unnamed_entities:
            validation['warnings'].append(f"Found {len(unnamed_entities)} entities with empty names")
        
        # Performance check
        if result.extraction_time > 5.0:  # 5 seconds threshold
            validation['warnings'].append(f"Extraction took {result.extraction_time:.2f}s (slow)")
        
        return validation