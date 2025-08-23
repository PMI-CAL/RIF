"""
Main relationship detection coordinator that orchestrates all relationship analyzers.

This is the primary interface for detecting relationships between code entities,
coordinating multiple specialized analyzers and handling cross-file resolution.
"""

import time
import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_analyzer import BaseRelationshipAnalyzer, RelationshipAnalysisContext
from .relationship_types import CodeRelationship, RelationshipDetectionResult, RelationshipType
from .import_analyzer import ImportExportAnalyzer
from .call_analyzer import FunctionCallAnalyzer  
from .inheritance_analyzer import InheritanceAnalyzer

from ..extraction.entity_types import CodeEntity
from ..parsing.parser_manager import ParserManager


class RelationshipDetector:
    """
    Main coordinator for relationship detection across all analyzers.
    
    Orchestrates multiple relationship analyzers, handles cross-file resolution,
    and provides a unified interface for relationship detection and storage.
    """
    
    def __init__(self, parser_manager: ParserManager, max_concurrent_files: int = 4):
        self.parser_manager = parser_manager
        self.max_concurrent_files = max_concurrent_files
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self.analyzers: List[BaseRelationshipAnalyzer] = [
            ImportExportAnalyzer(parser_manager),
            FunctionCallAnalyzer(parser_manager),
            InheritanceAnalyzer(parser_manager)
        ]
        
        # Analysis context for cross-file resolution
        self.analysis_context = RelationshipAnalysisContext(parser_manager)
        
        # Performance tracking
        self.analysis_metrics = {
            'files_processed': 0,
            'relationships_detected': 0,
            'analysis_time': 0.0,
            'analyzer_breakdown': {},
            'language_breakdown': {}
        }
    
    def detect_relationships_from_file(self, file_path: str, entities: List[CodeEntity]) -> RelationshipDetectionResult:
        """
        Detect all relationships in a single file.
        
        Args:
            file_path: Path to the source file
            entities: List of entities found in this file
            
        Returns:
            RelationshipDetectionResult containing all detected relationships
        """
        start_time = time.time()
        
        try:
            # Parse the file
            tree, language = self.parser_manager.parse_file(file_path)
            if not tree:
                return RelationshipDetectionResult(
                    file_path=file_path,
                    language="unknown",
                    relationships=[],
                    detection_time=0.0,
                    success=False,
                    error_message="Failed to parse file"
                )
            
            # Start file analysis in context
            self.analysis_context.start_file_analysis(file_path, language, entities)
            
            # Run all applicable analyzers
            all_relationships = []
            analyzer_results = {}
            
            for analyzer in self.analyzers:
                if analyzer.can_analyze(language):
                    analyzer_start = time.time()
                    try:
                        relationships = analyzer.analyze_ast(tree, file_path, language, entities)
                        all_relationships.extend(relationships)
                        
                        analyzer_time = time.time() - analyzer_start
                        analyzer_results[analyzer.__class__.__name__] = {
                            'relationships': len(relationships),
                            'time': analyzer_time
                        }
                        
                        self.logger.debug(
                            f"{analyzer.__class__.__name__} found {len(relationships)} relationships "
                            f"in {file_path} ({analyzer_time:.3f}s)"
                        )
                    except Exception as e:
                        self.logger.error(f"Error in {analyzer.__class__.__name__} for {file_path}: {e}")
                        analyzer_results[analyzer.__class__.__name__] = {
                            'relationships': 0,
                            'time': time.time() - analyzer_start,
                            'error': str(e)
                        }
            
            # Finish file analysis
            total_analysis_time = self.analysis_context.finish_file_analysis()
            detection_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(language, len(all_relationships), detection_time, analyzer_results)
            
            return RelationshipDetectionResult(
                file_path=file_path,
                language=language,
                relationships=all_relationships,
                detection_time=detection_time,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            detection_time = time.time() - start_time
            self.logger.error(f"Error detecting relationships in {file_path}: {e}")
            
            return RelationshipDetectionResult(
                file_path=file_path,
                language="unknown",
                relationships=[],
                detection_time=detection_time,
                success=False,
                error_message=str(e)
            )
    
    def detect_relationships_from_directory(self, directory_path: str, 
                                          extensions: Optional[List[str]] = None,
                                          recursive: bool = True,
                                          exclude_patterns: Optional[List[str]] = None) -> List[RelationshipDetectionResult]:
        """
        Detect relationships from all files in a directory.
        
        Args:
            directory_path: Path to the directory to analyze
            extensions: File extensions to include (e.g., ['.py', '.js'])
            recursive: Whether to search subdirectories
            exclude_patterns: Patterns to exclude (e.g., ['node_modules', '__pycache__'])
            
        Returns:
            List of RelationshipDetectionResult for each processed file
        """
        start_time = time.time()
        
        # Find all files to process
        files_to_process = self._find_files_to_process(
            directory_path, extensions, recursive, exclude_patterns
        )
        
        if not files_to_process:
            self.logger.warning(f"No files found to process in {directory_path}")
            return []
        
        self.logger.info(f"Processing {len(files_to_process)} files for relationship detection")
        
        # First pass: Extract entities from all files (needed for cross-file resolution)
        self.logger.info("Phase 1: Loading entities for cross-file resolution...")
        all_entities = self._load_entities_for_files(files_to_process)
        
        # Register all entities for cross-file resolution
        for entities in all_entities.values():
            self.analysis_context.cross_file_resolver.register_entities(entities)
        
        self.logger.info(
            f"Loaded {sum(len(entities) for entities in all_entities.values())} entities "
            f"from {len(all_entities)} files"
        )
        
        # Second pass: Detect relationships with concurrent processing
        self.logger.info("Phase 2: Detecting relationships...")
        results = []
        
        if self.max_concurrent_files > 1:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=self.max_concurrent_files) as executor:
                # Submit all file processing tasks
                future_to_file = {}
                for file_path in files_to_process:
                    entities = all_entities.get(file_path, [])
                    future = executor.submit(self.detect_relationships_from_file, file_path, entities)
                    future_to_file[future] = file_path
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.success:
                            self.logger.debug(
                                f"Completed {file_path}: {len(result.relationships)} relationships "
                                f"({result.detection_time:.3f}s)"
                            )
                        else:
                            self.logger.warning(f"Failed to process {file_path}: {result.error_message}")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        results.append(RelationshipDetectionResult(
                            file_path=file_path,
                            language="unknown",
                            relationships=[],
                            detection_time=0.0,
                            success=False,
                            error_message=str(e)
                        ))
        else:
            # Sequential processing
            for file_path in files_to_process:
                entities = all_entities.get(file_path, [])
                result = self.detect_relationships_from_file(file_path, entities)
                results.append(result)
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if r.success]
        total_relationships = sum(len(r.relationships) for r in successful_results)
        
        self.logger.info(
            f"Relationship detection complete: {len(successful_results)}/{len(results)} files processed, "
            f"{total_relationships} relationships detected in {total_time:.2f}s"
        )
        
        return results
    
    def get_supported_relationship_types(self) -> Set[RelationshipType]:
        """Get all relationship types supported by registered analyzers."""
        supported_types = set()
        for analyzer in self.analyzers:
            supported_types.update(analyzer.get_supported_relationship_types())
        return supported_types
    
    def get_supported_languages(self) -> Set[str]:
        """Get all languages supported by at least one analyzer."""
        supported_languages = set()
        for analyzer in self.analyzers:
            supported_languages.update(analyzer.supported_languages)
        return supported_languages
    
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get comprehensive analysis metrics."""
        return dict(self.analysis_metrics)
    
    def reset_metrics(self):
        """Reset analysis metrics."""
        self.analysis_metrics = {
            'files_processed': 0,
            'relationships_detected': 0,
            'analysis_time': 0.0,
            'analyzer_breakdown': {},
            'language_breakdown': {}
        }
    
    def add_analyzer(self, analyzer: BaseRelationshipAnalyzer):
        """Add a custom relationship analyzer."""
        self.analyzers.append(analyzer)
        self.logger.info(f"Added analyzer: {analyzer.__class__.__name__}")
    
    def remove_analyzer(self, analyzer_class: type):
        """Remove an analyzer by class type."""
        self.analyzers = [a for a in self.analyzers if not isinstance(a, analyzer_class)]
        self.logger.info(f"Removed analyzer: {analyzer_class.__name__}")
    
    def _find_files_to_process(self, directory_path: str, extensions: Optional[List[str]],
                              recursive: bool, exclude_patterns: Optional[List[str]]) -> List[str]:
        """Find all files that should be processed for relationship detection."""
        
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            return []
        
        # Default extensions based on supported languages
        if extensions is None:
            extensions = ['.py', '.js', '.jsx', '.mjs', '.cjs', '.go', '.rs']
        
        exclude_patterns = exclude_patterns or ['node_modules', '__pycache__', '.git', 'target', 'build']
        
        files = []
        
        # Use glob patterns for recursive search
        if recursive:
            for ext in extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in files:
            if file_path.is_file():
                # Check if any exclude pattern matches the path
                path_str = str(file_path)
                if not any(pattern in path_str for pattern in exclude_patterns):
                    filtered_files.append(str(file_path))
        
        return sorted(filtered_files)
    
    def _load_entities_for_files(self, file_paths: List[str]) -> Dict[str, List[CodeEntity]]:
        """Load entities for all files, required for cross-file resolution."""
        
        # This would typically load entities from the entity storage
        # For now, we'll need to extract entities on-demand
        from ..extraction.entity_extractor import EntityExtractor
        
        entity_extractor = EntityExtractor()
        all_entities = {}
        
        for file_path in file_paths:
            try:
                extraction_result = entity_extractor.extract_from_file(file_path)
                if extraction_result.success:
                    all_entities[file_path] = extraction_result.entities
                else:
                    self.logger.warning(f"Failed to extract entities from {file_path}: {extraction_result.error_message}")
                    all_entities[file_path] = []
            except Exception as e:
                self.logger.error(f"Error extracting entities from {file_path}: {e}")
                all_entities[file_path] = []
        
        return all_entities
    
    def _update_metrics(self, language: str, relationship_count: int, 
                       detection_time: float, analyzer_results: Dict[str, Any]):
        """Update analysis metrics with results from a file."""
        
        self.analysis_metrics['files_processed'] += 1
        self.analysis_metrics['relationships_detected'] += relationship_count
        self.analysis_metrics['analysis_time'] += detection_time
        
        # Update language breakdown
        if language not in self.analysis_metrics['language_breakdown']:
            self.analysis_metrics['language_breakdown'][language] = {
                'files': 0, 'relationships': 0, 'time': 0.0
            }
        
        lang_stats = self.analysis_metrics['language_breakdown'][language]
        lang_stats['files'] += 1
        lang_stats['relationships'] += relationship_count
        lang_stats['time'] += detection_time
        
        # Update analyzer breakdown
        for analyzer_name, analyzer_data in analyzer_results.items():
            if analyzer_name not in self.analysis_metrics['analyzer_breakdown']:
                self.analysis_metrics['analyzer_breakdown'][analyzer_name] = {
                    'relationships': 0, 'time': 0.0, 'files': 0, 'errors': 0
                }
            
            analyzer_stats = self.analysis_metrics['analyzer_breakdown'][analyzer_name]
            analyzer_stats['files'] += 1
            analyzer_stats['relationships'] += analyzer_data['relationships']
            analyzer_stats['time'] += analyzer_data['time']
            
            if 'error' in analyzer_data:
                analyzer_stats['errors'] += 1
    
    def validate_relationships(self, relationships: List[CodeRelationship]) -> Dict[str, Any]:
        """Validate detected relationships for consistency and quality."""
        
        validation_results = {
            'total_relationships': len(relationships),
            'valid_relationships': 0,
            'invalid_relationships': 0,
            'warnings': [],
            'errors': [],
            'relationship_type_counts': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        for relationship in relationships:
            try:
                # Basic validation
                if relationship.source_id == relationship.target_id:
                    validation_results['errors'].append(
                        f"Self-referential relationship: {relationship.id}"
                    )
                    validation_results['invalid_relationships'] += 1
                    continue
                
                if not (0.0 <= relationship.confidence <= 1.0):
                    validation_results['errors'].append(
                        f"Invalid confidence score {relationship.confidence}: {relationship.id}"
                    )
                    validation_results['invalid_relationships'] += 1
                    continue
                
                # Count by type
                rel_type = relationship.relationship_type.value
                if rel_type not in validation_results['relationship_type_counts']:
                    validation_results['relationship_type_counts'][rel_type] = 0
                validation_results['relationship_type_counts'][rel_type] += 1
                
                # Confidence distribution
                if relationship.confidence >= 0.8:
                    validation_results['confidence_distribution']['high'] += 1
                elif relationship.confidence >= 0.5:
                    validation_results['confidence_distribution']['medium'] += 1
                else:
                    validation_results['confidence_distribution']['low'] += 1
                
                validation_results['valid_relationships'] += 1
                
            except Exception as e:
                validation_results['errors'].append(f"Validation error for {relationship.id}: {e}")
                validation_results['invalid_relationships'] += 1
        
        return validation_results