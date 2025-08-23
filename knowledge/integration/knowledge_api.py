"""
KnowledgeAPI - Unified API Gateway

Provides a single interface for all knowledge operations across the hybrid pipeline.
Implements the agent-friendly API from the Master Coordination Plan with
natural language processing and intelligent query routing.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .hybrid_knowledge_system import HybridKnowledgeSystem


@dataclass
class QueryResult:
    """Standardized query result format."""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time_ms: float
    result_count: int
    error: Optional[str] = None


@dataclass
class ProcessingResult:
    """Standardized file processing result format."""
    success: bool
    file_path: str
    entities_extracted: int
    relationships_detected: int
    embeddings_generated: int
    processing_time_ms: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class KnowledgeAPI:
    """
    Unified API gateway for all hybrid knowledge system operations.
    
    Provides a clean, agent-friendly interface that abstracts the complexity
    of the four-component pipeline while maintaining high performance and
    comprehensive functionality.
    """
    
    def __init__(self, knowledge_system: HybridKnowledgeSystem):
        """Initialize the Knowledge API with a hybrid knowledge system."""
        self.knowledge_system = knowledge_system
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.default_query_config = {
            'performance_mode': 'BALANCED',  # FAST, BALANCED, COMPREHENSIVE
            'max_results': 50,
            'min_relevance_score': 0.1,
            'enable_caching': True,
            'timeout_ms': 5000
        }
        
        # Performance tracking
        self.api_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'avg_response_time_ms': 0.0,
            'files_processed': 0
        }
        
        self.logger.info("KnowledgeAPI initialized")
    
    # === Core Query Interface ===
    
    def query(self, 
              query_text: str,
              performance_mode: str = 'BALANCED',
              max_results: int = 50,
              filters: Dict[str, Any] = None,
              context: Dict[str, Any] = None) -> QueryResult:
        """
        Execute a natural language query against the knowledge system.
        
        Args:
            query_text: Natural language query string
            performance_mode: 'FAST', 'BALANCED', or 'COMPREHENSIVE'
            max_results: Maximum number of results to return
            filters: Optional filters (file types, languages, etc.)
            context: Optional context (active files, current task, etc.)
            
        Returns:
            QueryResult: Standardized query result
            
        Examples:
            >>> api.query("find authentication functions")
            >>> api.query("show me error handling patterns similar to try-catch")
            >>> api.query("what functions call processPayment")
        """
        start_time = time.time()
        self.api_metrics['total_queries'] += 1
        
        try:
            self.logger.info(f"Processing query: {query_text[:100]}...")
            
            # Prepare query parameters
            query_params = {
                'performance_mode': performance_mode,
                'max_results': max_results,
                'filters': filters or {},
                'context': context or {}
            }
            
            # Execute query through knowledge system
            raw_results = self.knowledge_system.process_query(query_text, **query_params)
            
            # Format results
            formatted_results = self._format_query_results(raw_results)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.api_metrics['successful_queries'] += 1
            self._update_response_time_metric(processing_time_ms)
            
            result = QueryResult(
                success=True,
                query=query_text,
                results=formatted_results,
                metadata={
                    'performance_mode': performance_mode,
                    'filters_applied': filters,
                    'context_used': bool(context),
                    'cached': getattr(raw_results, 'cached', False)
                },
                processing_time_ms=processing_time_ms,
                result_count=len(formatted_results)
            )
            
            if getattr(raw_results, 'cached', False):
                self.api_metrics['cache_hits'] += 1
            
            self.logger.info(f"Query completed in {processing_time_ms:.1f}ms, {len(formatted_results)} results")
            return result
            
        except Exception as e:
            self.api_metrics['failed_queries'] += 1
            processing_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"Query failed: {e}")
            return QueryResult(
                success=False,
                query=query_text,
                results=[],
                metadata={'error_type': type(e).__name__},
                processing_time_ms=processing_time_ms,
                result_count=0,
                error=str(e)
            )
    
    # === Specialized Query Methods ===
    
    def find_entities(self,
                     name_pattern: str = None,
                     entity_type: str = None,
                     file_pattern: str = None,
                     language: str = None) -> QueryResult:
        """
        Find code entities with specific criteria.
        
        Args:
            name_pattern: Pattern to match entity names
            entity_type: Type of entity (function, class, module, etc.)
            file_pattern: File path pattern
            language: Programming language
            
        Returns:
            QueryResult: Entities matching the criteria
        """
        # Build query from parameters
        query_parts = []
        if name_pattern:
            query_parts.append(f"find entities named {name_pattern}")
        if entity_type:
            query_parts.append(f"of type {entity_type}")
        if file_pattern:
            query_parts.append(f"in files matching {file_pattern}")
        if language:
            query_parts.append(f"written in {language}")
        
        query_text = " ".join(query_parts) if query_parts else "find all entities"
        
        filters = {}
        if entity_type:
            filters['entity_type'] = entity_type
        if file_pattern:
            filters['file_pattern'] = file_pattern
        if language:
            filters['language'] = language
        
        return self.query(query_text, filters=filters, performance_mode='FAST')
    
    def find_similar_code(self,
                         reference_text: str,
                         similarity_threshold: float = 0.7,
                         max_results: int = 20) -> QueryResult:
        """
        Find code similar to the provided reference.
        
        Args:
            reference_text: Code text to find similarities to
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            max_results: Maximum results to return
            
        Returns:
            QueryResult: Similar code entities
        """
        query_text = f"find code similar to: {reference_text[:200]}"
        
        return self.query(
            query_text,
            performance_mode='COMPREHENSIVE',
            max_results=max_results,
            filters={'min_similarity': similarity_threshold}
        )
    
    def analyze_dependencies(self,
                           entity_name: str,
                           direction: str = 'both',
                           max_depth: int = 3) -> QueryResult:
        """
        Analyze dependencies for a specific entity.
        
        Args:
            entity_name: Name of the entity to analyze
            direction: 'incoming', 'outgoing', or 'both'
            max_depth: Maximum depth of dependency traversal
            
        Returns:
            QueryResult: Dependency analysis results
        """
        if direction == 'incoming':
            query_text = f"what depends on {entity_name}"
        elif direction == 'outgoing':
            query_text = f"what does {entity_name} depend on"
        else:
            query_text = f"analyze all dependencies for {entity_name}"
        
        return self.query(
            query_text,
            performance_mode='COMPREHENSIVE',
            filters={'max_depth': max_depth}
        )
    
    def analyze_impact(self,
                      entity_name: str,
                      change_type: str = 'modification') -> QueryResult:
        """
        Analyze the impact of changes to an entity.
        
        Args:
            entity_name: Name of the entity to analyze
            change_type: Type of change ('modification', 'deletion', 'renaming')
            
        Returns:
            QueryResult: Impact analysis results
        """
        query_text = f"what breaks if I {change_type} {entity_name}"
        
        return self.query(
            query_text,
            performance_mode='COMPREHENSIVE',
            context={'change_type': change_type}
        )
    
    # === File Processing Interface ===
    
    def process_file(self, file_path: str, force_reprocess: bool = False) -> ProcessingResult:
        """
        Process a file through the complete knowledge pipeline.
        
        Args:
            file_path: Path to the file to process
            force_reprocess: Force reprocessing even if file hasn't changed
            
        Returns:
            ProcessingResult: Processing results and metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # Process through the knowledge system
            raw_result = self.knowledge_system.process_file(
                file_path,
                force_reprocess=force_reprocess
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.api_metrics['files_processed'] += 1
            
            result = ProcessingResult(
                success=raw_result.get('success', False),
                file_path=file_path,
                entities_extracted=len(raw_result.get('entities', [])),
                relationships_detected=len(raw_result.get('relationships', [])),
                embeddings_generated=len(raw_result.get('embeddings', {}).get('generated', [])),
                processing_time_ms=processing_time_ms,
                details=raw_result
            )
            
            self.logger.info(f"File processed in {processing_time_ms:.1f}ms: "
                           f"{result.entities_extracted} entities, "
                           f"{result.relationships_detected} relationships")
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"File processing failed: {e}")
            return ProcessingResult(
                success=False,
                file_path=file_path,
                entities_extracted=0,
                relationships_detected=0,
                embeddings_generated=0,
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def process_directory(self,
                         directory_path: str,
                         file_patterns: List[str] = None,
                         exclude_patterns: List[str] = None,
                         max_files: int = 1000) -> Dict[str, ProcessingResult]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to directory to process
            file_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: File patterns to exclude
            max_files: Maximum number of files to process
            
        Returns:
            Dict[str, ProcessingResult]: Results by file path
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        # Find files to process
        files_to_process = []
        
        if file_patterns:
            for pattern in file_patterns:
                files_to_process.extend(directory.rglob(pattern))
        else:
            # Default patterns for supported languages
            default_patterns = ['*.py', '*.js', '*.jsx', '*.mjs', '*.cjs', '*.go', '*.rs']
            for pattern in default_patterns:
                files_to_process.extend(directory.rglob(pattern))
        
        # Apply exclusion patterns
        if exclude_patterns:
            filtered_files = []
            for file_path in files_to_process:
                excluded = False
                for exclude_pattern in exclude_patterns:
                    if file_path.match(exclude_pattern):
                        excluded = True
                        break
                if not excluded:
                    filtered_files.append(file_path)
            files_to_process = filtered_files
        
        # Limit number of files
        files_to_process = files_to_process[:max_files]
        
        self.logger.info(f"Processing {len(files_to_process)} files from {directory_path}")
        
        # Process files
        results = {}
        for file_path in files_to_process:
            try:
                result = self.process_file(str(file_path))
                results[str(file_path)] = result
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                results[str(file_path)] = ProcessingResult(
                    success=False,
                    file_path=str(file_path),
                    entities_extracted=0,
                    relationships_detected=0,
                    embeddings_generated=0,
                    processing_time_ms=0,
                    error=str(e)
                )
        
        successful_files = sum(1 for r in results.values() if r.success)
        self.logger.info(f"Directory processing complete: {successful_files}/{len(files_to_process)} files successful")
        
        return results
    
    # === System Status and Management ===
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information."""
        system_status = self.knowledge_system.get_system_status()
        
        return {
            'system_healthy': system_status.healthy,
            'components': system_status.components_ready,
            'resource_usage': system_status.resource_usage,
            'error_count': system_status.error_count,
            'warnings': system_status.warnings,
            'api_metrics': self.api_metrics,
            'last_updated': system_status.last_updated
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        system_metrics = self.knowledge_system.get_metrics()
        
        return {
            'system_metrics': system_metrics,
            'api_metrics': self.api_metrics,
            'query_success_rate': (
                self.api_metrics['successful_queries'] / max(1, self.api_metrics['total_queries'])
            ),
            'cache_hit_rate': (
                self.api_metrics['cache_hits'] / max(1, self.api_metrics['successful_queries'])
            ),
            'avg_response_time_ms': self.api_metrics['avg_response_time_ms']
        }
    
    # === Helper Methods ===
    
    def _format_query_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """Format raw query results into standardized format."""
        if not raw_results:
            return []
        
        # Handle different result formats from the query planner
        if hasattr(raw_results, 'results'):
            results = raw_results.results
        elif isinstance(raw_results, list):
            results = raw_results
        else:
            results = [raw_results]
        
        formatted = []
        for result in results:
            if isinstance(result, dict):
                formatted.append(result)
            else:
                # Convert other types to dict representation
                formatted.append({
                    'type': type(result).__name__,
                    'content': str(result),
                    'raw': result
                })
        
        return formatted
    
    def _update_response_time_metric(self, response_time_ms: float):
        """Update average response time using exponential moving average."""
        if self.api_metrics['avg_response_time_ms'] == 0:
            self.api_metrics['avg_response_time_ms'] = response_time_ms
        else:
            # EMA with alpha = 0.1
            alpha = 0.1
            self.api_metrics['avg_response_time_ms'] = (
                (1 - alpha) * self.api_metrics['avg_response_time_ms'] + 
                alpha * response_time_ms
            )
    
    # === Agent-Specific Convenience Methods ===
    
    def quick_search(self, search_term: str) -> List[Dict[str, Any]]:
        """Quick search for agents that need fast responses."""
        result = self.query(search_term, performance_mode='FAST', max_results=10)
        return result.results
    
    def deep_analysis(self, analysis_query: str) -> Dict[str, Any]:
        """Deep analysis for agents that need comprehensive results."""
        result = self.query(analysis_query, performance_mode='COMPREHENSIVE', max_results=100)
        
        return {
            'results': result.results,
            'metadata': result.metadata,
            'processing_time_ms': result.processing_time_ms,
            'result_count': result.result_count,
            'success': result.success
        }
    
    def context_aware_search(self,
                           query: str,
                           active_files: List[str] = None,
                           current_task: str = None) -> List[Dict[str, Any]]:
        """Context-aware search for agents working on specific tasks."""
        context = {}
        if active_files:
            context['active_files'] = active_files
        if current_task:
            context['current_task'] = current_task
        
        result = self.query(query, context=context, performance_mode='BALANCED')
        return result.results