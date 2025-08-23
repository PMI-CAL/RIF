"""
LightRAG Knowledge System Adapter with Migration Compatibility

This module provides a concrete implementation of the KnowledgeInterface
using the existing LightRAG system with ChromaDB backend.

This adapter maintains full compatibility with existing LightRAG functionality
while providing the standardized interface required for agent decoupling.

Enhanced for Issue #36: Supports gradual migration from legacy knowledge systems
with query/response translation, performance monitoring, and context optimization.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime

# Import the interface
from .interface import KnowledgeInterface, KnowledgeStorageError, KnowledgeRetrievalError

# Import context optimizer for agent consumption (Issue #34)
try:
    from .context.optimizer import ContextOptimizer
    CONTEXT_OPTIMIZER_AVAILABLE = True
except ImportError:
    CONTEXT_OPTIMIZER_AVAILABLE = False

# Add LightRAG to path and import
current_dir = os.path.dirname(os.path.abspath(__file__))
lightrag_path = os.path.join(os.path.dirname(current_dir), 'lightrag')
sys.path.insert(0, lightrag_path)

try:
    from lightrag.core.lightrag_core import LightRAGCore
    LIGHTRAG_AVAILABLE = True
except ImportError as e:
    LIGHTRAG_AVAILABLE = False
    LIGHTRAG_IMPORT_ERROR = str(e)


class LightRAGKnowledgeAdapter(KnowledgeInterface):
    """
    LightRAG implementation of the KnowledgeInterface with Migration Compatibility.
    
    This adapter provides a bridge between the standardized KnowledgeInterface
    and the LightRAG system, enabling agents to use LightRAG through the
    interface without direct dependencies.
    
    Enhanced Features (Issue #36):
    - Semantic vector search using ChromaDB
    - Metadata filtering and categorization
    - Collection-based organization
    - Full CRUD operations
    - Automatic ID generation
    - Error handling and logging
    - **Migration Support**: Query/response translation for gradual migration
    - **Context Optimization**: Integration with Issue #34 context optimizer
    - **Performance Monitoring**: Track migration performance
    - **Translation Layer**: Convert between different knowledge system formats
    """
    
    def __init__(self, knowledge_path: Optional[str] = None, enable_migration_features: bool = True):
        """
        Initialize the LightRAG adapter with migration compatibility.
        
        Args:
            knowledge_path: Path to knowledge directory (defaults to RIF knowledge directory)
            enable_migration_features: Enable migration-specific features (Issue #36)
        """
        if not LIGHTRAG_AVAILABLE:
            raise RuntimeError(f"LightRAG not available: {LIGHTRAG_IMPORT_ERROR}")
        
        self.logger = logging.getLogger(f"{__name__}.LightRAGKnowledgeAdapter")
        self.migration_features_enabled = enable_migration_features
        
        # Initialize LightRAG core
        try:
            self.lightrag = LightRAGCore(knowledge_path)
            self.logger.info("LightRAG adapter initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LightRAG: {e}")
            raise RuntimeError(f"LightRAG initialization failed: {e}")
        
        # Initialize migration-specific components (Issue #36)
        if self.migration_features_enabled:
            self._init_migration_features()
    
    def _init_migration_features(self):
        """Initialize migration compatibility features."""
        # Context optimizer for agent consumption (Issue #34 integration)
        self.context_optimizer = None
        if CONTEXT_OPTIMIZER_AVAILABLE:
            try:
                self.context_optimizer = ContextOptimizer()
                self.logger.info("Context optimizer integrated successfully")
            except Exception as e:
                self.logger.warning(f"Context optimizer initialization failed: {e}")
        
        # Migration metrics tracking
        self.migration_metrics = {
            'queries_translated': 0,
            'responses_translated': 0,
            'performance_samples': [],
            'translation_errors': 0,
            'context_optimizations': 0
        }
        
        # Query translation registry
        self.query_translators = {}
        self.response_translators = {}
        
        # Register default translators
        self._register_default_translators()
        
        self.logger.info("Migration compatibility features initialized")
    
    def _register_default_translators(self):
        """Register default query and response translators for common legacy systems."""
        # Legacy file-based system translator
        self.query_translators['file_based'] = self._translate_file_based_query
        self.response_translators['file_based'] = self._translate_file_based_response
        
        # JSON-based system translator  
        self.query_translators['json_based'] = self._translate_json_based_query
        self.response_translators['json_based'] = self._translate_json_based_response
    
    def store_knowledge(self, 
                       collection: str, 
                       content: Union[str, Dict[str, Any]], 
                       metadata: Optional[Dict[str, Any]] = None,
                       doc_id: Optional[str] = None) -> Optional[str]:
        """
        Store knowledge item using LightRAG.
        
        Args:
            collection: Collection name
            content: Content to store
            metadata: Optional metadata
            doc_id: Optional document ID
            
        Returns:
            Document ID if successful, None on failure
        """
        try:
            # Validate collection
            if not self._is_valid_collection(collection):
                raise ValueError(f"Invalid collection: {collection}")
            
            # Convert content to string if needed
            if isinstance(content, dict):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content)
            
            # Ensure metadata is a dictionary
            metadata = metadata or {}
            
            # Add interface-specific metadata
            metadata.update({
                "interface_version": "1.0",
                "adapter": "LightRAGKnowledgeAdapter"
            })
            
            # Store using LightRAG
            result_id = self.lightrag.store_knowledge(
                collection_name=collection,
                content=content_str,
                metadata=metadata,
                doc_id=doc_id
            )
            
            self.logger.debug(f"Stored knowledge in {collection}: {result_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Failed to store knowledge in {collection}: {e}")
            raise KnowledgeStorageError(f"Storage failed: {e}")
    
    def retrieve_knowledge(self, 
                          query: str, 
                          collection: Optional[str] = None, 
                          n_results: int = 5,
                          filters: Optional[Dict[str, Any]] = None,
                          agent_type: Optional[str] = None,
                          legacy_system: Optional[str] = None,
                          optimize_for_agent: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge items using LightRAG semantic search with migration support.
        
        Enhanced for Issue #36: Supports query translation and context optimization.
        
        Args:
            query: Search query string
            collection: Optional collection filter
            n_results: Maximum results to return
            filters: Optional metadata filters
            agent_type: Agent type for context optimization (from Issue #34)
            legacy_system: Legacy system type for query translation
            optimize_for_agent: Apply context optimization if available
            
        Returns:
            List of matching documents, optionally optimized for agent consumption
        """
        start_time = time.time()
        
        try:
            # Step 1: Query translation if needed (Migration Feature)
            translated_query = query
            if self.migration_features_enabled and legacy_system and legacy_system in self.query_translators:
                translated_query = self._translate_query(query, legacy_system)
                self.migration_metrics['queries_translated'] += 1
            
            # Step 2: Validate collection if specified
            if collection and not self._is_valid_collection(collection):
                self.logger.warning(f"Invalid collection specified: {collection}")
                return []
            
            # Step 3: Use LightRAG retrieve with translated query
            raw_results = self.lightrag.retrieve_knowledge(
                query=translated_query,
                collection_name=collection,
                n_results=n_results * 2 if optimize_for_agent else n_results,  # Get more for optimization
                filters=filters
            )
            
            # Step 4: Apply context optimization for agent consumption (Issue #34 integration)
            if (self.migration_features_enabled and 
                optimize_for_agent and 
                self.context_optimizer and 
                agent_type):
                
                optimized_result = self.context_optimizer.optimize_for_agent(
                    results=raw_results,
                    query=translated_query,
                    agent_type=agent_type,
                    context={"collection": collection, "filters": filters},
                    min_results=min(3, n_results)
                )
                
                results = optimized_result['optimized_results'][:n_results]
                self.migration_metrics['context_optimizations'] += 1
                
                self.logger.debug(f"Context optimization applied: {len(raw_results)} -> {len(results)} results")
            else:
                results = raw_results[:n_results]
            
            # Step 5: Response translation if needed (Migration Feature)
            if self.migration_features_enabled and legacy_system and legacy_system in self.response_translators:
                results = self._translate_responses(results, legacy_system)
                self.migration_metrics['responses_translated'] += len(results)
            
            # Step 6: Track performance metrics
            if self.migration_features_enabled:
                duration = time.time() - start_time
                self.migration_metrics['performance_samples'].append({
                    'query_type': 'retrieve_knowledge',
                    'duration': duration,
                    'results_count': len(results),
                    'agent_type': agent_type,
                    'optimized': optimize_for_agent and self.context_optimizer is not None,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only last 100 samples
                if len(self.migration_metrics['performance_samples']) > 100:
                    self.migration_metrics['performance_samples'] = self.migration_metrics['performance_samples'][-100:]
            
            self.logger.debug(f"Retrieved {len(results)} results for query: {query[:50]}... (duration: {time.time() - start_time:.3f}s)")
            return results
            
        except Exception as e:
            if self.migration_features_enabled:
                self.migration_metrics['translation_errors'] += 1
            self.logger.error(f"Failed to retrieve knowledge for query '{query}': {e}")
            raise KnowledgeRetrievalError(f"Retrieval failed: {e}")
    
    def update_knowledge(self,
                        collection: str,
                        doc_id: str,
                        content: Optional[Union[str, Dict[str, Any]]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update existing knowledge item using LightRAG.
        
        Args:
            collection: Collection name
            doc_id: Document ID to update
            content: New content (None to keep existing)
            metadata: New metadata (None to keep existing)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate collection
            if not self._is_valid_collection(collection):
                raise ValueError(f"Invalid collection: {collection}")
            
            # Convert content to string if provided
            content_str = None
            if content is not None:
                if isinstance(content, dict):
                    content_str = json.dumps(content, indent=2)
                else:
                    content_str = str(content)
            
            # Update using LightRAG
            success = self.lightrag.update_knowledge(
                collection_name=collection,
                doc_id=doc_id,
                content=content_str,
                metadata=metadata
            )
            
            if success:
                self.logger.debug(f"Updated knowledge {doc_id} in {collection}")
            else:
                self.logger.warning(f"Failed to update {doc_id} in {collection}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge {doc_id}: {e}")
            return False
    
    def delete_knowledge(self, collection: str, doc_id: str) -> bool:
        """
        Delete knowledge item using LightRAG.
        
        Args:
            collection: Collection name
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate collection
            if not self._is_valid_collection(collection):
                raise ValueError(f"Invalid collection: {collection}")
            
            # Delete using LightRAG
            success = self.lightrag.delete_knowledge(collection, doc_id)
            
            if success:
                self.logger.debug(f"Deleted knowledge {doc_id} from {collection}")
            else:
                self.logger.warning(f"Failed to delete {doc_id} from {collection}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting knowledge {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics from LightRAG.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            return self.lightrag.get_collection_stats()
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get LightRAG system information.
        
        Returns:
            System information dictionary
        """
        return {
            "implementation": "LightRAGKnowledgeAdapter",
            "version": "1.0.0",
            "backend": "ChromaDB",
            "collections": list(self.lightrag.collections.keys()) if hasattr(self.lightrag, 'collections') else [],
            "features": [
                "semantic_search",
                "vector_embeddings", 
                "metadata_filtering",
                "collection_organization",
                "crud_operations",
                "automatic_id_generation"
            ],
            "knowledge_path": self.lightrag.knowledge_path,
            "db_path": self.lightrag.db_path
        }
    
    def _is_valid_collection(self, collection: str) -> bool:
        """
        Check if collection name is valid for LightRAG.
        
        Args:
            collection: Collection name to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not collection or not isinstance(collection, str):
            return False
        
        # Check against known LightRAG collections
        valid_collections = getattr(self.lightrag, 'collections', {})
        if valid_collections and collection not in valid_collections:
            # Allow common collections that might not be initialized yet
            common_collections = {
                'patterns', 'decisions', 'code_snippets', 'issue_resolutions',
                'learnings', 'checkpoints', 'metrics'
            }
            return collection in common_collections
        
        return True
    
    def export_collection(self, collection: str, output_path: str) -> bool:
        """
        Export collection to JSON file (LightRAG-specific feature).
        
        Args:
            collection: Collection name to export
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_valid_collection(collection):
                return False
            
            return self.lightrag.export_collection(collection, output_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting collection {collection}: {e}")
            return False
    
    def reset_collection(self, collection: str) -> bool:
        """
        Reset (clear) a collection (LightRAG-specific feature).
        
        Args:
            collection: Collection name to reset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_valid_collection(collection):
                return False
            
            return self.lightrag.reset_collection(collection)
            
        except Exception as e:
            self.logger.error(f"Error resetting collection {collection}: {e}")
            return False
    
    # Migration Compatibility Methods (Issue #36)
    
    def _translate_query(self, query: str, legacy_system: str) -> str:
        """Translate query from legacy system format to LightRAG format."""
        if legacy_system in self.query_translators:
            try:
                return self.query_translators[legacy_system](query)
            except Exception as e:
                self.logger.warning(f"Query translation failed for {legacy_system}: {e}")
        return query
    
    def _translate_responses(self, results: List[Dict[str, Any]], legacy_system: str) -> List[Dict[str, Any]]:
        """Translate responses to legacy system format."""
        if legacy_system in self.response_translators:
            try:
                return self.response_translators[legacy_system](results)
            except Exception as e:
                self.logger.warning(f"Response translation failed for {legacy_system}: {e}")
        return results
    
    def _translate_file_based_query(self, query: str) -> str:
        """Translate file-based system queries to semantic search format."""
        # Security: Sanitize input to prevent injection attacks
        import re
        
        # Check for dangerous SQL injection patterns
        if any(dangerous in query.upper() for dangerous in ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'UNION', '--']):
            self.logger.warning(f"Potentially malicious SQL query blocked: {query[:50]}")
            return "invalid query blocked for security"
        
        # Check for XSS patterns
        if '<script>' in query or 'javascript:' in query:
            self.logger.warning(f"Potentially malicious script query blocked: {query[:50]}")
            return "invalid query blocked for security"
            
        sanitized_query = re.sub(r'[<>"\';]', '', query)  # Remove dangerous characters
        
        # Convert file path queries to content-based queries
        if '/' in sanitized_query or '\\' in sanitized_query:
            # Extract filename and convert to content search
            filename = sanitized_query.split('/')[-1].split('\\')[-1]
            # Additional security: Only allow alphanumeric and common file characters
            safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
            return f"content related to {safe_filename}"
        return sanitized_query
    
    def _translate_file_based_response(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate responses to file-based system format."""
        translated_results = []
        for result in results:
            translated_result = result.copy()
            # Add file-like metadata
            if 'metadata' not in translated_result:
                translated_result['metadata'] = {}
            translated_result['metadata']['file_path'] = f"/knowledge/{result.get('collection', 'default')}/{result.get('id', 'unknown')}.txt"
            translated_results.append(translated_result)
        return translated_results
    
    def _translate_json_based_query(self, query: str) -> str:
        """Translate JSON-based queries to semantic search format."""
        # Security: Sanitize and validate JSON input
        import re
        
        # Remove dangerous patterns
        if '__proto__' in query or '<script>' in query or 'DROP' in query.upper():
            self.logger.warning(f"Potentially malicious query blocked: {query[:50]}")
            return "invalid query blocked for security"
        
        # Convert structured queries to natural language
        if query.startswith('{') and query.endswith('}'):
            try:
                query_obj = json.loads(query)
                # Security: Validate that this is a simple object without prototype pollution
                if isinstance(query_obj, dict) and not any(key.startswith('__') for key in query_obj.keys()):
                    parts = []
                    for key, value in query_obj.items():
                        # Sanitize key and value
                        safe_key = re.sub(r'[^a-zA-Z0-9_]', '', str(key))
                        safe_value = re.sub(r'[<>"\';]', '', str(value))
                        parts.append(f"{safe_key} {safe_value}")
                    return " ".join(parts)
            except json.JSONDecodeError:
                pass
        return re.sub(r'[<>"\';]', '', query)  # Sanitize any non-JSON query
    
    def _translate_json_based_response(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate responses to JSON-based system format."""
        translated_results = []
        for result in results:
            translated_result = {
                "document_id": result.get('id'),
                "text_content": result.get('content'),
                "metadata_fields": result.get('metadata', {}),
                "collection_name": result.get('collection'),
                "relevance_score": 1.0 - result.get('distance', 0.0),
                "source_system": "lightrag"
            }
            translated_results.append(translated_result)
        return translated_results
    
    def register_query_translator(self, system_type: str, translator_func: Callable[[str], str]):
        """Register a custom query translator for a legacy system type."""
        if not self.migration_features_enabled:
            self.logger.warning("Migration features not enabled - translator registration ignored")
            return
        
        self.query_translators[system_type] = translator_func
        self.logger.info(f"Registered query translator for {system_type}")
    
    def register_response_translator(self, system_type: str, translator_func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        """Register a custom response translator for a legacy system type."""
        if not self.migration_features_enabled:
            self.logger.warning("Migration features not enabled - translator registration ignored")
            return
        
        self.response_translators[system_type] = translator_func
        self.logger.info(f"Registered response translator for {system_type}")
    
    def get_migration_metrics(self) -> Dict[str, Any]:
        """Get migration performance and usage metrics."""
        if not self.migration_features_enabled:
            return {"migration_features": "disabled"}
        
        metrics = self.migration_metrics.copy()
        
        # Calculate performance statistics
        if metrics['performance_samples']:
            durations = [sample['duration'] for sample in metrics['performance_samples']]
            metrics['avg_response_time'] = sum(durations) / len(durations)
            metrics['max_response_time'] = max(durations)
            metrics['min_response_time'] = min(durations)
            
            # Breakdown by optimization
            optimized_samples = [s for s in metrics['performance_samples'] if s.get('optimized', False)]
            unoptimized_samples = [s for s in metrics['performance_samples'] if not s.get('optimized', False)]
            
            if optimized_samples:
                optimized_durations = [s['duration'] for s in optimized_samples]
                metrics['optimized_avg_time'] = sum(optimized_durations) / len(optimized_durations)
            
            if unoptimized_samples:
                unoptimized_durations = [s['duration'] for s in unoptimized_samples]
                metrics['unoptimized_avg_time'] = sum(unoptimized_durations) / len(unoptimized_durations)
        
        return metrics
    
    def retrieve_knowledge_for_agent(self,
                                   query: str,
                                   agent_type: str,
                                   collection: Optional[str] = None,
                                   n_results: int = 5,
                                   filters: Optional[Dict[str, Any]] = None,
                                   legacy_system: Optional[str] = None) -> Dict[str, Any]:
        """
        Convenience method for agent-optimized knowledge retrieval.
        
        This method provides a complete migration-aware knowledge retrieval
        specifically optimized for RIF agent consumption.
        
        Args:
            query: Search query string
            agent_type: Agent type for context optimization
            collection: Optional collection filter
            n_results: Maximum results to return
            filters: Optional metadata filters
            legacy_system: Legacy system for translation
            
        Returns:
            Dictionary with optimized results and metadata:
            {
                'results': [...],
                'optimization_applied': bool,
                'translation_applied': bool,
                'performance_info': {...},
                'context_info': {...}
            }
        """
        start_time = time.time()
        
        # Retrieve with full migration support
        results = self.retrieve_knowledge(
            query=query,
            collection=collection,
            n_results=n_results,
            filters=filters,
            agent_type=agent_type,
            legacy_system=legacy_system,
            optimize_for_agent=True
        )
        
        # Build comprehensive response
        response = {
            'results': results,
            'query_info': {
                'original_query': query,
                'agent_type': agent_type,
                'collection': collection,
                'legacy_system': legacy_system
            },
            'optimization_applied': self.context_optimizer is not None,
            'translation_applied': legacy_system is not None and legacy_system in self.query_translators,
            'performance_info': {
                'duration': time.time() - start_time,
                'results_count': len(results),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if self.migration_features_enabled:
            response['migration_metrics'] = self.get_migration_metrics()
        
        return response


class MockKnowledgeAdapter(KnowledgeInterface):
    """
    Mock implementation for testing purposes.
    
    This adapter provides a simple in-memory implementation that can be used
    for testing agents without requiring a full LightRAG setup.
    """
    
    def __init__(self):
        """Initialize mock adapter with in-memory storage."""
        self.logger = logging.getLogger(f"{__name__}.MockKnowledgeAdapter")
        self.storage = {}  # collection -> {doc_id: {content, metadata}}
        self.next_id = 1
        
    def store_knowledge(self, 
                       collection: str, 
                       content: Union[str, Dict[str, Any]], 
                       metadata: Optional[Dict[str, Any]] = None,
                       doc_id: Optional[str] = None) -> Optional[str]:
        """Store knowledge in memory."""
        if collection not in self.storage:
            self.storage[collection] = {}
        
        if doc_id is None:
            doc_id = f"mock_{self.next_id}"
            self.next_id += 1
        
        # Convert content to string if needed
        if isinstance(content, dict):
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)
        
        self.storage[collection][doc_id] = {
            "content": content_str,
            "metadata": metadata or {},
            "id": doc_id
        }
        
        return doc_id
    
    def retrieve_knowledge(self, 
                          query: str, 
                          collection: Optional[str] = None, 
                          n_results: int = 5,
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve knowledge with simple text matching."""
        results = []
        
        collections_to_search = [collection] if collection else self.storage.keys()
        
        for coll in collections_to_search:
            if coll not in self.storage:
                continue
            
            for doc_id, doc_data in self.storage[coll].items():
                content = doc_data["content"]
                metadata = doc_data["metadata"]
                
                # Apply filters if provided
                if filters:
                    matches_filters = all(
                        metadata.get(key) == value for key, value in filters.items()
                    )
                    if not matches_filters:
                        continue
                
                # Simple text matching
                if query.lower() in content.lower():
                    # Calculate simple relevance
                    relevance = content.lower().count(query.lower()) / len(content.split())
                    
                    results.append({
                        "collection": coll,
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "distance": 1.0 - min(relevance, 1.0)
                    })
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x["distance"])
        return results[:n_results]
    
    def update_knowledge(self,
                        collection: str,
                        doc_id: str,
                        content: Optional[Union[str, Dict[str, Any]]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update knowledge in memory."""
        if collection not in self.storage or doc_id not in self.storage[collection]:
            return False
        
        doc_data = self.storage[collection][doc_id]
        
        if content is not None:
            if isinstance(content, dict):
                doc_data["content"] = json.dumps(content, indent=2)
            else:
                doc_data["content"] = str(content)
        
        if metadata is not None:
            doc_data["metadata"].update(metadata)
        
        return True
    
    def delete_knowledge(self, collection: str, doc_id: str) -> bool:
        """Delete knowledge from memory."""
        if collection in self.storage and doc_id in self.storage[collection]:
            del self.storage[collection][doc_id]
            return True
        return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get mock collection statistics."""
        stats = {}
        for collection, docs in self.storage.items():
            stats[collection] = {
                "count": len(docs),
                "description": f"Mock collection {collection}"
            }
        return stats
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get mock system information."""
        return {
            "implementation": "MockKnowledgeAdapter",
            "version": "1.0.0",
            "backend": "in-memory",
            "collections": list(self.storage.keys()),
            "features": ["basic_storage", "basic_retrieval", "text_search"]
        }


# Register implementations with the factory
def register_implementations():
    """Register all available implementations with the factory."""
    from .interface import KnowledgeSystemFactory
    
    # Register LightRAG adapter if available
    if LIGHTRAG_AVAILABLE:
        KnowledgeSystemFactory.register_implementation("lightrag", LightRAGKnowledgeAdapter)
        KnowledgeSystemFactory.set_default_implementation("lightrag")
    
    # Always register mock for testing
    KnowledgeSystemFactory.register_implementation("mock", MockKnowledgeAdapter)
    
    # Set mock as default if LightRAG is not available
    if not LIGHTRAG_AVAILABLE:
        KnowledgeSystemFactory.set_default_implementation("mock")


# Auto-register implementations when module is imported
register_implementations()


# Convenience functions for backward compatibility and migration support
def get_lightrag_adapter(knowledge_path: Optional[str] = None, enable_migration_features: bool = True) -> LightRAGKnowledgeAdapter:
    """
    Get a LightRAG adapter instance with migration support.
    
    Args:
        knowledge_path: Optional knowledge path
        enable_migration_features: Enable Issue #36 migration features
        
    Returns:
        LightRAGKnowledgeAdapter instance with migration compatibility
    """
    return LightRAGKnowledgeAdapter(knowledge_path, enable_migration_features)

def get_migration_compatible_adapter(knowledge_path: Optional[str] = None) -> LightRAGKnowledgeAdapter:
    """
    Get a fully migration-compatible LightRAG adapter instance.
    
    This convenience function ensures all migration features are enabled
    and provides the complete Issue #36 functionality.
    
    Args:
        knowledge_path: Optional knowledge path
        
    Returns:
        LightRAGKnowledgeAdapter with full migration support
    """
    adapter = LightRAGKnowledgeAdapter(knowledge_path, enable_migration_features=True)
    
    # Verify context optimizer integration
    if not CONTEXT_OPTIMIZER_AVAILABLE:
        adapter.logger.warning("Context optimizer not available - some migration features may be limited")
    
    return adapter


def get_mock_adapter() -> MockKnowledgeAdapter:
    """
    Get a mock adapter instance for testing.
    
    Returns:
        MockKnowledgeAdapter instance
    """
    return MockKnowledgeAdapter()