#!/usr/bin/env python3
"""
Simplified Integration Layer for Issue #40 Master Coordination

This creates a working integration of all four pipeline components without 
complex dependency injection, focusing on demonstrating the coordination 
and providing the unified API interface for RIF agents.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing components
from knowledge.extraction.entity_extractor import EntityExtractor
from knowledge.relationships.relationship_detector import RelationshipDetector
from knowledge.embeddings.embedding_pipeline import EmbeddingPipeline
from knowledge.query import QueryPlanner, plan_and_execute_query
from knowledge.parsing.parser_manager import ParserManager
from knowledge.database.connection_manager import DuckDBConnectionManager, DatabaseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedKnowledgeSystem:
    """
    Simplified integration of the hybrid knowledge pipeline.
    
    Provides unified coordination of Issues #30-33 components:
    - Entity Extraction (Issue #30)
    - Relationship Detection (Issue #31) 
    - Vector Embeddings (Issue #32)
    - Query Planning (Issue #33)
    """
    
    def __init__(self, database_path: str = None):
        """Initialize the knowledge system."""
        # Use default database path if none provided
        self.database_path = database_path or "knowledge/chromadb/entities.duckdb"
        self.logger = logging.getLogger(__name__)
        
        # System state
        self._initialized = False
        self._components_ready = False
        
        # Performance metrics
        self.metrics = {
            'queries_processed': 0,
            'files_processed': 0,
            'entities_extracted': 0,
            'relationships_detected': 0,
            'embeddings_generated': 0,
            'system_start_time': time.time()
        }
        
        # Component instances (will be initialized on first use)
        self._entity_extractor = None
        self._relationship_detector = None
        self._embedding_pipeline = None
        self._query_planner = None
        self._parser_manager = None
        
        logger.info(f"SimplifiedKnowledgeSystem initialized with database: {self.database_path}")
    
    def initialize(self) -> bool:
        """Initialize all components."""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing simplified knowledge system components...")
            start_time = time.time()
            
            # Initialize parser manager (foundation for other components)
            self._parser_manager = ParserManager.get_instance()
            
            # Initialize entity extractor (Issue #30)
            self._entity_extractor = EntityExtractor()
            
            # Initialize relationship detector (Issue #31) 
            self._relationship_detector = RelationshipDetector(
                parser_manager=self._parser_manager,
                max_concurrent_files=2
            )
            
            # Initialize embedding pipeline (Issue #32)
            self._embedding_pipeline = EmbeddingPipeline(
                db_path=self.database_path,
                embedding_dim=384
            )
            
            # Initialize query planner (Issue #33)
            self._query_planner = QueryPlanner(
                db_path=self.database_path,
                enable_caching=True
            )
            
            self._initialized = True
            self._components_ready = True
            
            init_time = time.time() - start_time
            logger.info(f"Knowledge system initialized successfully in {init_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge system: {e}")
            return False
    
    def process_query(self, query: str, max_results: int = 50) -> Dict[str, Any]:
        """
        Process a natural language query through the hybrid search system.
        
        Args:
            query: Natural language query
            max_results: Maximum results to return
            
        Returns:
            Dict containing query results and metadata
        """
        if not self._ensure_initialized():
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        
        try:
            # Execute query through the query planner (Issue #33)
            results = self._query_planner.execute_query(query, max_results=max_results)
            
            # Update metrics
            self.metrics['queries_processed'] += 1
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Format response
            return {
                'success': True,
                'query': query,
                'results': [
                    {
                        'name': result.entity_name if hasattr(result, 'entity_name') else str(result),
                        'type': result.entity_type if hasattr(result, 'entity_type') else 'unknown',
                        'file': result.file_path if hasattr(result, 'file_path') else None,
                        'relevance': result.relevance_score if hasattr(result, 'relevance_score') else 1.0,
                        'metadata': getattr(result, 'metadata', {})
                    }
                    for result in results.results[:max_results]
                ],
                'total_found': results.total_found if hasattr(results, 'total_found') else len(results.results),
                'processing_time_ms': processing_time_ms,
                'cached': getattr(results, 'cached', False)
            }
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Query processing failed: {e}")
            
            return {
                'success': False,
                'query': query,
                'results': [],
                'total_found': 0,
                'processing_time_ms': processing_time_ms,
                'error': str(e)
            }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file through the complete pipeline.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Dict containing processing results
        """
        if not self._ensure_initialized():
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        file_path = str(Path(file_path).resolve())
        
        try:
            logger.info(f"Processing file through complete pipeline: {file_path}")
            
            # Step 1: Extract entities (Issue #30)
            entities = self._entity_extractor.extract_from_file(file_path)
            entity_count = len(entities.entities) if hasattr(entities, 'entities') else len(entities)
            
            # Step 2: Detect relationships (Issue #31)
            relationships = self._relationship_detector.detect_from_file(file_path)
            relationship_count = len(relationships.relationships) if hasattr(relationships, 'relationships') else len(relationships)
            
            # Step 3: Generate embeddings (Issue #32)
            embedding_result = self._embedding_pipeline.process_entities_by_file(file_path)
            embedding_count = len(embedding_result.get('generated', [])) if isinstance(embedding_result, dict) else 0
            
            # Update metrics
            self.metrics['files_processed'] += 1
            self.metrics['entities_extracted'] += entity_count
            self.metrics['relationships_detected'] += relationship_count
            self.metrics['embeddings_generated'] += embedding_count
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'file_path': file_path,
                'entities_extracted': entity_count,
                'relationships_detected': relationship_count,
                'embeddings_generated': embedding_count,
                'processing_time_seconds': processing_time,
                'components_used': ['entity_extraction', 'relationship_detection', 'embedding_generation']
            }
            
            logger.info(f"File processed successfully in {processing_time:.2f}s: "
                       f"{entity_count} entities, {relationship_count} relationships, {embedding_count} embeddings")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"File processing failed: {e}")
            
            return {
                'success': False,
                'file_path': file_path,
                'entities_extracted': 0,
                'relationships_detected': 0,
                'embeddings_generated': 0,
                'processing_time_seconds': processing_time,
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'initialized': self._initialized,
            'components_ready': self._components_ready,
            'components': {
                'entity_extractor': self._entity_extractor is not None,
                'relationship_detector': self._relationship_detector is not None,
                'embedding_pipeline': self._embedding_pipeline is not None,
                'query_planner': self._query_planner is not None,
                'parser_manager': self._parser_manager is not None
            },
            'metrics': self.metrics,
            'uptime_seconds': time.time() - self.metrics['system_start_time']
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        uptime = time.time() - self.metrics['system_start_time']
        
        return {
            'uptime_seconds': uptime,
            'queries_per_minute': (self.metrics['queries_processed'] * 60) / max(uptime, 1),
            'files_per_minute': (self.metrics['files_processed'] * 60) / max(uptime, 1),
            'total_entities': self.metrics['entities_extracted'],
            'total_relationships': self.metrics['relationships_detected'],
            'total_embeddings': self.metrics['embeddings_generated'],
            'query_cache_performance': self._query_planner.get_performance_metrics() if self._query_planner else {}
        }
    
    def _ensure_initialized(self) -> bool:
        """Ensure system is initialized."""
        if not self._initialized:
            return self.initialize()
        return True
    
    # Agent-friendly convenience methods
    
    def quick_search(self, search_term: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Quick search for agents needing fast responses."""
        result = self.process_query(search_term, max_results=max_results)
        return result.get('results', [])
    
    def find_entities(self, name_pattern: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Find entities by name pattern and type."""
        query = f"find entities named {name_pattern}"
        if entity_type:
            query += f" of type {entity_type}"
        
        result = self.process_query(query)
        return result.get('results', [])
    
    def analyze_dependencies(self, entity_name: str) -> List[Dict[str, Any]]:
        """Analyze dependencies for an entity."""
        query = f"analyze dependencies for {entity_name}"
        result = self.process_query(query)
        return result.get('results', [])
    
    def find_similar_code(self, reference_code: str) -> List[Dict[str, Any]]:
        """Find code similar to reference."""
        query = f"find code similar to: {reference_code[:100]}"
        result = self.process_query(query)
        return result.get('results', [])


class SimplifiedKnowledgeAPI:
    """
    Simplified API gateway that matches the interface expected by RIF agents.
    """
    
    def __init__(self, database_path: str = None):
        """Initialize the API with a knowledge system."""
        self.knowledge_system = SimplifiedKnowledgeSystem(database_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize system on creation
        if not self.knowledge_system.initialize():
            logger.warning("Knowledge system initialization failed - some features may not work")
    
    def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """Execute a natural language query."""
        max_results = kwargs.get('max_results', 50)
        return self.knowledge_system.process_query(query_text, max_results=max_results)
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file through the complete pipeline."""
        return self.knowledge_system.process_file(file_path)
    
    def process_directory(self, directory_path: str, file_patterns: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Process all files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Default patterns for supported languages
        patterns = file_patterns or ['*.py', '*.js', '*.jsx', '*.mjs', '*.cjs', '*.go', '*.rs']
        
        files_to_process = []
        for pattern in patterns:
            files_to_process.extend(directory.rglob(pattern))
        
        # Limit to prevent overwhelming the system
        files_to_process = files_to_process[:100]
        
        results = {}
        for file_path in files_to_process:
            try:
                result = self.process_file(str(file_path))
                results[str(file_path)] = result
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results[str(file_path)] = {
                    'success': False,
                    'file_path': str(file_path),
                    'error': str(e)
                }
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return self.knowledge_system.get_system_status()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.knowledge_system.get_performance_metrics()
    
    # Agent convenience methods
    def quick_search(self, search_term: str) -> List[Dict[str, Any]]:
        """Quick search for agents."""
        return self.knowledge_system.quick_search(search_term)
    
    def find_entities(self, name_pattern: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Find entities by pattern."""
        return self.knowledge_system.find_entities(name_pattern, entity_type)
    
    def analyze_dependencies(self, entity_name: str) -> List[Dict[str, Any]]:
        """Analyze entity dependencies."""
        return self.knowledge_system.analyze_dependencies(entity_name)
    
    def find_similar_code(self, reference_code: str) -> List[Dict[str, Any]]:
        """Find similar code."""
        return self.knowledge_system.find_similar_code(reference_code)


def create_knowledge_api(database_path: str = None) -> SimplifiedKnowledgeAPI:
    """
    Factory function to create a knowledge API instance.
    
    Args:
        database_path: Optional database path
        
    Returns:
        SimplifiedKnowledgeAPI: Ready-to-use API instance
    """
    return SimplifiedKnowledgeAPI(database_path)


# Example usage and testing
if __name__ == "__main__":
    # Test the simplified integration
    print("=" * 70)
    print("SIMPLIFIED KNOWLEDGE SYSTEM INTEGRATION TEST")
    print("=" * 70)
    
    # Create API instance
    api = create_knowledge_api()
    
    # Check system status
    status = api.get_system_status()
    print(f"System initialized: {status['initialized']}")
    print(f"Components ready: {status['components_ready']}")
    
    if status['initialized']:
        # Test queries
        test_queries = [
            "find authentication functions",
            "show me error handling patterns",
            "what functions call processPayment"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing Query: {query} ---")
            result = api.query(query)
            
            if result['success']:
                print(f"Found {result['total_found']} results in {result['processing_time_ms']:.1f}ms")
                for i, res in enumerate(result['results'][:3]):
                    print(f"  {i+1}. {res['name']} ({res['type']}) - relevance: {res['relevance']:.2f}")
            else:
                print(f"Query failed: {result.get('error', 'Unknown error')}")
        
        # Show performance metrics
        metrics = api.get_performance_metrics()
        print(f"\n--- Performance Metrics ---")
        print(f"Uptime: {metrics['uptime_seconds']:.1f}s")
        print(f"Queries processed: {status['metrics']['queries_processed']}")
        print(f"Files processed: {status['metrics']['files_processed']}")
        
        print("\n✅ Simplified integration test completed successfully!")
    else:
        print("❌ System initialization failed")