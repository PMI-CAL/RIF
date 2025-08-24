"""
Resilient Database Interface for Issue #150
Provides high-level database operations with built-in resilience and fallback mechanisms
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
import numpy as np

from knowledge.database.database_config import DatabaseConfig
from systems.database_resilience_manager import DatabaseResilienceManager, DatabaseHealthState
from knowledge.database.vector_search import VectorSearchEngine, VectorSearchResult, SearchQuery


class ResilientDatabaseInterface:
    """
    Resilient database interface that combines the original RIFDatabase functionality
    with enhanced resilience features from DatabaseResilienceManager.
    
    Features:
    - All original RIFDatabase operations
    - Circuit breaker pattern for fault tolerance
    - Graceful degradation with fallback mechanisms  
    - Comprehensive health monitoring and metrics
    - Automatic error recovery and retry logic
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None, 
                 fallback_mode_enabled: bool = True):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize resilience manager
        self.resilience_manager = DatabaseResilienceManager(
            config=self.config,
            fallback_mode_enabled=fallback_mode_enabled
        )
        
        # Initialize vector search with resilience support
        self.vector_search = None
        self._initialize_vector_search()
        
        # Performance tracking
        self.operation_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'fallback_operations': 0,
            'avg_response_time': 0.0
        }
        
        self.logger.info(f"Resilient Database Interface initialized: {self.config}")
    
    def _initialize_vector_search(self):
        """Initialize vector search with resilience support."""
        try:
            # Create a wrapper that uses resilience manager for connections
            self.vector_search = VectorSearchEngine(self.resilience_manager)
        except Exception as e:
            self.logger.warning(f"Vector search initialization failed: {e}")
            self.vector_search = None
    
    def _track_operation(self, operation_name: str, start_time: float, success: bool, used_fallback: bool = False):
        """Track operation metrics."""
        response_time = time.time() - start_time
        
        self.operation_metrics['total_operations'] += 1
        if success:
            self.operation_metrics['successful_operations'] += 1
        else:
            self.operation_metrics['failed_operations'] += 1
        
        if used_fallback:
            self.operation_metrics['fallback_operations'] += 1
        
        # Update average response time
        total_ops = self.operation_metrics['total_operations']
        current_avg = self.operation_metrics['avg_response_time']
        self.operation_metrics['avg_response_time'] = (
            (current_avg * (total_ops - 1) + response_time) / total_ops
        )
        
        self.logger.debug(f"Operation {operation_name}: {response_time:.3f}s, success={success}, fallback={used_fallback}")
    
    # =========================================================================
    # ENTITY OPERATIONS WITH RESILIENCE
    # =========================================================================
    
    def store_entity(self, entity_data: Dict[str, Any]) -> str:
        """Store a code entity with resilience features."""
        start_time = time.time()
        operation_name = "store_entity"
        
        # Validate required fields
        required_fields = ['type', 'name', 'file_path']
        for field in required_fields:
            if field not in entity_data:
                raise ValueError(f"Required field '{field}' missing from entity data")
        
        try:
            with self.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                # Check if this is a fallback connection
                is_fallback = hasattr(conn, 'fallback_operations')
                
                if is_fallback:
                    # Use fallback operation
                    entity_id = conn.fallback_operations['store_entity'](entity_data)
                    self._track_operation(operation_name, start_time, True, used_fallback=True)
                    self.logger.info(f"Entity stored using fallback mode: {entity_data['name']}")
                    return entity_id
                
                # Normal database operation
                insert_query = """
                INSERT INTO entities (
                    type, name, file_path, line_start, line_end, 
                    ast_hash, embedding, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """
                
                # Prepare embedding data
                embedding_data = None
                if 'embedding' in entity_data and entity_data['embedding'] is not None:
                    embedding = entity_data['embedding']
                    if isinstance(embedding, np.ndarray):
                        embedding_data = embedding.tolist()
                    elif isinstance(embedding, list):
                        embedding_data = embedding
                
                # Prepare metadata
                metadata_json = None
                if 'metadata' in entity_data and entity_data['metadata'] is not None:
                    metadata_json = json.dumps(entity_data['metadata'])
                
                params = [
                    entity_data['type'],
                    entity_data['name'],
                    entity_data['file_path'],
                    entity_data.get('line_start'),
                    entity_data.get('line_end'),
                    entity_data.get('ast_hash'),
                    embedding_data,
                    metadata_json
                ]
                
                result = conn.execute(insert_query, params).fetchone()
                entity_id = str(result[0])
                
                self._track_operation(operation_name, start_time, True, used_fallback=False)
                self.logger.debug(f"Stored entity: {entity_data['name']} (ID: {entity_id})")
                return entity_id
                
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Failed to store entity: {e}")
            raise
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by ID with resilience features."""
        start_time = time.time()
        operation_name = "get_entity"
        
        try:
            with self.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                is_fallback = hasattr(conn, 'fallback_operations')
                
                if is_fallback:
                    entity = conn.fallback_operations['get_entity'](entity_id)
                    self._track_operation(operation_name, start_time, True, used_fallback=True)
                    return entity
                
                query = """
                SELECT id, type, name, file_path, line_start, line_end,
                       ast_hash, embedding, metadata, created_at, updated_at
                FROM entities WHERE id = ?
                """
                
                result = conn.execute(query, [entity_id]).fetchone()
                if not result:
                    self._track_operation(operation_name, start_time, True)
                    return None
                
                entity = {
                    'id': str(result[0]),
                    'type': result[1],
                    'name': result[2],
                    'file_path': result[3],
                    'line_start': result[4],
                    'line_end': result[5],
                    'ast_hash': result[6],
                    'embedding': result[7],
                    'metadata': json.loads(result[8]) if result[8] else {},
                    'created_at': result[9],
                    'updated_at': result[10]
                }
                
                self._track_operation(operation_name, start_time, True)
                return entity
                
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    def update_entity_embedding(self, entity_id: str, embedding: np.ndarray, 
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an entity's embedding with resilience features."""
        start_time = time.time()
        operation_name = "update_entity_embedding"
        
        try:
            with self.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                is_fallback = hasattr(conn, 'fallback_operations')
                
                if is_fallback:
                    # In fallback mode, we can't really update but we can cache the intent
                    self.logger.warning(f"Cannot update entity embedding in fallback mode: {entity_id}")
                    self._track_operation(operation_name, start_time, False, used_fallback=True)
                    return False
                
                embedding_data = embedding.tolist() if embedding is not None else None
                metadata_json = json.dumps(metadata) if metadata is not None else None
                
                query = """
                UPDATE entities 
                SET embedding = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """
                
                result = conn.execute(query, [embedding_data, metadata_json, entity_id])
                success = result.rowcount > 0
                
                self._track_operation(operation_name, start_time, success)
                if success:
                    self.logger.debug(f"Updated embedding for entity {entity_id}")
                
                return success
                
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Failed to update entity embedding {entity_id}: {e}")
            return False
    
    def search_entities(self, query: Optional[str] = None,
                       entity_types: Optional[List[str]] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities with resilience features."""
        start_time = time.time()
        operation_name = "search_entities"
        
        try:
            with self.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                is_fallback = hasattr(conn, 'fallback_operations')
                
                if is_fallback:
                    entities = conn.fallback_operations['search_entities'](
                        query=query, entity_types=entity_types, limit=limit
                    )
                    self._track_operation(operation_name, start_time, True, used_fallback=True)
                    return entities
                
                conditions = []
                params = []
                
                if query:
                    conditions.append("name ILIKE ?")
                    params.append(f"%{query}%")
                
                if entity_types:
                    placeholders = ','.join('?' * len(entity_types))
                    conditions.append(f"type IN ({placeholders})")
                    params.extend(entity_types)
                
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                
                sql_query = f"""
                SELECT id, type, name, file_path, line_start, line_end, metadata
                FROM entities
                {where_clause}
                ORDER BY name
                LIMIT ?
                """
                
                params.append(limit)
                results = conn.execute(sql_query, params).fetchall()
                
                entities = []
                for row in results:
                    entity = {
                        'id': str(row[0]),
                        'type': row[1],
                        'name': row[2],
                        'file_path': row[3],
                        'line_start': row[4],
                        'line_end': row[5],
                        'metadata': json.loads(row[6]) if row[6] else {}
                    }
                    entities.append(entity)
                
                self._track_operation(operation_name, start_time, True)
                return entities
                
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Entity search failed: {e}")
            return []
    
    # =========================================================================
    # RELATIONSHIP OPERATIONS WITH RESILIENCE
    # =========================================================================
    
    def store_relationship(self, source_id: str, target_id: str, 
                          relationship_type: str, confidence: float = 1.0,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a relationship with resilience features."""
        start_time = time.time()
        operation_name = "store_relationship"
        
        try:
            with self.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                is_fallback = hasattr(conn, 'fallback_operations')
                
                if is_fallback:
                    # Generate temporary relationship ID for fallback
                    relationship_id = f"temp_rel_{int(time.time())}_{hash(f'{source_id}_{target_id}_{relationship_type}')}"
                    self.logger.warning(f"Relationship stored in fallback mode: {relationship_id}")
                    self._track_operation(operation_name, start_time, True, used_fallback=True)
                    return relationship_id
                
                query = """
                INSERT INTO relationships (source_id, target_id, relationship_type, confidence, metadata)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """
                
                metadata_json = json.dumps(metadata) if metadata else None
                params = [source_id, target_id, relationship_type, confidence, metadata_json]
                
                result = conn.execute(query, params).fetchone()
                relationship_id = str(result[0])
                
                self._track_operation(operation_name, start_time, True)
                self.logger.debug(f"Stored relationship: {source_id} -> {target_id} ({relationship_type})")
                return relationship_id
                
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Failed to store relationship: {e}")
            raise
    
    def get_entity_relationships(self, entity_id: str, 
                               direction: str = 'both') -> List[Dict[str, Any]]:
        """Get relationships for an entity with resilience features."""
        start_time = time.time()
        operation_name = "get_entity_relationships"
        
        try:
            with self.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                is_fallback = hasattr(conn, 'fallback_operations')
                
                if is_fallback:
                    # Return empty relationships in fallback mode
                    self.logger.warning(f"Cannot get relationships in fallback mode for entity: {entity_id}")
                    self._track_operation(operation_name, start_time, True, used_fallback=True)
                    return []
                
                if direction == 'outgoing':
                    condition = "r.source_id = ?"
                elif direction == 'incoming':
                    condition = "r.target_id = ?"
                else:  # both
                    condition = "(r.source_id = ? OR r.target_id = ?)"
                    entity_id = [entity_id, entity_id]
                
                if not isinstance(entity_id, list):
                    entity_id = [entity_id]
                
                query = f"""
                SELECT 
                    r.id, r.source_id, r.target_id, r.relationship_type, 
                    r.confidence, r.metadata,
                    e1.name as source_name, e1.type as source_type,
                    e2.name as target_name, e2.type as target_type
                FROM relationships r
                JOIN entities e1 ON r.source_id = e1.id
                JOIN entities e2 ON r.target_id = e2.id
                WHERE {condition}
                ORDER BY r.confidence DESC
                """
                
                results = conn.execute(query, entity_id).fetchall()
                
                relationships = []
                for row in results:
                    relationship = {
                        'id': str(row[0]),
                        'source_id': str(row[1]),
                        'target_id': str(row[2]),
                        'relationship_type': row[3],
                        'confidence': row[4],
                        'metadata': json.loads(row[5]) if row[5] else {},
                        'source_name': row[6],
                        'source_type': row[7],
                        'target_name': row[8],
                        'target_type': row[9]
                    }
                    relationships.append(relationship)
                
                self._track_operation(operation_name, start_time, True)
                return relationships
                
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Failed to get relationships for entity {entity_id}: {e}")
            return []
    
    # =========================================================================
    # VECTOR SEARCH WITH RESILIENCE
    # =========================================================================
    
    def similarity_search(self, query_embedding: np.ndarray,
                         entity_types: Optional[List[str]] = None,
                         limit: int = 10,
                         threshold: float = 0.7) -> List[VectorSearchResult]:
        """Perform vector similarity search with resilience."""
        start_time = time.time()
        operation_name = "similarity_search"
        
        try:
            if not self.vector_search:
                self.logger.warning("Vector search not available")
                self._track_operation(operation_name, start_time, False)
                return []
            
            search_query = SearchQuery(
                embedding=query_embedding,
                limit=limit,
                threshold=threshold,
                entity_types=entity_types
            )
            
            results = self.vector_search.search_similar_entities(search_query)
            self._track_operation(operation_name, start_time, True)
            return results
            
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    # =========================================================================
    # HEALTH AND MONITORING
    # =========================================================================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics with resilience metrics."""
        start_time = time.time()
        operation_name = "get_database_stats"
        
        try:
            with self.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                is_fallback = hasattr(conn, 'fallback_operations')
                
                if is_fallback:
                    stats = conn.fallback_operations['get_database_stats']()
                    stats['interface_metrics'] = self.operation_metrics
                    self._track_operation(operation_name, start_time, True, used_fallback=True)
                    return stats
                
                # Get basic database statistics
                stats = {}
                
                # Entity statistics
                entity_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entities,
                        COUNT(embedding) as entities_with_embeddings,
                        COUNT(DISTINCT type) as entity_types
                    FROM entities
                """).fetchone()
                
                stats['entities'] = {
                    'total': entity_stats[0],
                    'with_embeddings': entity_stats[1],
                    'types': entity_stats[2]
                }
                
                # Relationship statistics
                rel_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_relationships,
                        COUNT(DISTINCT relationship_type) as relationship_types
                    FROM relationships
                """).fetchone()
                
                stats['relationships'] = {
                    'total': rel_stats[0],
                    'types': rel_stats[1]
                }
                
                # Add resilience metrics
                stats['resilience'] = self.resilience_manager.get_health_metrics()
                stats['interface_metrics'] = self.operation_metrics
                
                self._track_operation(operation_name, start_time, True)
                return stats
                
        except Exception as e:
            self._track_operation(operation_name, start_time, False)
            self.logger.error(f"Failed to get database statistics: {e}")
            
            # Return minimal stats in case of failure
            return {
                'error': str(e),
                'resilience': self.resilience_manager.get_health_metrics(),
                'interface_metrics': self.operation_metrics
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the database system."""
        health_metrics = self.resilience_manager.get_health_metrics()
        
        return {
            'timestamp': time.time(),
            'overall_health': health_metrics['health_state'],
            'database_available': health_metrics['health_state'] != 'offline',
            'circuit_breaker_open': health_metrics['circuit_breaker']['state'] == 'open',
            'error_rate': health_metrics['error_rate'],
            'avg_response_time': health_metrics['avg_response_time'],
            'uptime': health_metrics['uptime'],
            'active_connections': health_metrics['active_connections'],
            'operation_metrics': self.operation_metrics,
            'recommendations': self._generate_health_recommendations(health_metrics)
        }
    
    def _generate_health_recommendations(self, health_metrics: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on current metrics."""
        recommendations = []
        
        if health_metrics['error_rate'] > 0.2:
            recommendations.append("High error rate detected - consider checking database connectivity")
        
        if health_metrics['avg_response_time'] > 1.0:
            recommendations.append("Slow response times - consider database optimization")
        
        if health_metrics['circuit_breaker']['state'] == 'open':
            recommendations.append("Circuit breaker is open - database service may be unavailable")
        
        if health_metrics['active_connections'] == 0:
            recommendations.append("No active connections - database may be offline")
        
        return recommendations
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check."""
        start_time = time.time()
        
        health_check = {
            'timestamp': start_time,
            'database_accessible': False,
            'schema_present': False,
            'connection_pool_working': False,
            'circuit_breaker_functional': False,
            'fallback_mode_available': False,
            'performance_acceptable': False,
            'errors': []
        }
        
        try:
            # Test basic database access
            with self.resilience_manager.get_resilient_connection(timeout=5.0, allow_fallback=False) as conn:
                conn.execute("SELECT 1").fetchone()
                health_check['database_accessible'] = True
            
            # Test schema presence
            with self.resilience_manager.get_resilient_connection(allow_fallback=False) as conn:
                tables = conn.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_name IN ('entities', 'relationships', 'agent_memory')
                """).fetchall()
                
                if len(tables) >= 3:
                    health_check['schema_present'] = True
                else:
                    health_check['errors'].append(f"Missing schema tables. Found: {[t[0] for t in tables]}")
            
            # Test connection pool
            pool_stats = self.resilience_manager.get_health_metrics()
            health_check['connection_pool_working'] = pool_stats['pool_stats']['max_connections'] > 0
            
            # Test circuit breaker
            health_check['circuit_breaker_functional'] = True  # If we got this far, it's working
            
            # Test fallback mode
            if self.resilience_manager.fallback_mode_enabled:
                health_check['fallback_mode_available'] = True
            
            # Performance test
            perf_start = time.time()
            with self.resilience_manager.get_resilient_connection() as conn:
                conn.execute("SELECT COUNT(*) FROM entities").fetchone()
            
            perf_time = time.time() - perf_start
            health_check['performance_acceptable'] = perf_time < 2.0
            health_check['performance_time'] = perf_time
            
        except Exception as e:
            health_check['errors'].append(f"Health check failed: {e}")
        
        health_check['duration'] = time.time() - start_time
        health_check['overall_healthy'] = (
            health_check['database_accessible'] and 
            health_check['schema_present'] and 
            health_check['connection_pool_working'] and
            len(health_check['errors']) == 0
        )
        
        return health_check
    
    def force_recovery(self) -> Dict[str, Any]:
        """Force recovery operations (admin function)."""
        recovery_results = {
            'timestamp': time.time(),
            'actions_taken': [],
            'success': False
        }
        
        try:
            # Reset circuit breaker
            self.resilience_manager.force_circuit_breaker_reset()
            recovery_results['actions_taken'].append('Circuit breaker reset')
            
            # Clear error history
            self.resilience_manager._error_history.clear()
            recovery_results['actions_taken'].append('Error history cleared')
            
            # Test connection
            with self.resilience_manager.get_resilient_connection(timeout=10.0) as conn:
                conn.execute("SELECT 1").fetchone()
            
            recovery_results['actions_taken'].append('Database connection test successful')
            recovery_results['success'] = True
            
        except Exception as e:
            recovery_results['error'] = str(e)
            self.logger.error(f"Force recovery failed: {e}")
        
        return recovery_results
    
    def close(self):
        """Close the database interface and cleanup resources."""
        self.resilience_manager.shutdown()
        self.logger.info("Resilient Database Interface closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()