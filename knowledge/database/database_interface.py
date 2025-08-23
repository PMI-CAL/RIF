"""
Unified database interface for the RIF knowledge system.
Combines connection management, vector search, and database operations.
Issue #26: Set up DuckDB as embedded database with vector search
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
import numpy as np

from .connection_manager import DuckDBConnectionManager
from .vector_search import VectorSearchEngine, VectorSearchResult, SearchQuery
from .database_config import DatabaseConfig


class RIFDatabase:
    """
    Unified database interface for the RIF knowledge system.
    
    Provides high-level operations for:
    - Entity storage and retrieval
    - Relationship management
    - Agent memory storage
    - Vector similarity search
    - Database maintenance
    
    Features:
    - Connection pooling with 500MB memory limit
    - VSS extension for vector search
    - Thread-safe operations
    - Performance monitoring
    - Automatic schema management
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.connection_manager = DuckDBConnectionManager(self.config)
        self.vector_search = VectorSearchEngine(self.connection_manager)
        
        self.logger.info(f"RIF Database initialized: {self.config}")
    
    # =========================================================================
    # ENTITY OPERATIONS
    # =========================================================================
    
    def store_entity(self, entity_data: Dict[str, Any]) -> str:
        """
        Store a code entity in the database.
        
        Args:
            entity_data: Dictionary with entity information
            
        Returns:
            Entity ID as string
        """
        required_fields = ['type', 'name', 'file_path']
        for field in required_fields:
            if field not in entity_data:
                raise ValueError(f"Required field '{field}' missing from entity data")
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Insert entity
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
                        embedding_data = embedding.tolist()  # Convert to list for DuckDB FLOAT[768]
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
                
                self.logger.debug(f"Stored entity: {entity_data['name']} (ID: {entity_id})")
                return entity_id
                
        except Exception as e:
            self.logger.error(f"Failed to store entity: {e}")
            raise
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by ID."""
        try:
            with self.connection_manager.get_connection() as conn:
                query = """
                SELECT id, type, name, file_path, line_start, line_end,
                       ast_hash, embedding, metadata, created_at, updated_at
                FROM entities WHERE id = ?
                """
                
                result = conn.execute(query, [entity_id]).fetchone()
                if not result:
                    return None
                
                # Convert result to dictionary
                entity = {
                    'id': str(result[0]),
                    'type': result[1],
                    'name': result[2],
                    'file_path': result[3],
                    'line_start': result[4],
                    'line_end': result[5],
                    'ast_hash': result[6],
                    'embedding': result[7],  # Raw bytes
                    'metadata': json.loads(result[8]) if result[8] else {},
                    'created_at': result[9],
                    'updated_at': result[10]
                }
                
                return entity
                
        except Exception as e:
            self.logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    def update_entity_embedding(self, entity_id: str, embedding: np.ndarray, 
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an entity's embedding and metadata."""
        try:
            with self.connection_manager.get_connection() as conn:
                embedding_data = embedding.tolist() if embedding is not None else None  # Convert to list
                metadata_json = json.dumps(metadata) if metadata is not None else None
                
                query = """
                UPDATE entities 
                SET embedding = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """
                
                result = conn.execute(query, [embedding_data, metadata_json, entity_id])
                success = result.rowcount > 0
                
                if success:
                    self.logger.debug(f"Updated embedding for entity {entity_id}")
                
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to update entity embedding {entity_id}: {e}")
            return False
    
    def search_entities(self, query: Optional[str] = None,
                       entity_types: Optional[List[str]] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name or type."""
        try:
            with self.connection_manager.get_connection() as conn:
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
                
                return entities
                
        except Exception as e:
            self.logger.error(f"Entity search failed: {e}")
            return []
    
    # =========================================================================
    # RELATIONSHIP OPERATIONS  
    # =========================================================================
    
    def store_relationship(self, source_id: str, target_id: str, 
                          relationship_type: str, confidence: float = 1.0,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a relationship between entities."""
        try:
            with self.connection_manager.get_connection() as conn:
                query = """
                INSERT INTO relationships (source_id, target_id, relationship_type, confidence, metadata)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """
                
                metadata_json = json.dumps(metadata) if metadata else None
                params = [source_id, target_id, relationship_type, confidence, metadata_json]
                
                result = conn.execute(query, params).fetchone()
                relationship_id = str(result[0])
                
                self.logger.debug(f"Stored relationship: {source_id} -> {target_id} ({relationship_type})")
                return relationship_id
                
        except Exception as e:
            self.logger.error(f"Failed to store relationship: {e}")
            raise
    
    def get_entity_relationships(self, entity_id: str, 
                               direction: str = 'both') -> List[Dict[str, Any]]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: Entity ID
            direction: 'outgoing', 'incoming', or 'both'
        """
        try:
            with self.connection_manager.get_connection() as conn:
                if direction == 'outgoing':
                    condition = "r.source_id = ?"
                elif direction == 'incoming':
                    condition = "r.target_id = ?"
                else:  # both
                    condition = "(r.source_id = ? OR r.target_id = ?)"
                    entity_id = [entity_id, entity_id]  # Duplicate for both conditions
                
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
                
                return relationships
                
        except Exception as e:
            self.logger.error(f"Failed to get relationships for entity {entity_id}: {e}")
            return []
    
    # =========================================================================
    # AGENT MEMORY OPERATIONS
    # =========================================================================
    
    def store_agent_memory(self, agent_type: str, context: str,
                          issue_number: Optional[int] = None,
                          decision: Optional[str] = None,
                          outcome: Optional[str] = None,
                          embedding: Optional[np.ndarray] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store agent memory/conversation data."""
        try:
            with self.connection_manager.get_connection() as conn:
                query = """
                INSERT INTO agent_memory (
                    agent_type, issue_number, context, decision, outcome, 
                    embedding, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """
                
                embedding_data = embedding.tolist() if embedding is not None else None
                metadata_json = json.dumps(metadata) if metadata else None
                
                params = [
                    agent_type, issue_number, context, decision, outcome,
                    embedding_data, metadata_json
                ]
                
                result = conn.execute(query, params).fetchone()
                memory_id = str(result[0])
                
                self.logger.debug(f"Stored agent memory for {agent_type} (ID: {memory_id})")
                return memory_id
                
        except Exception as e:
            self.logger.error(f"Failed to store agent memory: {e}")
            raise
    
    def get_agent_memories(self, agent_type: Optional[str] = None,
                          issue_number: Optional[int] = None,
                          outcome: Optional[str] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """Get agent memories with filtering."""
        try:
            with self.connection_manager.get_connection() as conn:
                conditions = []
                params = []
                
                if agent_type:
                    conditions.append("agent_type = ?")
                    params.append(agent_type)
                
                if issue_number:
                    conditions.append("issue_number = ?")
                    params.append(issue_number)
                
                if outcome:
                    conditions.append("outcome = ?")
                    params.append(outcome)
                
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                
                query = f"""
                SELECT id, agent_type, issue_number, context, decision, 
                       outcome, metadata, created_at
                FROM agent_memory
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
                """
                
                params.append(limit)
                results = conn.execute(query, params).fetchall()
                
                memories = []
                for row in results:
                    memory = {
                        'id': str(row[0]),
                        'agent_type': row[1],
                        'issue_number': row[2],
                        'context': row[3],
                        'decision': row[4],
                        'outcome': row[5],
                        'metadata': json.loads(row[6]) if row[6] else {},
                        'created_at': row[7]
                    }
                    memories.append(memory)
                
                return memories
                
        except Exception as e:
            self.logger.error(f"Failed to get agent memories: {e}")
            return []
    
    # =========================================================================
    # VECTOR SEARCH OPERATIONS
    # =========================================================================
    
    def similarity_search(self, query_embedding: np.ndarray,
                         entity_types: Optional[List[str]] = None,
                         limit: int = 10,
                         threshold: float = 0.7) -> List[VectorSearchResult]:
        """Perform vector similarity search."""
        search_query = SearchQuery(
            embedding=query_embedding,
            limit=limit,
            threshold=threshold,
            entity_types=entity_types
        )
        return self.vector_search.search_similar_entities(search_query)
    
    def hybrid_search(self, text_query: str, 
                     embedding_query: Optional[np.ndarray] = None,
                     entity_types: Optional[List[str]] = None,
                     limit: int = 10) -> List[VectorSearchResult]:
        """Perform hybrid text + vector search."""
        return self.vector_search.hybrid_search(
            text_query=text_query,
            embedding_query=embedding_query,
            entity_types=entity_types,
            limit=limit
        )
    
    def search_by_name(self, name_pattern: str,
                      entity_types: Optional[List[str]] = None,
                      limit: int = 10) -> List[VectorSearchResult]:
        """Search entities by name pattern."""
        return self.vector_search.search_by_entity_name(
            name_pattern=name_pattern,
            entity_types=entity_types,
            limit=limit
        )
    
    # =========================================================================
    # DATABASE MAINTENANCE
    # =========================================================================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            with self.connection_manager.get_connection() as conn:
                # Basic table counts
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
                
                # Agent memory statistics
                memory_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT agent_type) as agent_types,
                        COUNT(embedding) as memories_with_embeddings
                    FROM agent_memory
                """).fetchone()
                
                stats['agent_memory'] = {
                    'total': memory_stats[0],
                    'agent_types': memory_stats[1],
                    'with_embeddings': memory_stats[2]
                }
                
                # Connection pool statistics
                stats['connection_pool'] = self.connection_manager.get_pool_stats()
                
                # Vector search statistics
                stats['vector_search'] = self.vector_search.get_search_statistics()
                
                # VSS setup status
                stats['vss_status'] = self.vector_search.verify_vss_setup()
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get database statistics: {e}")
            return {'error': str(e)}
    
    def run_maintenance(self) -> Dict[str, Any]:
        """Run database maintenance tasks."""
        results = {}
        
        try:
            # Clean up idle connections
            self.connection_manager.cleanup_idle_connections()
            results['connection_cleanup'] = 'completed'
            
            # Analyze tables for query optimization
            with self.connection_manager.get_connection() as conn:
                conn.execute("ANALYZE")
            results['analyze_tables'] = 'completed'
            
            # Checkpoint WAL if enabled
            if self.config.auto_checkpoint:
                with self.connection_manager.get_connection() as conn:
                    conn.execute("CHECKPOINT")
                results['checkpoint'] = 'completed'
            
            self.logger.info("Database maintenance completed successfully")
            
        except Exception as e:
            self.logger.error(f"Database maintenance failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def verify_setup(self) -> Dict[str, Any]:
        """Verify that the database is properly set up and configured."""
        verification = {
            'database_accessible': False,
            'schema_present': False,
            'vss_setup': False,
            'connection_pool_working': False,
            'memory_limit_applied': False,
            'performance_acceptable': False,
            'errors': []
        }
        
        try:
            # Test basic database access
            with self.connection_manager.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
                verification['database_accessible'] = True
            
            # Check schema tables exist
            with self.connection_manager.get_connection() as conn:
                tables = conn.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_name IN ('entities', 'relationships', 'agent_memory')
                """).fetchall()
                
                if len(tables) >= 3:
                    verification['schema_present'] = True
                else:
                    verification['errors'].append(f"Missing schema tables. Found: {[t[0] for t in tables]}")
            
            # Check VSS setup
            vss_status = self.vector_search.verify_vss_setup()
            verification['vss_setup'] = (
                vss_status['vss_extension_loaded'] and 
                vss_status['vss_functions_available']
            )
            if not verification['vss_setup']:
                verification['errors'].extend(vss_status['error_messages'])
            
            # Test connection pool
            pool_stats = self.connection_manager.get_pool_stats()
            verification['connection_pool_working'] = pool_stats['max_connections'] > 0
            
            # Check memory configuration
            with self.connection_manager.get_connection() as conn:
                result = conn.execute("SELECT current_setting('memory_limit')").fetchone()
                if result and '500MB' in str(result[0]):
                    verification['memory_limit_applied'] = True
                else:
                    verification['errors'].append(f"Memory limit not set to 500MB. Current: {result}")
            
            # Basic performance test
            import time
            start_time = time.time()
            with self.connection_manager.get_connection() as conn:
                conn.execute("SELECT COUNT(*) FROM entities").fetchone()
            query_time = time.time() - start_time
            
            verification['performance_acceptable'] = query_time < 1.0  # Should be under 1 second
            if not verification['performance_acceptable']:
                verification['errors'].append(f"Query performance poor: {query_time:.2f}s")
            
        except Exception as e:
            verification['errors'].append(f"Verification failed: {e}")
        
        return verification
    
    def close(self):
        """Close the database and cleanup resources."""
        self.connection_manager.shutdown()
        self.logger.info("RIF Database closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()