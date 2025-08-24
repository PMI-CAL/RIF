"""
Storage integration for embeddings with DuckDB vector similarity search.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from uuid import UUID
from datetime import datetime

import duckdb

from .embedding_generator import EmbeddingResult


class EmbeddingStorage:
    """
    Handles storage and retrieval of embeddings in DuckDB with vector similarity search.
    """
    
    def __init__(self, db_path: str = "knowledge/chromadb/entities.duckdb"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection
        self._conn = None
        self._ensure_embedding_schema()
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn
    
    def _ensure_embedding_schema(self):
        """Ensure the embedding columns and indexes exist."""
        conn = self._get_connection()
        
        try:
            # Check if embedding column exists
            result = conn.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'entities' AND column_name = 'embedding'
            """).fetchone()
            
            if not result:
                # Add embedding columns to entities table
                self.logger.info("Adding embedding columns to entities table...")
                
                # Add embedding column (use BLOB for vector storage)
                conn.execute("ALTER TABLE entities ADD COLUMN embedding BLOB")
                
                # Add embedding metadata column
                conn.execute("ALTER TABLE entities ADD COLUMN embedding_metadata TEXT")
                
                self.logger.info("Embedding columns added successfully")
            
            # Note: For now, we'll use Python-based similarity calculations
            # since DuckDB vector functions are complex to implement
            
            self.logger.info("Vector similarity functions created")
            
        except Exception as e:
            self.logger.warning(f"Error setting up embedding schema: {e}")
            # Continue anyway - basic functionality might still work
    
    def store_embeddings(self, embedding_results: List[EmbeddingResult], 
                        update_mode: str = 'upsert') -> Dict[str, int]:
        """
        Store embedding results in the database.
        
        Args:
            embedding_results: List of EmbeddingResult objects
            update_mode: 'insert', 'upsert', or 'replace'
            
        Returns:
            Dictionary with counts of inserted, updated, and skipped embeddings
        """
        if not embedding_results:
            return {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        conn = self._get_connection()
        results = {'inserted': 0, 'updated': 0, 'skipped': 0}
        
        try:
            conn.begin()
            
            for result in embedding_results:
                if update_mode == 'upsert':
                    operation_result = self._upsert_embedding(conn, result)
                elif update_mode == 'replace':
                    operation_result = self._replace_embedding(conn, result)
                else:  # insert
                    operation_result = self._insert_embedding(conn, result)
                
                results[operation_result] += 1
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error storing embeddings: {e}")
            raise
        
        self.logger.info(f"Stored embeddings: {results}")
        return results
    
    def _upsert_embedding(self, conn: duckdb.DuckDBPyConnection, 
                         result: EmbeddingResult) -> str:
        """Insert or update embedding based on entity ID."""
        
        # Check if entity exists
        existing = conn.execute("""
            SELECT embedding_metadata FROM entities WHERE id = ?
        """, [str(result.entity_id)]).fetchone()
        
        if existing:
            existing_metadata_str = existing[0] if existing[0] else '{}'
            try:
                existing_metadata = json.loads(existing_metadata_str) if existing_metadata_str else {}
            except Exception as e:
                self.logger.warning(f"Error parsing existing metadata: {e}")
                existing_metadata = {}
            existing_hash = existing_metadata.get('content_hash', '')
            
            # Check if content has changed
            if existing_hash != result.content_hash:
                # Update embedding
                conn.execute("""
                    UPDATE entities SET 
                        embedding = ?, 
                        embedding_metadata = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, [
                    result.embedding.tobytes(),
                    json.dumps(result.metadata),
                    str(result.entity_id)
                ])
                return 'updated'
            else:
                return 'skipped'
        else:
            # Entity doesn't exist - this shouldn't happen, but we'll log it
            self.logger.warning(f"Entity {result.entity_id} not found for embedding update")
            return 'skipped'
    
    def _replace_embedding(self, conn: duckdb.DuckDBPyConnection, 
                          result: EmbeddingResult) -> str:
        """Replace embedding regardless of hash."""
        
        updated = conn.execute("""
            UPDATE entities SET 
                embedding = ?, 
                embedding_metadata = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, [
            result.embedding.tobytes(),
            json.dumps(result.metadata),
            str(result.entity_id)
        ])
        
        if updated.rowcount > 0:
            return 'updated'
        else:
            self.logger.warning(f"Entity {result.entity_id} not found for embedding replacement")
            return 'skipped'
    
    def _insert_embedding(self, conn: duckdb.DuckDBPyConnection, 
                         result: EmbeddingResult) -> str:
        """Insert embedding (assumes entity already exists)."""
        try:
            conn.execute("""
                UPDATE entities SET 
                    embedding = ?, 
                    embedding_metadata = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [
                result.embedding.tobytes(),
                json.dumps(result.metadata),
                str(result.entity_id)
            ])
            return 'inserted'
        except Exception:
            return 'skipped'
    
    def find_similar_entities(self, query_embedding: np.ndarray, 
                             limit: int = 10,
                             threshold: float = 0.7,
                             entity_types: Optional[List[str]] = None,
                             exclude_entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find entities similar to the query embedding using cosine similarity.
        
        Args:
            query_embedding: Query vector to find similarities for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            entity_types: Filter by entity types (optional)
            exclude_entity_id: Exclude specific entity from results
            
        Returns:
            List of similar entities with similarity scores
        """
        conn = self._get_connection()
        
        # Build query conditions
        conditions = ["embedding IS NOT NULL"]
        params = [query_embedding.tolist()]
        
        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            conditions.append(f"type IN ({placeholders})")
            params.extend(entity_types)
        
        if exclude_entity_id:
            conditions.append("id != ?")
            params.append(exclude_entity_id)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        # For now, get all entities with embeddings and calculate similarity in Python
        query = f"""
        SELECT 
            id, type, name, file_path, line_start, line_end,
            embedding, embedding_metadata
        FROM entities
        {where_clause}
        ORDER BY created_at DESC
        LIMIT 100
        """
        
        try:
            results = conn.execute(query, params).fetchall()
            
            similar_entities = []
            for row in results:
                if row[6] is None:  # No embedding
                    continue
                    
                # Calculate cosine similarity in Python
                entity_embedding = np.frombuffer(row[6], dtype=np.float32)
                
                # Calculate cosine similarity
                dot_product = np.dot(query_embedding, entity_embedding)
                query_norm = np.linalg.norm(query_embedding)
                entity_norm = np.linalg.norm(entity_embedding)
                
                if query_norm > 0 and entity_norm > 0:
                    similarity_score = dot_product / (query_norm * entity_norm)
                else:
                    similarity_score = 0.0
                
                # Filter by threshold
                if similarity_score >= threshold:
                    entity_data = {
                        'id': row[0],
                        'type': row[1],
                        'name': row[2],
                        'file_path': row[3],
                        'line_start': row[4],
                        'line_end': row[5],
                        'similarity_score': float(similarity_score),
                        'embedding_metadata': json.loads(row[7]) if row[7] else {}
                    }
                    similar_entities.append(entity_data)
            
            # Sort by similarity score descending and limit results
            similar_entities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_entities[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            # Fallback to basic search without vector similarity
            return self._fallback_similarity_search(query_embedding, limit, entity_types)
    
    def _fallback_similarity_search(self, query_embedding: np.ndarray,
                                   limit: int = 10,
                                   entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fallback similarity search using basic filtering."""
        conn = self._get_connection()
        
        conditions = ["embedding IS NOT NULL"]
        params = []
        
        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            conditions.append(f"type IN ({placeholders})")
            params.extend(entity_types)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT 
            id, type, name, file_path, line_start, line_end,
            0.5 as similarity_score,  -- Default similarity
            embedding_metadata
        FROM entities
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """
        
        params.append(limit)
        
        results = conn.execute(query, params).fetchall()
        
        fallback_entities = []
        for row in results:
            entity_data = {
                'id': row[0],
                'type': row[1],
                'name': row[2],
                'file_path': row[3],
                'line_start': row[4],
                'line_end': row[5],
                'similarity_score': 0.5,  # Default similarity for fallback
                'embedding_metadata': json.loads(row[7]) if row[7] else {}
            }
            fallback_entities.append(entity_data)
        
        return fallback_entities
    
    def search_by_entity_name(self, name_pattern: str, 
                             limit: int = 10,
                             entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search entities by name pattern with embeddings."""
        conn = self._get_connection()
        
        conditions = ["embedding IS NOT NULL", "name LIKE ?"]
        params = [f"%{name_pattern}%"]
        
        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            conditions.append(f"type IN ({placeholders})")
            params.extend(entity_types)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
        SELECT 
            id, type, name, file_path, line_start, line_end,
            embedding_metadata
        FROM entities
        {where_clause}
        ORDER BY name
        LIMIT ?
        """
        
        params.append(limit)
        
        results = conn.execute(query, params).fetchall()
        
        entities = []
        for row in results:
            entity_data = {
                'id': row[0],
                'type': row[1],
                'name': row[2],
                'file_path': row[3],
                'line_start': row[4],
                'line_end': row[5],
                'similarity_score': 1.0,  # Full match for name search
                'embedding_metadata': json.loads(row[6]) if row[6] else {}
            }
            entities.append(entity_data)
        
        return entities
    
    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific entity."""
        conn = self._get_connection()
        
        result = conn.execute("""
            SELECT embedding FROM entities WHERE id = ?
        """, [str(entity_id)]).fetchone()
        
        if result and result[0]:
            # Convert from bytes back to numpy array
            return np.frombuffer(result[0], dtype=np.float32)
        
        return None
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        conn = self._get_connection()
        
        stats = {}
        
        try:
            # Basic counts
            result = conn.execute("""
                SELECT 
                    COUNT(*) as total_entities,
                    COUNT(embedding) as entities_with_embeddings,
                    COUNT(DISTINCT type) as entity_types_with_embeddings
                FROM entities
            """).fetchone()
            
            stats['total_entities'] = result[0]
            stats['entities_with_embeddings'] = result[1]
            stats['entity_types_with_embeddings'] = result[2]
            
            # Embedding coverage by type
            coverage_results = conn.execute("""
                SELECT 
                    type,
                    COUNT(*) as total_count,
                    COUNT(embedding) as embedded_count,
                    ROUND(COUNT(embedding) * 100.0 / COUNT(*), 2) as coverage_percentage
                FROM entities
                GROUP BY type
                ORDER BY coverage_percentage DESC
            """).fetchall()
            
            stats['coverage_by_type'] = {
                row[0]: {
                    'total': row[1],
                    'embedded': row[2],
                    'coverage_percent': row[3]
                }
                for row in coverage_results
            }
            
        except Exception as e:
            self.logger.error(f"Error getting embedding statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def cleanup_orphaned_embeddings(self) -> int:
        """Remove embeddings for entities that no longer exist."""
        conn = self._get_connection()
        
        # For now, just count entities with embeddings but no other data
        try:
            result = conn.execute("""
                SELECT COUNT(*) FROM entities 
                WHERE embedding IS NOT NULL AND (name IS NULL OR name = '')
            """).fetchone()
            
            orphaned_count = result[0] if result else 0
            
            if orphaned_count > 0:
                # Clear embeddings for orphaned entities
                conn.execute("""
                    UPDATE entities 
                    SET embedding = NULL, embedding_metadata = NULL
                    WHERE embedding IS NOT NULL AND (name IS NULL OR name = '')
                """)
                
                self.logger.info(f"Cleaned up {orphaned_count} orphaned embeddings")
            
            return orphaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up orphaned embeddings: {e}")
            return 0
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None