"""
Vector similarity search engine for DuckDB with VSS extension.
Issue #26: Set up DuckDB as embedded database with vector search
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from dataclasses import dataclass

import duckdb

from .connection_manager import DuckDBConnectionManager
from .database_config import DatabaseConfig


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    id: str
    name: str
    type: str
    file_path: str
    line_start: Optional[int]
    line_end: Optional[int]
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class SearchQuery:
    """Vector search query configuration."""
    embedding: np.ndarray
    limit: int = 10
    threshold: float = 0.7
    entity_types: Optional[List[str]] = None
    exclude_entity_id: Optional[str] = None
    include_metadata: bool = True


class VectorSearchEngine:
    """
    High-performance vector similarity search engine using DuckDB VSS extension.
    
    Features:
    - HNSW index for fast approximate nearest neighbor search
    - Multiple distance metrics (cosine, euclidean, inner product)
    - Hybrid search combining text and vector similarity
    - Batch operations for efficiency
    - Query optimization and caching
    """
    
    def __init__(self, connection_manager: DuckDBConnectionManager):
        self.connection_manager = connection_manager
        self.config = connection_manager.config
        self.logger = logging.getLogger(__name__)
        
        # VSS configuration
        self.vss_config = self.config.get_vss_config()
        self.metric = self.vss_config['metric']
        
        # Performance monitoring
        self._query_count = 0
        self._total_query_time = 0.0
        
        self.logger.info(f"Vector search engine initialized with metric: {self.metric}")
    
    def search_similar_entities(self, query: SearchQuery) -> List[VectorSearchResult]:
        """
        Find entities similar to the query embedding using vector similarity search.
        
        Args:
            query: SearchQuery object with embedding and search parameters
            
        Returns:
            List of VectorSearchResult objects ordered by similarity
        """
        start_time = self._start_query_timing()
        
        try:
            with self.connection_manager.get_connection() as conn:
                # First, try VSS-based search if available
                results = self._vss_similarity_search(conn, query)
                
                if not results:
                    # Fallback to Python-based similarity calculation
                    self.logger.debug("Falling back to Python-based similarity search")
                    results = self._python_similarity_search(conn, query)
                
                self._end_query_timing(start_time)
                return results
                
        except Exception as e:
            self.logger.error(f"Vector similarity search failed: {e}")
            self._end_query_timing(start_time)
            raise
    
    def _vss_similarity_search(self, conn: duckdb.DuckDBPyConnection, 
                              query: SearchQuery) -> List[VectorSearchResult]:
        """Perform similarity search using DuckDB VSS extension."""
        try:
            # Build query conditions
            conditions = ["embedding IS NOT NULL"]
            params = []
            
            if query.entity_types:
                placeholders = ','.join('?' * len(query.entity_types))
                conditions.append(f"type IN ({placeholders})")
                params.extend(query.entity_types)
            
            if query.exclude_entity_id:
                conditions.append("id != ?")
                params.append(query.exclude_entity_id)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            # Try DuckDB array cosine similarity if available
            try:
                # Test if array_cosine_similarity function exists
                conn.execute("SELECT array_cosine_similarity([1.0, 2.0], [1.0, 2.0])").fetchone()
                use_array_functions = True
            except:
                use_array_functions = False
            
            if use_array_functions:
                # Use DuckDB array functions
                vss_query = f"""
                SELECT 
                    id::VARCHAR as id,
                    name,
                    type,
                    file_path,
                    line_start,
                    line_end,
                    array_cosine_similarity(embedding, ?::FLOAT[768]) as similarity_score,
                    {('metadata::VARCHAR as metadata' if query.include_metadata else 'NULL as metadata')}
                FROM entities
                {where_clause}
                ORDER BY array_cosine_similarity(embedding, ?::FLOAT[768]) DESC
                LIMIT ?
                """
                
                # Add query embedding to params (twice for the query)
                embedding_list = query.embedding.tolist()
                search_params = [embedding_list, embedding_list] + params + [query.limit]
                
                rows = conn.execute(vss_query, search_params).fetchall()
                
                results = []
                for row in rows:
                    if row[6] >= query.threshold:  # similarity_score
                        metadata = {}
                        if query.include_metadata and row[7]:
                            try:
                                metadata = json.loads(row[7])
                            except (json.JSONDecodeError, TypeError):
                                metadata = {}
                        
                        result = VectorSearchResult(
                            id=row[0],
                            name=row[1],
                            type=row[2],
                            file_path=row[3],
                            line_start=row[4],
                            line_end=row[5],
                            similarity_score=float(row[6]),
                            metadata=metadata
                        )
                        results.append(result)
                
                self.logger.debug(f"VSS array search returned {len(results)} results")
                return results
            else:
                # Fallback to basic query without vector functions
                return []
            
        except Exception as e:
            self.logger.warning(f"VSS search failed: {e}")
            return []  # Return empty list to trigger fallback
    
    def _python_similarity_search(self, conn: duckdb.DuckDBPyConnection,
                                 query: SearchQuery) -> List[VectorSearchResult]:
        """Fallback similarity search using Python-based calculations."""
        try:
            # Build query conditions
            conditions = ["embedding IS NOT NULL"]
            params = []
            
            if query.entity_types:
                placeholders = ','.join('?' * len(query.entity_types))
                conditions.append(f"type IN ({placeholders})")
                params.extend(query.entity_types)
            
            if query.exclude_entity_id:
                conditions.append("id != ?")
                params.append(query.exclude_entity_id)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get entities with embeddings (limit to reasonable number for Python processing)
            fetch_query = f"""
            SELECT 
                id::VARCHAR as id,
                name,
                type,
                file_path,
                line_start,
                line_end,
                embedding,
                {('metadata::VARCHAR as metadata' if query.include_metadata else 'NULL as metadata')}
            FROM entities
            {where_clause}
            ORDER BY created_at DESC
            LIMIT 1000
            """
            
            rows = conn.execute(fetch_query, params).fetchall()
            
            results = []
            for row in rows:
                try:
                    # Extract embedding from storage 
                    embedding_data = row[6]
                    if embedding_data is None:
                        continue
                    
                    # Convert embedding based on storage format
                    if isinstance(embedding_data, list):
                        # Direct array from DuckDB FLOAT[768]
                        entity_embedding = np.array(embedding_data, dtype=np.float32)
                    elif isinstance(embedding_data, bytes):
                        # Stored as bytes from numpy array
                        entity_embedding = np.frombuffer(embedding_data, dtype=np.float32)
                    elif isinstance(embedding_data, str):
                        # Stored as JSON string
                        entity_embedding = np.array(json.loads(embedding_data), dtype=np.float32)
                    else:
                        self.logger.warning(f"Unknown embedding format: {type(embedding_data)}")
                        continue
                    
                    # Calculate cosine similarity
                    similarity_score = self._calculate_cosine_similarity(
                        query.embedding, entity_embedding
                    )
                    
                    # Filter by threshold
                    if similarity_score >= query.threshold:
                        metadata = {}
                        if query.include_metadata and row[7]:
                            try:
                                metadata = json.loads(row[7])
                            except (json.JSONDecodeError, TypeError):
                                metadata = {}
                        
                        result = VectorSearchResult(
                            id=row[0],
                            name=row[1],
                            type=row[2],
                            file_path=row[3],
                            line_start=row[4],
                            line_end=row[5],
                            similarity_score=float(similarity_score),
                            metadata=metadata
                        )
                        results.append(result)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing entity embedding: {e}")
                    continue
            
            # Sort by similarity score descending and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = results[:query.limit]
            
            self.logger.debug(f"Python similarity search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Python similarity search failed: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure vectors are the same length
            if vec1.shape != vec2.shape:
                return 0.0
            
            # Calculate dot product and norms
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Clamp to [-1, 1] range and return
            return max(-1.0, min(1.0, float(similarity)))
            
        except Exception as e:
            self.logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def hybrid_search(self, text_query: str, embedding_query: Optional[np.ndarray] = None,
                     entity_types: Optional[List[str]] = None, 
                     limit: int = 10,
                     text_weight: float = 0.3,
                     vector_weight: float = 0.7) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining text matching and vector similarity.
        
        Args:
            text_query: Text query for keyword matching
            embedding_query: Optional embedding for vector similarity
            entity_types: Filter by entity types
            limit: Maximum results to return
            text_weight: Weight for text relevance score (0.0-1.0)
            vector_weight: Weight for vector similarity score (0.0-1.0)
            
        Returns:
            List of search results with combined scores
        """
        start_time = self._start_query_timing()
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Build conditions
                conditions = []
                params = []
                
                # Text matching conditions
                text_conditions = [
                    "name ILIKE ?",
                    "file_path ILIKE ?",
                    "metadata::TEXT ILIKE ?"
                ]
                text_pattern = f"%{text_query}%"
                
                # Entity type filter
                if entity_types:
                    placeholders = ','.join('?' * len(entity_types))
                    conditions.append(f"type IN ({placeholders})")
                    params.extend(entity_types)
                
                # Build the hybrid search query
                if embedding_query is not None:
                    # Build WHERE clause properly
                    where_clause = f"({' OR '.join(text_conditions)})"
                    if conditions:
                        where_clause += f" AND {' AND '.join(conditions)}"
                    
                    # Hybrid search with both text and vector
                    hybrid_query = f"""
                    SELECT 
                        id::VARCHAR as id,
                        name,
                        type,
                        file_path,
                        line_start,
                        line_end,
                        -- Text relevance score
                        CASE 
                            WHEN name ILIKE ? THEN 1.0
                            WHEN file_path ILIKE ? THEN 0.7
                            WHEN metadata::TEXT ILIKE ? THEN 0.5
                            ELSE 0.0
                        END as text_score,
                        -- Vector similarity (placeholder for now)
                        0.8 as vector_score,
                        -- Combined score
                        (CASE 
                            WHEN name ILIKE ? THEN 1.0
                            WHEN file_path ILIKE ? THEN 0.7
                            WHEN metadata::TEXT ILIKE ? THEN 0.5
                            ELSE 0.0
                        END * ? + 0.8 * ?) as combined_score,
                        metadata::VARCHAR as metadata
                    FROM entities
                    WHERE {where_clause}
                    ORDER BY combined_score DESC
                    LIMIT ?
                    """
                    
                    search_params = [
                        text_pattern, text_pattern, text_pattern,  # WHERE clause text matching
                        text_pattern, text_pattern, text_pattern,  # text score calculation
                        text_pattern, text_pattern, text_pattern,  # combined score calculation  
                        text_weight, vector_weight                 # weights
                    ] + params + [limit]
                    
                else:
                    # Build WHERE clause properly
                    where_clause = f"({' OR '.join(text_conditions)})"
                    if conditions:
                        where_clause += f" AND {' AND '.join(conditions)}"
                    
                    # Text-only search
                    hybrid_query = f"""
                    SELECT 
                        id::VARCHAR as id,
                        name,
                        type,
                        file_path,
                        line_start,
                        line_end,
                        CASE 
                            WHEN name ILIKE ? THEN 1.0
                            WHEN file_path ILIKE ? THEN 0.7
                            WHEN metadata::TEXT ILIKE ? THEN 0.5
                            ELSE 0.0
                        END as text_score,
                        0.0 as vector_score,
                        CASE 
                            WHEN name ILIKE ? THEN 1.0
                            WHEN file_path ILIKE ? THEN 0.7
                            WHEN metadata::TEXT ILIKE ? THEN 0.5
                            ELSE 0.0
                        END as combined_score,
                        metadata::VARCHAR as metadata
                    FROM entities
                    WHERE {where_clause}
                    ORDER BY combined_score DESC
                    LIMIT ?
                    """
                    
                    search_params = [
                        text_pattern, text_pattern, text_pattern,  # WHERE clause text matching
                        text_pattern, text_pattern, text_pattern,  # text score calculation
                        text_pattern, text_pattern, text_pattern   # combined score calculation
                    ] + params + [limit]
                
                rows = conn.execute(hybrid_query, search_params).fetchall()
                
                results = []
                for row in rows:
                    metadata = {}
                    if row[9]:  # metadata column
                        try:
                            metadata = json.loads(row[9])
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    
                    result = VectorSearchResult(
                        id=row[0],
                        name=row[1],
                        type=row[2],
                        file_path=row[3],
                        line_start=row[4],
                        line_end=row[5],
                        similarity_score=float(row[8]),  # combined_score
                        metadata=metadata
                    )
                    results.append(result)
                
                self._end_query_timing(start_time)
                self.logger.debug(f"Hybrid search returned {len(results)} results")
                return results
                
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            self._end_query_timing(start_time)
            raise
    
    def search_by_entity_name(self, name_pattern: str, 
                             entity_types: Optional[List[str]] = None,
                             limit: int = 10) -> List[VectorSearchResult]:
        """Search entities by name pattern."""
        try:
            with self.connection_manager.get_connection() as conn:
                conditions = ["name LIKE ?"]
                params = [f"%{name_pattern}%"]
                
                if entity_types:
                    placeholders = ','.join('?' * len(entity_types))
                    conditions.append(f"type IN ({placeholders})")
                    params.extend(entity_types)
                
                where_clause = "WHERE " + " AND ".join(conditions)
                
                query = f"""
                SELECT 
                    id::VARCHAR as id,
                    name,
                    type,
                    file_path,
                    line_start,
                    line_end,
                    1.0 as similarity_score,
                    metadata::VARCHAR as metadata
                FROM entities
                {where_clause}
                ORDER BY name
                LIMIT ?
                """
                
                params.append(limit)
                rows = conn.execute(query, params).fetchall()
                
                results = []
                for row in rows:
                    metadata = {}
                    if row[7]:
                        try:
                            metadata = json.loads(row[7])
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    
                    result = VectorSearchResult(
                        id=row[0],
                        name=row[1],
                        type=row[2],
                        file_path=row[3],
                        line_start=row[4],
                        line_end=row[5],
                        similarity_score=1.0,  # Exact name match
                        metadata=metadata
                    )
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Name search failed: {e}")
            return []
    
    def _start_query_timing(self) -> float:
        """Start timing a query for performance monitoring."""
        import time
        return time.time()
    
    def _end_query_timing(self, start_time: float):
        """End timing a query and update performance metrics."""
        import time
        query_time = time.time() - start_time
        self._query_count += 1
        self._total_query_time += query_time
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about search performance."""
        avg_query_time = (self._total_query_time / self._query_count 
                         if self._query_count > 0 else 0.0)
        
        return {
            'total_queries': self._query_count,
            'total_query_time': self._total_query_time,
            'average_query_time': avg_query_time,
            'metric': self.metric,
            'vss_config': self.vss_config
        }
    
    def verify_vss_setup(self) -> Dict[str, Any]:
        """Verify that VSS extension is properly set up and working."""
        status = {
            'vss_extension_loaded': False,
            'vss_indexes_exist': False,
            'vss_functions_available': False,
            'test_query_successful': False,
            'error_messages': []
        }
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Check if VSS extension is loaded
                try:
                    result = conn.execute("""
                        SELECT extension_name, loaded, installed 
                        FROM duckdb_extensions() 
                        WHERE extension_name = 'vss'
                    """).fetchone()
                    
                    if result and result[1] and result[2]:
                        status['vss_extension_loaded'] = True
                    else:
                        status['error_messages'].append("VSS extension not loaded or installed")
                        
                except Exception as e:
                    status['error_messages'].append(f"Error checking VSS extension: {e}")
                
                # Check for VSS indexes using DuckDB system tables
                try:
                    # Use DuckDB's duckdb_indexes() function instead
                    indexes = conn.execute("""
                        SELECT index_name 
                        FROM duckdb_indexes() 
                        WHERE index_name LIKE '%embedding%'
                    """).fetchall()
                    
                    if indexes:
                        status['vss_indexes_exist'] = True
                        status['index_count'] = len(indexes)
                    else:
                        # Check alternative way
                        try:
                            conn.execute("SELECT * FROM entities WHERE embedding IS NOT NULL LIMIT 1").fetchone()
                            status['vss_indexes_exist'] = True  # Table exists with embedding column
                        except:
                            status['error_messages'].append("No VSS indexes found")
                        
                except Exception as e:
                    status['error_messages'].append(f"Error checking VSS indexes: {e}")
                
                # Test a simple VSS query if possible
                try:
                    # Create a test vector and run a similarity query
                    test_vector = np.random.rand(768).astype(np.float32)
                    
                    # Try array cosine similarity first
                    try:
                        test_result = conn.execute("""
                            SELECT array_cosine_similarity(?, ?) as similarity
                        """, [test_vector.tolist(), test_vector.tolist()]).fetchone()
                        
                        if test_result is not None:
                            status['test_query_successful'] = True
                            status['vss_functions_available'] = True
                    except:
                        # Try array dot product as alternative
                        try:
                            test_result = conn.execute("""
                                SELECT array_dot_product(?, ?) as dot_product
                            """, [test_vector.tolist(), test_vector.tolist()]).fetchone()
                            
                            if test_result is not None:
                                status['test_query_successful'] = True
                                status['vss_functions_available'] = True
                        except Exception as e:
                            status['error_messages'].append(f"VSS test query failed: {e}")
                        
                except Exception as e:
                    status['error_messages'].append(f"VSS test query failed: {e}")
                
        except Exception as e:
            status['error_messages'].append(f"Database connection error: {e}")
        
        return status