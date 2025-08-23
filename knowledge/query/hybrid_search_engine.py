"""
Hybrid Search Engine - Issue #33
Combines vector similarity and graph traversal searches
"""

import asyncio
import time
import logging
import sqlite3
import duckdb
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .query_parser import StructuredQuery
from .strategy_planner import ExecutionPlan, SearchCostEstimate
from ..embeddings.embedding_storage import EmbeddingStorage
from ..extraction.storage_integration import EntityStorage


@dataclass
class SearchResult:
    """Individual search result from any search method"""
    entity_id: str
    entity_name: str
    entity_type: str
    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    # Scoring information
    relevance_score: float = 0.0
    confidence_score: float = 1.0
    source_strategy: str = "unknown"  # vector, graph, direct
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None


@dataclass
class HybridSearchResults:
    """Combined results from hybrid search"""
    results: List[SearchResult]
    total_found: int
    execution_time_ms: int
    
    # Strategy breakdown
    vector_results: int = 0
    graph_results: int = 0
    direct_results: int = 0
    
    # Performance metrics
    vector_search_time_ms: int = 0
    graph_search_time_ms: int = 0
    fusion_time_ms: int = 0
    
    # Quality metrics
    average_relevance: float = 0.0
    result_diversity: float = 0.0


class VectorSearchEngine:
    """Vector similarity search component"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embedding_storage = EmbeddingStorage(db_path)
        self.logger = logging.getLogger(__name__)
    
    def search(self, query_text: str, limit: int = 50, 
              threshold: float = 0.7, entity_types: List[str] = None) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query_text: Text to search for semantically
            limit: Maximum results to return
            threshold: Similarity threshold (0-1)
            entity_types: Filter by entity types
            
        Returns:
            List of search results sorted by relevance
        """
        self.logger.debug(f"Vector search for: {query_text}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_storage.embedding_generator.generate_embedding(query_text)
            
            if query_embedding is None:
                return []
            
            # Search for similar embeddings
            similar_entities = self.embedding_storage.find_similar_entities(
                query_embedding.embedding,
                limit=limit,
                threshold=threshold
            )
            
            results = []
            for entity_info in similar_entities:
                # Filter by entity type if specified
                if entity_types and entity_info.get('type') not in entity_types:
                    continue
                
                result = SearchResult(
                    entity_id=entity_info['entity_id'],
                    entity_name=entity_info['name'],
                    entity_type=entity_info['type'],
                    file_path=entity_info['file_path'],
                    line_start=entity_info.get('line_start'),
                    line_end=entity_info.get('line_end'),
                    relevance_score=entity_info['similarity_score'],
                    confidence_score=0.9,  # High confidence for vector search
                    source_strategy="vector",
                    content=entity_info.get('content')
                )
                results.append(result)
            
            self.logger.debug(f"Vector search found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return []


class GraphSearchEngine:
    """Graph traversal search component"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.entity_storage = EntityStorage(db_path)
        self.logger = logging.getLogger(__name__)
    
    def search(self, start_entities: List[Dict[str, Any]], 
              relationship_types: List[str],
              direction: str = "both",
              max_depth: int = 3,
              min_confidence: float = 0.5) -> List[SearchResult]:
        """
        Perform graph traversal search.
        
        Args:
            start_entities: Starting points for traversal
            relationship_types: Types of relationships to follow
            direction: 'outgoing', 'incoming', or 'both'
            max_depth: Maximum traversal depth
            min_confidence: Minimum relationship confidence
            
        Returns:
            List of search results with relationship context
        """
        self.logger.debug(f"Graph search from {len(start_entities)} entities")
        
        try:
            # Find starting entity IDs
            start_entity_ids = []
            for entity_spec in start_entities:
                entity_id = self._find_entity_id(entity_spec)
                if entity_id:
                    start_entity_ids.append(entity_id)
            
            if not start_entity_ids:
                return []
            
            # Perform graph traversal
            related_entities = self._traverse_graph(
                start_entity_ids, relationship_types, direction, 
                max_depth, min_confidence
            )
            
            # Convert to search results
            results = []
            for entity_info in related_entities:
                result = SearchResult(
                    entity_id=entity_info['id'],
                    entity_name=entity_info['name'],
                    entity_type=entity_info['type'],
                    file_path=entity_info['file_path'],
                    line_start=entity_info.get('line_start'),
                    line_end=entity_info.get('line_end'),
                    relevance_score=entity_info.get('relationship_strength', 0.8),
                    confidence_score=entity_info.get('confidence', 0.8),
                    source_strategy="graph",
                    metadata={
                        'depth': entity_info.get('depth', 0),
                        'relationship_path': entity_info.get('path', []),
                        'relationship_type': entity_info.get('relationship_type')
                    }
                )
                results.append(result)
            
            self.logger.debug(f"Graph search found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Graph search error: {e}")
            return []
    
    def _find_entity_id(self, entity_spec: Dict[str, Any]) -> Optional[str]:
        """Find entity ID from specification"""
        try:
            conn = duckdb.connect(self.db_path)
            
            # Build query conditions
            conditions = []
            params = []
            
            if entity_spec.get('name'):
                conditions.append("name = ?")
                params.append(entity_spec['name'])
            
            if entity_spec.get('type'):
                conditions.append("type = ?")
                params.append(entity_spec['type'])
                
            if entity_spec.get('file_path'):
                conditions.append("file_path = ?")
                params.append(entity_spec['file_path'])
            
            if not conditions:
                return None
            
            query = f"SELECT id FROM entities WHERE {' AND '.join(conditions)} LIMIT 1"
            
            result = conn.execute(query, params).fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"Error finding entity ID: {e}")
            return None
    
    def _traverse_graph(self, start_ids: List[str], relationship_types: List[str],
                       direction: str, max_depth: int, min_confidence: float) -> List[Dict[str, Any]]:
        """Perform recursive graph traversal"""
        try:
            conn = duckdb.connect(self.db_path)
            
            # Build CTE query for recursive traversal
            relationship_filter = "AND relationship_type IN ({})".format(
                ','.join(f"'{rt}'" for rt in relationship_types)
            ) if relationship_types else ""
            
            confidence_filter = f"AND confidence >= {min_confidence}"
            
            # Direction determines which relationships to follow
            if direction == "outgoing":
                relationship_join = "r.source_id = current.entity_id AND r.target_id = e.id"
                start_join = "r.source_id = e.id"
            elif direction == "incoming":
                relationship_join = "r.target_id = current.entity_id AND r.source_id = e.id"
                start_join = "r.target_id = e.id"
            else:  # both
                relationship_join = """
                    ((r.source_id = current.entity_id AND r.target_id = e.id) OR 
                     (r.target_id = current.entity_id AND r.source_id = e.id))
                """
                start_join = "(r.source_id = e.id OR r.target_id = e.id)"
            
            start_ids_str = ','.join(f"'{id}'" for id in start_ids)
            
            query = f"""
            WITH RECURSIVE graph_traversal AS (
                -- Base case: starting entities
                SELECT 
                    e.id as entity_id,
                    e.name,
                    e.type,
                    e.file_path,
                    e.line_start,
                    e.line_end,
                    0 as depth,
                    ARRAY[e.id] as path,
                    1.0 as relationship_strength,
                    1.0 as confidence,
                    '' as relationship_type
                FROM entities e
                WHERE e.id IN ({start_ids_str})
                
                UNION ALL
                
                -- Recursive case: follow relationships
                SELECT 
                    e.id as entity_id,
                    e.name,
                    e.type,
                    e.file_path,
                    e.line_start,
                    e.line_end,
                    current.depth + 1 as depth,
                    array_append(current.path, e.id) as path,
                    current.relationship_strength * r.confidence as relationship_strength,
                    r.confidence,
                    r.relationship_type
                FROM graph_traversal current
                JOIN relationships r ON {relationship_join}
                JOIN entities e ON e.id = CASE 
                    WHEN '{direction}' = 'outgoing' THEN r.target_id
                    WHEN '{direction}' = 'incoming' THEN r.source_id
                    ELSE CASE WHEN r.source_id = current.entity_id THEN r.target_id ELSE r.source_id END
                END
                WHERE current.depth < {max_depth}
                  AND e.id != ALL(current.path)  -- Prevent cycles
                  {relationship_filter}
                  {confidence_filter}
            )
            SELECT * FROM graph_traversal 
            WHERE depth > 0  -- Exclude starting entities
            ORDER BY relationship_strength DESC, depth ASC
            """
            
            results = conn.execute(query).fetchall()
            conn.close()
            
            # Convert results to dictionaries
            entities = []
            columns = ['id', 'name', 'type', 'file_path', 'line_start', 'line_end', 
                      'depth', 'path', 'relationship_strength', 'confidence', 'relationship_type']
            
            for row in results:
                entity_dict = dict(zip(columns, row))
                entities.append(entity_dict)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Graph traversal error: {e}")
            return []


class DirectSearchEngine:
    """Direct entity lookup component"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def search(self, entities: List[Dict[str, Any]], 
              exact_match: bool = True) -> List[SearchResult]:
        """
        Perform direct entity lookups.
        
        Args:
            entities: Entities to search for directly
            exact_match: Whether to require exact name matches
            
        Returns:
            List of directly matched entities
        """
        self.logger.debug(f"Direct search for {len(entities)} entities")
        
        try:
            conn = duckdb.connect(self.db_path)
            results = []
            
            for entity_spec in entities:
                # Build query conditions
                conditions = []
                params = []
                
                name = entity_spec.get('name', '')
                if name:
                    if exact_match:
                        conditions.append("name = ?")
                        params.append(name)
                    else:
                        conditions.append("name ILIKE ?")
                        params.append(f"%{name}%")
                
                if entity_spec.get('type'):
                    conditions.append("type = ?")
                    params.append(entity_spec['type'])
                    
                if entity_spec.get('file_path'):
                    conditions.append("file_path LIKE ?")
                    params.append(f"%{entity_spec['file_path']}%")
                
                if not conditions:
                    continue
                
                query = f"""
                SELECT id, name, type, file_path, line_start, line_end, metadata
                FROM entities 
                WHERE {' AND '.join(conditions)}
                LIMIT 10
                """
                
                rows = conn.execute(query, params).fetchall()
                
                for row in rows:
                    result = SearchResult(
                        entity_id=row[0],
                        entity_name=row[1],
                        entity_type=row[2],
                        file_path=row[3],
                        line_start=row[4],
                        line_end=row[5],
                        relevance_score=1.0,  # Perfect match for direct search
                        confidence_score=1.0,
                        source_strategy="direct",
                        metadata=json.loads(row[6]) if row[6] else {}
                    )
                    results.append(result)
            
            conn.close()
            
            self.logger.debug(f"Direct search found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Direct search error: {e}")
            return []


class ResultFuser:
    """Combines and ranks results from multiple search strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fuse_results(self, vector_results: List[SearchResult],
                    graph_results: List[SearchResult],
                    direct_results: List[SearchResult],
                    fusion_strategy: str = "weighted_merge",
                    limit: int = 20) -> List[SearchResult]:
        """
        Fuse results from multiple search strategies.
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search  
            direct_results: Results from direct search
            fusion_strategy: How to combine results
            limit: Maximum results to return
            
        Returns:
            Fused and ranked results
        """
        self.logger.debug(f"Fusing {len(vector_results)} vector + {len(graph_results)} graph + {len(direct_results)} direct results")
        
        if fusion_strategy == "single_source":
            # Return results from single source (first non-empty)
            for results in [direct_results, vector_results, graph_results]:
                if results:
                    return results[:limit]
            return []
        
        elif fusion_strategy == "rank_fusion":
            return self._rank_fusion(vector_results, graph_results, direct_results, limit)
        
        else:  # weighted_merge (default)
            return self._weighted_merge(vector_results, graph_results, direct_results, limit)
    
    def _weighted_merge(self, vector_results: List[SearchResult],
                       graph_results: List[SearchResult],
                       direct_results: List[SearchResult],
                       limit: int) -> List[SearchResult]:
        """Merge results with weighted scores"""
        # Combine all results with entity deduplication
        entity_results = {}  # entity_id -> best result
        
        # Process direct results (highest weight)
        for result in direct_results:
            result.relevance_score = result.relevance_score * 1.0  # Weight: 1.0
            entity_results[result.entity_id] = result
        
        # Process vector results (semantic weight)
        for result in vector_results:
            weighted_score = result.relevance_score * 0.8  # Weight: 0.8
            
            if result.entity_id in entity_results:
                # Combine scores if entity already exists
                existing = entity_results[result.entity_id]
                existing.relevance_score = max(existing.relevance_score, weighted_score)
                # Add vector metadata
                existing.metadata['vector_score'] = result.relevance_score
            else:
                result.relevance_score = weighted_score
                entity_results[result.entity_id] = result
        
        # Process graph results (structural weight)
        for result in graph_results:
            weighted_score = result.relevance_score * 0.7  # Weight: 0.7
            
            if result.entity_id in entity_results:
                # Combine scores if entity already exists
                existing = entity_results[result.entity_id]
                existing.relevance_score = max(existing.relevance_score, weighted_score)
                # Add graph metadata
                existing.metadata['graph_score'] = result.relevance_score
                existing.metadata.update(result.metadata)
            else:
                result.relevance_score = weighted_score
                entity_results[result.entity_id] = result
        
        # Sort by combined relevance score
        final_results = sorted(entity_results.values(), 
                             key=lambda x: x.relevance_score, 
                             reverse=True)
        
        return final_results[:limit]
    
    def _rank_fusion(self, vector_results: List[SearchResult],
                    graph_results: List[SearchResult],
                    direct_results: List[SearchResult],
                    limit: int) -> List[SearchResult]:
        """Fuse results using reciprocal rank fusion"""
        entity_scores = {}  # entity_id -> (result, combined_score)
        
        # Process each result list
        for results, weight in [(direct_results, 1.0), (vector_results, 0.8), (graph_results, 0.7)]:
            for rank, result in enumerate(results):
                # Reciprocal rank fusion: 1 / (rank + k), k=60 is common
                rrf_score = weight / (rank + 60)
                
                if result.entity_id in entity_scores:
                    # Accumulate scores
                    existing_result, existing_score = entity_scores[result.entity_id]
                    entity_scores[result.entity_id] = (existing_result, existing_score + rrf_score)
                else:
                    entity_scores[result.entity_id] = (result, rrf_score)
        
        # Sort by combined RRF score and update relevance scores
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1][1], reverse=True)
        
        final_results = []
        for entity_id, (result, rrf_score) in sorted_entities[:limit]:
            result.relevance_score = rrf_score
            final_results.append(result)
        
        return final_results


class HybridSearchEngine:
    """
    Main hybrid search engine coordinating all search strategies.
    """
    
    def __init__(self, db_path: str = "knowledge/chromadb/entities.duckdb"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize search engines
        self.vector_engine = VectorSearchEngine(db_path)
        self.graph_engine = GraphSearchEngine(db_path)
        self.direct_engine = DirectSearchEngine(db_path)
        self.result_fuser = ResultFuser()
        
        # Performance tracking
        self.search_metrics = {
            'total_searches': 0,
            'average_latency_ms': 0,
            'cache_hits': 0
        }
    
    def search(self, query: StructuredQuery, execution_plan: ExecutionPlan) -> HybridSearchResults:
        """
        Execute hybrid search according to execution plan.
        
        Args:
            query: Structured query from parser
            execution_plan: Detailed execution plan from strategy planner
            
        Returns:
            Combined search results with performance metrics
        """
        start_time = time.time()
        self.logger.info(f"Executing hybrid search: {query.original_query}")
        
        # Initialize result containers
        vector_results = []
        graph_results = []
        direct_results = []
        
        # Performance tracking
        vector_time_ms = 0
        graph_time_ms = 0
        fusion_time_ms = 0
        
        try:
            if execution_plan.parallel_execution:
                # Execute searches in parallel
                results_dict = self._execute_parallel_searches(query, execution_plan)
                vector_results = results_dict.get('vector', [])
                graph_results = results_dict.get('graph', [])
                direct_results = results_dict.get('direct', [])
                
                # Parallel timing is approximated
                vector_time_ms = results_dict.get('vector_time_ms', 0)
                graph_time_ms = results_dict.get('graph_time_ms', 0)
                
            else:
                # Execute searches sequentially
                if execution_plan.direct_enabled:
                    direct_start = time.time()
                    direct_results = self.direct_engine.search(
                        execution_plan.direct_entities,
                        exact_match=True
                    )
                    # Sequential timing is additive but we want max for total
                    
                if execution_plan.vector_enabled:
                    vector_start = time.time()
                    vector_results = self.vector_engine.search(
                        execution_plan.vector_query,
                        limit=execution_plan.vector_limit,
                        threshold=execution_plan.vector_threshold
                    )
                    vector_time_ms = int((time.time() - vector_start) * 1000)
                
                if execution_plan.graph_enabled:
                    graph_start = time.time()
                    graph_results = self.graph_engine.search(
                        execution_plan.graph_start_entities,
                        execution_plan.graph_relationship_types,
                        execution_plan.graph_direction,
                        execution_plan.graph_max_depth
                    )
                    graph_time_ms = int((time.time() - graph_start) * 1000)
            
            # Fuse results
            fusion_start = time.time()
            fused_results = self.result_fuser.fuse_results(
                vector_results, graph_results, direct_results,
                execution_plan.fusion_strategy, execution_plan.result_limit
            )
            fusion_time_ms = int((time.time() - fusion_start) * 1000)
            
            # Calculate metrics
            total_time_ms = int((time.time() - start_time) * 1000)
            
            # Build response
            hybrid_results = HybridSearchResults(
                results=fused_results,
                total_found=len(fused_results),
                execution_time_ms=total_time_ms,
                vector_results=len(vector_results),
                graph_results=len(graph_results),
                direct_results=len(direct_results),
                vector_search_time_ms=vector_time_ms,
                graph_search_time_ms=graph_time_ms,
                fusion_time_ms=fusion_time_ms,
                average_relevance=sum(r.relevance_score for r in fused_results) / len(fused_results) if fused_results else 0.0,
                result_diversity=self._calculate_diversity(fused_results)
            )
            
            # Update metrics
            self._update_metrics(hybrid_results)
            
            self.logger.info(f"Search completed in {total_time_ms}ms, found {len(fused_results)} results")
            return hybrid_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {e}")
            # Return empty results on error
            return HybridSearchResults(
                results=[],
                total_found=0,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _execute_parallel_searches(self, query: StructuredQuery, 
                                 execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute multiple searches in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=execution_plan.max_workers) as executor:
            # Submit search tasks
            future_to_strategy = {}
            
            if execution_plan.direct_enabled:
                future = executor.submit(
                    self.direct_engine.search,
                    execution_plan.direct_entities,
                    True  # exact_match
                )
                future_to_strategy[future] = 'direct'
            
            if execution_plan.vector_enabled:
                future = executor.submit(
                    self.vector_engine.search,
                    execution_plan.vector_query,
                    execution_plan.vector_limit,
                    execution_plan.vector_threshold
                )
                future_to_strategy[future] = 'vector'
            
            if execution_plan.graph_enabled:
                future = executor.submit(
                    self.graph_engine.search,
                    execution_plan.graph_start_entities,
                    execution_plan.graph_relationship_types,
                    execution_plan.graph_direction,
                    execution_plan.graph_max_depth
                )
                future_to_strategy[future] = 'graph'
            
            # Collect results as they complete
            for future in as_completed(future_to_strategy, timeout=execution_plan.timeout_per_search_ms/1000):
                strategy = future_to_strategy[future]
                try:
                    search_results = future.result()
                    results[strategy] = search_results
                    self.logger.debug(f"{strategy} search completed with {len(search_results)} results")
                except Exception as e:
                    self.logger.error(f"{strategy} search failed: {e}")
                    results[strategy] = []
        
        return results
    
    def _calculate_diversity(self, results: List[SearchResult]) -> float:
        """Calculate result diversity (different files, types, etc.)"""
        if not results:
            return 0.0
        
        unique_files = set(r.file_path for r in results)
        unique_types = set(r.entity_type for r in results)
        
        # Diversity score based on variety
        file_diversity = len(unique_files) / len(results)
        type_diversity = len(unique_types) / len(results)
        
        return (file_diversity + type_diversity) / 2.0
    
    def _update_metrics(self, results: HybridSearchResults):
        """Update performance metrics"""
        self.search_metrics['total_searches'] += 1
        
        # Update average latency using exponential moving average
        alpha = 0.1
        if self.search_metrics['average_latency_ms'] == 0:
            self.search_metrics['average_latency_ms'] = results.execution_time_ms
        else:
            old_avg = self.search_metrics['average_latency_ms']
            new_avg = alpha * results.execution_time_ms + (1 - alpha) * old_avg
            self.search_metrics['average_latency_ms'] = int(new_avg)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.search_metrics.copy()


# Convenience function
def hybrid_search(query_text: str, db_path: str = "knowledge/chromadb/entities.duckdb") -> HybridSearchResults:
    """Convenience function for end-to-end hybrid search"""
    from .query_parser import parse_query
    from .strategy_planner import plan_query_execution
    
    # Parse query
    structured_query = parse_query(query_text)
    
    # Plan execution
    execution_plan = plan_query_execution(structured_query)
    
    # Execute search
    search_engine = HybridSearchEngine(db_path)
    return search_engine.search(structured_query, execution_plan)


# Example usage
if __name__ == "__main__":
    # Test hybrid search
    results = hybrid_search("find authentication functions similar to login")
    
    print(f"Found {results.total_found} results in {results.execution_time_ms}ms")
    for result in results.results[:5]:
        print(f"- {result.entity_name} ({result.entity_type}) - {result.relevance_score:.2f}")