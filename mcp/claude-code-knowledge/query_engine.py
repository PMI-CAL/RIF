"""
Query Engine for Claude Code Knowledge MCP Server.

Provides optimized query interface over the existing RIF DuckDB knowledge graph,
leveraging the extended schema from Phase 1 for Claude-specific knowledge retrieval.

Features:
- Direct integration with RIFDatabase
- Cached query execution for performance
- Semantic search using vector embeddings
- Claude-specific entity type filtering
- Relationship traversal for complex queries
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add RIF root to Python path
rif_root = Path(__file__).parents[2] 
sys.path.insert(0, str(rif_root))

try:
    from knowledge.database.database_interface import RIFDatabase
    from knowledge.database.vector_search import VectorSearchResult, SearchQuery
except ImportError as e:
    logging.error(f"Failed to import RIF database components: {e}")
    raise


@dataclass
class QueryResult:
    """Result of a knowledge graph query."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    execution_time_ms: float
    total_results: int
    query_type: str
    cache_hit: bool = False


@dataclass
class SemanticQuery:
    """Semantic query with vector similarity."""
    text_query: str
    entity_types: List[str]
    similarity_threshold: float = 0.7
    limit: int = 10


@dataclass
class RelationshipQuery:
    """Query for entity relationships."""
    entity_id: str
    relationship_types: Optional[List[str]] = None
    direction: str = "both"  # "incoming", "outgoing", "both"
    depth: int = 1


class ClaudeKnowledgeQueryEngine:
    """
    Optimized query engine for Claude Code knowledge.
    
    Provides high-level query methods tailored for the 5 MCP tools:
    - Compatibility checking
    - Pattern recommendations
    - Alternative finding
    - Architecture validation
    - Limitation queries
    """
    
    def __init__(self, rif_db: RIFDatabase, enable_caching: bool = True):
        self.rif_db = rif_db
        self.enable_caching = enable_caching
        self.logger = logging.getLogger(__name__)
        
        # Query cache for performance
        self.query_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Claude-specific entity type mappings
        self.claude_entity_types = {
            'capabilities': 'claude_capability',
            'limitations': 'claude_limitation', 
            'tools': 'claude_tool',
            'patterns': 'implementation_pattern',
            'anti_patterns': 'anti_pattern'
        }
        
        # Common relationship types for Claude knowledge
        self.claude_relationships = {
            'supports': 'Tool/capability supports pattern',
            'conflicts_with': 'Pattern conflicts with limitation',
            'alternative_to': 'Pattern is alternative to anti-pattern',
            'validates': 'Tool validates implementation',
            'requires': 'Pattern requires capability',
            'incompatible_with': 'Approach incompatible with limitation'
        }
        
        self.logger.info("Claude Knowledge Query Engine initialized")
    
    def _get_cache_key(self, query_type: str, **params) -> str:
        """Generate cache key for query."""
        # Create deterministic key from query type and parameters
        param_str = "_".join(f"{k}:{v}" for k, v in sorted(params.items()))
        return f"{query_type}:{hash(param_str)}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if not self.enable_caching or cache_key not in self.query_cache:
            return None
        
        result, timestamp = self.query_cache[cache_key]
        
        # Check if cache entry is still valid
        if time.time() - timestamp < self.cache_ttl:
            self.cache_hits += 1
            return result
        else:
            # Remove expired entry
            del self.query_cache[cache_key]
            return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache query result."""
        if self.enable_caching:
            self.query_cache[cache_key] = (result, time.time())
            self.cache_misses += 1
            
            # Limit cache size
            if len(self.query_cache) > 100:
                # Remove oldest entry
                oldest_key = min(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k][1])
                del self.query_cache[oldest_key]
    
    async def find_compatibility_conflicts(self, concepts: List[str]) -> QueryResult:
        """
        Find limitations that conflict with proposed concepts.
        
        Used by check_compatibility tool to identify incompatible approaches.
        """
        start_time = time.time()
        cache_key = self._get_cache_key("compatibility_conflicts", concepts=",".join(concepts))
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result
        
        try:
            conflicts = []
            relationships = []
            
            for concept in concepts:
                # Search for limitations related to this concept
                limitation_results = self.rif_db.search_entities(
                    query=concept,
                    entity_types=[self.claude_entity_types['limitations']],
                    limit=10
                )
                
                for limitation in limitation_results:
                    entity = self.rif_db.get_entity(limitation['id'])
                    if entity:
                        conflicts.append({
                            'entity': entity,
                            'concept': concept,
                            'conflict_type': 'limitation',
                            'severity': entity.get('metadata', {}).get('severity', 'medium')
                        })
                
                # Also check for anti-patterns related to concept
                antipattern_results = self.rif_db.search_entities(
                    query=concept,
                    entity_types=[self.claude_entity_types['anti_patterns']],
                    limit=5
                )
                
                for antipattern in antipattern_results:
                    entity = self.rif_db.get_entity(antipattern['id'])
                    if entity:
                        conflicts.append({
                            'entity': entity,
                            'concept': concept,
                            'conflict_type': 'anti_pattern',
                            'severity': 'high'  # Anti-patterns are always high severity
                        })
            
            result = QueryResult(
                entities=conflicts,
                relationships=relationships,
                execution_time_ms=(time.time() - start_time) * 1000,
                total_results=len(conflicts),
                query_type="compatibility_conflicts"
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Compatibility conflict query failed: {e}")
            raise
    
    async def find_implementation_patterns(self, technology: str, task_type: str, limit: int = 5) -> QueryResult:
        """
        Find implementation patterns for technology and task type.
        
        Used by recommend_pattern tool for pattern suggestions.
        """
        start_time = time.time()
        cache_key = self._get_cache_key("implementation_patterns", 
                                       technology=technology, task_type=task_type, limit=limit)
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result
        
        try:
            # Use hybrid search for best results
            search_query = f"{technology} {task_type}"
            
            pattern_results = self.rif_db.hybrid_search(
                text_query=search_query,
                entity_types=[self.claude_entity_types['patterns']],
                limit=limit * 2  # Get extra for filtering
            )
            
            patterns = []
            relationships = []
            
            for result in pattern_results[:limit]:
                entity = self.rif_db.get_entity(str(result.entity_id))
                if entity:
                    # Get supporting capabilities and tools
                    entity_relationships = self.rif_db.get_entity_relationships(
                        entity['id'], direction='incoming'
                    )
                    
                    supporting_tools = []
                    supporting_capabilities = []
                    
                    for rel in entity_relationships:
                        if rel['relationship_type'] == 'supports':
                            related_entity = self.rif_db.get_entity(rel['source_id'])
                            if related_entity:
                                if related_entity['type'] == self.claude_entity_types['tools']:
                                    supporting_tools.append(related_entity['name'])
                                elif related_entity['type'] == self.claude_entity_types['capabilities']:
                                    supporting_capabilities.append(related_entity['name'])
                    
                    patterns.append({
                        'entity': entity,
                        'similarity_score': result.similarity_score,
                        'supporting_tools': supporting_tools,
                        'supporting_capabilities': supporting_capabilities,
                        'usage_count': entity.get('metadata', {}).get('usage_count', 0)
                    })
                    
                    relationships.extend(entity_relationships)
            
            result = QueryResult(
                entities=patterns,
                relationships=relationships,
                execution_time_ms=(time.time() - start_time) * 1000,
                total_results=len(patterns),
                query_type="implementation_patterns"
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern search query failed: {e}")
            raise
    
    async def find_alternatives(self, problematic_approach: str) -> QueryResult:
        """
        Find alternatives to problematic approaches.
        
        Used by find_alternatives tool to suggest compatible approaches.
        """
        start_time = time.time()
        cache_key = self._get_cache_key("alternatives", approach=problematic_approach)
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result
        
        try:
            alternatives = []
            relationships = []
            
            # Strategy 1: Find entities with explicit "alternative_to" relationships
            # First find entities related to the problematic approach
            related_entities = self.rif_db.search_entities(
                query=problematic_approach,
                entity_types=[
                    self.claude_entity_types['anti_patterns'],
                    self.claude_entity_types['limitations']
                ],
                limit=10
            )
            
            for entity_data in related_entities:
                entity_relationships = self.rif_db.get_entity_relationships(
                    entity_data['id'], direction='outgoing'
                )
                
                for rel in entity_relationships:
                    if rel['relationship_type'] == 'alternative_to':
                        alternative_entity = self.rif_db.get_entity(rel['target_id'])
                        if alternative_entity:
                            alternatives.append({
                                'entity': alternative_entity,
                                'confidence': rel.get('confidence', 0.8),
                                'source': 'relationship',
                                'relationship_type': 'alternative_to'
                            })
                            relationships.append(rel)
            
            # Strategy 2: Find similar patterns using vector search
            similar_patterns = self.rif_db.hybrid_search(
                text_query=f"alternative to {problematic_approach}",
                entity_types=[self.claude_entity_types['patterns']],
                limit=10
            )
            
            for result in similar_patterns:
                entity = self.rif_db.get_entity(str(result.entity_id))
                if entity:
                    # Avoid duplicates
                    if not any(alt['entity']['id'] == entity['id'] for alt in alternatives):
                        alternatives.append({
                            'entity': entity,
                            'confidence': result.similarity_score,
                            'source': 'similarity',
                            'similarity_score': result.similarity_score
                        })
            
            result = QueryResult(
                entities=alternatives,
                relationships=relationships,
                execution_time_ms=(time.time() - start_time) * 1000,
                total_results=len(alternatives),
                query_type="alternatives"
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Alternatives query failed: {e}")
            raise
    
    async def validate_architectural_components(self, components: List[str]) -> QueryResult:
        """
        Validate architectural components against Claude Code constraints.
        
        Used by validate_architecture tool for design validation.
        """
        start_time = time.time()
        cache_key = self._get_cache_key("architectural_validation", components=",".join(components))
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result
        
        try:
            validation_results = []
            all_relationships = []
            
            for component in components:
                component_issues = []
                
                # Check for limitations related to this architectural component
                limitations = self.rif_db.search_entities(
                    query=component,
                    entity_types=[self.claude_entity_types['limitations']],
                    limit=5
                )
                
                for limitation in limitations:
                    entity = self.rif_db.get_entity(limitation['id'])
                    if entity:
                        component_issues.append({
                            'type': 'limitation',
                            'entity': entity,
                            'severity': entity.get('metadata', {}).get('severity', 'medium'),
                            'description': entity.get('metadata', {}).get('description', '')
                        })
                
                # Check for anti-patterns
                anti_patterns = self.rif_db.search_entities(
                    query=component,
                    entity_types=[self.claude_entity_types['anti_patterns']],
                    limit=3
                )
                
                for anti_pattern in anti_patterns:
                    entity = self.rif_db.get_entity(anti_pattern['id'])
                    if entity:
                        component_issues.append({
                            'type': 'anti_pattern',
                            'entity': entity,
                            'severity': 'high',
                            'description': entity.get('metadata', {}).get('description', '')
                        })
                
                # Get recommended patterns for this component
                recommended_patterns = self.rif_db.search_entities(
                    query=component,
                    entity_types=[self.claude_entity_types['patterns']],
                    limit=3
                )
                
                component_recommendations = []
                for pattern in recommended_patterns:
                    entity = self.rif_db.get_entity(pattern['id'])
                    if entity:
                        component_recommendations.append({
                            'entity': entity,
                            'confidence': 0.7  # Default confidence for text search
                        })
                
                validation_results.append({
                    'component': component,
                    'valid': len(component_issues) == 0,
                    'issues': component_issues,
                    'recommendations': component_recommendations,
                    'issue_count': len(component_issues)
                })
            
            result = QueryResult(
                entities=validation_results,
                relationships=all_relationships,
                execution_time_ms=(time.time() - start_time) * 1000,
                total_results=len(validation_results),
                query_type="architectural_validation"
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Architectural validation query failed: {e}")
            raise
    
    async def query_capability_limitations(self, capability_area: str, severity_filter: Optional[str] = None) -> QueryResult:
        """
        Query limitations for specific capability area.
        
        Used by query_limitations tool for limitation information.
        """
        start_time = time.time()
        cache_key = self._get_cache_key("capability_limitations", 
                                       area=capability_area, severity=severity_filter)
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result
        
        try:
            # Search for limitations in the capability area
            limitations = self.rif_db.search_entities(
                query=capability_area,
                entity_types=[self.claude_entity_types['limitations']],
                limit=20
            )
            
            filtered_limitations = []
            relationships = []
            
            for limitation in limitations:
                entity = self.rif_db.get_entity(limitation['id'])
                if entity:
                    metadata = entity.get('metadata', {})
                    severity = metadata.get('severity', 'medium')
                    
                    # Apply severity filter
                    if severity_filter and severity != severity_filter:
                        continue
                    
                    # Get alternative approaches for this limitation
                    entity_relationships = self.rif_db.get_entity_relationships(
                        entity['id'], direction='outgoing'
                    )
                    
                    workarounds = []
                    alternatives = []
                    
                    for rel in entity_relationships:
                        if rel['relationship_type'] == 'alternative_to':
                            alternative_entity = self.rif_db.get_entity(rel['target_id'])
                            if alternative_entity:
                                alternatives.append({
                                    'entity': alternative_entity,
                                    'confidence': rel.get('confidence', 0.7)
                                })
                                relationships.append(rel)
                    
                    # Extract workarounds from metadata
                    workarounds = metadata.get('workarounds', [])
                    
                    filtered_limitations.append({
                        'entity': entity,
                        'severity': severity,
                        'category': metadata.get('category', ''),
                        'impact': metadata.get('impact', ''),
                        'workarounds': workarounds,
                        'alternatives': alternatives,
                        'documentation_link': metadata.get('documentation_link', '')
                    })
            
            # Sort by severity (high -> medium -> low)
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            filtered_limitations.sort(
                key=lambda x: severity_order.get(x['severity'], 0),
                reverse=True
            )
            
            result = QueryResult(
                entities=filtered_limitations,
                relationships=relationships,
                execution_time_ms=(time.time() - start_time) * 1000,
                total_results=len(filtered_limitations),
                query_type="capability_limitations"
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Capability limitations query failed: {e}")
            raise
    
    async def get_entity_with_relationships(self, entity_id: str, relationship_depth: int = 1) -> QueryResult:
        """
        Get entity with its relationships up to specified depth.
        
        Utility method for complex relationship queries.
        """
        start_time = time.time()
        
        try:
            entity = self.rif_db.get_entity(entity_id)
            if not entity:
                return QueryResult(
                    entities=[],
                    relationships=[],
                    execution_time_ms=(time.time() - start_time) * 1000,
                    total_results=0,
                    query_type="entity_with_relationships"
                )
            
            all_relationships = []
            related_entities = {entity_id: entity}
            
            # Get relationships iteratively up to specified depth
            current_entities = [entity_id]
            
            for depth in range(relationship_depth):
                next_entities = []
                
                for current_entity_id in current_entities:
                    relationships = self.rif_db.get_entity_relationships(
                        current_entity_id, direction='both'
                    )
                    
                    for rel in relationships:
                        # Avoid duplicate relationships
                        if rel not in all_relationships:
                            all_relationships.append(rel)
                        
                        # Get related entities
                        for related_id in [rel['source_id'], rel['target_id']]:
                            if related_id not in related_entities and related_id != current_entity_id:
                                related_entity = self.rif_db.get_entity(related_id)
                                if related_entity:
                                    related_entities[related_id] = related_entity
                                    next_entities.append(related_id)
                
                current_entities = next_entities
            
            result = QueryResult(
                entities=list(related_entities.values()),
                relationships=all_relationships,
                execution_time_ms=(time.time() - start_time) * 1000,
                total_results=len(related_entities),
                query_type="entity_with_relationships"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Entity relationship query failed: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_enabled': self.enable_caching,
            'cache_size': len(self.query_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_ttl_seconds': self.cache_ttl
        }
    
    def clear_cache(self):
        """Clear query cache."""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Query cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on query engine."""
        try:
            # Test basic database connectivity
            test_entities = self.rif_db.search_entities(
                entity_types=[self.claude_entity_types['capabilities']],
                limit=1
            )
            
            # Test vector search
            test_search = self.rif_db.hybrid_search(
                text_query="test",
                entity_types=[self.claude_entity_types['patterns']],
                limit=1
            )
            
            return {
                'status': 'healthy',
                'database_connected': True,
                'vector_search_available': len(test_search) >= 0,  # May be 0 but should not error
                'claude_entities_available': len(test_entities) > 0,
                'cache_stats': self.get_cache_stats()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_connected': False,
                'vector_search_available': False,
                'claude_entities_available': False
            }