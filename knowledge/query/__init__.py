"""
Hybrid Query Planning System - Issue #33

Main interface for the hybrid search query planner that combines:
- Natural language query parsing
- Intelligent strategy planning  
- Vector similarity search
- Graph relationship traversal
- Advanced result ranking

Optimized for <100ms P95 latency as specified in requirements.
"""

from .query_parser import (
    QueryParser, 
    StructuredQuery, 
    QueryIntent, 
    SearchStrategy,
    parse_query
)

from .strategy_planner import (
    StrategyPlanner,
    ExecutionPlan,
    ExecutionMode,
    ResourceConstraints,
    plan_query_execution
)

from .hybrid_search_engine import (
    HybridSearchEngine,
    SearchResult,
    HybridSearchResults,
    hybrid_search
)

from .result_ranker import (
    ResultRanker,
    RankingContext,
    RankingWeights,
    create_ranking_context,
    rank_search_results
)

__all__ = [
    # Core classes
    'QueryPlanner',
    'QueryParser', 
    'StrategyPlanner',
    'HybridSearchEngine',
    'ResultRanker',
    
    # Data structures
    'StructuredQuery',
    'ExecutionPlan', 
    'SearchResult',
    'HybridSearchResults',
    'RankingContext',
    
    # Enums
    'QueryIntent',
    'SearchStrategy',
    'ExecutionMode',
    
    # Configuration
    'ResourceConstraints',
    'RankingWeights',
    
    # Convenience functions
    'plan_and_execute_query',
    'parse_query',
    'hybrid_search',
    'plan_query_execution',
    'create_ranking_context',
    'rank_search_results'
]


class QueryPlanner:
    """
    Main query planner interface optimized for performance.
    
    This is the primary interface for the hybrid search system, providing
    end-to-end query processing with intelligent caching and optimization.
    """
    
    def __init__(self, db_path: str = "knowledge/chromadb/entities.duckdb",
                 enable_caching: bool = True,
                 performance_mode: ExecutionMode = ExecutionMode.BALANCED):
        self.db_path = db_path
        self.enable_caching = enable_caching
        self.performance_mode = performance_mode
        
        # Initialize components
        self.parser = QueryParser()
        self.planner = StrategyPlanner()
        self.search_engine = HybridSearchEngine(db_path)
        self.ranker = ResultRanker()
        
        # Performance optimizations
        self.query_cache = {} if enable_caching else None
        self.performance_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'average_latency_ms': 0,
            'p95_latency_ms': 0,
            'latency_samples': []
        }
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    def execute_query(self, query_text: str, 
                     context: RankingContext = None,
                     max_results: int = 20) -> HybridSearchResults:
        """
        Execute complete hybrid search query.
        
        Args:
            query_text: Natural language query
            context: Optional ranking context for personalization
            max_results: Maximum results to return
            
        Returns:
            Comprehensive search results with performance metrics
        """
        import time
        start_time = time.time()
        
        try:
            # Check cache first
            if self.enable_caching:
                cache_key = self._generate_cache_key(query_text, context)
                if cache_key in self.query_cache:
                    cached_result = self.query_cache[cache_key]
                    self.performance_metrics['cache_hits'] += 1
                    self.logger.debug(f"Cache hit for query: {query_text}")
                    return cached_result
            
            # Parse natural language query
            structured_query = self.parser.parse_query(query_text)
            structured_query.result_limit = max_results
            
            # Plan optimal execution strategy
            execution_plan = self.planner.plan_execution(structured_query, self.performance_mode)
            
            # Execute hybrid search
            search_results = self.search_engine.search(structured_query, execution_plan)
            
            # Rank and optimize results
            if context is None:
                context = create_ranking_context(query_text, structured_query.intent.primary_intent.value)
            
            ranked_results = self.ranker.rank_results(search_results.results, context)
            
            # Update search results with ranked results
            search_results.results = ranked_results[:max_results]
            search_results.total_found = len(ranked_results)
            
            # Cache successful results
            if self.enable_caching and len(ranked_results) > 0:
                self.query_cache[cache_key] = search_results
                
                # Limit cache size (LRU-style eviction)
                if len(self.query_cache) > 1000:
                    # Remove oldest 100 entries
                    old_keys = list(self.query_cache.keys())[:100]
                    for old_key in old_keys:
                        del self.query_cache[old_key]
            
            # Update performance metrics
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_performance_metrics(execution_time_ms)
            
            self.logger.info(f"Query executed in {execution_time_ms}ms: {query_text}")
            return search_results
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_performance_metrics(execution_time_ms)
            self.logger.error(f"Query execution failed in {execution_time_ms}ms: {e}")
            
            # Return empty results on error
            return HybridSearchResults(
                results=[],
                total_found=0,
                execution_time_ms=execution_time_ms
            )
    
    def explain_query(self, query_text: str) -> dict:
        """
        Explain how a query would be processed (for debugging/optimization).
        
        Args:
            query_text: Query to explain
            
        Returns:
            Dictionary with query processing explanation
        """
        structured_query = self.parser.parse_query(query_text)
        execution_plan = self.planner.plan_execution(structured_query, self.performance_mode)
        
        return {
            'original_query': query_text,
            'parsed_intent': structured_query.intent.primary_intent.value,
            'entities_found': [e.name for e in structured_query.intent.entities],
            'concepts_extracted': structured_query.intent.concepts,
            'execution_strategy': execution_plan.strategy.value,
            'vector_enabled': execution_plan.vector_enabled,
            'graph_enabled': execution_plan.graph_enabled,
            'parallel_execution': execution_plan.parallel_execution,
            'estimated_latency_ms': execution_plan.estimated_cost.latency_ms if execution_plan.estimated_cost else None,
            'filters': {
                'entity_types': list(structured_query.intent.filters.entity_types),
                'file_patterns': structured_query.intent.filters.file_patterns
            }
        }
    
    def optimize_performance(self) -> dict:
        """
        Perform performance optimization and return recommendations.
        
        Returns:
            Dictionary with optimization recommendations
        """
        metrics = self.get_performance_metrics()
        recommendations = []
        
        # Analyze performance patterns
        if metrics['p95_latency_ms'] > 500:  # Above 500ms P95
            recommendations.append({
                'issue': 'High P95 latency',
                'current_value': f"{metrics['p95_latency_ms']}ms",
                'recommendation': 'Consider switching to FAST execution mode or increase caching'
            })
        
        if metrics['cache_hit_rate'] < 0.3:  # Low cache hit rate
            recommendations.append({
                'issue': 'Low cache hit rate',
                'current_value': f"{metrics['cache_hit_rate']:.1%}",
                'recommendation': 'Review query patterns - consider query normalization'
            })
        
        if metrics['average_latency_ms'] > 200:  # High average latency
            recommendations.append({
                'issue': 'High average latency',
                'current_value': f"{metrics['average_latency_ms']}ms",
                'recommendation': 'Review database indexes and consider result limit reduction'
            })
        
        # Auto-optimize if performance is poor
        if metrics['p95_latency_ms'] > 500 and self.performance_mode != ExecutionMode.FAST:
            self.performance_mode = ExecutionMode.FAST
            recommendations.append({
                'issue': 'Auto-optimization triggered',
                'action': 'Switched to FAST execution mode',
                'expected_improvement': 'Reduce latency by 30-50%'
            })
        
        return {
            'current_metrics': metrics,
            'recommendations': recommendations,
            'optimization_applied': len([r for r in recommendations if 'action' in r]) > 0
        }
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Calculate cache hit rate
        total_queries = metrics['total_queries']
        cache_hits = metrics['cache_hits']
        metrics['cache_hit_rate'] = cache_hits / total_queries if total_queries > 0 else 0.0
        
        # Calculate P95 latency
        if len(metrics['latency_samples']) >= 20:  # Need sufficient samples
            import numpy as np
            metrics['p95_latency_ms'] = int(np.percentile(metrics['latency_samples'], 95))
        else:
            metrics['p95_latency_ms'] = metrics['average_latency_ms']
        
        # Don't return raw samples in metrics
        del metrics['latency_samples']
        
        return metrics
    
    def clear_cache(self):
        """Clear query cache"""
        if self.query_cache is not None:
            self.query_cache.clear()
            self.logger.info("Query cache cleared")
    
    def _generate_cache_key(self, query_text: str, context: RankingContext = None) -> str:
        """Generate cache key for query"""
        import hashlib
        
        # Include query text and relevant context in cache key
        cache_input = query_text.lower().strip()
        
        if context:
            # Include context that affects results
            cache_input += f"|files:{sorted(context.active_files)}"
            cache_input += f"|entities:{sorted(context.recent_entities)}"
        
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _update_performance_metrics(self, execution_time_ms: int):
        """Update performance tracking"""
        metrics = self.performance_metrics
        
        metrics['total_queries'] += 1
        
        # Update average latency using exponential moving average
        alpha = 0.1
        if metrics['average_latency_ms'] == 0:
            metrics['average_latency_ms'] = execution_time_ms
        else:
            old_avg = metrics['average_latency_ms']
            new_avg = alpha * execution_time_ms + (1 - alpha) * old_avg
            metrics['average_latency_ms'] = int(new_avg)
        
        # Keep recent latency samples for P95 calculation
        metrics['latency_samples'].append(execution_time_ms)
        if len(metrics['latency_samples']) > 1000:  # Keep last 1000 samples
            metrics['latency_samples'] = metrics['latency_samples'][-1000:]


# Convenience function for end-to-end query execution
def plan_and_execute_query(query_text: str, 
                          db_path: str = "knowledge/chromadb/entities.duckdb",
                          max_results: int = 20,
                          context: RankingContext = None) -> HybridSearchResults:
    """
    Convenience function for complete query planning and execution.
    
    Args:
        query_text: Natural language query
        db_path: Path to knowledge database
        max_results: Maximum results to return  
        context: Optional ranking context
        
    Returns:
        Complete search results with performance metrics
    """
    planner = QueryPlanner(db_path, enable_caching=True, performance_mode=ExecutionMode.BALANCED)
    return planner.execute_query(query_text, context, max_results)


# Example usage and testing
if __name__ == "__main__":
    # Test the complete query planning system
    test_queries = [
        "find authentication functions similar to login",
        "what functions call processPayment",
        "what breaks if I change the User class",
        "show me error handling patterns in Python files"
    ]
    
    planner = QueryPlanner()
    
    for query in test_queries:
        print(f"\n--- Testing Query: {query} ---")
        
        # Explain the query first
        explanation = planner.explain_query(query)
        print(f"Intent: {explanation['parsed_intent']}")
        print(f"Strategy: {explanation['execution_strategy']}")
        print(f"Entities: {explanation['entities_found']}")
        
        # Execute the query
        results = planner.execute_query(query, max_results=5)
        print(f"Found {results.total_found} results in {results.execution_time_ms}ms")
        
        for i, result in enumerate(results.results[:3]):
            print(f"  {i+1}. {result.entity_name} ({result.entity_type}) - {result.relevance_score:.2f}")
    
    # Show performance metrics
    metrics = planner.get_performance_metrics()
    print(f"\n--- Performance Metrics ---")
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"Average latency: {metrics['average_latency_ms']}ms")
    print(f"P95 latency: {metrics['p95_latency_ms']}ms")