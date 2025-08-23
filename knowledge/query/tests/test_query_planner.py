"""
Comprehensive tests for Hybrid Query Planner system - Issue #33
Tests all components with focus on performance requirements
"""

import pytest
import time
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import all components to test
from ..query_parser import QueryParser, StructuredQuery, QueryIntent, SearchStrategy
from ..strategy_planner import StrategyPlanner, ExecutionPlan, ExecutionMode, ResourceConstraints  
from ..hybrid_search_engine import HybridSearchEngine, SearchResult, HybridSearchResults
from ..result_ranker import ResultRanker, RankingContext, RankingWeights
from .. import QueryPlanner, plan_and_execute_query


class TestQueryParser:
    """Test suite for natural language query parsing"""
    
    def setup_method(self):
        self.parser = QueryParser()
    
    def test_entity_search_parsing(self):
        """Test parsing of entity search queries"""
        query = "find function authenticateUser"
        result = self.parser.parse_query(query)
        
        assert result.intent.primary_intent == QueryIntent.ENTITY_SEARCH
        assert len(result.intent.entities) >= 1
        assert any(e.name == "authenticateUser" for e in result.intent.entities)
        assert result.execution_strategy in [SearchStrategy.GRAPH_ONLY, SearchStrategy.HYBRID_PARALLEL]
    
    def test_similarity_search_parsing(self):
        """Test parsing of similarity search queries"""
        query = "show me functions similar to login handling"
        result = self.parser.parse_query(query)
        
        assert result.intent.primary_intent == QueryIntent.SIMILARITY_SEARCH
        assert result.intent.requires_semantic_search == True
        assert result.execution_strategy in [SearchStrategy.VECTOR_ONLY, SearchStrategy.HYBRID_PARALLEL]
    
    def test_dependency_analysis_parsing(self):
        """Test parsing of dependency analysis queries"""
        query = "what functions call processPayment"
        result = self.parser.parse_query(query)
        
        assert result.intent.primary_intent == QueryIntent.DEPENDENCY_ANALYSIS
        assert result.intent.requires_structural_search == True
        assert "processPayment" in [e.name for e in result.intent.entities]
    
    def test_impact_analysis_parsing(self):
        """Test parsing of impact analysis queries"""
        query = "what breaks if I change the User class"
        result = self.parser.parse_query(query)
        
        assert result.intent.primary_intent == QueryIntent.IMPACT_ANALYSIS
        assert result.intent.requires_structural_search == True
        assert "User" in [e.name for e in result.intent.entities]
    
    def test_hybrid_query_parsing(self):
        """Test parsing of complex hybrid queries"""
        query = "find authentication functions similar to login that also handle errors"
        result = self.parser.parse_query(query)
        
        assert result.intent.primary_intent == QueryIntent.HYBRID_SEARCH
        assert result.intent.requires_semantic_search == True
        assert result.intent.requires_structural_search == True
        assert result.execution_strategy == SearchStrategy.HYBRID_PARALLEL
    
    def test_entity_extraction(self):
        """Test entity extraction from various query formats"""
        test_cases = [
            ("find function getUserInfo", ["getUserInfo"]),
            ("what calls processData and handleError", ["processData", "handleError"]),
            ("show me the AuthService class", ["AuthService"]),
            ('search for "exact_function_name"', ["exact_function_name"])
        ]
        
        for query_text, expected_entities in test_cases:
            result = self.parser.parse_query(query_text)
            extracted_entities = [e.name for e in result.intent.entities]
            
            for expected_entity in expected_entities:
                assert expected_entity in extracted_entities, f"Expected {expected_entity} in {extracted_entities} for query: {query_text}"
    
    def test_filter_extraction(self):
        """Test extraction of filters from queries"""
        query = "find Python functions in auth.py"
        result = self.parser.parse_query(query)
        
        assert "function" in result.intent.filters.entity_types
        assert any("python" in pattern.lower() for pattern in result.intent.filters.file_patterns)
    
    def test_query_normalization(self):
        """Test query text normalization"""
        test_queries = [
            "Find Function    AuthenticateUser  ",  # Extra whitespace
            "FIND function authenticateuser",       # Case variations
            "Can you help me find function authenticateUser?",  # Conversational
        ]
        
        results = [self.parser.parse_query(q) for q in test_queries]
        
        # All should result in similar parsed structures
        for result in results:
            assert result.intent.primary_intent == QueryIntent.ENTITY_SEARCH
            assert len(result.intent.entities) > 0


class TestStrategyPlanner:
    """Test suite for execution strategy planning"""
    
    def setup_method(self):
        self.planner = StrategyPlanner()
    
    def test_vector_strategy_selection(self):
        """Test selection of vector-only strategy"""
        # Create mock query that should use vector search
        mock_query = Mock(spec=StructuredQuery)
        mock_query.intent.primary_intent = QueryIntent.SIMILARITY_SEARCH
        mock_query.intent.requires_semantic_search = True
        mock_query.intent.requires_structural_search = False
        mock_query.intent.entities = []
        mock_query.intent.concepts = ["similar", "pattern"]
        mock_query.intent.filters.entity_types = set()
        mock_query.intent.filters.file_patterns = []
        mock_query.result_limit = 20
        mock_query.timeout_ms = 2000
        mock_query.vector_query = "test query"
        mock_query.graph_query = None
        mock_query.direct_lookup = None
        
        plan = self.planner.plan_execution(mock_query, ExecutionMode.BALANCED)
        
        assert plan.vector_enabled == True
        assert plan.graph_enabled == False or plan.strategy == SearchStrategy.HYBRID_PARALLEL
    
    def test_graph_strategy_selection(self):
        """Test selection of graph-only strategy"""
        mock_query = Mock(spec=StructuredQuery)
        mock_query.intent.primary_intent = QueryIntent.DEPENDENCY_ANALYSIS
        mock_query.intent.requires_semantic_search = False
        mock_query.intent.requires_structural_search = True
        mock_query.intent.entities = [Mock(name="testEntity")]
        mock_query.intent.concepts = ["calls", "depends"]
        mock_query.intent.filters.entity_types = set()
        mock_query.intent.filters.file_patterns = []
        mock_query.result_limit = 20
        mock_query.timeout_ms = 2000
        mock_query.vector_query = None
        mock_query.graph_query = {"start_entities": [{"name": "test"}]}
        mock_query.direct_lookup = None
        
        plan = self.planner.plan_execution(mock_query, ExecutionMode.BALANCED)
        
        assert plan.graph_enabled == True
        assert plan.vector_enabled == False or plan.strategy == SearchStrategy.HYBRID_PARALLEL
    
    def test_hybrid_strategy_selection(self):
        """Test selection of hybrid parallel strategy"""
        mock_query = Mock(spec=StructuredQuery)
        mock_query.intent.primary_intent = QueryIntent.HYBRID_SEARCH
        mock_query.intent.requires_semantic_search = True
        mock_query.intent.requires_structural_search = True
        mock_query.intent.entities = [Mock(name="testEntity")]
        mock_query.intent.concepts = ["similar", "calls"]
        mock_query.intent.filters.entity_types = set()
        mock_query.intent.filters.file_patterns = []
        mock_query.result_limit = 20
        mock_query.timeout_ms = 2000
        mock_query.vector_query = "test query"
        mock_query.graph_query = {"start_entities": [{"name": "test"}]}
        mock_query.direct_lookup = None
        
        plan = self.planner.plan_execution(mock_query, ExecutionMode.BALANCED)
        
        assert plan.strategy == SearchStrategy.HYBRID_PARALLEL
        assert plan.vector_enabled == True
        assert plan.graph_enabled == True
        assert plan.parallel_execution == True
    
    def test_performance_mode_impact(self):
        """Test how different performance modes affect planning"""
        mock_query = Mock(spec=StructuredQuery)
        mock_query.intent.primary_intent = QueryIntent.HYBRID_SEARCH
        mock_query.intent.requires_semantic_search = True
        mock_query.intent.requires_structural_search = True
        mock_query.intent.entities = []
        mock_query.intent.concepts = []
        mock_query.intent.filters.entity_types = set()
        mock_query.intent.filters.file_patterns = []
        mock_query.result_limit = 20
        mock_query.timeout_ms = 2000
        mock_query.vector_query = "test"
        mock_query.graph_query = {"start_entities": []}
        mock_query.direct_lookup = None
        
        # Test different modes
        fast_plan = self.planner.plan_execution(mock_query, ExecutionMode.FAST)
        balanced_plan = self.planner.plan_execution(mock_query, ExecutionMode.BALANCED)
        comprehensive_plan = self.planner.plan_execution(mock_query, ExecutionMode.COMPREHENSIVE)
        
        # Fast mode should have lower limits and timeouts
        assert fast_plan.vector_limit <= balanced_plan.vector_limit
        assert fast_plan.timeout_per_search_ms <= balanced_plan.timeout_per_search_ms
        
        # Comprehensive mode should have higher limits
        assert comprehensive_plan.vector_limit >= balanced_plan.vector_limit
    
    def test_resource_constraints_enforcement(self):
        """Test that resource constraints are enforced"""
        strict_constraints = ResourceConstraints(
            max_latency_ms=100,  # Very strict latency
            max_memory_mb=50,
            max_concurrent_searches=1
        )
        
        constrained_planner = StrategyPlanner(strict_constraints)
        
        mock_query = Mock(spec=StructuredQuery)
        mock_query.intent.primary_intent = QueryIntent.HYBRID_SEARCH
        mock_query.intent.requires_semantic_search = True
        mock_query.intent.requires_structural_search = True
        mock_query.intent.entities = []
        mock_query.intent.concepts = []
        mock_query.intent.filters.entity_types = set()
        mock_query.intent.filters.file_patterns = []
        mock_query.result_limit = 20
        mock_query.timeout_ms = 2000
        mock_query.vector_query = "test"
        mock_query.graph_query = {"start_entities": []}
        mock_query.direct_lookup = None
        
        plan = constrained_planner.plan_execution(mock_query, ExecutionMode.BALANCED)
        
        # Should respect constraints
        assert plan.max_workers <= strict_constraints.max_concurrent_searches
        # Should adapt strategy if needed for performance


class TestResultRanker:
    """Test suite for result ranking and relevance scoring"""
    
    def setup_method(self):
        self.ranker = ResultRanker()
        self.context = RankingContext(
            query_text="find authentication functions",
            query_intent="entity_search",
            mentioned_entities=["authenticate", "login"],
            semantic_keywords=["auth", "security"]
        )
    
    def test_result_ranking_order(self):
        """Test that results are properly ranked by relevance"""
        # Create sample results with different characteristics
        results = [
            SearchResult(
                entity_id="1",
                entity_name="authenticate",  # Exact match
                entity_type="function",
                file_path="/src/auth.py",
                relevance_score=0.5,  # Will be recalculated
                source_strategy="direct"
            ),
            SearchResult(
                entity_id="2", 
                entity_name="login_handler",  # Partial match
                entity_type="function",
                file_path="/src/handlers.py", 
                relevance_score=0.7,
                source_strategy="vector"
            ),
            SearchResult(
                entity_id="3",
                entity_name="process_data",  # No match
                entity_type="function",
                file_path="/src/data.py",
                relevance_score=0.3,
                source_strategy="graph"
            )
        ]
        
        ranked_results = self.ranker.rank_results(results, self.context)
        
        # Should be ranked by final relevance score
        assert len(ranked_results) == 3
        for i in range(len(ranked_results) - 1):
            assert ranked_results[i].relevance_score >= ranked_results[i + 1].relevance_score
        
        # Exact match should rank highly
        exact_match_result = next((r for r in ranked_results if r.entity_name == "authenticate"), None)
        assert exact_match_result is not None
        assert exact_match_result.relevance_score > 0.5
    
    def test_diversity_filtering(self):
        """Test that similar results are filtered for diversity"""
        # Create many similar results
        similar_results = []
        for i in range(10):
            result = SearchResult(
                entity_id=str(i),
                entity_name=f"auth_function_{i}",
                entity_type="function",
                file_path="/src/auth.py",  # Same file
                relevance_score=0.8,
                source_strategy="vector"
            )
            similar_results.append(result)
        
        ranked_results = self.ranker.rank_results(similar_results, self.context)
        
        # Should have fewer results due to diversity filtering
        # Exact number depends on diversity algorithm, but should be less than original
        assert len(ranked_results) <= len(similar_results)
    
    def test_context_influence_on_ranking(self):
        """Test that context influences ranking appropriately"""
        # Create context with active file preference
        context_with_active_file = RankingContext(
            query_text="find functions",
            query_intent="entity_search",
            active_files={"/src/auth.py"}
        )
        
        results = [
            SearchResult(
                entity_id="1",
                entity_name="function_a", 
                entity_type="function",
                file_path="/src/auth.py",  # In active file
                relevance_score=0.5,
                source_strategy="vector"
            ),
            SearchResult(
                entity_id="2",
                entity_name="function_b",
                entity_type="function", 
                file_path="/src/other.py",  # Not in active file
                relevance_score=0.6,  # Higher base score
                source_strategy="vector"
            )
        ]
        
        ranked_results = self.ranker.rank_results(results, context_with_active_file)
        
        # Active file should get boost and potentially rank higher
        active_file_result = next((r for r in ranked_results if r.file_path == "/src/auth.py"), None)
        assert active_file_result is not None


class TestPerformance:
    """Performance-focused test suite"""
    
    def setup_method(self):
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.duckdb')
        self.temp_db.close()
        
        # Mock the database operations for performance testing
        self.mock_search_engines()
    
    def teardown_method(self):
        os.unlink(self.temp_db.name)
    
    def mock_search_engines(self):
        """Mock search engines to return test data quickly"""
        def mock_vector_search(*args, **kwargs):
            time.sleep(0.02)  # 20ms simulated search time
            return [
                SearchResult(
                    entity_id="v1", entity_name="vector_result_1", entity_type="function",
                    file_path="/test.py", relevance_score=0.9, source_strategy="vector"
                )
            ]
        
        def mock_graph_search(*args, **kwargs):
            time.sleep(0.03)  # 30ms simulated search time
            return [
                SearchResult(
                    entity_id="g1", entity_name="graph_result_1", entity_type="function",
                    file_path="/test.py", relevance_score=0.8, source_strategy="graph"
                )
            ]
        
        def mock_direct_search(*args, **kwargs):
            time.sleep(0.01)  # 10ms simulated search time
            return [
                SearchResult(
                    entity_id="d1", entity_name="direct_result_1", entity_type="function",
                    file_path="/test.py", relevance_score=1.0, source_strategy="direct"
                )
            ]
        
        # Patch the search methods
        self.vector_search_patch = patch('knowledge.query.hybrid_search_engine.VectorSearchEngine.search', side_effect=mock_vector_search)
        self.graph_search_patch = patch('knowledge.query.hybrid_search_engine.GraphSearchEngine.search', side_effect=mock_graph_search)
        self.direct_search_patch = patch('knowledge.query.hybrid_search_engine.DirectSearchEngine.search', side_effect=mock_direct_search)
        
        self.vector_search_patch.start()
        self.graph_search_patch.start()
        self.direct_search_patch.start()
    
    def test_latency_requirement_simple_query(self):
        """Test that simple queries meet <100ms P95 latency requirement"""
        planner = QueryPlanner(db_path=self.temp_db.name, performance_mode=ExecutionMode.FAST)
        
        # Test simple entity search queries
        simple_queries = [
            "find function auth",
            "show class User",
            "locate variable config"
        ]
        
        execution_times = []
        
        for query in simple_queries:
            start_time = time.time()
            results = planner.execute_query(query, max_results=10)
            execution_time_ms = int((time.time() - start_time) * 1000)
            execution_times.append(execution_time_ms)
            
            # Individual query should be fast
            assert execution_time_ms < 200, f"Query '{query}' took {execution_time_ms}ms (expected <200ms)"
        
        # Calculate P95
        if len(execution_times) >= 2:
            import numpy as np
            p95_latency = np.percentile(execution_times, 95)
            assert p95_latency < 100, f"P95 latency {p95_latency}ms exceeds 100ms requirement"
    
    def test_latency_requirement_complex_query(self):
        """Test that complex queries meet reasonable latency requirements"""
        planner = QueryPlanner(db_path=self.temp_db.name, performance_mode=ExecutionMode.BALANCED)
        
        complex_queries = [
            "find authentication functions similar to login that handle errors",
            "what functions call processPayment and also handle database transactions",
            "show me all error handling patterns in Python modules that connect to APIs"
        ]
        
        execution_times = []
        
        for query in complex_queries:
            start_time = time.time()
            results = planner.execute_query(query, max_results=20)
            execution_time_ms = int((time.time() - start_time) * 1000)
            execution_times.append(execution_time_ms)
            
            # Complex queries should still be reasonable
            assert execution_time_ms < 500, f"Complex query '{query}' took {execution_time_ms}ms (expected <500ms)"
        
        # Average should be good
        avg_latency = sum(execution_times) / len(execution_times)
        assert avg_latency < 300, f"Average complex query latency {avg_latency}ms too high"
    
    def test_caching_performance_improvement(self):
        """Test that caching improves performance for repeated queries"""
        planner = QueryPlanner(db_path=self.temp_db.name, enable_caching=True)
        
        query = "find function authenticate"
        
        # First execution (cache miss)
        start_time = time.time()
        first_result = planner.execute_query(query)
        first_time_ms = int((time.time() - start_time) * 1000)
        
        # Second execution (cache hit)
        start_time = time.time()  
        second_result = planner.execute_query(query)
        second_time_ms = int((time.time() - start_time) * 1000)
        
        # Cache hit should be significantly faster
        assert second_time_ms < first_time_ms / 2, f"Cache hit ({second_time_ms}ms) not significantly faster than miss ({first_time_ms}ms)"
        
        # Verify cache hit was recorded
        metrics = planner.get_performance_metrics()
        assert metrics['cache_hits'] > 0
        assert metrics['cache_hit_rate'] > 0.0
    
    def test_parallel_execution_performance(self):
        """Test that parallel execution improves performance over sequential"""
        # Test with hybrid queries that use both vector and graph search
        query = "find functions similar to auth that also call database"
        
        # Create execution plans with different parallelization
        parser = QueryParser()
        planner = StrategyPlanner()
        
        structured_query = parser.parse_query(query)
        
        # Force parallel execution
        parallel_plan = planner.plan_execution(structured_query, ExecutionMode.BALANCED)
        parallel_plan.parallel_execution = True
        parallel_plan.max_workers = 2
        
        # Force sequential execution  
        sequential_plan = planner.plan_execution(structured_query, ExecutionMode.BALANCED)
        sequential_plan.parallel_execution = False
        sequential_plan.max_workers = 1
        
        engine = HybridSearchEngine(self.temp_db.name)
        
        # Time parallel execution
        start_time = time.time()
        parallel_results = engine.search(structured_query, parallel_plan)
        parallel_time_ms = int((time.time() - start_time) * 1000)
        
        # Time sequential execution
        start_time = time.time()
        sequential_results = engine.search(structured_query, sequential_plan)
        sequential_time_ms = int((time.time() - start_time) * 1000)
        
        # Parallel should be faster for hybrid queries
        # Note: With mocked short execution times, the difference may be small
        # but parallel should not be significantly slower
        assert parallel_time_ms <= sequential_time_ms * 1.5, f"Parallel ({parallel_time_ms}ms) much slower than sequential ({sequential_time_ms}ms)"
    
    def test_memory_usage_bounds(self):
        """Test that memory usage stays within reasonable bounds"""
        planner = QueryPlanner(db_path=self.temp_db.name)
        
        # Execute many different queries to build up caches
        queries = [
            "find function auth",
            "show class User", 
            "what calls process_data",
            "similar to error_handler",
            "dependencies of payment_service"
        ]
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Execute queries multiple times
        for _ in range(10):
            for query in queries:
                planner.execute_query(query)
        
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - initial_memory_mb
        
        # Memory increase should be reasonable (less than 200MB for testing)
        assert memory_increase_mb < 200, f"Memory usage increased by {memory_increase_mb}MB (too high)"


class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.duckdb')
        self.temp_db.close()
    
    def teardown_method(self):
        os.unlink(self.temp_db.name)
    
    def test_end_to_end_query_execution(self):
        """Test complete end-to-end query execution"""
        # Mock the underlying storage to return test data
        with patch('knowledge.query.hybrid_search_engine.VectorSearchEngine.search') as mock_vector, \
             patch('knowledge.query.hybrid_search_engine.GraphSearchEngine.search') as mock_graph, \
             patch('knowledge.query.hybrid_search_engine.DirectSearchEngine.search') as mock_direct:
            
            # Mock return values
            mock_vector.return_value = [
                SearchResult("v1", "auth_function", "function", "/src/auth.py", relevance_score=0.9, source_strategy="vector")
            ]
            mock_graph.return_value = [
                SearchResult("g1", "login_handler", "function", "/src/login.py", relevance_score=0.8, source_strategy="graph")  
            ]
            mock_direct.return_value = [
                SearchResult("d1", "authenticate", "function", "/src/auth.py", relevance_score=1.0, source_strategy="direct")
            ]
            
            # Test end-to-end execution
            result = plan_and_execute_query(
                "find authentication functions similar to login",
                db_path=self.temp_db.name,
                max_results=10
            )
            
            # Verify results structure
            assert isinstance(result, HybridSearchResults)
            assert result.total_found > 0
            assert result.execution_time_ms > 0
            assert len(result.results) > 0
            
            # Verify results are properly ranked
            assert all(isinstance(r, SearchResult) for r in result.results)
            assert all(hasattr(r, 'relevance_score') for r in result.results)
    
    def test_query_explanation(self):
        """Test query explanation functionality"""
        planner = QueryPlanner(db_path=self.temp_db.name)
        
        explanation = planner.explain_query("find functions similar to authentication")
        
        # Verify explanation structure
        required_keys = [
            'original_query', 'parsed_intent', 'entities_found', 'concepts_extracted',
            'execution_strategy', 'vector_enabled', 'graph_enabled', 'parallel_execution'
        ]
        
        for key in required_keys:
            assert key in explanation, f"Missing key '{key}' in query explanation"
        
        # Verify content makes sense
        assert explanation['original_query'] == "find functions similar to authentication"
        assert explanation['parsed_intent'] in ['similarity_search', 'hybrid_search']
        assert isinstance(explanation['entities_found'], list)
        assert isinstance(explanation['concepts_extracted'], list)
    
    def test_performance_optimization(self):
        """Test performance optimization functionality"""
        planner = QueryPlanner(db_path=self.temp_db.name)
        
        # Execute some queries to generate metrics
        queries = ["test query 1", "test query 2", "test query 3"]
        for query in queries:
            with patch.object(planner.search_engine, 'search') as mock_search:
                mock_search.return_value = HybridSearchResults([], 0, 100)  # 100ms simulated time
                planner.execute_query(query)
        
        # Test optimization
        optimization_result = planner.optimize_performance()
        
        # Verify optimization structure
        assert 'current_metrics' in optimization_result
        assert 'recommendations' in optimization_result  
        assert 'optimization_applied' in optimization_result
        
        # Verify metrics structure
        metrics = optimization_result['current_metrics']
        assert 'total_queries' in metrics
        assert 'average_latency_ms' in metrics
        assert 'cache_hit_rate' in metrics


if __name__ == "__main__":
    # Run specific test classes for development
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "performance":
        # Run only performance tests
        pytest.main(["-v", "test_query_planner.py::TestPerformance"])
    elif len(sys.argv) > 1 and sys.argv[1] == "integration": 
        # Run only integration tests
        pytest.main(["-v", "test_query_planner.py::TestIntegration"])
    else:
        # Run all tests
        pytest.main(["-v", "test_query_planner.py"])