#!/usr/bin/env python3
"""
Test Suite for DPIBS Database Schema + Performance Optimization Layer
Issue #138: Database Schema + Performance Optimization Layer

Comprehensive tests for database schema, performance optimization, 
caching layer, and <100ms cached query performance validation.
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sqlite3
import duckdb

# Import components under test
import sys
sys.path.append('/Users/cal/DEV/RIF')

from database.optimization.caching import DatabaseCachingLayer, CacheLevel, create_database_caching_layer
from systems.context_database_schema import ContextDatabaseSchema


class TestDPIBSDatabaseSchema:
    """Test suite for DPIBS Database Schema"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        temp_dir = tempfile.mkdtemp()
        return temp_dir
    
    @pytest.fixture
    def database_schema(self, temp_db_path):
        """Create database schema instance for testing"""
        return ContextDatabaseSchema(base_path=temp_db_path)
    
    @pytest.fixture
    def caching_layer(self, temp_db_path):
        """Create database caching layer for testing"""
        cache_db_path = os.path.join(temp_db_path, "test_cache.db")
        return create_database_caching_layer(
            db_path=temp_db_path,
            cache_db_path=cache_db_path,
            max_memory_cache_size=100,
            default_ttl=300
        )
    
    # Database Schema Tests
    
    def test_database_schema_initialization(self, database_schema):
        """Test database schema initializes correctly"""
        assert database_schema is not None
        assert os.path.exists(database_schema.context_db_path)
        assert os.path.exists(database_schema.performance_db_path)
    
    def test_context_optimization_table_creation(self, database_schema, temp_db_path):
        """Test context optimization tables are created with correct structure"""
        db_path = os.path.join(temp_db_path, "context_intelligence.duckdb")
        
        with duckdb.connect(db_path) as conn:
            # Test context_optimizations table exists
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'context_optimizations'
            """).fetchall()
            assert len(tables) == 1
            
            # Test table structure
            columns = conn.execute("""
                SELECT column_name, data_type FROM information_schema.columns 
                WHERE table_name = 'context_optimizations'
            """).fetchall()
            
            column_names = [col[0] for col in columns]
            assert 'optimization_id' in column_names
            assert 'agent_type' in column_names
            assert 'relevance_score' in column_names
            assert 'optimization_time_ms' in column_names
    
    def test_context_cache_metadata_table(self, database_schema, temp_db_path):
        """Test context cache metadata table structure"""
        db_path = os.path.join(temp_db_path, "context_intelligence.duckdb")
        
        with duckdb.connect(db_path) as conn:
            # Insert test data
            conn.execute("""
                INSERT INTO context_cache_metadata 
                (cache_key, cache_level, data_type, size_bytes, hit_count)
                VALUES ('test_key', 'L2', 'context_data', 1024, 5)
            """)
            
            # Verify data can be retrieved
            result = conn.execute("""
                SELECT cache_key, cache_level, hit_count FROM context_cache_metadata
                WHERE cache_key = 'test_key'
            """).fetchone()
            
            assert result is not None
            assert result[0] == 'test_key'
            assert result[1] == 'L2'
            assert result[2] == 5
    
    def test_agent_context_deliveries_table(self, database_schema, temp_db_path):
        """Test agent context deliveries table functionality"""
        db_path = os.path.join(temp_db_path, "context_intelligence.duckdb")
        
        with duckdb.connect(db_path) as conn:
            # Insert test delivery record
            conn.execute("""
                INSERT INTO agent_context_deliveries 
                (delivery_id, request_id, agent_type, response_time_ms, cache_hit, total_context_size)
                VALUES ('test_delivery', 'req_123', 'rif-implementer', 150.5, true, 2048)
            """)
            
            # Test query performance tracking
            result = conn.execute("""
                SELECT agent_type, response_time_ms, cache_hit 
                FROM agent_context_deliveries
                WHERE delivery_id = 'test_delivery'
            """).fetchone()
            
            assert result[0] == 'rif-implementer'
            assert result[1] == 150.5
            assert result[2] is True
    
    def test_performance_metrics_insertion(self, database_schema):
        """Test performance metrics can be inserted and retrieved"""
        test_metric = {
            'metric_id': 'test_metric_001',
            'service_name': 'context-optimization',
            'operation_name': 'optimize_context',
            'duration_ms': 125.7,
            'success': True,
            'cache_level': 'L2',
            'request_size': 1024
        }
        
        # Insert metric
        database_schema.insert_performance_metric(test_metric)
        
        # Verify insertion using direct SQLite connection
        with sqlite3.connect(database_schema.performance_db_path) as conn:
            cursor = conn.execute("""
                SELECT service_name, duration_ms, success 
                FROM performance_metrics 
                WHERE metric_id = ?
            """, (test_metric['metric_id'],))
            
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 'context-optimization'
            assert result[1] == 125.7
            assert result[2] == 1  # Boolean stored as integer
    
    def test_context_optimization_analytics(self, database_schema):
        """Test context optimization analytics functionality"""
        # Insert test optimization data
        test_optimization = {
            'optimization_id': 'opt_test_001',
            'agent_type': 'rif-implementer',
            'task_context': {'description': 'Test optimization'},
            'original_context_size': 5000,
            'optimized_context_size': 2500,
            'relevance_score': 0.85,
            'optimization_time_ms': 145.2,
            'cache_hit': False,
            'context_items': ['item1', 'item2']
        }
        
        database_schema.insert_context_optimization(test_optimization)
        
        # Get analytics
        analytics = database_schema.get_context_optimization_analytics(hours=1)
        
        assert 'agent_statistics' in analytics
        assert 'performance_trends' in analytics
        assert analytics['time_period_hours'] == 1
        
        if analytics['agent_statistics']:
            stats = analytics['agent_statistics'][0]
            assert stats['agent_type'] == 'rif-implementer'
            assert stats['optimization_count'] == 1
    
    # Database Performance Tests
    
    @pytest.mark.asyncio
    async def test_database_caching_layer_initialization(self, caching_layer):
        """Test database caching layer initializes correctly"""
        assert caching_layer is not None
        assert len(caching_layer.l1_cache) == 0
        assert caching_layer.max_memory_cache_size == 100
        assert caching_layer.default_ttl == 300
    
    @pytest.mark.asyncio
    async def test_l1_cache_operations(self, caching_layer):
        """Test L1 (memory) cache operations"""
        test_data = {"agent_type": "rif-implementer", "context": "test data"}
        
        # Test cache set
        success = await caching_layer.set_cached_data("test_key", test_data, ttl=300, cache_levels=[CacheLevel.L1])
        assert success is True
        assert len(caching_layer.l1_cache) == 1
        
        # Test cache get
        cached_data = await caching_layer.get_cached_data("test_key", cache_levels=[CacheLevel.L1])
        assert cached_data is not None
        assert cached_data["agent_type"] == "rif-implementer"
    
    @pytest.mark.asyncio
    async def test_l2_cache_operations(self, caching_layer):
        """Test L2 (SQLite) cache operations"""
        test_data = {"performance_data": {"response_time": 125.5}}
        
        # Test L2 cache set
        success = await caching_layer.set_cached_data("l2_test_key", test_data, ttl=600, cache_levels=[CacheLevel.L2])
        assert success is True
        
        # Test L2 cache get
        cached_data = await caching_layer.get_cached_data("l2_test_key", cache_levels=[CacheLevel.L2])
        assert cached_data is not None
        assert cached_data["performance_data"]["response_time"] == 125.5
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_100ms(self, caching_layer):
        """Test cache operations meet <100ms performance target"""
        test_data = {"optimization_result": "cached context data"}
        
        # Set data first
        await caching_layer.set_cached_data("perf_test_key", test_data, ttl=300)
        
        # Test cache retrieval performance
        start_time = time.time()
        cached_data = await caching_layer.get_cached_data("perf_test_key")
        cache_time = (time.time() - start_time) * 1000
        
        assert cache_time < 100, f"Cache retrieval took {cache_time:.2f}ms, exceeds 100ms target"
        assert cached_data is not None
    
    @pytest.mark.asyncio
    async def test_cache_hierarchy_promotion(self, caching_layer):
        """Test cache hierarchy promotion (L3 -> L2 -> L1)"""
        test_data = {"hierarchical_data": "test promotion"}
        
        # Set data in L3 only
        await caching_layer.set_cached_data("hierarchy_key", test_data, cache_levels=[CacheLevel.L3])
        
        # Get data - should promote to L2 and L1
        cached_data = await caching_layer.get_cached_data("hierarchy_key")
        
        assert cached_data is not None
        
        # Check data is now in L1 cache
        l1_data = await caching_layer.get_cached_data("hierarchy_key", cache_levels=[CacheLevel.L1])
        assert l1_data is not None
        assert l1_data["hierarchical_data"] == "test promotion"
    
    @pytest.mark.asyncio
    async def test_cache_expiration_handling(self, caching_layer):
        """Test cache expiration and cleanup"""
        test_data = {"expiring_data": "will expire"}
        
        # Set data with very short TTL
        await caching_layer.set_cached_data("expire_key", test_data, ttl=1)
        
        # Verify data is initially available
        cached_data = await caching_layer.get_cached_data("expire_key")
        assert cached_data is not None
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Data should be expired and return None
        expired_data = await caching_layer.get_cached_data("expire_key")
        assert expired_data is None
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, caching_layer):
        """Test cache invalidation functionality"""
        test_data_1 = {"data": "first"}
        test_data_2 = {"data": "second"}
        
        # Set multiple cache entries
        await caching_layer.set_cached_data("invalidate_1", test_data_1)
        await caching_layer.set_cached_data("invalidate_2", test_data_2)
        
        # Verify data is cached
        assert await caching_layer.get_cached_data("invalidate_1") is not None
        assert await caching_layer.get_cached_data("invalidate_2") is not None
        
        # Invalidate specific key
        invalidated_count = await caching_layer.invalidate_cache(cache_key="invalidate_1")
        assert invalidated_count > 0
        
        # Verify specific key is invalidated but other remains
        assert await caching_layer.get_cached_data("invalidate_1") is None
        assert await caching_layer.get_cached_data("invalidate_2") is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, caching_layer):
        """Test cache handles concurrent operations correctly"""
        async def cache_operation(key: str, data: Dict[str, Any]):
            await caching_layer.set_cached_data(key, data)
            return await caching_layer.get_cached_data(key)
        
        # Create concurrent cache operations
        tasks = [
            cache_operation(f"concurrent_key_{i}", {"index": i, "data": f"test_{i}"})
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = (time.time() - start_time) * 1000
        
        # Verify all operations completed successfully
        assert len(results) == 10
        assert all(result is not None for result in results)
        assert total_time < 500, f"Concurrent operations took {total_time:.2f}ms"
        
        # Verify data integrity
        for i, result in enumerate(results):
            assert result["index"] == i
            assert result["data"] == f"test_{i}"
    
    @pytest.mark.asyncio
    async def test_cache_memory_management(self, caching_layer):
        """Test cache memory management and LRU eviction"""
        # Fill cache beyond capacity
        for i in range(caching_layer.max_memory_cache_size + 20):  # Exceed capacity
            await caching_layer.set_cached_data(f"memory_key_{i}", {"data": f"value_{i}"}, cache_levels=[CacheLevel.L1])
        
        # Cache should not exceed max size due to LRU eviction
        assert len(caching_layer.l1_cache) <= caching_layer.max_memory_cache_size
        
        # Most recent entries should still be cached
        recent_data = await caching_layer.get_cached_data(f"memory_key_{caching_layer.max_memory_cache_size + 19}", cache_levels=[CacheLevel.L1])
        assert recent_data is not None
        
        # Earliest entries should be evicted
        old_data = await caching_layer.get_cached_data("memory_key_0", cache_levels=[CacheLevel.L1])
        assert old_data is None
    
    # Integration Tests
    
    @pytest.mark.asyncio
    async def test_database_schema_caching_integration(self, database_schema, caching_layer):
        """Test integration between database schema and caching layer"""
        # Insert optimization data via schema
        test_optimization = {
            'optimization_id': 'integration_test_001',
            'agent_type': 'rif-validator',
            'task_context': {'description': 'Integration test'},
            'original_context_size': 3000,
            'optimized_context_size': 1500,
            'relevance_score': 0.92,
            'optimization_time_ms': 89.3,
            'cache_hit': True,
            'context_items': ['integration_item']
        }
        
        database_schema.insert_context_optimization(test_optimization)
        
        # Cache the optimization result
        cache_key = f"opt_{test_optimization['agent_type']}_{test_optimization['optimization_id']}"
        await caching_layer.set_cached_data(cache_key, test_optimization)
        
        # Retrieve from cache
        cached_optimization = await caching_layer.get_cached_data(cache_key)
        assert cached_optimization is not None
        assert cached_optimization['agent_type'] == 'rif-validator'
        assert cached_optimization['relevance_score'] == 0.92
    
    @pytest.mark.asyncio
    async def test_performance_analytics_integration(self, database_schema, caching_layer):
        """Test performance analytics with caching"""
        # Generate performance metrics
        for i in range(5):
            metric = {
                'metric_id': f'analytics_metric_{i}',
                'service_name': 'context-optimization',
                'operation_name': f'operation_{i}',
                'duration_ms': 100 + (i * 25),
                'success': True,
                'cache_level': 'L1' if i % 2 == 0 else 'L2'
            }
            database_schema.insert_performance_metric(metric)
        
        # Get performance analytics
        performance_summary = database_schema.get_performance_summary(hours=1)
        assert 'service_performance' in performance_summary
        
        # Cache the analytics result
        analytics_key = "performance_analytics_1h"
        await caching_layer.set_cached_data(analytics_key, performance_summary)
        
        # Verify cached analytics retrieval is fast
        start_time = time.time()
        cached_analytics = await caching_layer.get_cached_data(analytics_key)
        retrieval_time = (time.time() - start_time) * 1000
        
        assert retrieval_time < 50, f"Cached analytics retrieval took {retrieval_time:.2f}ms"
        assert cached_analytics is not None
        assert 'service_performance' in cached_analytics
    
    # Performance Benchmarks
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_database_operations_benchmark(self, database_schema):
        """Benchmark database operations for performance validation"""
        # Benchmark optimization data insertion
        insertion_times = []
        for i in range(50):
            test_optimization = {
                'optimization_id': f'benchmark_opt_{i}',
                'agent_type': 'rif-implementer',
                'task_context': {'description': f'Benchmark test {i}'},
                'original_context_size': 4000,
                'optimized_context_size': 2000,
                'relevance_score': 0.8,
                'optimization_time_ms': 150.0 + i,
                'cache_hit': i % 2 == 0,
                'context_items': [f'item_{i}']
            }
            
            start_time = time.time()
            database_schema.insert_context_optimization(test_optimization)
            insertion_time = (time.time() - start_time) * 1000
            insertion_times.append(insertion_time)
        
        avg_insertion_time = sum(insertion_times) / len(insertion_times)
        max_insertion_time = max(insertion_times)
        
        print(f"\nDatabase Insertion Benchmark:")
        print(f"Average insertion time: {avg_insertion_time:.2f}ms")
        print(f"Maximum insertion time: {max_insertion_time:.2f}ms")
        
        # Assert performance targets
        assert avg_insertion_time < 50, f"Average insertion time {avg_insertion_time:.2f}ms exceeds 50ms target"
        assert max_insertion_time < 200, f"Maximum insertion time {max_insertion_time:.2f}ms too high"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_performance_benchmark(self, caching_layer):
        """Benchmark cache performance across all levels"""
        test_data = {"benchmark_data": "performance_test" * 100}  # Larger test data
        
        # Benchmark L1 cache performance
        l1_times = []
        for i in range(100):
            cache_key = f"l1_benchmark_{i}"
            
            # Set operation
            start_time = time.time()
            await caching_layer.set_cached_data(cache_key, test_data, cache_levels=[CacheLevel.L1])
            set_time = (time.time() - start_time) * 1000
            
            # Get operation
            start_time = time.time()
            await caching_layer.get_cached_data(cache_key, cache_levels=[CacheLevel.L1])
            get_time = (time.time() - start_time) * 1000
            
            l1_times.append({"set": set_time, "get": get_time})
        
        avg_l1_get = sum(op["get"] for op in l1_times) / len(l1_times)
        avg_l1_set = sum(op["set"] for op in l1_times) / len(l1_times)
        
        print(f"\nL1 Cache Benchmark:")
        print(f"Average GET time: {avg_l1_get:.3f}ms")
        print(f"Average SET time: {avg_l1_set:.3f}ms")
        
        # Performance assertions for L1 cache
        assert avg_l1_get < 1.0, f"L1 cache GET too slow: {avg_l1_get:.3f}ms"
        assert avg_l1_set < 5.0, f"L1 cache SET too slow: {avg_l1_set:.3f}ms"
    
    # Cleanup and resource management tests
    
    @pytest.mark.asyncio
    async def test_cache_cleanup_and_maintenance(self, caching_layer):
        """Test cache cleanup and maintenance operations"""
        # Add test data with various expiration times
        await caching_layer.set_cached_data("cleanup_1", {"data": "test1"}, ttl=1)
        await caching_layer.set_cached_data("cleanup_2", {"data": "test2"}, ttl=300)
        await caching_layer.set_cached_data("cleanup_3", {"data": "test3"}, ttl=600)
        
        # Wait for some entries to expire
        await asyncio.sleep(2)
        
        # Run cleanup
        cleanup_results = await caching_layer.cleanup_expired_cache()
        
        assert "l1_cleaned" in cleanup_results
        assert "l2_cleaned" in cleanup_results
        assert "l3_cleaned" in cleanup_results
        
        # Verify expired entries are removed
        expired_data = await caching_layer.get_cached_data("cleanup_1")
        assert expired_data is None
        
        # Verify non-expired entries remain
        valid_data = await caching_layer.get_cached_data("cleanup_2")
        assert valid_data is not None
    
    def test_database_schema_cleanup(self, database_schema):
        """Test database schema cleanup operations"""
        # Insert old test data
        old_optimization = {
            'optimization_id': 'old_opt_001',
            'agent_type': 'rif-implementer',
            'task_context': {'description': 'Old test data'},
            'original_context_size': 1000,
            'optimized_context_size': 500,
            'relevance_score': 0.7,
            'optimization_time_ms': 200.0,
            'cache_hit': False,
            'context_items': ['old_item']
        }
        
        database_schema.insert_context_optimization(old_optimization)
        
        # Run cleanup (with short retention for testing)
        cleanup_results = database_schema.cleanup_old_data(days_to_keep=0)  # Clean everything for test
        
        # In a real scenario, this would clean old data
        # For test purposes, we verify the cleanup function runs without error
        print(f"Cleanup completed: {cleanup_results}")


# Test Runner Configuration
if __name__ == "__main__":
    import pytest
    
    # Run tests with coverage and performance benchmarks
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--cov=database.optimization.caching",  # Coverage for caching
        "--cov=systems.context_database_schema",  # Coverage for schema
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage report
        "-m", "not benchmark"  # Skip benchmark tests in normal run
    ])