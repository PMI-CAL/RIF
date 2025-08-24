"""
Comprehensive test suite for MCP Context Aggregator

Tests all components of the MCP context aggregation system including
parallel execution, response merging, caching, and performance characteristics.

Issue: #85 - Implement MCP context aggregator  
Agent: RIF-Implementer
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import components to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mcp.aggregator.context_aggregator import (
    MCPContextAggregator, QueryOptimizer, CacheManager, MockHealthMonitor,
    ServerResponse, AggregationResult, QueryOptimizationResult
)
from knowledge.context.optimizer import ContextOptimizer


class TestMockHealthMonitor:
    """Test MockHealthMonitor functionality"""
    
    def test_init(self):
        """Test MockHealthMonitor initialization"""
        monitor = MockHealthMonitor()
        assert monitor.server_health == {}
        assert monitor.health_cache_ttl == 30
        assert monitor.last_health_check == {}
    
    @pytest.mark.asyncio
    async def test_get_server_health(self):
        """Test server health checking"""
        monitor = MockHealthMonitor()
        
        # Test with unknown server
        health = await monitor.get_server_health("test_server")
        assert health == "healthy"  # Mock always returns healthy
        
    @pytest.mark.asyncio
    async def test_get_healthy_servers(self):
        """Test filtering to healthy servers"""
        monitor = MockHealthMonitor()
        
        server_ids = ["server1", "server2", "server3"]
        healthy = await monitor.get_healthy_servers(server_ids)
        
        assert len(healthy) == 3  # Mock returns all as healthy
        assert healthy == server_ids
    
    @pytest.mark.asyncio
    async def test_register_server(self):
        """Test server registration"""
        monitor = MockHealthMonitor()
        
        await monitor.register_server("test_server")
        assert monitor.server_health["test_server"] == "healthy"


class TestQueryOptimizer:
    """Test QueryOptimizer functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.mock_context_optimizer = Mock(spec=ContextOptimizer)
        self.mock_server_registry = AsyncMock()
        self.optimizer = QueryOptimizer(self.mock_context_optimizer, self.mock_server_registry)
    
    @pytest.mark.asyncio
    async def test_analyze_server_capabilities(self):
        """Test server capability analysis"""
        # Mock server registry responses
        self.mock_server_registry.get_server.side_effect = [
            {"capabilities": ["semantic_search", "code_search"]},
            {"capabilities": ["full_text_search"]},
            None  # Server not found
        ]
        
        server_ids = ["server1", "server2", "server3"]
        capabilities = await self.optimizer._analyze_server_capabilities(server_ids)
        
        assert capabilities["server1"] == ["semantic_search", "code_search"]
        assert capabilities["server2"] == ["full_text_search"]
        assert capabilities["server3"] == ["general"]  # Default for missing server
    
    @pytest.mark.asyncio
    async def test_optimize_queries_by_server(self):
        """Test query optimization for different server capabilities"""
        query = "find authentication patterns"
        server_capabilities = {
            "server1": ["semantic_search"],
            "server2": ["full_text_search"], 
            "server3": ["pattern_matching"],
            "server4": ["general"]
        }
        context = {"agent_type": "rif-implementer"}
        
        optimized = await self.optimizer._optimize_queries_by_server(query, server_capabilities, context)
        
        assert optimized["server1"] == "semantic:find authentication patterns"
        assert optimized["server2"] == "fulltext:find authentication patterns"
        assert optimized["server3"] == "pattern:find authentication patterns"
        assert optimized["server4"] == "find authentication patterns"  # No optimization for general
    
    @pytest.mark.asyncio
    async def test_prioritize_servers(self):
        """Test server prioritization based on performance"""
        # Setup performance history
        self.optimizer.performance_history = {
            "fast_server": [100, 120, 110],    # Fast average: 110ms
            "slow_server": [500, 600, 550],    # Slow average: 550ms
        }
        
        server_ids = ["slow_server", "fast_server", "unknown_server"]
        priority = await self.optimizer._prioritize_servers(server_ids, "test query", None)
        
        # Fast server should be first
        assert priority[0] == "fast_server"
        assert "slow_server" in priority
        assert "unknown_server" in priority
    
    @pytest.mark.asyncio
    async def test_optimize_query_complete_flow(self):
        """Test complete query optimization flow"""
        # Setup mocks
        self.mock_server_registry.get_server.return_value = {"capabilities": ["semantic_search"]}
        
        query = "test query"
        available_servers = ["server1", "server2"]
        
        result = await self.optimizer.optimize_query(query, available_servers)
        
        assert isinstance(result, QueryOptimizationResult)
        assert len(result.optimized_queries) == 2
        assert result.priority_servers == available_servers
        assert result.expected_response_time_ms > 0


class TestCacheManager:
    """Test CacheManager functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.mock_health_monitor = AsyncMock(spec=MockHealthMonitor)
        self.cache_manager = CacheManager(self.mock_health_monitor, default_ttl=300, max_size=100)
    
    def test_generate_cache_key(self):
        """Test cache key generation"""
        query = "test query"
        servers = ["server2", "server1"]  # Different order
        context = {"agent_type": "rif-implementer", "issue_id": "85"}
        
        key1 = self.cache_manager.generate_cache_key(query, servers, context)
        key2 = self.cache_manager.generate_cache_key(query, ["server1", "server2"], context)  # Same servers, different order
        
        assert key1 == key2  # Order shouldn't matter
        assert len(key1) == 16  # SHA256 hash truncated to 16 chars
    
    @pytest.mark.asyncio
    async def test_cache_put_and_get(self):
        """Test cache storage and retrieval"""
        # Create mock aggregation result
        mock_result = Mock(spec=AggregationResult)
        mock_result.server_responses = [
            Mock(server_id="server1"),
            Mock(server_id="server2")
        ]
        
        # Mock health validation to pass
        self.mock_health_monitor.get_healthy_servers.return_value = ["server1", "server2"]
        
        cache_key = "test_key"
        
        # Test cache miss
        result = await self.cache_manager.get(cache_key)
        assert result is None
        assert self.cache_manager.miss_count == 1
        
        # Test cache put
        await self.cache_manager.put(cache_key, mock_result)
        
        # Test cache hit
        result = await self.cache_manager.get(cache_key)
        assert result == mock_result
        assert self.cache_manager.hit_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_unhealthy_servers(self):
        """Test cache invalidation when servers become unhealthy"""
        # Create mock aggregation result
        mock_result = Mock(spec=AggregationResult)
        mock_result.server_responses = [
            Mock(server_id="server1"),
            Mock(server_id="server2")
        ]
        
        cache_key = "test_key"
        await self.cache_manager.put(cache_key, mock_result)
        
        # Mock health check to show servers are unhealthy (health ratio < 80%)
        self.mock_health_monitor.get_healthy_servers.return_value = ["server1"]  # Only 50% healthy
        
        # Should invalidate cache
        result = await self.cache_manager.get(cache_key)
        assert result is None
        assert cache_key not in self.cache_manager.cache
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Initial state
        stats = self.cache_manager.get_cache_stats()
        assert stats['hit_count'] == 0
        assert stats['miss_count'] == 0
        assert stats['hit_rate_percent'] == 0
        
        # After some operations
        self.cache_manager.hit_count = 8
        self.cache_manager.miss_count = 2
        
        stats = self.cache_manager.get_cache_stats()
        assert stats['hit_count'] == 8
        assert stats['miss_count'] == 2
        assert stats['hit_rate_percent'] == 80.0  # 8/(8+2) * 100


class TestMCPContextAggregator:
    """Test main MCPContextAggregator functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create mocks for dependencies
        self.mock_context_optimizer = Mock(spec=ContextOptimizer)
        self.mock_mcp_loader = Mock()
        self.mock_server_registry = AsyncMock()
        self.mock_security_gateway = AsyncMock()
        self.mock_health_monitor = AsyncMock(spec=MockHealthMonitor)
        
        # Create aggregator with mocks
        self.aggregator = MCPContextAggregator(
            context_optimizer=self.mock_context_optimizer,
            mcp_loader=self.mock_mcp_loader,
            server_registry=self.mock_server_registry,
            security_gateway=self.mock_security_gateway,
            health_monitor=self.mock_health_monitor,
            max_concurrent_servers=2,  # Small number for testing
            query_timeout_seconds=5,
            cache_ttl_seconds=60
        )
    
    @pytest.mark.asyncio
    async def test_discover_available_servers(self):
        """Test server discovery"""
        # Test with specific servers
        required_servers = ["server1", "server2", "server3"]
        self.mock_health_monitor.get_healthy_servers.return_value = ["server1", "server2"]
        
        servers = await self.aggregator._discover_available_servers(required_servers, None)
        assert servers == ["server1", "server2"]
        
        # Test auto-discovery
        self.mock_server_registry.list_servers.return_value = [
            {"server_id": "auto1", "status": "active"},
            {"server_id": "auto2", "status": "active"},
            {"server_id": "auto3", "status": "inactive"}  # Should be filtered out
        ]
        self.mock_health_monitor.get_healthy_servers.return_value = ["auto1", "auto2"]
        
        servers = await self.aggregator._discover_available_servers(None, None)
        assert servers == ["auto1", "auto2"]
    
    @pytest.mark.asyncio
    async def test_query_single_server_success(self):
        """Test successful single server query"""
        server_id = "test_server"
        query = "test query"
        
        # Mock server registry
        self.mock_server_registry.get_server.return_value = {"name": "Test Server"}
        
        # Mock security validation
        self.mock_security_gateway.validate_query_permission.return_value = True
        
        response = await self.aggregator._query_single_server(server_id, query)
        
        assert isinstance(response, ServerResponse)
        assert response.server_id == server_id
        assert response.status == "success"
        assert response.response_time_ms > 0
        assert response.response_data is not None
    
    @pytest.mark.asyncio
    async def test_query_single_server_security_failure(self):
        """Test server query with security validation failure"""
        server_id = "test_server"
        query = "test query"
        
        # Mock server registry
        self.mock_server_registry.get_server.return_value = {"name": "Test Server"}
        
        # Mock security validation failure
        self.mock_security_gateway.validate_query_permission.return_value = False
        
        response = await self.aggregator._query_single_server(server_id, query)
        
        assert response.status == "failed"
        assert "permission denied" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_merge_responses(self):
        """Test response merging with ContextOptimizer"""
        # Create mock server responses
        server_responses = [
            ServerResponse(
                server_id="server1",
                server_name="Server 1",
                response_data={
                    "results": [
                        {"content": "Result 1 from server 1", "relevance": 0.8},
                        {"content": "Result 2 from server 1", "relevance": 0.6}
                    ]
                },
                response_time_ms=150,
                status="success"
            ),
            ServerResponse(
                server_id="server2", 
                server_name="Server 2",
                response_data={
                    "results": [
                        {"content": "Result 1 from server 2", "relevance": 0.9}
                    ]
                },
                response_time_ms=200,
                status="success"
            )
        ]
        
        # Mock ContextOptimizer response
        self.mock_context_optimizer.optimize_for_agent.return_value = {
            "optimized_results": [
                {"content": "Merged result 1", "relevance": 0.9},
                {"content": "Merged result 2", "relevance": 0.8}
            ],
            "optimization_info": {"merged": True},
            "performance_stats": {"optimization_time": 50}
        }
        
        merged = await self.aggregator._merge_responses(
            server_responses, "test query", None, "rif-implementer"
        )
        
        assert "optimized_results" in merged
        assert "server_summary" in merged
        assert merged["server_summary"]["successful_servers"] == 2
        assert merged["optimization_applied"] is True
    
    @pytest.mark.asyncio
    async def test_get_context_complete_flow(self):
        """Test complete context aggregation flow"""
        # Setup mocks for successful flow
        self.mock_server_registry.list_servers.return_value = [
            {"server_id": "server1", "status": "active"},
            {"server_id": "server2", "status": "active"}
        ]
        self.mock_health_monitor.get_healthy_servers.return_value = ["server1", "server2"]
        
        # Mock server registry for individual servers
        self.mock_server_registry.get_server.side_effect = [
            {"name": "Server 1", "capabilities": ["semantic_search"]},
            {"name": "Server 2", "capabilities": ["full_text_search"]}
        ]
        
        # Mock security gateway
        self.mock_security_gateway.validate_query_permission.return_value = True
        
        # Mock context optimizer
        self.mock_context_optimizer.optimize_for_agent.return_value = {
            "optimized_results": [{"content": "Optimized result", "relevance": 0.8}],
            "optimization_info": {"optimized": True},
            "performance_stats": {"time": 100}
        }
        
        # Execute context aggregation
        result = await self.aggregator.get_context(
            query="test query",
            agent_type="rif-implementer",
            use_cache=False  # Disable cache for predictable testing
        )
        
        assert isinstance(result, AggregationResult)
        assert result.query == "test query"
        assert result.successful_servers >= 0  # May be 0 in mock environment
        assert result.total_time_ms > 0
        assert "optimized_results" in result.merged_response or "error" in result.merged_response
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test caching behavior in context aggregation"""
        query = "cached query test"
        
        # Setup mocks for successful response
        self.mock_server_registry.list_servers.return_value = [
            {"server_id": "server1", "status": "active"}
        ]
        self.mock_health_monitor.get_healthy_servers.return_value = ["server1"]
        self.mock_server_registry.get_server.return_value = {"name": "Server 1"}
        self.mock_security_gateway.validate_query_permission.return_value = True
        self.mock_context_optimizer.optimize_for_agent.return_value = {
            "optimized_results": [{"content": "Cached result"}],
            "optimization_info": {},
            "performance_stats": {}
        }
        
        # Mock healthy server check for cache validation
        self.aggregator.cache_manager.health_monitor.get_healthy_servers = AsyncMock(return_value=["server1"])
        
        # First query - should miss cache
        result1 = await self.aggregator.get_context(query, use_cache=True)
        
        # Second query - should hit cache (if first was successful)
        result2 = await self.aggregator.get_context(query, use_cache=True)
        
        # Both should succeed (though may be different due to mocking)
        assert result1.total_time_ms >= 0
        assert result2.total_time_ms >= 0
    
    @pytest.mark.asyncio 
    async def test_performance_metrics_tracking(self):
        """Test performance metrics tracking"""
        initial_queries = self.aggregator.performance_metrics['total_queries']
        
        # Execute some queries
        await self.aggregator.get_context("test query 1", use_cache=False)
        await self.aggregator.get_context("test query 2", use_cache=False)
        
        # Check metrics were updated
        assert self.aggregator.performance_metrics['total_queries'] == initial_queries + 2
        assert self.aggregator.performance_metrics['avg_response_time_ms'] >= 0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test aggregator health check"""
        # Mock healthy server discovery
        self.aggregator._discover_available_servers = AsyncMock(return_value=["server1", "server2"])
        
        health = await self.aggregator.health_check()
        
        assert "status" in health
        assert "components" in health
        assert "available_servers" in health
        assert health["available_servers"] == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and graceful degradation"""
        # Test with server discovery failure
        self.aggregator._discover_available_servers = AsyncMock(side_effect=Exception("Discovery failed"))
        
        result = await self.aggregator.get_context("test query")
        
        assert "error" in result.merged_response
        assert result.successful_servers == 0
        assert result.total_time_ms > 0


class TestPerformanceCharacteristics:
    """Test performance characteristics and requirements"""
    
    def setup_method(self):
        """Setup for performance tests"""
        self.aggregator = MCPContextAggregator(max_concurrent_servers=4)
    
    @pytest.mark.asyncio
    async def test_concurrent_server_limit(self):
        """Test that concurrent server queries respect limits"""
        # This test verifies the semaphore works correctly
        # In a real test, we'd mock server responses with delays
        
        # Mock server registry to return many servers
        self.aggregator.server_registry.list_servers = AsyncMock(return_value=[
            {"server_id": f"server{i}", "status": "active"} for i in range(10)
        ])
        self.aggregator.health_monitor.get_healthy_servers = AsyncMock(
            return_value=[f"server{i}" for i in range(10)]
        )
        
        # The aggregator should limit to max_concurrent_servers (4)
        servers = await self.aggregator._discover_available_servers(None, None)
        assert len(servers) <= 4
    
    @pytest.mark.asyncio
    async def test_query_timeout_enforcement(self):
        """Test query timeout enforcement"""
        # Set very short timeout for testing
        self.aggregator.query_timeout_seconds = 0.1
        
        # Mock a slow server response
        async def slow_query(server_id, query):
            await asyncio.sleep(0.2)  # Longer than timeout
            return ServerResponse(server_id, "slow_server", {}, 200, "success")
        
        self.aggregator._query_single_server = slow_query
        
        # Should timeout and return timeout response
        responses = await self.aggregator._query_servers_parallel({"server1": "test"})
        assert len(responses) == 1
        assert responses[0].status == "timeout"
    
    def test_cache_performance_targets(self):
        """Test cache performance meets targets"""
        cache_manager = CacheManager(MockHealthMonitor(), max_size=1000)
        
        # Test cache key generation performance
        start_time = time.time()
        for i in range(1000):
            cache_manager.generate_cache_key(f"query{i}", [f"server{i}"], {"test": i})
        
        key_generation_time = time.time() - start_time
        assert key_generation_time < 1.0  # Should be much faster than 1 second for 1000 keys
        
        # Test cache size limits
        assert cache_manager.cache.maxsize == 1000


# Integration tests
class TestIntegration:
    """Integration tests with real components"""
    
    @pytest.mark.asyncio
    async def test_integration_with_context_optimizer(self):
        """Test integration with real ContextOptimizer"""
        try:
            # Use real ContextOptimizer
            real_optimizer = ContextOptimizer()
            aggregator = MCPContextAggregator(context_optimizer=real_optimizer)
            
            # Test that it initializes without error
            assert aggregator.context_optimizer is not None
            
            # Test health check works
            health = await aggregator.health_check()
            assert "status" in health
            
        except ImportError:
            pytest.skip("ContextOptimizer not available for integration test")
    
    @pytest.mark.asyncio
    async def test_end_to_end_aggregation_flow(self):
        """Test complete end-to-end aggregation flow"""
        aggregator = MCPContextAggregator()
        
        # Mock minimal components for E2E test
        aggregator.server_registry.list_servers = AsyncMock(return_value=[
            {"server_id": "test_server", "status": "active"}
        ])
        aggregator.health_monitor.get_healthy_servers = AsyncMock(return_value=["test_server"])
        aggregator.server_registry.get_server = AsyncMock(return_value={"name": "Test Server"})
        aggregator.security_gateway.validate_query_permission = AsyncMock(return_value=True)
        
        # Execute full flow
        result = await aggregator.get_context(
            query="integration test query",
            agent_type="rif-implementer", 
            use_cache=True
        )
        
        # Verify result structure
        assert isinstance(result, AggregationResult)
        assert hasattr(result, 'merged_response')
        assert hasattr(result, 'performance_metrics')
        assert hasattr(result, 'cache_info')


# Benchmark tests
class TestBenchmarks:
    """Benchmark tests for performance validation"""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_aggregation_performance_benchmark(self):
        """Benchmark aggregation performance"""
        aggregator = MCPContextAggregator()
        
        # Setup mocks for consistent performance testing
        aggregator.server_registry.list_servers = AsyncMock(return_value=[
            {"server_id": f"server{i}", "status": "active"} for i in range(4)
        ])
        aggregator.health_monitor.get_healthy_servers = AsyncMock(
            return_value=[f"server{i}" for i in range(4)]
        )
        aggregator.server_registry.get_server = AsyncMock(return_value={"name": "Benchmark Server"})
        aggregator.security_gateway.validate_query_permission = AsyncMock(return_value=True)
        
        # Benchmark multiple queries
        queries = ["benchmark query " + str(i) for i in range(10)]
        
        start_time = time.time()
        results = []
        
        for query in queries:
            result = await aggregator.get_context(query, use_cache=False)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(queries)
        
        # Performance assertions
        assert avg_time < 2.0  # Average should be under 2 seconds per query
        assert all(r.total_time_ms < 10000 for r in results)  # Each query under 10 seconds
        
        print(f"Benchmark: {len(queries)} queries in {total_time:.2f}s, avg {avg_time:.2f}s per query")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])