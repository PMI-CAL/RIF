#!/usr/bin/env python3
"""
Test Suite for DPIBS Core API Framework
Issue #137: Core API Framework + Context Optimization Engine

Comprehensive tests for API routing, context optimization integration,
and performance validation (<200ms response time requirement).
"""

import pytest
import asyncio
import time
import json
from typing import Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

# Import components under test
import sys
import os
sys.path.append('/Users/cal/DEV/RIF')

from api.core.routing import APIFramework, create_api_framework
from api.context.optimization import ContextOptimizationAPI, create_context_optimization_api


class TestDPIBSAPIFramework:
    """Test suite for DPIBS API Framework"""
    
    @pytest.fixture
    def api_framework(self):
        """Create API framework instance for testing"""
        return create_api_framework(
            knowledge_base_path="/tmp/test_knowledge",
            performance_monitoring=True
        )
    
    @pytest.fixture  
    def test_client(self, api_framework):
        """Create test client for API testing"""
        return TestClient(api_framework.app)
    
    @pytest.fixture
    def context_optimization_api(self):
        """Create context optimization API for testing"""
        return create_context_optimization_api(
            knowledge_base_path="/tmp/test_knowledge",
            performance_monitoring=True
        )
    
    # API Framework Core Tests
    
    def test_api_framework_initialization(self, api_framework):
        """Test API framework initializes correctly"""
        assert api_framework is not None
        assert hasattr(api_framework, 'app')
        assert hasattr(api_framework, 'context_optimizer')
        assert hasattr(api_framework, 'performance_monitoring')
        assert api_framework.performance_monitoring is True
    
    def test_health_check_endpoint(self, test_client):
        """Test health check endpoint responds correctly"""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_context_request_endpoint_valid_agent(self, test_client):
        """Test context request endpoint with valid agent type"""
        request_data = {
            "agent_type": "rif-implementer",
            "task_context": {
                "description": "Test context optimization",
                "complexity": "medium"
            },
            "issue_number": 137
        }
        
        with patch('api.core.routing.ContextOptimizer') as mock_optimizer:
            # Mock context optimizer response
            mock_agent_context = Mock()
            mock_agent_context.relevant_knowledge = [Mock(), Mock()]
            mock_agent_context.total_size = 2500
            mock_agent_context.context_window_utilization = 0.75
            
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize_for_agent.return_value = mock_agent_context
            mock_optimizer_instance.format_context_for_agent.return_value = "Formatted context"
            mock_optimizer.return_value = mock_optimizer_instance
            
            response = test_client.post("/api/v1/context/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_type"] == "rif-implementer"
        assert "context" in data
        assert "metadata" in data
        assert "performance" in data
        assert data["metadata"]["total_items"] == 2
        assert data["metadata"]["context_size"] == 2500
    
    def test_context_request_endpoint_invalid_agent(self, test_client):
        """Test context request endpoint with invalid agent type"""
        request_data = {
            "agent_type": "invalid-agent",
            "task_context": {"description": "Test"}
        }
        
        response = test_client.post("/api/v1/context/request", json=request_data)
        assert response.status_code == 400
        assert "Invalid agent type" in response.json()["detail"]
    
    def test_agent_context_endpoint(self, test_client):
        """Test agent-specific context endpoint"""
        with patch('api.core.routing.ContextOptimizer') as mock_optimizer:
            mock_agent_context = Mock()
            mock_agent_context.relevant_knowledge = []
            mock_agent_context.total_size = 1000
            
            mock_optimizer_instance = Mock()
            mock_optimizer_instance.optimize_for_agent.return_value = mock_agent_context
            mock_optimizer_instance.format_context_for_agent.return_value = "Agent context"
            mock_optimizer.return_value = mock_optimizer_instance
            
            response = test_client.get("/api/v1/context/agent/rif-analyst?issue_number=137")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "rif-analyst"
        assert "context" in data
        assert "metadata" in data
    
    def test_context_optimization_endpoint(self, test_client):
        """Test context optimization feedback endpoint"""
        optimization_data = {
            "agent_type": "rif-implementer", 
            "context_used": True,
            "decisions_made": 5,
            "problems_found": 2
        }
        
        with patch('api.core.routing.ContextOptimizer') as mock_optimizer:
            mock_optimizer_instance = Mock()
            mock_optimizer.return_value = mock_optimizer_instance
            
            response = test_client.put("/api/v1/context/optimize/test-session", json=optimization_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["optimization_applied"] is True
    
    def test_performance_metrics_endpoint(self, test_client):
        """Test performance metrics endpoint"""
        # Add some mock metrics to the framework
        with patch.object(TestClient, 'get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "total_requests": 100,
                "average_duration_ms": 150.5,
                "target_compliance": {"under_200ms": 95, "compliance_rate": 0.95}
            }
            
            response = test_client.get("/api/v1/metrics/performance")
        
        # For real test, check actual response
        response = test_client.get("/api/v1/metrics/performance")
        assert response.status_code == 200
    
    # Performance Tests
    
    @pytest.mark.asyncio
    async def test_context_optimization_performance(self, context_optimization_api):
        """Test context optimization meets <200ms performance target"""
        request_data = {
            "agent_type": "rif-implementer",
            "task_context": {
                "description": "Performance test context optimization",
                "complexity": "medium",
                "type": "implementation"
            },
            "issue_number": 137
        }
        
        # Warm up caches
        await context_optimization_api.optimize_agent_context(**request_data)
        
        # Performance test - should be <200ms
        start_time = time.time()
        result = await context_optimization_api.optimize_agent_context(**request_data)
        optimization_time = (time.time() - start_time) * 1000
        
        assert optimization_time < 200, f"Optimization took {optimization_time:.2f}ms, exceeds 200ms target"
        assert result is not None
        assert "performance" in result
        assert result["performance"]["target_met"] is True
    
    @pytest.mark.asyncio
    async def test_batch_context_optimization(self, context_optimization_api):
        """Test batch context optimization performance"""
        requests = [
            {
                "agent_type": "rif-implementer",
                "task_context": {"description": f"Test request {i}"},
                "use_cache": True
            }
            for i in range(5)
        ]
        
        start_time = time.time()
        results = await context_optimization_api.batch_optimize_contexts(requests, max_concurrent=4)
        batch_time = (time.time() - start_time) * 1000
        
        assert len(results) == 5
        assert batch_time < 1000, f"Batch optimization took {batch_time:.2f}ms, too slow"
        
        # Check individual results
        for result in results:
            if not isinstance(result, dict) or "error" not in result:
                assert "performance" in result
    
    # Integration Tests
    
    def test_middleware_performance_tracking(self, test_client):
        """Test that performance middleware tracks request metrics"""
        # Make a request that will be tracked
        response = test_client.get("/api/v1/health")
        
        # Check that performance headers are present
        assert "X-Response-Time" in response.headers
        assert "X-Request-ID" in response.headers
        
        # Verify response time is reasonable
        response_time = float(response.headers["X-Response-Time"].replace("ms", ""))
        assert response_time < 100  # Health check should be very fast
    
    def test_error_handling_standardization(self, test_client):
        """Test that errors are handled with standardized format"""
        # Test 404 error
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test validation error  
        response = test_client.post("/api/v1/context/request", json={"invalid": "data"})
        assert response.status_code == 422  # FastAPI validation error
    
    @pytest.mark.asyncio
    async def test_context_cache_efficiency(self, context_optimization_api):
        """Test context caching improves performance"""
        request_data = {
            "agent_type": "rif-implementer", 
            "task_context": {"description": "Cache efficiency test"},
            "use_cache": True
        }
        
        # First request (cache miss)
        start_time = time.time()
        result1 = await context_optimization_api.optimize_agent_context(**request_data)
        first_time = (time.time() - start_time) * 1000
        
        # Second request (should be cache hit)
        start_time = time.time()
        result2 = await context_optimization_api.optimize_agent_context(**request_data) 
        second_time = (time.time() - start_time) * 1000
        
        # Cache hit should be significantly faster
        assert second_time < first_time * 0.5, f"Cache hit ({second_time:.2f}ms) not much faster than miss ({first_time:.2f}ms)"
        assert result2["performance"]["cache_hit"] is True
    
    # Edge Case Tests
    
    @pytest.mark.asyncio
    async def test_concurrent_context_requests(self, context_optimization_api):
        """Test handling concurrent context optimization requests"""
        async def make_request(agent_type: str):
            return await context_optimization_api.optimize_agent_context(
                agent_type=agent_type,
                task_context={"description": f"Concurrent test for {agent_type}"}
            )
        
        # Test concurrent requests for different agents
        tasks = [
            make_request("rif-implementer"),
            make_request("rif-analyst"), 
            make_request("rif-validator"),
            make_request("rif-implementer")  # Duplicate for cache testing
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = (time.time() - start_time) * 1000
        
        assert len(results) == 4
        assert all(result is not None for result in results)
        assert total_time < 800, f"Concurrent requests took {total_time:.2f}ms, too slow"
    
    @pytest.mark.asyncio
    async def test_large_context_handling(self, context_optimization_api):
        """Test handling of large context data"""
        large_task_context = {
            "description": "Large context test" * 100,  # Large description
            "complexity": "high",
            "requirements": ["req" + str(i) for i in range(50)],  # Many requirements
            "metadata": {"key" + str(i): "value" + str(i) for i in range(20)}  # Large metadata
        }
        
        result = await context_optimization_api.optimize_agent_context(
            agent_type="rif-implementer",
            task_context=large_task_context
        )
        
        assert result is not None
        assert "context" in result
        assert result["performance"]["response_time_ms"] < 500  # Allow more time for large context
    
    # Quality Tests
    
    @pytest.mark.asyncio
    async def test_context_quality_indicators(self, context_optimization_api):
        """Test context optimization quality indicators"""
        result = await context_optimization_api.optimize_agent_context(
            agent_type="rif-implementer",
            task_context={
                "description": "Quality indicator test",
                "issue_type": "feature_implementation"
            }
        )
        
        assert "quality_indicators" in result
        quality = result["quality_indicators"]
        
        # Check quality scores are valid ranges
        assert 0 <= quality["relevance_score"] <= 1
        assert 0 <= quality["completeness_score"] <= 1 
        assert 0 <= quality["freshness_score"] <= 1
    
    def test_api_documentation_generation(self, api_framework):
        """Test API documentation is generated correctly"""
        app = api_framework.app
        
        # Check OpenAPI schema is available
        openapi_schema = app.openapi()
        assert openapi_schema is not None
        assert "paths" in openapi_schema
        assert "/api/v1/context/request" in openapi_schema["paths"]
        assert "/api/v1/health" in openapi_schema["paths"]
    
    # Reliability Tests
    
    @pytest.mark.asyncio
    async def test_optimization_error_recovery(self, context_optimization_api):
        """Test context optimization error handling and recovery"""
        # Test with invalid context data that might cause errors
        with patch.object(context_optimization_api.context_optimizer, 'optimize_for_agent') as mock_optimize:
            mock_optimize.side_effect = Exception("Simulated optimization error")
            
            with pytest.raises(Exception) as exc_info:
                await context_optimization_api.optimize_agent_context(
                    agent_type="rif-implementer",
                    task_context={"description": "Error test"}
                )
            
            assert "Context optimization failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_cache_performance_optimization(self, context_optimization_api):
        """Test cache performance optimization functionality"""
        # Add some test data to cache
        await context_optimization_api.set_cached_data(
            "test_key_1", {"data": "test1"}, ttl=1  # Short TTL
        )
        await context_optimization_api.set_cached_data(
            "test_key_2", {"data": "test2"}, ttl=300
        )
        
        # Wait for first entry to expire
        await asyncio.sleep(2)
        
        # Run cache optimization
        optimization_result = await context_optimization_api.optimize_cache_performance()
        
        assert "cache_stats" in optimization_result
        assert "optimization_time_ms" in optimization_result
        assert optimization_result["optimization_time_ms"] < 100


# Performance Benchmarking Test Class
class TestDPIBSPerformanceBenchmarks:
    """Performance benchmarking tests for DPIBS API Framework"""
    
    @pytest.fixture
    def benchmark_api(self):
        """Create API framework optimized for benchmarking"""
        return create_api_framework(
            knowledge_base_path="/tmp/benchmark_knowledge",
            performance_monitoring=True
        )
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_context_optimization_benchmark(self, benchmark_api):
        """Benchmark context optimization under various loads"""
        context_api = create_context_optimization_api(
            knowledge_base_path="/tmp/benchmark_knowledge"
        )
        
        test_scenarios = [
            {"agent_type": "rif-implementer", "complexity": "low"},
            {"agent_type": "rif-analyst", "complexity": "medium"}, 
            {"agent_type": "rif-validator", "complexity": "high"}
        ]
        
        benchmark_results = []
        
        for scenario in test_scenarios:
            times = []
            for _ in range(10):  # 10 iterations per scenario
                start_time = time.time()
                await context_api.optimize_agent_context(
                    agent_type=scenario["agent_type"],
                    task_context={"description": f"Benchmark test", "complexity": scenario["complexity"]}
                )
                times.append((time.time() - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            benchmark_results.append({
                "scenario": scenario,
                "avg_time_ms": avg_time,
                "max_time_ms": max_time,
                "min_time_ms": min_time,
                "target_met": avg_time < 200
            })
        
        # Assert all scenarios meet performance targets
        for result in benchmark_results:
            assert result["target_met"], f"Scenario {result['scenario']} failed: {result['avg_time_ms']:.2f}ms > 200ms"
            assert result["max_time_ms"] < 500, f"Max time too high: {result['max_time_ms']:.2f}ms"
        
        print("\nBenchmark Results:")
        for result in benchmark_results:
            print(f"{result['scenario']}: avg={result['avg_time_ms']:.2f}ms, max={result['max_time_ms']:.2f}ms")


# Test Runner Configuration
if __name__ == "__main__":
    import pytest
    
    # Run tests with coverage and detailed output
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--cov=api.core.routing",  # Coverage for routing
        "--cov=api.context.optimization",  # Coverage for optimization
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage report
        "-m", "not benchmark"  # Skip benchmark tests in normal run
    ])