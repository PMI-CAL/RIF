"""
Comprehensive MCP Integration Tests

Main integration test suite implementing the requirements from GitHub Issue #86.
Tests parallel query performance, failure recovery, and system resilience.

Issue: #86 - Build MCP integration tests
Component: Core Integration Tests
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

from .test_base import MCPIntegrationTestBase, IntegrationTestConfig, MockServerManager, managed_mock_servers
from .enhanced_mock_server import MockServerConfig, HealthState
from .performance_metrics import PerformanceMetrics, default_collector
from .mock_response_templates import ResponseScenario, TestScenarios


class MCPIntegrationTests(MCPIntegrationTestBase):
    """
    Main integration test class implementing requirements from Issue #86
    
    Tests:
    - Parallel query performance
    - Failure recovery mechanisms
    - Server health monitoring
    - Throughput benchmarking
    """
    
    def __init__(self):
        """Initialize integration tests with default configuration"""
        config = IntegrationTestConfig(
            test_name="mcp_integration_tests",
            timeout_seconds=30.0,
            max_concurrent_requests=20,
            expected_success_rate=0.95,
            max_response_time_ms=1000.0,
            enable_metrics_collection=True,
            mock_server_configs={
                'github': MockServerConfig(
                    server_id='mock_github',
                    server_type='github',
                    name='GitHub MCP Server',
                    capabilities=['repository_info', 'issues', 'pull_requests'],
                    base_response_time_ms=150,
                    max_concurrent_requests=15
                ),
                'memory': MockServerConfig(
                    server_id='mock_memory',
                    server_type='memory',
                    name='Memory MCP Server',
                    capabilities=['memory_storage', 'retrieval', 'search'],
                    base_response_time_ms=80,
                    max_concurrent_requests=20
                ),
                'sequential_thinking': MockServerConfig(
                    server_id='mock_thinking',
                    server_type='sequential_thinking',
                    name='Sequential Thinking MCP Server',
                    capabilities=['reasoning', 'analysis'],
                    base_response_time_ms=300,
                    max_concurrent_requests=5
                )
            }
        )
        super().__init__(config)
    
    async def setup_mock_servers(self):
        """Setup mock servers with enhanced configuration"""
        await super().setup_mock_servers()
        
        # Verify all servers are healthy
        for server_name, server in self.mock_servers.items():
            health = await server.health_check()
            assert health == HealthState.HEALTHY.value, f"Server {server_name} not healthy: {health}"
    
    async def test_parallel_query_performance(self) -> Dict[str, Any]:
        """Test parallel query performance as specified in Issue #86"""
        
        async def parallel_query_scenario():
            """Execute parallel queries to multiple servers"""
            github_server = self.get_mock_server('github')
            memory_server = self.get_mock_server('memory')
            thinking_server = self.get_mock_server('sequential_thinking')
            
            # Simulate MCP Context Aggregator behavior
            tasks = [
                github_server.execute_tool('get_repository_info', {'repo': 'PMI-CAL/RIF'}),
                github_server.execute_tool('list_issues', {'state': 'open'}),
                memory_server.execute_tool('retrieve_memory', {'key': 'integration_patterns'}),
                memory_server.execute_tool('search_memory', {'query': 'MCP testing'}),
                thinking_server.execute_tool('start_reasoning', {'problem': 'integration strategy'})
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            # Verify results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            return {
                'duration': duration,
                'total_queries': len(tasks),
                'successful_queries': len(successful_results),
                'results': results
            }
        
        # Execute test with metrics
        result = await self.execute_with_metrics(
            parallel_query_scenario,
            "parallel_query_performance"
        )
        
        # Assertions from Issue #86
        assert result['duration'] < 1.0, f"Parallel queries took {result['duration']:.3f}s, should be < 1.0s"
        assert result['successful_queries'] >= 4, f"Only {result['successful_queries']}/5 queries succeeded"
        
        return {
            "test": "parallel_query_performance",
            "status": "passed",
            "duration_seconds": result['duration'],
            "success_rate": result['successful_queries'] / result['total_queries'],
            "requirements_met": result['duration'] < 1.0
        }
    
    async def test_failure_recovery(self) -> Dict[str, Any]:
        """Test failure recovery mechanisms as specified in Issue #86"""
        
        github_server = self.get_mock_server('github')
        
        # Simulate server failure
        original_health = await github_server.health_check()
        assert original_health == HealthState.HEALTHY.value
        
        # Set server to unhealthy state
        github_server.set_health(False)
        failed_health = await github_server.health_check()
        assert failed_health == HealthState.UNHEALTHY.value
        
        # Simulate monitoring system detecting failure and triggering recovery
        recovery_start = time.time()
        await github_server.restart()  # This simulates MCPHealthMonitor recovery
        recovery_duration = time.time() - recovery_start
        
        # Verify recovery
        recovered_health = await github_server.health_check()
        assert recovered_health == HealthState.HEALTHY.value
        assert github_server.restart_called, "Restart method should have been called"
        
        # Test that server works after recovery
        result = await github_server.execute_tool('get_repository_info', {'repo': 'test'})
        assert 'repository' in result
        
        return {
            "test": "failure_recovery",
            "status": "passed",
            "recovery_duration_seconds": recovery_duration,
            "server_recovered": recovered_health == HealthState.HEALTHY.value,
            "restart_called": github_server.restart_called,
            "post_recovery_functional": 'repository' in result
        }
    
    async def test_server_health_monitoring(self) -> Dict[str, Any]:
        """Test continuous health monitoring capabilities"""
        
        health_checks = []
        
        for server_name, server in self.mock_servers.items():
            # Check initial health
            initial_health = await server.health_check()
            health_checks.append({
                'server': server_name,
                'timestamp': time.time(),
                'health': initial_health,
                'status': 'initial_check'
            })
            
            # Simulate load to test degradation
            await server.execute_tool('test_load', {})
            load_health = await server.health_check()
            health_checks.append({
                'server': server_name,
                'timestamp': time.time(),
                'health': load_health,
                'status': 'under_load'
            })
        
        # Verify health monitoring works
        healthy_servers = [hc for hc in health_checks if hc['health'] == HealthState.HEALTHY.value]
        
        return {
            "test": "server_health_monitoring",
            "status": "passed",
            "total_health_checks": len(health_checks),
            "healthy_checks": len(healthy_servers),
            "health_monitoring_functional": len(healthy_servers) > 0,
            "detailed_checks": health_checks
        }
    
    async def benchmark_throughput(self) -> List[Dict[str, Any]]:
        """Benchmark throughput as specified in Issue #86"""
        
        results = []
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            # Prepare benchmark scenario
            async def benchmark_operation():
                github_server = self.get_mock_server('github')
                return await github_server.execute_tool('list_issues', {'state': 'open'})
            
            # Execute concurrent requests
            start_time = time.time()
            tasks = [benchmark_operation() for _ in range(concurrency * 2)]  # 2 requests per concurrency level
            benchmark_results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            # Analyze results
            successful_requests = [r for r in benchmark_results if not isinstance(r, Exception)]
            throughput = len(successful_requests) / duration if duration > 0 else 0
            
            results.append({
                'concurrency': concurrency,
                'total_requests': len(tasks),
                'successful_requests': len(successful_requests),
                'duration_seconds': duration,
                'throughput_requests_per_second': throughput,
                'success_rate': len(successful_requests) / len(tasks) if tasks else 0
            })
        
        return results
    
    async def run_test_scenario(self) -> Dict[str, Any]:
        """Run complete integration test scenario"""
        results = {}
        
        # Run all test scenarios
        try:
            results['parallel_query_performance'] = await self.test_parallel_query_performance()
        except Exception as e:
            results['parallel_query_performance'] = {'status': 'failed', 'error': str(e)}
        
        try:
            results['failure_recovery'] = await self.test_failure_recovery()
        except Exception as e:
            results['failure_recovery'] = {'status': 'failed', 'error': str(e)}
        
        try:
            results['health_monitoring'] = await self.test_server_health_monitoring()
        except Exception as e:
            results['health_monitoring'] = {'status': 'failed', 'error': str(e)}
        
        try:
            results['throughput_benchmarks'] = await self.benchmark_throughput()
        except Exception as e:
            results['throughput_benchmarks'] = {'status': 'failed', 'error': str(e)}
        
        # Add performance requirements check
        try:
            self.assert_performance_requirements()
            results['performance_requirements'] = {'status': 'passed'}
        except Exception as e:
            results['performance_requirements'] = {'status': 'failed', 'error': str(e)}
        
        return results


# Pytest test functions implementing the Issue #86 requirements

@pytest.mark.asyncio
async def test_setup_mock_servers():
    """Test that mock servers can be set up correctly"""
    async with managed_mock_servers(
        {'type': 'github', 'name': 'github'},
        {'type': 'memory', 'name': 'memory'},
        {'type': 'sequential_thinking', 'name': 'sequential_thinking'}
    ) as manager:
        
        # Verify servers are created
        assert manager.get_server('github') is not None
        assert manager.get_server('memory') is not None
        assert manager.get_server('sequential_thinking') is not None
        
        # Verify servers are healthy
        for server_name in ['github', 'memory', 'sequential_thinking']:
            server = manager.get_server(server_name)
            health = await server.health_check()
            assert health == HealthState.HEALTHY.value


@pytest.mark.asyncio
async def test_parallel_query_performance():
    """Main test for parallel query performance from Issue #86"""
    integration_test = MCPIntegrationTests()
    await integration_test.setup_method()
    
    try:
        result = await integration_test.test_parallel_query_performance()
        
        # Verify Issue #86 requirements
        assert result['status'] == 'passed'
        assert result['duration_seconds'] < 1.0
        assert result['success_rate'] >= 0.8  # Allow some flexibility for test environment
        
    finally:
        await integration_test.teardown_method()


@pytest.mark.asyncio
async def test_failure_recovery():
    """Main test for failure recovery from Issue #86"""
    integration_test = MCPIntegrationTests()
    await integration_test.setup_method()
    
    try:
        result = await integration_test.test_failure_recovery()
        
        # Verify Issue #86 requirements
        assert result['status'] == 'passed'
        assert result['server_recovered'] is True
        assert result['restart_called'] is True
        assert result['post_recovery_functional'] is True
        
    finally:
        await integration_test.teardown_method()


@pytest.mark.asyncio
async def test_benchmark_throughput():
    """Main test for throughput benchmarking from Issue #86"""
    integration_test = MCPIntegrationTests()
    await integration_test.setup_method()
    
    try:
        results = await integration_test.benchmark_throughput()
        
        # Verify Issue #86 requirements
        assert len(results) == 4  # Should test concurrency levels [1, 5, 10, 20]
        
        # Check that throughput generally increases with concurrency (within reason)
        concurrency_1 = next(r for r in results if r['concurrency'] == 1)
        concurrency_5 = next(r for r in results if r['concurrency'] == 5)
        
        assert concurrency_1['throughput_requests_per_second'] > 0
        assert concurrency_5['throughput_requests_per_second'] >= concurrency_1['throughput_requests_per_second']
        
        # All tests should have reasonable success rates
        for result in results:
            assert result['success_rate'] >= 0.8
        
    finally:
        await integration_test.teardown_method()


@pytest.mark.asyncio
async def test_comprehensive_integration_scenario():
    """Comprehensive test running all scenarios from Issue #86"""
    integration_test = MCPIntegrationTests()
    await integration_test.setup_method()
    
    try:
        results = await integration_test.run_test_scenario()
        
        # Verify all scenarios ran
        assert 'parallel_query_performance' in results
        assert 'failure_recovery' in results
        assert 'health_monitoring' in results
        assert 'throughput_benchmarks' in results
        assert 'performance_requirements' in results
        
        # Count successful scenarios
        successful_scenarios = sum(
            1 for result in results.values() 
            if isinstance(result, dict) and result.get('status') == 'passed'
        )
        
        # At least 80% of scenarios should pass
        assert successful_scenarios >= len(results) * 0.8
        
        # Get final test results
        test_results = integration_test.get_test_results()
        assert test_results['test_name'] == 'mcp_integration_tests'
        assert 'performance_metrics' in test_results
        assert 'server_metrics' in test_results
        
    finally:
        await integration_test.teardown_method()


@pytest.mark.asyncio
async def test_mcp_context_aggregator_simulation():
    """Simulate MCP Context Aggregator behavior for integration testing"""
    
    # This test simulates the behavior that would be tested once Issue #85 is complete
    async with managed_mock_servers(
        {'type': 'github', 'name': 'github'},
        {'type': 'memory', 'name': 'memory'}
    ) as manager:
        
        github_server = manager.get_server('github')
        memory_server = manager.get_server('memory')
        
        # Simulate context aggregation request
        aggregator_query = "Get PR info and related issues"
        required_servers = ['github', 'memory']
        
        # Execute parallel queries (simulating aggregator)
        start_time = time.time()
        tasks = [
            github_server.execute_tool('get_pull_request', {'pr_number': 1}),
            github_server.execute_tool('list_issues', {'state': 'open'}),
            memory_server.execute_tool('get_context', {'query': aggregator_query})
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Verify simulated aggregator behavior
        assert duration < 1.0, f"Aggregator simulation took {duration:.3f}s"
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2, "At least 2/3 aggregator queries should succeed"
        
        # Verify we got expected data structures
        github_results = [r for r in successful_results if 'pull_request' in r or 'issues' in r]
        memory_results = [r for r in successful_results if 'context' in r]
        
        assert len(github_results) >= 1, "Should get GitHub data"
        assert len(memory_results) >= 1, "Should get memory/context data"


# Performance regression test
@pytest.mark.asyncio 
async def test_performance_regression():
    """Test for performance regressions in MCP integration"""
    
    # Create baseline metrics
    baseline_metrics = PerformanceMetrics("baseline_test")
    current_metrics = PerformanceMetrics("regression_test")
    
    async with managed_mock_servers(
        {'type': 'github', 'name': 'github', 'config': {'base_response_time_ms': 100}}
    ) as manager:
        
        github_server = manager.get_server('github')
        
        # Establish baseline
        for _ in range(10):
            start_time = time.time()
            await github_server.execute_tool('get_repository_info', {'repo': 'test'})
            duration_ms = (time.time() - start_time) * 1000
            baseline_metrics.record_operation("get_repository_info", duration_ms, True)
        
        # Current performance test
        for _ in range(10):
            start_time = time.time()
            await github_server.execute_tool('get_repository_info', {'repo': 'test'})
            duration_ms = (time.time() - start_time) * 1000
            current_metrics.record_operation("get_repository_info", duration_ms, True)
        
        # Compare performance
        comparison = current_metrics.get_benchmark_comparison(baseline_metrics)
        
        # Assert no significant regression (allow 10% variance)
        assert comparison['overall_performance'] != 'regressed', "Performance regression detected"
        assert not comparison['performance_regression']['response_time_regression'], "Response time regressed"
        assert not comparison['performance_regression']['success_rate_regression'], "Success rate regressed"


if __name__ == "__main__":
    # Run tests directly for development
    asyncio.run(test_parallel_query_performance())
    asyncio.run(test_failure_recovery()) 
    asyncio.run(test_benchmark_throughput())
    print("All integration tests completed successfully!")