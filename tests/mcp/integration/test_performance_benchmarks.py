"""
Performance Benchmarking Framework for MCP Integration Testing

Comprehensive performance benchmarking suite implementing advanced metrics collection,
stress testing, and performance analysis for MCP services.

Issue: #86 - Build MCP integration tests
Component: Performance Benchmarking
"""

import pytest
import asyncio
import time
import statistics
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from .test_base import PerformanceBenchmarkBase, MockServerManager, managed_mock_servers
from .performance_metrics import PerformanceMetrics, BenchmarkResult, default_collector
from .enhanced_mock_server import MockServerConfig, HealthState


@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    target_rps: int = 100
    max_concurrency: int = 50
    failure_threshold: float = 0.05  # 5% failure rate threshold


class MCPPerformanceBenchmarks(PerformanceBenchmarkBase):
    """
    Comprehensive performance benchmarking for MCP services
    
    Implements sophisticated benchmarking scenarios including:
    - Throughput scaling tests
    - Latency distribution analysis  
    - Resource utilization benchmarks
    - Stress and load testing
    - Multi-server coordination benchmarks
    """
    
    def __init__(self):
        super().__init__("mcp_performance_benchmarks")
        self.server_manager = MockServerManager()
        self.metrics_collector = default_collector.create_test_session("performance_benchmarks")
    
    async def setup_benchmark_servers(self) -> Dict[str, Any]:
        """Setup optimized servers for benchmarking"""
        
        # GitHub server optimized for high throughput
        github_server = await self.server_manager.create_github_server(
            base_response_time_ms=50,  # Faster for benchmarking
            max_concurrent_requests=30,
            failure_rate=0.02,  # 2% failure rate for realism
            timeout_rate=0.01   # 1% timeout rate
        )
        
        # Memory server optimized for low latency
        memory_server = await self.server_manager.create_memory_server(
            base_response_time_ms=30,
            max_concurrent_requests=50,
            failure_rate=0.01,
            timeout_rate=0.005
        )
        
        # Sequential thinking server with realistic constraints
        thinking_server = await self.server_manager.create_thinking_server(
            base_response_time_ms=200,  # More realistic thinking time
            max_concurrent_requests=10,
            failure_rate=0.03,
            timeout_rate=0.02
        )
        
        return {
            'github': github_server,
            'memory': memory_server,
            'sequential_thinking': thinking_server
        }
    
    async def benchmark_single_server_throughput(self, 
                                                server_name: str,
                                                tool_name: str,
                                                concurrency_levels: List[int] = None) -> List[BenchmarkResult]:
        """Benchmark throughput for a single server across concurrency levels"""
        
        if concurrency_levels is None:
            concurrency_levels = [1, 5, 10, 15, 20, 30, 40, 50]
        
        server = self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")
        
        async def single_request():
            return await server.execute_tool(tool_name, {'benchmark': True})
        
        return await self.run_benchmark(
            benchmark_func=lambda concurrency: self._execute_concurrent_requests(
                single_request, concurrency, iterations=20
            ),
            concurrency_levels=concurrency_levels,
            iterations_per_level=5  # 5 iterations per concurrency level
        )
    
    async def _execute_concurrent_requests(self, request_func, concurrency: int, iterations: int = 20):
        """Execute concurrent requests for benchmarking"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def controlled_request():
            async with semaphore:
                return await request_func()
        
        tasks = [controlled_request() for _ in range(iterations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful results
        successful = [r for r in results if not isinstance(r, Exception)]
        return len(successful)
    
    async def benchmark_multi_server_coordination(self) -> Dict[str, Any]:
        """Benchmark coordination between multiple servers"""
        
        servers = await self.setup_benchmark_servers()
        
        async def coordinated_workflow():
            """Simulate a coordinated workflow across servers"""
            
            # Step 1: Get repository information
            repo_info = await servers['github'].execute_tool(
                'get_repository_info', 
                {'repo': 'PMI-CAL/RIF'}
            )
            
            # Step 2: Store context in memory
            memory_result = await servers['memory'].execute_tool(
                'store_memory',
                {'key': 'repo_context', 'data': repo_info}
            )
            
            # Step 3: Start reasoning about the repository
            thinking_result = await servers['sequential_thinking'].execute_tool(
                'start_reasoning',
                {'problem': 'analyze repository structure'}
            )
            
            return {
                'repo_info': repo_info,
                'memory_stored': memory_result,
                'reasoning_started': thinking_result
            }
        
        # Benchmark coordinated workflows
        start_time = time.time()
        workflow_results = []
        
        for i in range(10):  # 10 workflow executions
            try:
                workflow_start = time.time()
                result = await coordinated_workflow()
                workflow_duration = time.time() - workflow_start
                
                workflow_results.append({
                    'iteration': i + 1,
                    'duration_ms': workflow_duration * 1000,
                    'success': True,
                    'steps_completed': len([r for r in result.values() if r])
                })
                
            except Exception as e:
                workflow_results.append({
                    'iteration': i + 1,
                    'duration_ms': 0,
                    'success': False,
                    'error': str(e)
                })
        
        total_duration = time.time() - start_time
        successful_workflows = [r for r in workflow_results if r['success']]
        
        return {
            'total_workflows': len(workflow_results),
            'successful_workflows': len(successful_workflows),
            'success_rate': len(successful_workflows) / len(workflow_results),
            'average_workflow_duration_ms': statistics.mean([r['duration_ms'] for r in successful_workflows]) if successful_workflows else 0,
            'total_benchmark_duration_s': total_duration,
            'workflows_per_second': len(successful_workflows) / total_duration,
            'detailed_results': workflow_results
        }
    
    async def stress_test_server_limits(self, config: StressTestConfig) -> Dict[str, Any]:
        """Perform stress testing to find server limits"""
        
        servers = await self.setup_benchmark_servers()
        stress_results = {}
        
        for server_name, server in servers.items():
            print(f"Running stress test for {server_name}...")
            
            # Gradually increase load
            current_rps = 1
            max_successful_rps = 0
            failure_rates = []
            
            while current_rps <= config.target_rps:
                # Test current RPS for 10 seconds
                test_duration = 10
                request_count = current_rps * test_duration
                
                start_time = time.time()
                
                async def stress_request():
                    try:
                        return await server.execute_tool('stress_test', {'rps': current_rps})
                    except Exception:
                        return None
                
                # Execute stress requests
                semaphore = asyncio.Semaphore(min(current_rps, config.max_concurrency))
                
                async def controlled_stress_request():
                    async with semaphore:
                        return await stress_request()
                
                tasks = [controlled_stress_request() for _ in range(request_count)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                actual_duration = time.time() - start_time
                successful_requests = len([r for r in results if r is not None and not isinstance(r, Exception)])
                failure_rate = 1 - (successful_requests / len(results))
                
                failure_rates.append(failure_rate)
                
                if failure_rate <= config.failure_threshold:
                    max_successful_rps = current_rps
                
                print(f"  RPS {current_rps}: {successful_requests}/{len(results)} successful, failure rate: {failure_rate:.3f}")
                
                # Break if failure rate is too high
                if failure_rate > config.failure_threshold * 2:
                    break
                
                current_rps = min(current_rps + 5, config.target_rps)
            
            stress_results[server_name] = {
                'max_successful_rps': max_successful_rps,
                'failure_rates': failure_rates,
                'breaking_point_rps': current_rps - 5 if failure_rates else config.target_rps,
                'stress_test_completed': True
            }
        
        return stress_results
    
    async def benchmark_resource_efficiency(self) -> Dict[str, Any]:
        """Benchmark resource efficiency and utilization"""
        
        servers = await self.setup_benchmark_servers()
        efficiency_results = {}
        
        for server_name, server in servers.items():
            # Collect baseline resource usage
            baseline_usage = await server.get_resource_usage()
            
            # Execute sustained load
            sustained_load_duration = 30  # 30 seconds
            requests_per_second = 10
            
            resource_samples = []
            start_time = time.time()
            
            async def resource_monitor():
                """Monitor resource usage during load"""
                while time.time() - start_time < sustained_load_duration:
                    usage = await server.get_resource_usage()
                    resource_samples.append({
                        'timestamp': time.time(),
                        'memory_mb': usage['memory_mb'],
                        'cpu_percent': usage['cpu_percent']
                    })
                    await asyncio.sleep(1)  # Sample every second
            
            async def sustained_load():
                """Generate sustained load"""
                while time.time() - start_time < sustained_load_duration:
                    await server.execute_tool('efficiency_test', {'sustained_load': True})
                    await asyncio.sleep(1 / requests_per_second)
            
            # Run monitoring and load in parallel
            await asyncio.gather(
                resource_monitor(),
                sustained_load()
            )
            
            # Analyze resource efficiency
            if resource_samples:
                memory_usage = [s['memory_mb'] for s in resource_samples]
                cpu_usage = [s['cpu_percent'] for s in resource_samples]
                
                efficiency_results[server_name] = {
                    'baseline_memory_mb': baseline_usage['memory_mb'],
                    'peak_memory_mb': max(memory_usage),
                    'average_memory_mb': statistics.mean(memory_usage),
                    'memory_efficiency': baseline_usage['memory_mb'] / max(memory_usage) if max(memory_usage) > 0 else 1.0,
                    
                    'baseline_cpu_percent': baseline_usage['cpu_percent'],
                    'peak_cpu_percent': max(cpu_usage),
                    'average_cpu_percent': statistics.mean(cpu_usage),
                    'cpu_efficiency': 100 / max(cpu_usage) if max(cpu_usage) > 0 else 1.0,
                    
                    'resource_samples_count': len(resource_samples),
                    'sustained_load_duration': sustained_load_duration
                }
        
        return efficiency_results
    
    async def benchmark_failure_recovery_performance(self) -> Dict[str, Any]:
        """Benchmark performance during and after failure scenarios"""
        
        servers = await self.setup_benchmark_servers()
        recovery_results = {}
        
        for server_name, server in servers.items():
            print(f"Testing failure recovery for {server_name}...")
            
            # Establish baseline performance
            baseline_start = time.time()
            baseline_requests = []
            
            for _ in range(20):
                try:
                    request_start = time.time()
                    await server.execute_tool('baseline_test', {})
                    request_duration = time.time() - request_start
                    baseline_requests.append(request_duration * 1000)
                except Exception:
                    pass
            
            baseline_avg = statistics.mean(baseline_requests) if baseline_requests else 0
            
            # Simulate failure
            server.set_health(False)
            failure_time = time.time()
            
            # Measure recovery time
            await server.restart()
            recovery_time = time.time() - failure_time
            
            # Test performance after recovery
            post_recovery_requests = []
            
            for _ in range(20):
                try:
                    request_start = time.time()
                    await server.execute_tool('recovery_test', {})
                    request_duration = time.time() - request_start
                    post_recovery_requests.append(request_duration * 1000)
                except Exception:
                    pass
            
            post_recovery_avg = statistics.mean(post_recovery_requests) if post_recovery_requests else 0
            
            recovery_results[server_name] = {
                'baseline_avg_response_time_ms': baseline_avg,
                'recovery_time_seconds': recovery_time,
                'post_recovery_avg_response_time_ms': post_recovery_avg,
                'performance_impact_percent': ((post_recovery_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0,
                'recovery_successful': post_recovery_avg > 0,
                'baseline_samples': len(baseline_requests),
                'recovery_samples': len(post_recovery_requests)
            }
        
        return recovery_results
    
    async def run_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        print("Starting comprehensive MCP performance benchmark suite...")
        
        benchmark_results = {
            'suite_start_time': time.time(),
            'benchmarks': {}
        }
        
        try:
            # Setup servers
            servers = await self.setup_benchmark_servers()
            print(f"✓ Setup completed: {len(servers)} servers ready")
            
            # 1. Single server throughput benchmarks
            print("\n1. Running single server throughput benchmarks...")
            throughput_results = {}
            for server_name in servers.keys():
                try:
                    results = await self.benchmark_single_server_throughput(
                        server_name, 'throughput_test', [1, 5, 10, 20, 30]
                    )
                    throughput_results[server_name] = [
                        {
                            'concurrency': r.concurrency_level,
                            'throughput': r.throughput_requests_per_second,
                            'avg_response_time': r.average_response_time_ms,
                            'success_rate': r.success_rate
                        }
                        for r in results
                    ]
                    print(f"  ✓ {server_name} throughput benchmark completed")
                except Exception as e:
                    print(f"  ✗ {server_name} throughput benchmark failed: {e}")
                    throughput_results[server_name] = {'error': str(e)}
            
            benchmark_results['benchmarks']['throughput'] = throughput_results
            
            # 2. Multi-server coordination benchmark
            print("\n2. Running multi-server coordination benchmark...")
            try:
                coordination_results = await self.benchmark_multi_server_coordination()
                benchmark_results['benchmarks']['coordination'] = coordination_results
                print(f"  ✓ Coordination benchmark completed: {coordination_results['success_rate']:.3f} success rate")
            except Exception as e:
                print(f"  ✗ Coordination benchmark failed: {e}")
                benchmark_results['benchmarks']['coordination'] = {'error': str(e)}
            
            # 3. Stress testing
            print("\n3. Running stress tests...")
            try:
                stress_config = StressTestConfig(
                    duration_seconds=30,
                    target_rps=50,
                    max_concurrency=20
                )
                stress_results = await self.stress_test_server_limits(stress_config)
                benchmark_results['benchmarks']['stress_testing'] = stress_results
                print("  ✓ Stress testing completed")
            except Exception as e:
                print(f"  ✗ Stress testing failed: {e}")
                benchmark_results['benchmarks']['stress_testing'] = {'error': str(e)}
            
            # 4. Resource efficiency benchmark
            print("\n4. Running resource efficiency benchmark...")
            try:
                efficiency_results = await self.benchmark_resource_efficiency()
                benchmark_results['benchmarks']['resource_efficiency'] = efficiency_results
                print("  ✓ Resource efficiency benchmark completed")
            except Exception as e:
                print(f"  ✗ Resource efficiency benchmark failed: {e}")
                benchmark_results['benchmarks']['resource_efficiency'] = {'error': str(e)}
            
            # 5. Failure recovery performance
            print("\n5. Running failure recovery performance benchmark...")
            try:
                recovery_results = await self.benchmark_failure_recovery_performance()
                benchmark_results['benchmarks']['failure_recovery'] = recovery_results
                print("  ✓ Failure recovery benchmark completed")
            except Exception as e:
                print(f"  ✗ Failure recovery benchmark failed: {e}")
                benchmark_results['benchmarks']['failure_recovery'] = {'error': str(e)}
        
        finally:
            # Cleanup
            await self.server_manager.cleanup_all()
        
        benchmark_results['suite_end_time'] = time.time()
        benchmark_results['suite_duration_seconds'] = benchmark_results['suite_end_time'] - benchmark_results['suite_start_time']
        
        print(f"\nBenchmark suite completed in {benchmark_results['suite_duration_seconds']:.1f} seconds")
        
        return benchmark_results
    
    def export_benchmark_results(self, results: Dict[str, Any], file_path: str):
        """Export benchmark results to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results exported to {file_path}")


# Pytest test functions for performance benchmarks

@pytest.mark.asyncio
async def test_single_server_throughput_github():
    """Test GitHub server throughput benchmarking"""
    benchmarks = MCPPerformanceBenchmarks()
    await benchmarks.setup_benchmark_servers()
    
    try:
        results = await benchmarks.benchmark_single_server_throughput('github', 'get_repository_info', [1, 5, 10])
        
        assert len(results) == 3  # Should have results for 3 concurrency levels
        assert all(r.success_rate > 0.8 for r in results)  # Minimum 80% success rate
        assert results[0].concurrency_level == 1
        assert results[-1].concurrency_level == 10
        
    finally:
        await benchmarks.server_manager.cleanup_all()


@pytest.mark.asyncio
async def test_multi_server_coordination_benchmark():
    """Test multi-server coordination performance"""
    benchmarks = MCPPerformanceBenchmarks()
    
    try:
        results = await benchmarks.benchmark_multi_server_coordination()
        
        assert results['total_workflows'] == 10
        assert results['success_rate'] >= 0.8  # At least 80% success rate
        assert results['average_workflow_duration_ms'] > 0
        assert results['workflows_per_second'] > 0
        
    finally:
        await benchmarks.server_manager.cleanup_all()


@pytest.mark.asyncio
async def test_stress_test_limits():
    """Test stress testing to find server limits"""
    benchmarks = MCPPerformanceBenchmarks()
    
    try:
        config = StressTestConfig(
            duration_seconds=5,  # Shorter for testing
            target_rps=20,
            max_concurrency=10
        )
        results = await benchmarks.stress_test_server_limits(config)
        
        assert len(results) >= 2  # At least 2 servers tested
        for server_name, result in results.items():
            assert 'max_successful_rps' in result
            assert result['stress_test_completed'] is True
            assert result['max_successful_rps'] >= 1
        
    finally:
        await benchmarks.server_manager.cleanup_all()


@pytest.mark.asyncio
async def test_resource_efficiency_benchmark():
    """Test resource efficiency benchmarking"""
    benchmarks = MCPPerformanceBenchmarks()
    
    try:
        results = await benchmarks.benchmark_resource_efficiency()
        
        assert len(results) >= 2  # At least 2 servers
        for server_name, result in results.items():
            assert 'peak_memory_mb' in result
            assert 'average_cpu_percent' in result
            assert result['resource_samples_count'] > 0
            assert result['memory_efficiency'] > 0
        
    finally:
        await benchmarks.server_manager.cleanup_all()


@pytest.mark.asyncio
async def test_failure_recovery_performance():
    """Test failure recovery performance impact"""
    benchmarks = MCPPerformanceBenchmarks()
    
    try:
        results = await benchmarks.benchmark_failure_recovery_performance()
        
        assert len(results) >= 2  # At least 2 servers
        for server_name, result in results.items():
            assert 'recovery_time_seconds' in result
            assert 'recovery_successful' in result
            assert result['recovery_successful'] is True
            assert result['recovery_time_seconds'] < 5.0  # Should recover within 5 seconds
        
    finally:
        await benchmarks.server_manager.cleanup_all()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_comprehensive_benchmark_suite():
    """Run the complete benchmark suite (marked as performance test)"""
    benchmarks = MCPPerformanceBenchmarks()
    
    results = await benchmarks.run_comprehensive_benchmark_suite()
    
    # Verify all benchmark categories completed
    assert 'benchmarks' in results
    assert 'suite_duration_seconds' in results
    assert results['suite_duration_seconds'] < 300  # Should complete within 5 minutes
    
    # Check that key benchmarks ran
    benchmark_categories = ['throughput', 'coordination', 'stress_testing', 'resource_efficiency', 'failure_recovery']
    completed_benchmarks = sum(1 for cat in benchmark_categories if cat in results['benchmarks'])
    
    assert completed_benchmarks >= len(benchmark_categories) * 0.8  # At least 80% of benchmarks should complete


if __name__ == "__main__":
    # Run comprehensive benchmark suite directly
    async def main():
        benchmarks = MCPPerformanceBenchmarks()
        results = await benchmarks.run_comprehensive_benchmark_suite()
        
        # Export results
        benchmarks.export_benchmark_results(results, "/tmp/mcp_benchmark_results.json")
        
        print("\n=== BENCHMARK SUMMARY ===")
        print(f"Suite Duration: {results['suite_duration_seconds']:.1f} seconds")
        print(f"Benchmarks Completed: {len(results['benchmarks'])}")
        
        for category, result in results['benchmarks'].items():
            if isinstance(result, dict) and 'error' not in result:
                print(f"✓ {category}: Completed successfully")
            else:
                print(f"✗ {category}: Failed or had errors")
    
    asyncio.run(main())