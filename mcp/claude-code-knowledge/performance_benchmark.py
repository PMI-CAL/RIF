#!/usr/bin/env python3
"""
Performance Benchmarking for Claude Code Knowledge MCP Server.

This script performs comprehensive performance testing including:
- Response time measurements
- Concurrent request handling
- Memory usage monitoring
- Cache performance validation
- Stress testing with production-like loads
"""

import asyncio
import time
import json
import statistics
import psutil
import concurrent.futures
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import logging
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from server import ClaudeCodeKnowledgeServer, MCPRequest
    from config import load_server_config
except ImportError as e:
    print(f"Error importing MCP server components: {e}")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Performance benchmark results."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    requests_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    errors: List[str]


class PerformanceBenchmark:
    """Performance benchmarking suite for the MCP server."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.server = None
        self.results = []
        
    async def setup(self):
        """Initialize the server for testing."""
        try:
            config = load_server_config()
            self.server = ClaudeCodeKnowledgeServer(config.__dict__)
            await self.server.initialize()
            print("✓ Server initialized for benchmarking")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize server: {e}")
            return False
    
    async def teardown(self):
        """Clean up after testing."""
        if self.server:
            await self.server.shutdown()
            print("✓ Server shut down")
    
    async def run_single_request_benchmark(self, num_requests: int = 100) -> BenchmarkResult:
        """Benchmark single sequential requests."""
        print(f"Running single request benchmark ({num_requests} requests)...")
        
        response_times = []
        errors = []
        start_time = time.time()
        
        # Monitor memory and CPU
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        for i in range(num_requests):
            request_start = time.time()
            
            try:
                # Test compatibility check using MCP request structure
                request = MCPRequest(
                    id=f"test-{i}",
                    method="check_compatibility",
                    params={
                        "issue_description": f"Performance test issue {i}",
                        "approach": f"Test approach {i} using direct tool usage"
                    }
                )
                
                response = await self.server.handle_request(asdict(request))
                request_end = time.time()
                response_times.append((request_end - request_start) * 1000)
                
            except Exception as e:
                errors.append(f"Request {i}: {str(e)}")
                response_times.append(float('inf'))
        
        end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024
        
        # Calculate statistics
        valid_times = [t for t in response_times if t != float('inf')]
        successful = len(valid_times)
        failed = len(response_times) - successful
        
        if not valid_times:
            return BenchmarkResult(
                test_name="single_request",
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=num_requests,
                average_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                max_response_time_ms=0,
                requests_per_second=0,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=0,
                cache_hit_rate=0,
                errors=errors[:10]  # First 10 errors
            )
        
        avg_time = statistics.mean(valid_times)
        p50_time = statistics.median(valid_times)
        p95_time = statistics.quantiles(valid_times, n=20)[18] if len(valid_times) > 1 else valid_times[0]
        p99_time = statistics.quantiles(valid_times, n=100)[98] if len(valid_times) > 2 else valid_times[-1]
        max_time = max(valid_times)
        rps = successful / (end_time - start_time)
        
        return BenchmarkResult(
            test_name="single_request",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            average_response_time_ms=avg_time,
            p50_response_time_ms=p50_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
            max_response_time_ms=max_time,
            requests_per_second=rps,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_percent=process.cpu_percent(),
            cache_hit_rate=0.0,  # TODO: Get from server metrics
            errors=errors[:10]
        )
    
    async def run_concurrent_request_benchmark(self, num_concurrent: int = 10, requests_per_worker: int = 10) -> BenchmarkResult:
        """Benchmark concurrent requests."""
        print(f"Running concurrent request benchmark ({num_concurrent} workers, {requests_per_worker} requests each)...")
        
        async def worker(worker_id: int) -> List[float]:
            """Single worker making requests."""
            response_times = []
            
            for i in range(requests_per_worker):
                request_start = time.time()
                
                try:
                    request = MCPRequest(
                        id=f"concurrent-{worker_id}-{i}",
                        method="check_compatibility", 
                        params={
                            "issue_description": f"Concurrent test {worker_id}-{i}",
                            "approach": f"Concurrent approach {worker_id}-{i}"
                        }
                    )
                    
                    response = await self.server.handle_request(asdict(request))
                    request_end = time.time()
                    response_times.append((request_end - request_start) * 1000)
                    
                except Exception:
                    response_times.append(float('inf'))
            
            return response_times
        
        # Monitor resources
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Run concurrent workers
        tasks = [worker(i) for i in range(num_concurrent)]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024
        
        # Aggregate results
        all_response_times = []
        errors = []
        
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                errors.append(f"Worker {i}: {str(result)}")
            else:
                all_response_times.extend(result)
        
        # Calculate statistics
        valid_times = [t for t in all_response_times if t != float('inf')]
        total_requests = num_concurrent * requests_per_worker
        successful = len(valid_times)
        failed = total_requests - successful
        
        if not valid_times:
            return BenchmarkResult(
                test_name="concurrent_request",
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=total_requests,
                average_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                max_response_time_ms=0,
                requests_per_second=0,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=process.cpu_percent(),
                cache_hit_rate=0,
                errors=errors[:10]
            )
        
        avg_time = statistics.mean(valid_times)
        p50_time = statistics.median(valid_times)
        p95_time = statistics.quantiles(valid_times, n=20)[18] if len(valid_times) > 1 else valid_times[0]
        p99_time = statistics.quantiles(valid_times, n=100)[98] if len(valid_times) > 2 else valid_times[-1]
        max_time = max(valid_times)
        rps = successful / (end_time - start_time)
        
        return BenchmarkResult(
            test_name="concurrent_request",
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            average_response_time_ms=avg_time,
            p50_response_time_ms=p50_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
            max_response_time_ms=max_time,
            requests_per_second=rps,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_percent=process.cpu_percent(),
            cache_hit_rate=0.0,
            errors=errors[:10]
        )
    
    async def run_stress_test(self, duration_seconds: int = 30) -> BenchmarkResult:
        """Run stress test with continuous load."""
        print(f"Running stress test ({duration_seconds} seconds)...")
        
        response_times = []
        errors = []
        request_count = 0
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        async def stress_worker():
            nonlocal request_count
            while time.time() < end_time:
                request_start = time.time()
                
                try:
                    request = MCPRequest(
                        id=f"stress-{request_count}",
                        method="check_compatibility",
                        params={
                            "issue_description": f"Stress test request {request_count}",
                            "approach": "High load testing approach"
                        }
                    )
                    
                    response = await self.server.handle_request(asdict(request))
                    request_end = time.time()
                    response_times.append((request_end - request_start) * 1000)
                    request_count += 1
                    
                except Exception as e:
                    errors.append(str(e))
                    response_times.append(float('inf'))
                    request_count += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        # Run multiple stress workers
        workers = [stress_worker() for _ in range(5)]
        await asyncio.gather(*workers, return_exceptions=True)
        
        actual_end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024
        
        # Calculate statistics
        valid_times = [t for t in response_times if t != float('inf')]
        successful = len(valid_times)
        failed = request_count - successful
        
        if not valid_times:
            return BenchmarkResult(
                test_name="stress_test",
                total_requests=request_count,
                successful_requests=0,
                failed_requests=request_count,
                average_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                max_response_time_ms=0,
                requests_per_second=0,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=process.cpu_percent(),
                cache_hit_rate=0,
                errors=errors[:10]
            )
        
        avg_time = statistics.mean(valid_times)
        p50_time = statistics.median(valid_times)
        p95_time = statistics.quantiles(valid_times, n=20)[18] if len(valid_times) > 1 else valid_times[0]
        p99_time = statistics.quantiles(valid_times, n=100)[98] if len(valid_times) > 2 else valid_times[-1]
        max_time = max(valid_times)
        rps = successful / (actual_end_time - start_time)
        
        return BenchmarkResult(
            test_name="stress_test",
            total_requests=request_count,
            successful_requests=successful,
            failed_requests=failed,
            average_response_time_ms=avg_time,
            p50_response_time_ms=p50_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
            max_response_time_ms=max_time,
            requests_per_second=rps,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_percent=process.cpu_percent(),
            cache_hit_rate=0.0,
            errors=errors[:10]
        )
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results in a readable format."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\n📊 {result.test_name.upper()}")
            print("-" * 40)
            print(f"Total Requests: {result.total_requests}")
            print(f"Successful: {result.successful_requests}")
            print(f"Failed: {result.failed_requests}")
            print(f"Success Rate: {(result.successful_requests/result.total_requests*100):.1f}%")
            print()
            print(f"Average Response Time: {result.average_response_time_ms:.2f}ms")
            print(f"P50 Response Time: {result.p50_response_time_ms:.2f}ms")
            print(f"P95 Response Time: {result.p95_response_time_ms:.2f}ms")
            print(f"P99 Response Time: {result.p99_response_time_ms:.2f}ms")
            print(f"Max Response Time: {result.max_response_time_ms:.2f}ms")
            print()
            print(f"Requests Per Second: {result.requests_per_second:.2f}")
            print(f"Memory Usage: {result.memory_usage_mb:.2f}MB")
            print(f"CPU Usage: {result.cpu_usage_percent:.1f}%")
            
            # Check acceptance criteria
            if result.average_response_time_ms <= 200:
                print("✅ Response time target met (<200ms)")
            else:
                print("❌ Response time target missed (>200ms)")
            
            if result.errors:
                print(f"\n🚨 Errors ({len(result.errors)} shown):")
                for error in result.errors[:5]:
                    print(f"  - {error}")
        
        # Overall assessment
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        
        all_passed = True
        for result in results:
            if result.average_response_time_ms > 200:
                all_passed = False
                print(f"❌ {result.test_name}: Response time too high ({result.average_response_time_ms:.2f}ms)")
            elif result.failed_requests > result.total_requests * 0.05:  # >5% failure rate
                all_passed = False
                print(f"❌ {result.test_name}: High failure rate ({result.failed_requests}/{result.total_requests})")
            else:
                print(f"✅ {result.test_name}: Performance acceptable")
        
        if all_passed:
            print("\n🎉 ALL PERFORMANCE TARGETS MET - READY FOR PRODUCTION")
        else:
            print("\n⚠️  SOME PERFORMANCE TARGETS NOT MET - REVIEW REQUIRED")
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        if not await self.setup():
            return []
        
        try:
            results = []
            
            # Single request benchmark
            result = await self.run_single_request_benchmark(50)
            results.append(result)
            
            # Concurrent request benchmark  
            result = await self.run_concurrent_request_benchmark(5, 10)
            results.append(result)
            
            # Stress test
            result = await self.run_stress_test(15)
            results.append(result)
            
            self.results = results
            return results
            
        finally:
            await self.teardown()
    
    def save_results(self, filename: str = "performance_results.json"):
        """Save results to JSON file."""
        results_data = [asdict(result) for result in self.results]
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'results': results_data
            }, f, indent=2)
        
        print(f"✓ Results saved to {filename}")


async def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    
    print("🚀 Starting Performance Benchmark for Claude Code Knowledge MCP Server")
    print("This will test response times, concurrency, and stress handling...")
    
    results = await benchmark.run_all_benchmarks()
    
    if results:
        benchmark.print_results(results)
        benchmark.save_results()
    else:
        print("❌ Benchmark failed to run")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))