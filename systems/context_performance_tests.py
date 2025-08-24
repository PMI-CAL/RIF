#!/usr/bin/env python3
"""
Context Intelligence Platform - Performance Tests
Issue #119: DPIBS Architecture Phase 1

Comprehensive test suite for validating sub-200ms performance targets
and ensuring all quality gates are met.
"""

import asyncio
import json
import time
import statistics
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

# Add systems directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all Context Intelligence Platform components
try:
    from context_intelligence_platform import ContextIntelligencePlatform, AgentType
    from context_api_gateway import ContextAPIGateway
    from event_service_bus import EventServiceBus, EventType, EventPriority
    from context_database_schema import ContextDatabaseSchema
    from context_integration_interface import ContextIntegrationInterface
    PLATFORM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Platform components not available: {e}")
    PLATFORM_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTest:
    """Individual performance test definition"""
    test_id: str
    test_name: str
    target_ms: float
    description: str
    concurrent_requests: int = 1
    iterations: int = 10
    warmup_iterations: int = 3

@dataclass
class TestResult:
    """Performance test result"""
    test_id: str
    success: bool
    avg_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    target_ms: float
    passed: bool
    error_count: int
    cache_hit_rate: Optional[float] = None
    throughput_rps: Optional[float] = None
    memory_usage_mb: Optional[float] = None

class ContextIntelligencePerformanceTester:
    """Comprehensive performance tester for Context Intelligence Platform"""
    
    def __init__(self):
        self.platform = None
        self.gateway = None  
        self.event_bus = None
        self.schema_manager = None
        self.integration_interface = None
        self.test_results = []
        self.overall_stats = {}
        
        # Test configuration
        self.performance_tests = self._define_performance_tests()
        
    def _define_performance_tests(self) -> List[PerformanceTest]:
        """Define comprehensive performance test suite"""
        return [
            # Core Context Optimization Tests
            PerformanceTest(
                test_id="ctx_opt_single",
                test_name="Context Optimization - Single Request",
                target_ms=50.0,
                description="Single agent context optimization should complete in <50ms",
                concurrent_requests=1,
                iterations=20
            ),
            PerformanceTest(
                test_id="ctx_opt_concurrent", 
                test_name="Context Optimization - Concurrent Requests",
                target_ms=100.0,
                description="10 concurrent context optimizations should average <100ms",
                concurrent_requests=10,
                iterations=10
            ),
            
            # Agent Context Delivery Tests
            PerformanceTest(
                test_id="agent_delivery_single",
                test_name="Agent Context Delivery - Single Agent",
                target_ms=150.0,
                description="Complete agent context delivery in <150ms",
                concurrent_requests=1,
                iterations=15
            ),
            PerformanceTest(
                test_id="agent_delivery_multi",
                test_name="Agent Context Delivery - Multiple Agents",
                target_ms=180.0,
                description="Multiple concurrent agent deliveries averaging <180ms",
                concurrent_requests=5,
                iterations=10
            ),
            
            # API Gateway Tests
            PerformanceTest(
                test_id="api_gateway_auth",
                test_name="API Gateway - Authenticated Request",
                target_ms=20.0,
                description="API Gateway authentication and routing in <20ms",
                concurrent_requests=1,
                iterations=25
            ),
            PerformanceTest(
                test_id="api_gateway_concurrent",
                test_name="API Gateway - Concurrent Load",
                target_ms=50.0,
                description="API Gateway handling 15 concurrent requests in <50ms avg",
                concurrent_requests=15,
                iterations=8
            ),
            
            # Event Service Bus Tests  
            PerformanceTest(
                test_id="event_processing",
                test_name="Event Bus - Event Processing",
                target_ms=10.0,
                description="Individual event processing in <10ms",
                concurrent_requests=1,
                iterations=30
            ),
            PerformanceTest(
                test_id="event_throughput",
                test_name="Event Bus - High Throughput",
                target_ms=25.0,
                description="High throughput event processing averaging <25ms",
                concurrent_requests=20,
                iterations=5
            ),
            
            # Cache Performance Tests
            PerformanceTest(
                test_id="cache_l1_hit",
                test_name="Cache L1 - Cache Hit",
                target_ms=5.0,
                description="L1 cache hit response in <5ms",
                concurrent_requests=1,
                iterations=30
            ),
            PerformanceTest(
                test_id="cache_l2_hit",
                test_name="Cache L2 - Cache Hit", 
                target_ms=15.0,
                description="L2 cache hit response in <15ms",
                concurrent_requests=1,
                iterations=25
            ),
            
            # Integration Interface Tests
            PerformanceTest(
                test_id="integration_legacy",
                test_name="Integration - Legacy Interface",
                target_ms=100.0,
                description="Legacy interface compatibility in <100ms",
                concurrent_requests=1,
                iterations=15
            ),
            PerformanceTest(
                test_id="integration_modern",
                test_name="Integration - Modern Interface", 
                target_ms=80.0,
                description="Modern async interface in <80ms",
                concurrent_requests=1,
                iterations=15
            ),
            
            # End-to-End Performance Tests
            PerformanceTest(
                test_id="e2e_full_context",
                test_name="End-to-End - Full Context Request",
                target_ms=200.0,
                description="Complete context request pipeline in <200ms (P95)",
                concurrent_requests=1,
                iterations=20,
                warmup_iterations=5
            ),
            PerformanceTest(
                test_id="e2e_concurrent_agents",
                test_name="End-to-End - Concurrent Agents",
                target_ms=200.0,
                description="Multiple agents requesting context concurrently <200ms avg",
                concurrent_requests=8,
                iterations=10,
                warmup_iterations=3
            )
        ]
    
    async def setup_test_environment(self):
        """Setup test environment with all components"""
        if not PLATFORM_AVAILABLE:
            raise Exception("Context Intelligence Platform components not available for testing")
        
        print("Setting up test environment...")
        
        # Initialize core platform
        self.platform = ContextIntelligencePlatform()
        await asyncio.sleep(0.1)  # Allow initialization
        
        # Initialize API Gateway
        self.gateway = ContextAPIGateway(self.platform)
        
        # Initialize Event Service Bus
        self.event_bus = EventServiceBus()
        await self.event_bus.start()
        
        # Initialize Database Schema
        self.schema_manager = ContextDatabaseSchema()
        
        # Initialize Integration Interface
        self.integration_interface = ContextIntegrationInterface(enable_platform=True)
        
        # Warmup - run a few requests to initialize caches
        print("Warming up caches...")
        await self._warmup_system()
        
        print("✓ Test environment ready")
    
    async def _warmup_system(self):
        """Warmup system caches and connections"""
        warmup_tasks = []
        
        for agent_type in [AgentType.IMPLEMENTER, AgentType.VALIDATOR, AgentType.ANALYST]:
            task_context = {"description": "Warmup request", "warmup": True}
            warmup_task = self.platform.process_context_request(agent_type, task_context, 119)
            warmup_tasks.append(warmup_task)
        
        # Wait for warmup requests
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        # API Gateway warmup
        await self.gateway.handle_request("/health", "GET", {}, {})
        
        # Brief pause to allow caches to settle
        await asyncio.sleep(0.5)
    
    async def run_performance_test(self, test: PerformanceTest) -> TestResult:
        """Run individual performance test"""
        print(f"Running {test.test_name}...")
        
        # Warmup iterations (not counted in results)
        for i in range(test.warmup_iterations):
            await self._execute_test_iteration(test, is_warmup=True)
        
        # Actual test iterations
        iteration_times = []
        error_count = 0
        cache_hits = 0
        total_requests = 0
        
        start_time = time.time()
        
        for i in range(test.iterations):
            try:
                if test.concurrent_requests > 1:
                    # Concurrent execution
                    tasks = []
                    for _ in range(test.concurrent_requests):
                        task = self._execute_test_iteration(test, is_warmup=False)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process concurrent results
                    for result in results:
                        total_requests += 1
                        if isinstance(result, Exception):
                            error_count += 1
                            logger.error(f"Test iteration failed: {result}")
                        else:
                            iteration_times.append(result['duration_ms'])
                            if result.get('cache_hit'):
                                cache_hits += 1
                else:
                    # Single request execution
                    result = await self._execute_test_iteration(test, is_warmup=False)
                    total_requests += 1
                    
                    if isinstance(result, Exception):
                        error_count += 1
                        logger.error(f"Test iteration failed: {result}")
                    else:
                        iteration_times.append(result['duration_ms'])
                        if result.get('cache_hit'):
                            cache_hits += 1
                            
            except Exception as e:
                error_count += 1
                logger.error(f"Test iteration failed: {e}")
        
        total_test_time = time.time() - start_time
        
        # Calculate statistics
        if iteration_times:
            avg_time = statistics.mean(iteration_times)
            p95_time = self._percentile(iteration_times, 95)
            p99_time = self._percentile(iteration_times, 99)
            min_time = min(iteration_times)
            max_time = max(iteration_times)
            std_dev = statistics.stdev(iteration_times) if len(iteration_times) > 1 else 0.0
            
            # For P95 tests, use P95 time for pass/fail, otherwise use average
            test_metric = p95_time if "P95" in test.description else avg_time
            passed = test_metric <= test.target_ms and error_count == 0
            
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else None
            throughput_rps = total_requests / total_test_time if total_test_time > 0 else None
            
        else:
            # All iterations failed
            avg_time = p95_time = p99_time = min_time = max_time = std_dev = float('inf')
            passed = False
            cache_hit_rate = throughput_rps = None
        
        result = TestResult(
            test_id=test.test_id,
            success=len(iteration_times) > 0,
            avg_time_ms=avg_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            target_ms=test.target_ms,
            passed=passed,
            error_count=error_count,
            cache_hit_rate=cache_hit_rate,
            throughput_rps=throughput_rps
        )
        
        # Print result
        status = "✓ PASS" if result.passed else "✗ FAIL"
        metric = result.p95_time_ms if "P95" in test.description else result.avg_time_ms
        print(f"  {status} - {metric:.1f}ms (target: {test.target_ms}ms)")
        if not result.passed:
            print(f"    Errors: {error_count}, StdDev: {std_dev:.1f}ms")
        
        return result
    
    async def _execute_test_iteration(self, test: PerformanceTest, is_warmup: bool = False) -> Dict[str, Any]:
        """Execute single test iteration"""
        start_time = time.time()
        cache_hit = False
        
        try:
            if test.test_id.startswith("ctx_opt_"):
                # Context optimization test
                agent_type = AgentType.IMPLEMENTER
                task_context = {"description": f"Test context optimization {test.test_id}", "test": True}
                response = await self.platform.process_context_request(agent_type, task_context, 119)
                cache_hit = response.cache_hit
                
            elif test.test_id.startswith("agent_delivery_"):
                # Agent delivery test
                agent_type = AgentType.VALIDATOR
                task_context = {"description": f"Test agent delivery {test.test_id}", "test": True}
                response = await self.platform.process_context_request(agent_type, task_context, 119)
                cache_hit = response.cache_hit
                
            elif test.test_id.startswith("api_gateway_"):
                # API Gateway test
                headers = {"Authorization": "Bearer agent_token_123"}
                body = {
                    "task_context": {"description": f"Test API Gateway {test.test_id}", "test": True},
                    "issue_number": 119
                }
                response = await self.gateway.handle_request("/context/rif-implementer", "POST", headers, body)
                cache_hit = response.cache_hit
                
            elif test.test_id.startswith("event_"):
                # Event processing test
                event_id = self.event_bus.publish_context_update(
                    f"test_context_{test.test_id}",
                    [f"test_key_{int(time.time())}"],
                    priority=EventPriority.NORMAL
                )
                await asyncio.sleep(0.05)  # Allow processing time
                
            elif test.test_id.startswith("cache_"):
                # Cache test - simulate cache hit
                agent_type = AgentType.IMPLEMENTER
                task_context = {"description": f"Cache test {test.test_id}", "cache_test": True}
                
                # First request to populate cache
                if not is_warmup:
                    await self.platform.process_context_request(agent_type, task_context, 119)
                    
                # Second request should hit cache
                response = await self.platform.process_context_request(agent_type, task_context, 119)
                cache_hit = response.cache_hit
                
            elif test.test_id.startswith("integration_"):
                # Integration interface test
                if "legacy" in test.test_id:
                    # Legacy interface
                    task_context = {"description": f"Legacy test {test.test_id}", "test": True}
                    context_data = self.integration_interface.optimize_for_agent("rif-implementer", task_context, 119)
                    cache_hit = "cache" in context_data.get("source", "")
                else:
                    # Modern interface
                    task_context = {"description": f"Modern test {test.test_id}", "test": True}
                    response = await self.integration_interface.get_optimized_context("rif-implementer", task_context, 119)
                    cache_hit = "cache_hit" in response.get("formatted_context", "")
                    
            elif test.test_id.startswith("e2e_"):
                # End-to-end test
                agent_type = AgentType.IMPLEMENTER if "single" in test.test_id else AgentType.VALIDATOR
                task_context = {"description": f"E2E test {test.test_id}", "test": True, "e2e": True}
                response = await self.platform.process_context_request(agent_type, task_context, 119)
                cache_hit = response.cache_hit
                
            else:
                raise ValueError(f"Unknown test type: {test.test_id}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "duration_ms": duration_ms,
                "cache_hit": cache_hit,
                "success": True
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Test iteration failed: {e}")
            return {
                "duration_ms": duration_ms,
                "cache_hit": False,
                "success": False,
                "error": str(e)
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        f = int(k)
        c = k - f
        
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        else:
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete performance test suite"""
        print("=== Context Intelligence Platform Performance Tests ===\n")
        
        try:
            await self.setup_test_environment()
        except Exception as e:
            return {"error": f"Failed to setup test environment: {e}", "tests_run": 0}
        
        # Run all performance tests
        self.test_results = []
        
        for test in self.performance_tests:
            try:
                result = await self.run_performance_test(test)
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"Test {test.test_id} failed with exception: {e}")
                failed_result = TestResult(
                    test_id=test.test_id,
                    success=False,
                    avg_time_ms=float('inf'),
                    p95_time_ms=float('inf'),
                    p99_time_ms=float('inf'),
                    min_time_ms=float('inf'),
                    max_time_ms=float('inf'),
                    std_dev_ms=0.0,
                    target_ms=test.target_ms,
                    passed=False,
                    error_count=1
                )
                self.test_results.append(failed_result)
        
        # Generate summary
        summary = self._generate_test_summary()
        
        # Cleanup
        try:
            if self.event_bus:
                await self.event_bus.stop()
        except:
            pass
        
        return summary
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        # Calculate overall statistics
        all_times = []
        for result in self.test_results:
            if result.success and result.avg_time_ms != float('inf'):
                all_times.append(result.avg_time_ms)
        
        overall_avg = statistics.mean(all_times) if all_times else 0
        overall_p95 = self._percentile(all_times, 95) if all_times else 0
        
        # Categorize results by component
        component_results = {
            "context_optimization": [r for r in self.test_results if r.test_id.startswith("ctx_opt_")],
            "agent_delivery": [r for r in self.test_results if r.test_id.startswith("agent_delivery_")],
            "api_gateway": [r for r in self.test_results if r.test_id.startswith("api_gateway_")],
            "event_bus": [r for r in self.test_results if r.test_id.startswith("event_")],
            "caching": [r for r in self.test_results if r.test_id.startswith("cache_")],
            "integration": [r for r in self.test_results if r.test_id.startswith("integration_")],
            "end_to_end": [r for r in self.test_results if r.test_id.startswith("e2e_")]
        }
        
        component_stats = {}
        for component, results in component_results.items():
            if results:
                passed = len([r for r in results if r.passed])
                avg_times = [r.avg_time_ms for r in results if r.success and r.avg_time_ms != float('inf')]
                component_stats[component] = {
                    "total_tests": len(results),
                    "passed": passed,
                    "pass_rate": passed / len(results),
                    "avg_time_ms": statistics.mean(avg_times) if avg_times else float('inf')
                }
        
        # Performance compliance check
        sub_200ms_compliance = True
        compliance_details = []
        
        for result in self.test_results:
            if result.test_id.startswith("e2e_"):  # Focus on end-to-end tests
                metric = result.p95_time_ms if "P95" in [t.description for t in self.performance_tests if t.test_id == result.test_id][0] else result.avg_time_ms
                if metric > 200:
                    sub_200ms_compliance = False
                    compliance_details.append(f"{result.test_id}: {metric:.1f}ms > 200ms")
        
        summary = {
            "test_suite": "Context Intelligence Platform Performance",
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "pass_rate": len(passed_tests) / len(self.test_results) if self.test_results else 0,
            "overall_performance": {
                "average_time_ms": overall_avg,
                "p95_time_ms": overall_p95,
                "sub_200ms_compliance": sub_200ms_compliance,
                "compliance_details": compliance_details
            },
            "component_performance": component_stats,
            "detailed_results": [asdict(r) for r in self.test_results]
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted test summary"""
        print("\n=== Performance Test Summary ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        
        print(f"\nOverall Performance:")
        perf = summary['overall_performance']
        print(f"  Average Time: {perf['average_time_ms']:.1f}ms")
        print(f"  P95 Time: {perf['p95_time_ms']:.1f}ms")
        print(f"  Sub-200ms Compliance: {'✓ PASS' if perf['sub_200ms_compliance'] else '✗ FAIL'}")
        
        if perf['compliance_details']:
            print("  Compliance Issues:")
            for detail in perf['compliance_details']:
                print(f"    - {detail}")
        
        print(f"\nComponent Performance:")
        for component, stats in summary['component_performance'].items():
            status = "✓" if stats['pass_rate'] == 1.0 else "✗"
            avg_time = "∞" if stats['avg_time_ms'] == float('inf') else f"{stats['avg_time_ms']:.1f}ms"
            print(f"  {status} {component.replace('_', ' ').title()}: {stats['passed']}/{stats['total_tests']} ({stats['pass_rate']:.1%}) - {avg_time}")
        
        print(f"\nDetailed Results:")
        for result in summary['detailed_results']:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {status} {result['test_id']}: {result['avg_time_ms']:.1f}ms (target: {result['target_ms']}ms)")

# Async test runner
async def run_performance_tests(save_results: bool = False) -> Dict[str, Any]:
    """Run complete performance test suite"""
    tester = ContextIntelligencePerformanceTester()
    summary = await tester.run_all_tests()
    
    if save_results:
        # Save results to file
        results_file = f"/Users/cal/DEV/RIF/systems/context/performance_test_results_{int(time.time())}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return summary

# CLI Interface
async def main():
    """Main function for performance testing CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context Intelligence Platform Performance Tests")
    parser.add_argument("--run", action="store_true", help="Run full performance test suite")
    parser.add_argument("--test", type=str, help="Run specific test by ID")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of tests")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--benchmark", action="store_true", help="Run extended benchmark")
    
    args = parser.parse_args()
    
    if not PLATFORM_AVAILABLE:
        print("Error: Context Intelligence Platform components not available")
        print("Make sure all required modules are installed and accessible")
        return
    
    if args.run or args.quick:
        print("Starting Context Intelligence Platform Performance Tests...")
        
        if args.quick:
            # Quick test subset
            tester = ContextIntelligencePerformanceTester()
            tester.performance_tests = [t for t in tester.performance_tests 
                                      if t.test_id in ["ctx_opt_single", "agent_delivery_single", 
                                                     "api_gateway_auth", "e2e_full_context"]]
        
        summary = await run_performance_tests(save_results=args.save)
        
        # Print summary
        tester = ContextIntelligencePerformanceTester()
        tester.print_summary(summary)
        
        # Exit code based on results
        exit_code = 0 if summary['pass_rate'] == 1.0 else 1
        sys.exit(exit_code)
        
    elif args.test:
        # Run specific test
        tester = ContextIntelligencePerformanceTester()
        specific_tests = [t for t in tester.performance_tests if t.test_id == args.test]
        
        if not specific_tests:
            print(f"Test '{args.test}' not found")
            print("Available tests:")
            for test in tester.performance_tests:
                print(f"  {test.test_id}: {test.test_name}")
            return
        
        await tester.setup_test_environment()
        result = await tester.run_performance_test(specific_tests[0])
        
        print(f"\nTest Result:")
        print(f"  Passed: {result.passed}")
        print(f"  Average Time: {result.avg_time_ms:.1f}ms")
        print(f"  P95 Time: {result.p95_time_ms:.1f}ms")
        print(f"  Target: {result.target_ms}ms")
        print(f"  Error Count: {result.error_count}")
        
    elif args.benchmark:
        print("Running extended benchmark...")
        
        # Extended benchmark configuration
        tester = ContextIntelligencePerformanceTester()
        
        # Increase iterations for more thorough testing
        for test in tester.performance_tests:
            test.iterations = min(test.iterations * 3, 50)
            test.warmup_iterations = max(test.warmup_iterations, 5)
        
        summary = await tester.run_all_tests()
        tester.print_summary(summary)
        
        if args.save:
            benchmark_file = f"/Users/cal/DEV/RIF/systems/context/benchmark_results_{int(time.time())}.json"
            with open(benchmark_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nBenchmark results saved to: {benchmark_file}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())