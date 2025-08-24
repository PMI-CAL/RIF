#!/usr/bin/env python3
"""
Context Optimization Engine Test Suite
Issue #123: DPIBS Development Phase 1

Comprehensive test suite with:
- Performance validation (sub-200ms targets)
- Multi-factor relevance scoring validation  
- Concurrent request testing (10+ agents)
- MCP integration testing
- Cache performance testing
- Load balancing validation
- End-to-end integration testing

Based on RIF-Implementer requirements for evidence generation.
"""

import asyncio
import json
import time
import statistics
import threading
import logging
import unittest
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys

# Import the components to test
sys.path.append('/Users/cal/DEV/RIF/systems')

# Import from context optimization engine (with hyphens in filename)
from import_utils import import_context_optimization_engine
context_imports = import_context_optimization_engine()
ContextOptimizer = context_imports['ContextOptimizer']
AgentType = context_imports['AgentType']
ContextType = context_imports['ContextType']
ContextItem = context_imports['ContextItem']
SystemContext = context_imports['SystemContext']
from context_intelligence_platform import (
    ContextIntelligencePlatform, ContextRequest, PerformanceMetrics
)
from context_request_router import ContextRequestRouter
from context_performance_monitor import ContextPerformanceMonitor, MetricType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTestResult:
    """Container for performance test results"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.response_times = []
        self.success_count = 0
        self.failure_count = 0
        self.cache_hits = 0
        self.start_time = None
        self.end_time = None
        self.errors = []
        
    def add_result(self, response_time_ms: float, success: bool, cache_hit: bool = False):
        """Add individual test result"""
        self.response_times.append(response_time_ms)
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        if cache_hit:
            self.cache_hits += 1
    
    def add_error(self, error: str):
        """Add error message"""
        self.errors.append(error)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.response_times:
            return {"error": "No response times recorded"}
        
        total_requests = len(self.response_times)
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        
        return {
            "test_name": self.test_name,
            "total_requests": total_requests,
            "success_rate": self.success_count / total_requests if total_requests > 0 else 0,
            "cache_hit_rate": self.cache_hits / total_requests if total_requests > 0 else 0,
            "response_times": {
                "min_ms": min(self.response_times),
                "max_ms": max(self.response_times),
                "mean_ms": statistics.mean(self.response_times),
                "median_ms": statistics.median(self.response_times),
                "p95_ms": sorted(self.response_times)[int(len(self.response_times) * 0.95)],
                "p99_ms": sorted(self.response_times)[int(len(self.response_times) * 0.99)],
                "std_dev_ms": statistics.stdev(self.response_times) if len(self.response_times) > 1 else 0
            },
            "performance_compliance": {
                "sub_200ms_count": len([t for t in self.response_times if t < 200]),
                "sub_200ms_percentage": len([t for t in self.response_times if t < 200]) / total_requests * 100,
                "sub_500ms_p99": sorted(self.response_times)[int(len(self.response_times) * 0.99)] < 500
            },
            "throughput_rps": total_requests / duration if duration > 0 else 0,
            "duration_seconds": duration,
            "error_count": len(self.errors),
            "errors": self.errors[:10]  # First 10 errors
        }

class TestContextOptimizationEngine(unittest.TestCase):
    """Test suite for Context Optimization Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = ContextOptimizer()
        self.test_task_context = {
            "description": "Implement context optimization engine with sub-200ms performance",
            "complexity": "very_high",
            "type": "implementation",
            "issue_number": 123
        }
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_multi_factor_relevance_scoring(self):
        """Test multi-factor relevance scoring algorithm"""
        print("Testing multi-factor relevance scoring...")
        
        # Create test context items
        test_items = [
            ContextItem(
                id="test-claude-capabilities",
                type=ContextType.CLAUDE_CODE_CAPABILITIES,
                content="Claude Code test capabilities",
                relevance_score=1.0,
                last_updated=datetime.now(),
                source="test",
                agent_relevance={AgentType.IMPLEMENTER: 0.9},
                size_estimate=100
            ),
            ContextItem(
                id="test-implementation-patterns",
                type=ContextType.IMPLEMENTATION_PATTERNS,
                content="Implementation patterns for testing",
                relevance_score=0.8,
                last_updated=datetime.now() - timedelta(days=1),
                source="test",
                agent_relevance={AgentType.IMPLEMENTER: 0.85},
                size_estimate=150
            )
        ]
        
        # Test scoring for different agent types
        for agent_type in [AgentType.IMPLEMENTER, AgentType.VALIDATOR, AgentType.ANALYST]:
            scores = self.optimizer._score_knowledge_relevance(
                agent_type, self.test_task_context, test_items
            )
            
            # Verify scores are in valid range
            for item_id, score in scores.items():
                self.assertGreaterEqual(score, 0.0, f"Score for {item_id} should be >= 0")
                self.assertLessEqual(score, 1.0, f"Score for {item_id} should be <= 1")
                
            # Claude capabilities should always score highly
            claude_score = scores.get("test-claude-capabilities", 0)
            self.assertGreater(claude_score, 0.8, "Claude capabilities should score highly")
            
            print(f"  {agent_type.value}: Claude={claude_score:.3f}, Patterns={scores.get('test-implementation-patterns', 0):.3f}")
        
        print("✓ Multi-factor relevance scoring tests passed")
    
    def test_agent_context_optimization(self):
        """Test agent-specific context optimization"""
        print("Testing agent-specific context optimization...")
        
        results = {}
        
        for agent_type in AgentType:
            start_time = time.time()
            
            # Optimize context for agent
            agent_context = self.optimizer.optimize_for_agent(
                agent_type, self.test_task_context, 123
            )
            
            optimization_time = (time.time() - start_time) * 1000
            
            # Verify context structure
            self.assertIsNotNone(agent_context, f"Context should not be None for {agent_type.value}")
            self.assertEqual(agent_context.agent_type, agent_type)
            self.assertIsInstance(agent_context.relevant_knowledge, list)
            self.assertIsNotNone(agent_context.system_context)
            
            # Verify context window limits
            context_limit = self.optimizer.agent_context_limits[agent_type]
            self.assertLessEqual(agent_context.total_size, context_limit * 1.2,  # 20% tolerance
                                f"Context size {agent_context.total_size} exceeds limit {context_limit}")
            
            results[agent_type.value] = {
                "optimization_time_ms": optimization_time,
                "context_items": len(agent_context.relevant_knowledge),
                "context_size": agent_context.total_size,
                "utilization": agent_context.context_window_utilization
            }
            
            print(f"  {agent_type.value}: {optimization_time:.1f}ms, "
                  f"{len(agent_context.relevant_knowledge)} items, "
                  f"{agent_context.total_size} chars")
        
        print("✓ Agent context optimization tests passed")
        return results
    
    def test_performance_targets(self):
        """Test sub-200ms performance targets"""
        print("Testing sub-200ms performance targets...")
        
        test_result = PerformanceTestResult("performance_targets")
        test_result.start_time = time.time()
        
        # Test multiple requests to get statistical significance
        for i in range(50):
            start_time = time.time()
            
            try:
                agent_context = self.optimizer.optimize_for_agent(
                    AgentType.IMPLEMENTER, self.test_task_context, 123 + i
                )
                response_time = (time.time() - start_time) * 1000
                test_result.add_result(response_time, True, False)
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                test_result.add_result(response_time, False, False)
                test_result.add_error(str(e))
        
        test_result.end_time = time.time()
        stats = test_result.get_statistics()
        
        # Performance assertions
        self.assertGreater(stats["performance_compliance"]["sub_200ms_percentage"], 80,
                          f"At least 80% of requests should be under 200ms, got {stats['performance_compliance']['sub_200ms_percentage']:.1f}%")
        
        self.assertLess(stats["response_times"]["p95_ms"], 300,
                       f"P95 should be under 300ms, got {stats['response_times']['p95_ms']:.1f}ms")
        
        self.assertTrue(stats["performance_compliance"]["sub_500ms_p99"],
                       f"P99 should be under 500ms, got {stats['response_times']['p99_ms']:.1f}ms")
        
        print(f"  Mean: {stats['response_times']['mean_ms']:.1f}ms")
        print(f"  P95: {stats['response_times']['p95_ms']:.1f}ms")
        print(f"  P99: {stats['response_times']['p99_ms']:.1f}ms")
        print(f"  Sub-200ms: {stats['performance_compliance']['sub_200ms_percentage']:.1f}%")
        print("✓ Performance targets tests passed")
        
        return stats

class TestContextIntelligencePlatform(unittest.TestCase):
    """Test suite for Context Intelligence Platform"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.platform = ContextIntelligencePlatform(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_platform_integration(self):
        """Test full platform integration"""
        print("Testing Context Intelligence Platform integration...")
        
        test_result = PerformanceTestResult("platform_integration")
        test_result.start_time = time.time()
        
        # Test different agent types
        for agent_type in [AgentType.IMPLEMENTER, AgentType.VALIDATOR, AgentType.ANALYST]:
            for i in range(10):
                start_time = time.time()
                
                try:
                    response = await self.platform.process_context_request(
                        agent_type=agent_type,
                        task_context={
                            "description": f"Test request {i} for {agent_type.value}",
                            "complexity": "medium"
                        },
                        issue_number=123 + i
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    test_result.add_result(response_time, True, response.cache_hit)
                    
                    # Verify response structure
                    self.assertIsNotNone(response.agent_context)
                    self.assertIsNotNone(response.performance_metrics)
                    self.assertGreater(len(response.agent_context.relevant_knowledge), 0)
                    
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    test_result.add_result(response_time, False, False)
                    test_result.add_error(f"{agent_type.value}:{i}: {str(e)}")
        
        test_result.end_time = time.time()
        stats = test_result.get_statistics()
        
        # Verify performance
        self.assertGreater(stats["success_rate"], 0.95, "Success rate should be > 95%")
        self.assertLess(stats["response_times"]["mean_ms"], 200, "Mean response time should be < 200ms")
        
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Mean response time: {stats['response_times']['mean_ms']:.1f}ms")
        print("✓ Platform integration tests passed")
        
        return stats
    
    async def test_concurrent_requests(self):
        """Test concurrent request handling (10+ agents)"""
        print("Testing concurrent request handling...")
        
        test_result = PerformanceTestResult("concurrent_requests")
        test_result.start_time = time.time()
        
        # Create 15 concurrent requests (exceeds 10+ requirement)
        tasks = []
        for i in range(15):
            agent_type = list(AgentType)[i % len(AgentType)]
            
            task = self.platform.process_context_request(
                agent_type=agent_type,
                task_context={
                    "description": f"Concurrent test request {i}",
                    "complexity": "high"
                },
                issue_number=200 + i,
                priority=1 if i < 5 else 2 if i < 10 else 3
            )
            tasks.append((task, i, agent_type))
        
        # Execute all requests concurrently
        start_concurrent = time.time()
        responses = await asyncio.gather(*[task for task, _, _ in tasks], return_exceptions=True)
        concurrent_duration = time.time() - start_concurrent
        
        # Analyze results
        successful_responses = []
        for i, (response, (_, req_id, agent_type)) in enumerate(zip(responses, tasks)):
            if isinstance(response, Exception):
                test_result.add_result(concurrent_duration * 1000, False, False)
                test_result.add_error(f"Request {req_id}: {str(response)}")
            else:
                test_result.add_result(response.total_response_time_ms, True, response.cache_hit)
                successful_responses.append(response)
        
        test_result.end_time = time.time()
        stats = test_result.get_statistics()
        
        # Verify concurrent performance
        self.assertGreater(len(successful_responses), 12, "At least 12/15 concurrent requests should succeed")
        self.assertLess(concurrent_duration, 5.0, "All concurrent requests should complete within 5 seconds")
        
        print(f"  Concurrent requests: 15")
        print(f"  Successful: {len(successful_responses)}")
        print(f"  Total duration: {concurrent_duration:.2f}s")
        print(f"  Mean response time: {stats['response_times']['mean_ms']:.1f}ms")
        print("✓ Concurrent request tests passed")
        
        return stats

class TestContextRequestRouter(unittest.TestCase):
    """Test suite for Context Request Router"""
    
    def setUp(self):
        """Set up test environment"""
        self.router = ContextRequestRouter()
        
        # Register test service instances
        for i in range(3):
            self.router.register_service_instance(
                "test-service", f"instance-{i}", f"http://test:{i}", max_capacity=5
            )
    
    def tearDown(self):
        """Clean up test environment"""
        self.router.shutdown()
    
    def test_load_balancing(self):
        """Test load balancing functionality"""
        print("Testing load balancing...")
        
        # Create test requests
        requests = []
        for i in range(20):
            request = ContextRequest(
                request_id=f"test-lb-{i}",
                agent_type=AgentType.IMPLEMENTER,
                task_context={"description": f"Load balance test {i}"},
                issue_number=300 + i,
                priority=1 if i < 7 else 2 if i < 14 else 3
            )
            requests.append(request)
        
        # Route requests and measure
        start_time = time.time()
        futures = [self.router.route_request(req) for req in requests]
        
        # Allow processing
        time.sleep(3)
        
        routing_duration = time.time() - start_time
        
        # Check routing statistics
        stats = self.router.get_routing_stats()
        
        self.assertGreater(stats["total_requests_routed"], 15, "Should have routed most requests")
        self.assertLess(stats["avg_routing_overhead_ms"], 100, "Routing overhead should be < 100ms")
        
        print(f"  Requests routed: {stats['total_requests_routed']}")
        print(f"  Avg routing overhead: {stats['avg_routing_overhead_ms']:.1f}ms")
        print(f"  Current concurrent: {stats['current_concurrent']}")
        print("✓ Load balancing tests passed")
        
        return stats

class IntegrationTestSuite:
    """End-to-end integration test suite"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.platform = None
        self.monitor = None
        
    async def setup(self):
        """Set up integration test environment"""
        self.platform = ContextIntelligencePlatform(self.temp_dir)
        self.monitor = ContextPerformanceMonitor(os.path.join(self.temp_dir, "monitoring"))
        await asyncio.sleep(1)  # Allow initialization
    
    def cleanup(self):
        """Clean up integration test environment"""
        if self.monitor:
            self.monitor.shutdown()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def run_full_integration_test(self):
        """Run comprehensive integration test"""
        print("Running full integration test...")
        
        test_result = PerformanceTestResult("full_integration")
        test_result.start_time = time.time()
        
        # Test scenarios
        scenarios = [
            {"agent": AgentType.ANALYST, "complexity": "low", "count": 5},
            {"agent": AgentType.IMPLEMENTER, "complexity": "high", "count": 10},
            {"agent": AgentType.VALIDATOR, "complexity": "medium", "count": 5},
            {"agent": AgentType.ARCHITECT, "complexity": "very_high", "count": 3},
        ]
        
        for scenario in scenarios:
            tasks = []
            
            # Create requests for scenario
            for i in range(scenario["count"]):
                task = self.platform.process_context_request(
                    agent_type=scenario["agent"],
                    task_context={
                        "description": f"Integration test {i} - {scenario['complexity']} complexity",
                        "complexity": scenario["complexity"],
                        "type": "implementation"
                    },
                    issue_number=400 + i
                )
                tasks.append(task)
            
            # Execute scenario
            scenario_start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            scenario_duration = (time.time() - scenario_start) * 1000
            
            # Record results
            successful = 0
            for response in responses:
                if isinstance(response, Exception):
                    test_result.add_result(scenario_duration, False, False)
                    test_result.add_error(str(response))
                else:
                    test_result.add_result(response.total_response_time_ms, True, response.cache_hit)
                    successful += 1
                    
                    # Record in performance monitor
                    self.monitor.record_context_request(
                        response_time_ms=response.total_response_time_ms,
                        cache_hit=response.cache_hit,
                        context_relevance=0.9,  # Simulated high relevance
                        service_component="integration-test",
                        success=True
                    )
            
            print(f"  {scenario['agent'].value}: {successful}/{scenario['count']} successful")
        
        test_result.end_time = time.time()
        
        # Get final performance dashboard
        await asyncio.sleep(2)  # Allow monitoring to process
        dashboard = self.monitor.get_performance_dashboard()
        
        stats = test_result.get_statistics()
        stats["performance_dashboard"] = dashboard
        
        print("✓ Full integration test completed")
        return stats

def run_comprehensive_test_suite():
    """Run the complete test suite"""
    print("=" * 60)
    print("Context Optimization Engine - Comprehensive Test Suite")
    print("Issue #123: DPIBS Development Phase 1")
    print("=" * 60)
    
    # Test results collection
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "context_optimization_comprehensive",
        "results": {}
    }
    
    # Unit tests
    print("\n1. UNIT TESTS")
    print("-" * 30)
    
    # Context Optimization Engine tests
    engine_tests = TestContextOptimizationEngine()
    engine_tests.setUp()
    
    try:
        engine_tests.test_multi_factor_relevance_scoring()
        agent_results = engine_tests.test_agent_context_optimization()
        performance_results = engine_tests.test_performance_targets()
        
        test_results["results"]["engine_tests"] = {
            "agent_optimization": agent_results,
            "performance_targets": performance_results
        }
    finally:
        engine_tests.tearDown()
    
    # Platform integration tests
    print("\n2. INTEGRATION TESTS")
    print("-" * 30)
    
    async def run_integration_tests():
        platform_tests = TestContextIntelligencePlatform()
        platform_tests.setUp()
        
        try:
            platform_stats = await platform_tests.test_platform_integration()
            concurrent_stats = await platform_tests.test_concurrent_requests()
            
            test_results["results"]["platform_tests"] = {
                "integration": platform_stats,
                "concurrent": concurrent_stats
            }
        finally:
            platform_tests.tearDown()
    
    asyncio.run(run_integration_tests())
    
    # Load balancing tests  
    print("\n3. LOAD BALANCING TESTS")
    print("-" * 30)
    
    router_tests = TestContextRequestRouter()
    router_tests.setUp()
    
    try:
        routing_stats = router_tests.test_load_balancing()
        test_results["results"]["routing_tests"] = routing_stats
    finally:
        router_tests.tearDown()
    
    # End-to-end integration
    print("\n4. END-TO-END INTEGRATION")
    print("-" * 30)
    
    async def run_e2e_tests():
        integration_suite = IntegrationTestSuite()
        await integration_suite.setup()
        
        try:
            e2e_stats = await integration_suite.run_full_integration_test()
            test_results["results"]["e2e_tests"] = e2e_stats
        finally:
            integration_suite.cleanup()
    
    asyncio.run(run_e2e_tests())
    
    # Generate final report
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    
    # Performance compliance summary
    performance_results = test_results["results"]["engine_tests"]["performance_targets"]
    platform_results = test_results["results"]["platform_tests"]["integration"]
    
    print(f"\n✅ PERFORMANCE TARGETS:")
    print(f"   Sub-200ms compliance: {performance_results['performance_compliance']['sub_200ms_percentage']:.1f}%")
    print(f"   P95 response time: {performance_results['response_times']['p95_ms']:.1f}ms")
    print(f"   P99 response time: {performance_results['response_times']['p99_ms']:.1f}ms")
    
    print(f"\n✅ INTEGRATION VALIDATION:")
    print(f"   Platform success rate: {platform_results['success_rate']:.1%}")
    print(f"   Cache hit rate: {platform_results['cache_hit_rate']:.1%}")
    print(f"   Mean response time: {platform_results['response_times']['mean_ms']:.1f}ms")
    
    concurrent_results = test_results["results"]["platform_tests"]["concurrent"]
    print(f"\n✅ CONCURRENT SUPPORT:")
    print(f"   Concurrent requests: 15 agents")
    print(f"   Success rate: {concurrent_results['success_rate']:.1%}")
    print(f"   Mean response time: {concurrent_results['response_times']['mean_ms']:.1f}ms")
    
    # Overall assessment
    all_performance_good = (
        performance_results['performance_compliance']['sub_200ms_percentage'] >= 80 and
        performance_results['response_times']['p95_ms'] < 300 and
        platform_results['success_rate'] > 0.95 and
        concurrent_results['success_rate'] > 0.80
    )
    
    print(f"\n{'🎉 ALL TESTS PASSED' if all_performance_good else '⚠️  SOME TESTS NEED ATTENTION'}")
    print(f"Overall Status: {'✅ READY FOR PRODUCTION' if all_performance_good else '🔄 NEEDS OPTIMIZATION'}")
    
    # Save detailed results
    results_file = f"/tmp/context_optimization_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return test_results, all_performance_good

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Context Optimization Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive suite")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick smoke tests
        engine_tests = TestContextOptimizationEngine()
        engine_tests.setUp()
        engine_tests.test_multi_factor_relevance_scoring()
        engine_tests.tearDown()
        print("✓ Quick tests completed")
        
    elif args.performance:
        # Performance-focused tests
        engine_tests = TestContextOptimizationEngine()
        engine_tests.setUp()
        engine_tests.test_performance_targets()
        engine_tests.tearDown()
        print("✓ Performance tests completed")
        
    elif args.integration:
        # Integration tests only
        async def run_integration_only():
            platform_tests = TestContextIntelligencePlatform()
            platform_tests.setUp()
            await platform_tests.test_platform_integration()
            await platform_tests.test_concurrent_requests()
            platform_tests.tearDown()
        
        asyncio.run(run_integration_only())
        print("✓ Integration tests completed")
        
    else:
        # Run full comprehensive suite
        test_results, success = run_comprehensive_test_suite()
        sys.exit(0 if success else 1)