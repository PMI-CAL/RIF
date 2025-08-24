"""
Base Classes for MCP Integration Testing

Provides foundational test classes, utilities, and frameworks for comprehensive
MCP integration testing with sophisticated mock management and metrics collection.

Issue: #86 - Build MCP integration tests
Component: Test Base Classes
"""

import asyncio
import pytest
import time
import statistics
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from unittest.mock import AsyncMock, patch

from .enhanced_mock_server import EnhancedMockMCPServer, MockServerConfig, HealthState
from .mock_response_templates import ResponseTemplateManager, ResponseScenario, TestScenarios
from .performance_metrics import PerformanceMetrics, BenchmarkResult, TestMetrics


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests"""
    test_name: str
    timeout_seconds: float = 30.0
    max_concurrent_requests: int = 10
    expected_success_rate: float = 0.95
    max_response_time_ms: float = 1000.0
    enable_metrics_collection: bool = True
    mock_server_configs: Dict[str, MockServerConfig] = field(default_factory=dict)


class MCPIntegrationTestBase(ABC):
    """
    Base class for MCP integration tests
    
    Provides common functionality for:
    - Mock server management
    - Performance metrics collection  
    - Test scenario execution
    - Result validation
    """
    
    def __init__(self, config: IntegrationTestConfig):
        """Initialize base test class"""
        self.config = config
        self.mock_servers: Dict[str, EnhancedMockMCPServer] = {}
        self.template_manager = ResponseTemplateManager()
        self.performance_metrics = PerformanceMetrics(config.test_name)
        self._test_start_time = 0.0
        self._cleanup_tasks: List[Callable] = []
    
    async def setup_method(self):
        """Setup method called before each test"""
        self._test_start_time = time.time()
        await self.setup_mock_servers()
        self.performance_metrics.reset()
    
    async def teardown_method(self):
        """Teardown method called after each test"""
        await self.cleanup_mock_servers()
        
        # Execute any registered cleanup tasks
        for cleanup_task in self._cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(cleanup_task):
                    await cleanup_task()
                else:
                    cleanup_task()
            except Exception as e:
                print(f"Warning: Cleanup task failed: {e}")
        
        self._cleanup_tasks.clear()
    
    def register_cleanup(self, cleanup_task: Callable):
        """Register a cleanup task to be executed during teardown"""
        self._cleanup_tasks.append(cleanup_task)
    
    async def setup_mock_servers(self):
        """Setup mock servers based on configuration"""
        for server_name, server_config in self.config.mock_server_configs.items():
            mock_server = EnhancedMockMCPServer(server_config)
            await mock_server.initialize()
            self.mock_servers[server_name] = mock_server
    
    async def cleanup_mock_servers(self):
        """Clean up all mock servers"""
        cleanup_tasks = []
        for server in self.mock_servers.values():
            cleanup_tasks.append(server.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.mock_servers.clear()
    
    def get_mock_server(self, server_name: str) -> Optional[EnhancedMockMCPServer]:
        """Get mock server by name"""
        return self.mock_servers.get(server_name)
    
    async def execute_with_metrics(self, operation: Callable, operation_name: str) -> Any:
        """Execute operation with metrics collection"""
        start_time = time.time()
        error = None
        result = None
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
            success = True
        except Exception as e:
            error = e
            success = False
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            if self.config.enable_metrics_collection:
                self.performance_metrics.record_operation(
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=success,
                    error_type=type(error).__name__ if error else None
                )
        
        return result
    
    async def simulate_concurrent_requests(self, 
                                          request_func: Callable,
                                          concurrency_level: int,
                                          request_count: int = None) -> List[Any]:
        """Simulate concurrent requests for performance testing"""
        if request_count is None:
            request_count = concurrency_level * 5
        
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrency_level)
        
        async def controlled_request():
            async with semaphore:
                return await request_func()
        
        # Execute requests
        tasks = [controlled_request() for _ in range(request_count)]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def wait_for_server_health(self, 
                                   server_name: str, 
                                   expected_health: HealthState,
                                   timeout_seconds: float = 10.0) -> bool:
        """Wait for server to reach expected health state"""
        server = self.get_mock_server(server_name)
        if not server:
            return False
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            current_health = await server.health_check()
            if current_health == expected_health.value:
                return True
            await asyncio.sleep(0.1)
        
        return False
    
    def assert_performance_requirements(self):
        """Assert that performance requirements are met"""
        metrics = self.performance_metrics.get_summary()
        
        # Check success rate
        if metrics['success_rate'] < self.config.expected_success_rate:
            raise AssertionError(
                f"Success rate {metrics['success_rate']:.3f} below required "
                f"{self.config.expected_success_rate:.3f}"
            )
        
        # Check response time
        if metrics['average_response_time_ms'] > self.config.max_response_time_ms:
            raise AssertionError(
                f"Average response time {metrics['average_response_time_ms']:.1f}ms exceeds "
                f"maximum {self.config.max_response_time_ms:.1f}ms"
            )
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get comprehensive test results"""
        test_duration = time.time() - self._test_start_time
        
        return {
            "test_name": self.config.test_name,
            "test_duration_seconds": test_duration,
            "performance_metrics": self.performance_metrics.get_summary(),
            "server_metrics": {
                name: server.get_metrics_summary()
                for name, server in self.mock_servers.items()
            },
            "requirements_met": {
                "success_rate": self.performance_metrics.get_summary()['success_rate'] >= self.config.expected_success_rate,
                "response_time": self.performance_metrics.get_summary()['average_response_time_ms'] <= self.config.max_response_time_ms
            }
        }
    
    @abstractmethod
    async def run_test_scenario(self) -> Dict[str, Any]:
        """Run the specific test scenario - to be implemented by subclasses"""
        pass


class MockServerManager:
    """Utility class for managing multiple mock servers"""
    
    def __init__(self):
        self.servers: Dict[str, EnhancedMockMCPServer] = {}
        self.template_manager = ResponseTemplateManager()
    
    async def create_server(self, 
                           server_name: str,
                           server_type: str,
                           **config_overrides) -> EnhancedMockMCPServer:
        """Create and initialize a mock server"""
        config = MockServerConfig(
            server_id=f"mock_{server_name}",
            server_type=server_type,
            name=server_name,
            **config_overrides
        )
        
        server = EnhancedMockMCPServer(config)
        await server.initialize()
        self.servers[server_name] = server
        
        return server
    
    async def create_github_server(self, **config_overrides) -> EnhancedMockMCPServer:
        """Create GitHub MCP mock server with realistic defaults"""
        defaults = {
            'capabilities': ['repository_info', 'issues', 'pull_requests', 'comments'],
            'base_response_time_ms': 150,
            'max_concurrent_requests': 15
        }
        defaults.update(config_overrides)
        
        return await self.create_server("github", "github", **defaults)
    
    async def create_memory_server(self, **config_overrides) -> EnhancedMockMCPServer:
        """Create Memory MCP mock server with realistic defaults"""
        defaults = {
            'capabilities': ['memory_storage', 'memory_retrieval', 'context_search'],
            'base_response_time_ms': 80,
            'max_concurrent_requests': 20
        }
        defaults.update(config_overrides)
        
        return await self.create_server("memory", "memory", **defaults)
    
    async def create_thinking_server(self, **config_overrides) -> EnhancedMockMCPServer:
        """Create Sequential Thinking MCP mock server with realistic defaults"""
        defaults = {
            'capabilities': ['reasoning', 'analysis', 'decision_making'],
            'base_response_time_ms': 300,
            'max_concurrent_requests': 5
        }
        defaults.update(config_overrides)
        
        return await self.create_server("sequential_thinking", "sequential_thinking", **defaults)
    
    def get_server(self, server_name: str) -> Optional[EnhancedMockMCPServer]:
        """Get server by name"""
        return self.servers.get(server_name)
    
    async def simulate_server_failure(self, server_name: str, duration_seconds: float = 5.0):
        """Simulate server failure for testing recovery scenarios"""
        server = self.get_server(server_name)
        if server:
            server.set_health(False)
            await asyncio.sleep(duration_seconds)
            await server.restart()
    
    async def cleanup_all(self):
        """Clean up all managed servers"""
        cleanup_tasks = []
        for server in self.servers.values():
            cleanup_tasks.append(server.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.servers.clear()
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all servers"""
        return {
            name: server.get_metrics_summary()
            for name, server in self.servers.items()
        }


class PerformanceBenchmarkBase:
    """Base class for performance benchmarking"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[BenchmarkResult] = []
    
    async def run_benchmark(self, 
                          benchmark_func: Callable,
                          concurrency_levels: List[int] = None,
                          iterations_per_level: int = 10) -> List[BenchmarkResult]:
        """Run performance benchmark across different concurrency levels"""
        
        if concurrency_levels is None:
            concurrency_levels = [1, 5, 10, 20]
        
        results = []
        
        for concurrency in concurrency_levels:
            level_results = []
            
            for iteration in range(iterations_per_level):
                start_time = time.time()
                
                try:
                    if asyncio.iscoroutinefunction(benchmark_func):
                        result = await benchmark_func(concurrency)
                    else:
                        result = benchmark_func(concurrency)
                    
                    duration_ms = (time.time() - start_time) * 1000
                    level_results.append(duration_ms)
                    
                except Exception as e:
                    print(f"Benchmark iteration failed: {e}")
                    continue
            
            if level_results:
                benchmark_result = BenchmarkResult(
                    concurrency_level=concurrency,
                    total_requests=concurrency * iterations_per_level,
                    successful_requests=len(level_results),
                    average_response_time_ms=statistics.mean(level_results),
                    min_response_time_ms=min(level_results),
                    max_response_time_ms=max(level_results),
                    throughput_requests_per_second=len(level_results) / (sum(level_results) / 1000) if level_results else 0,
                    success_rate=len(level_results) / iterations_per_level,
                    timestamp=time.time()
                )
                
                results.append(benchmark_result)
        
        self.results.extend(results)
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all benchmark results"""
        if not self.results:
            return {"message": "No benchmark results available"}
        
        return {
            "benchmark_name": self.name,
            "total_runs": len(self.results),
            "concurrency_levels": [r.concurrency_level for r in self.results],
            "best_throughput": max(r.throughput_requests_per_second for r in self.results),
            "average_success_rate": statistics.mean(r.success_rate for r in self.results),
            "response_time_stats": {
                "min": min(r.min_response_time_ms for r in self.results),
                "max": max(r.max_response_time_ms for r in self.results),
                "average": statistics.mean(r.average_response_time_ms for r in self.results)
            },
            "detailed_results": [
                {
                    "concurrency": r.concurrency_level,
                    "throughput": r.throughput_requests_per_second,
                    "success_rate": r.success_rate,
                    "avg_response_time_ms": r.average_response_time_ms
                }
                for r in self.results
            ]
        }


@asynccontextmanager
async def managed_mock_servers(*server_configs) -> AsyncGenerator[MockServerManager, None]:
    """Async context manager for automatic mock server lifecycle management"""
    manager = MockServerManager()
    
    try:
        # Create all requested servers
        for config in server_configs:
            if isinstance(config, dict):
                server_type = config.get('type', 'generic')
                server_name = config.get('name', server_type)
                await manager.create_server(server_name, server_type, **config.get('config', {}))
        
        yield manager
        
    finally:
        # Cleanup all servers
        await manager.cleanup_all()


# Pytest fixtures for common test scenarios
@pytest.fixture
async def mock_server_manager():
    """Pytest fixture providing a managed mock server manager"""
    manager = MockServerManager()
    yield manager
    await manager.cleanup_all()


@pytest.fixture
async def basic_mcp_servers(mock_server_manager):
    """Pytest fixture providing basic GitHub, Memory, and Sequential Thinking servers"""
    github_server = await mock_server_manager.create_github_server()
    memory_server = await mock_server_manager.create_memory_server()
    thinking_server = await mock_server_manager.create_thinking_server()
    
    return {
        'github': github_server,
        'memory': memory_server,
        'sequential_thinking': thinking_server
    }


@pytest.fixture
def integration_test_config():
    """Pytest fixture providing default integration test configuration"""
    return IntegrationTestConfig(
        test_name="default_integration_test",
        timeout_seconds=30.0,
        max_concurrent_requests=10,
        expected_success_rate=0.95,
        max_response_time_ms=1000.0,
        enable_metrics_collection=True
    )