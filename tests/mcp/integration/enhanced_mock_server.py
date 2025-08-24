"""
Enhanced Mock MCP Server Framework for Integration Testing

Provides sophisticated mock servers with configurable responses, performance simulation,
and failure scenarios for comprehensive MCP integration testing.

Issue: #86 - Build MCP integration tests
Component: Enhanced Mock Server Framework
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import AsyncMock


class HealthState(Enum):
    """Server health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass
class MockServerConfig:
    """Configuration for enhanced mock server"""
    server_id: str
    server_type: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    
    # Performance simulation
    base_response_time_ms: int = 100
    response_time_variance: float = 0.3
    max_concurrent_requests: int = 10
    
    # Resource simulation
    memory_usage_mb: int = 64
    cpu_usage_percent: int = 5
    
    # Failure simulation
    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    recovery_time_ms: int = 1000
    
    # Custom response handlers
    custom_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """Track request metrics"""
    timestamp: float
    request_type: str
    response_time_ms: float
    success: bool
    error_type: Optional[str] = None


class EnhancedMockMCPServer:
    """
    Enhanced mock MCP server with sophisticated simulation capabilities
    
    Features:
    - Configurable response scenarios
    - Performance variation simulation
    - Health state management
    - Resource usage tracking
    - Failure and recovery scenarios
    - Request metrics collection
    """
    
    def __init__(self, config: MockServerConfig):
        """Initialize enhanced mock server"""
        self.config = config
        self.server_id = config.server_id
        self.name = config.name
        self.server_type = config.server_type
        
        # State tracking
        self.is_running = False
        self.health_state = HealthState.HEALTHY
        self.current_requests = 0
        self.request_metrics: List[RequestMetrics] = []
        
        # Failure simulation state
        self._failure_state = False
        self._recovery_task: Optional[asyncio.Task] = None
        self._restart_called = False
        
        # Custom response handlers
        self._response_handlers: Dict[str, Callable] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default response handlers based on server type"""
        if self.server_type == "github":
            self._response_handlers.update({
                "get_repository_info": self._mock_github_repo_response,
                "list_issues": self._mock_github_issues_response,
                "get_pull_request": self._mock_github_pr_response,
                "create_issue_comment": self._mock_github_comment_response
            })
        elif self.server_type == "memory":
            self._response_handlers.update({
                "store_memory": self._mock_memory_store_response,
                "retrieve_memory": self._mock_memory_retrieve_response,
                "search_memory": self._mock_memory_search_response,
                "get_context": self._mock_memory_context_response
            })
        elif self.server_type == "sequential_thinking":
            self._response_handlers.update({
                "start_reasoning": self._mock_thinking_start_response,
                "continue_reasoning": self._mock_thinking_continue_response,
                "get_conclusion": self._mock_thinking_conclusion_response
            })
    
    async def initialize(self):
        """Initialize the mock server"""
        await self._simulate_startup_delay()
        self.is_running = True
        self.health_state = HealthState.HEALTHY
    
    async def _simulate_startup_delay(self):
        """Simulate server startup time"""
        startup_time = random.uniform(0.1, 0.5)
        await asyncio.sleep(startup_time)
    
    async def health_check(self) -> str:
        """Perform health check with sophisticated state management"""
        if not self.is_running:
            return HealthState.UNHEALTHY.value
        
        if self._failure_state:
            return HealthState.UNHEALTHY.value
        
        # Simulate load-based health degradation
        if self.current_requests > self.config.max_concurrent_requests * 0.8:
            self.health_state = HealthState.DEGRADED
        elif self.current_requests > self.config.max_concurrent_requests:
            self.health_state = HealthState.UNHEALTHY
        else:
            self.health_state = HealthState.HEALTHY
        
        return self.health_state.value
    
    def set_health(self, healthy: bool):
        """Manually set health state for testing"""
        self._failure_state = not healthy
        if not healthy:
            self.health_state = HealthState.UNHEALTHY
        else:
            self.health_state = HealthState.HEALTHY
    
    @property
    def restart_called(self) -> bool:
        """Check if restart was called"""
        return self._restart_called
    
    async def restart(self):
        """Restart the server (for testing failure recovery)"""
        self._restart_called = True
        self.is_running = False
        await asyncio.sleep(0.1)  # Simulate restart time
        self._failure_state = False
        self.health_state = HealthState.HEALTHY
        self.is_running = True
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with enhanced simulation"""
        start_time = time.time()
        
        # Track concurrent requests
        self.current_requests += 1
        
        try:
            # Simulate failure scenarios
            if await self._should_simulate_failure():
                raise Exception(f"Simulated failure in {tool_name}")
            
            if await self._should_simulate_timeout():
                await asyncio.sleep(10)  # Simulate timeout
                raise asyncio.TimeoutError(f"Timeout in {tool_name}")
            
            # Simulate processing time
            await self._simulate_processing_time()
            
            # Get response from handler
            if tool_name in self._response_handlers:
                result = await self._response_handlers[tool_name](parameters)
            else:
                result = await self._default_tool_response(tool_name, parameters)
            
            # Record metrics
            response_time_ms = (time.time() - start_time) * 1000
            self._record_metrics(tool_name, response_time_ms, True)
            
            return result
            
        except Exception as e:
            # Record failure metrics
            response_time_ms = (time.time() - start_time) * 1000
            self._record_metrics(tool_name, response_time_ms, False, str(type(e).__name__))
            raise
            
        finally:
            self.current_requests -= 1
    
    async def _should_simulate_failure(self) -> bool:
        """Check if we should simulate a failure"""
        return random.random() < self.config.failure_rate
    
    async def _should_simulate_timeout(self) -> bool:
        """Check if we should simulate a timeout"""
        return random.random() < self.config.timeout_rate
    
    async def _simulate_processing_time(self):
        """Simulate variable processing time"""
        base_time = self.config.base_response_time_ms / 1000
        variance = base_time * self.config.response_time_variance
        processing_time = random.uniform(base_time - variance, base_time + variance)
        await asyncio.sleep(max(0.001, processing_time))
    
    def _record_metrics(self, request_type: str, response_time_ms: float, 
                       success: bool, error_type: Optional[str] = None):
        """Record request metrics"""
        metric = RequestMetrics(
            timestamp=time.time(),
            request_type=request_type,
            response_time_ms=response_time_ms,
            success=success,
            error_type=error_type
        )
        self.request_metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.request_metrics) > 1000:
            self.request_metrics = self.request_metrics[-1000:]
    
    # Default response handlers for different server types
    
    async def _mock_github_repo_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock GitHub repository response"""
        return {
            "repository": {
                "name": "test-repo",
                "full_name": "test-org/test-repo",
                "description": "Mock repository for testing",
                "html_url": "https://github.com/test-org/test-repo",
                "default_branch": "main",
                "open_issues_count": random.randint(1, 50),
                "stargazers_count": random.randint(10, 1000),
                "language": "Python"
            }
        }
    
    async def _mock_github_issues_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock GitHub issues list response"""
        issues = []
        for i in range(random.randint(1, 10)):
            issues.append({
                "number": i + 1,
                "title": f"Mock Issue #{i + 1}",
                "state": random.choice(["open", "closed"]),
                "html_url": f"https://github.com/test-org/test-repo/issues/{i + 1}",
                "user": {"login": "mock-user"},
                "labels": [{"name": "bug"}, {"name": "enhancement"}][random.randint(0, 1):random.randint(1, 2)]
            })
        return {"issues": issues}
    
    async def _mock_github_pr_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock GitHub pull request response"""
        return {
            "pull_request": {
                "number": parameters.get("pr_number", 1),
                "title": "Mock Pull Request",
                "state": "open",
                "html_url": "https://github.com/test-org/test-repo/pull/1",
                "user": {"login": "mock-user"},
                "head": {"ref": "feature-branch"},
                "base": {"ref": "main"},
                "mergeable": True
            }
        }
    
    async def _mock_github_comment_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock GitHub comment creation response"""
        return {
            "comment": {
                "id": random.randint(1000000, 9999999),
                "html_url": "https://github.com/test-org/test-repo/issues/1#issuecomment-123456789",
                "body": parameters.get("body", "Mock comment"),
                "user": {"login": "mock-bot"}
            }
        }
    
    async def _mock_memory_store_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock memory storage response"""
        return {
            "stored": True,
            "memory_id": f"mem_{random.randint(1000, 9999)}",
            "key": parameters.get("key", "mock_key"),
            "timestamp": time.time()
        }
    
    async def _mock_memory_retrieve_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock memory retrieval response"""
        return {
            "found": True,
            "data": {
                "content": "Mock memory content",
                "metadata": {"created": time.time() - 3600},
                "relevance_score": random.uniform(0.7, 1.0)
            }
        }
    
    async def _mock_memory_search_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock memory search response"""
        results = []
        for i in range(random.randint(1, 5)):
            results.append({
                "memory_id": f"mem_{random.randint(1000, 9999)}",
                "content": f"Mock search result {i + 1}",
                "relevance_score": random.uniform(0.5, 1.0),
                "timestamp": time.time() - random.randint(0, 86400)
            })
        return {"results": results, "total_count": len(results)}
    
    async def _mock_memory_context_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock memory context response"""
        return {
            "context": {
                "summary": "Mock context summary",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "related_memories": random.randint(3, 10),
                "confidence_score": random.uniform(0.6, 0.9)
            }
        }
    
    async def _mock_thinking_start_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock reasoning start response"""
        return {
            "reasoning_session": {
                "session_id": f"think_{random.randint(1000, 9999)}",
                "problem": parameters.get("problem", "Mock problem"),
                "status": "started",
                "initial_thoughts": ["Initial analysis", "Problem breakdown", "Approach consideration"]
            }
        }
    
    async def _mock_thinking_continue_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock reasoning continuation response"""
        return {
            "reasoning_step": {
                "step_number": parameters.get("step", 1),
                "thoughts": ["Continuing analysis", "Evaluating options", "Considering implications"],
                "conclusions": ["Intermediate conclusion"],
                "next_steps": ["Further analysis needed"]
            }
        }
    
    async def _mock_thinking_conclusion_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock reasoning conclusion response"""
        return {
            "conclusion": {
                "final_answer": "Mock reasoning conclusion",
                "confidence_level": random.uniform(0.7, 0.95),
                "reasoning_chain": ["Step 1", "Step 2", "Step 3", "Conclusion"],
                "alternative_approaches": ["Alternative 1", "Alternative 2"]
            }
        }
    
    async def _default_tool_response(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Default response for unknown tools"""
        return {
            "tool": tool_name,
            "result": "success",
            "output": f"Mock execution of {tool_name}",
            "parameters_count": len(parameters),
            "execution_time_ms": random.randint(50, 300)
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        if not self.request_metrics:
            return {"total_requests": 0}
        
        successful_requests = [m for m in self.request_metrics if m.success]
        failed_requests = [m for m in self.request_metrics if not m.success]
        
        response_times = [m.response_time_ms for m in self.request_metrics]
        
        return {
            "total_requests": len(self.request_metrics),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(self.request_metrics) if self.request_metrics else 0,
            "average_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "current_concurrent_requests": self.current_requests
        }
    
    async def cleanup(self):
        """Clean up server resources"""
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
        
        self.is_running = False
        await asyncio.sleep(0.05)  # Simulate cleanup time
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive server status"""
        return {
            "server_id": self.server_id,
            "name": self.name,
            "server_type": self.server_type,
            "running": self.is_running,
            "health_state": self.health_state.value,
            "current_requests": self.current_requests,
            "capabilities": self.config.capabilities,
            "metrics": self.get_metrics_summary(),
            "config": {
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "failure_rate": self.config.failure_rate,
                "timeout_rate": self.config.timeout_rate,
                "base_response_time_ms": self.config.base_response_time_ms
            }
        }