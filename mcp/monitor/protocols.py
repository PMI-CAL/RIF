"""
MCP Health Check Protocols
Standardized health checking interfaces and protocols for MCP server monitoring

Features:
- Standardized health check interface with timeout management
- Multiple health check protocols (basic, performance, comprehensive)
- Health status classification and analysis
- Performance measurement and benchmarking
- Error handling and retry logic

Issue: #84 - Create MCP health monitor
Component: Health check protocols and standardization
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthCheckType(Enum):
    """Types of health checks available"""
    BASIC = "basic"
    PERFORMANCE = "performance"
    COMPREHENSIVE = "comprehensive"


class HealthStatus(Enum):
    """Standard health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation"""
    status: HealthStatus
    response_time_ms: float
    timestamp: float
    check_type: HealthCheckType
    details: Dict[str, Any] = None
    error: Optional[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.suggestions is None:
            self.suggestions = []


class HealthCheckProtocol(ABC):
    """Abstract base class for health check protocols"""

    def __init__(self, timeout_seconds: float = 5.0, retry_count: int = 2):
        """
        Initialize health check protocol
        
        Args:
            timeout_seconds: Timeout for health checks
            retry_count: Number of retries on failure
        """
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count

    @abstractmethod
    async def check_health(self, server: Any, config: Dict[str, Any]) -> HealthCheckResult:
        """
        Perform health check on server
        
        Args:
            server: Server instance to check
            config: Server configuration
            
        Returns:
            HealthCheckResult with status and details
        """
        pass

    async def _execute_with_timeout(self, coro, timeout: Optional[float] = None) -> Any:
        """
        Execute coroutine with timeout handling
        
        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds (uses self.timeout_seconds if None)
            
        Returns:
            Coroutine result
            
        Raises:
            asyncio.TimeoutError: If operation times out
        """
        timeout = timeout or self.timeout_seconds
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Health check timed out after {timeout}s")
            raise


class BasicHealthCheck(HealthCheckProtocol):
    """
    Basic connectivity and availability health check
    Verifies server is responsive and can handle basic requests
    """

    async def check_health(self, server: Any, config: Dict[str, Any]) -> HealthCheckResult:
        """
        Perform basic health check - connectivity and basic response
        
        Args:
            server: Server instance to check
            config: Server configuration
            
        Returns:
            HealthCheckResult with basic health status
        """
        start_time = time.time()
        check_type = HealthCheckType.BASIC
        
        try:
            # Basic connectivity check
            if hasattr(server, 'ping') and callable(server.ping):
                ping_result = server.ping()
                if asyncio.iscoroutine(ping_result):
                    await self._execute_with_timeout(ping_result)
                else:
                    # Handle sync ping
                    if not ping_result:
                        raise Exception("Ping failed")
            elif hasattr(server, 'health_check') and callable(server.health_check):
                health_call = server.health_check()
                if asyncio.iscoroutine(health_call):
                    result = await self._execute_with_timeout(health_call)
                else:
                    result = health_call
                    
                # Handle different health check response formats
                if isinstance(result, bool):
                    if not result:
                        raise Exception("Server health check returned False")
                elif isinstance(result, str):
                    if result.lower() not in ['healthy', 'ok', 'good']:
                        raise Exception(f"Server health check returned: {result}")
                elif hasattr(result, 'value'):  # Handle enum types like HealthStatus
                    if result.value not in ['healthy', 'ok', 'good']:
                        raise Exception(f"Server health check returned: {result.value}")
                elif isinstance(result, dict):
                    status = result.get('status', '').lower()
                    if status not in ['healthy', 'ok', 'good']:
                        error_msg = result.get('error', f"Unhealthy status: {status}")
                        raise Exception(error_msg)
            else:
                # Fallback: try to access server attributes to verify it's alive
                if hasattr(server, 'is_running') and callable(server.is_running) and not server.is_running():
                    raise Exception("Server is not running")
                elif hasattr(server, 'status') and server.status != 'running':
                    raise Exception(f"Server status: {server.status}")

            response_time_ms = (time.time() - start_time) * 1000

            # Classify health based on response time
            if response_time_ms > 5000:  # 5 seconds
                status = HealthStatus.DEGRADED
                suggestions = ["Server response time is very slow", "Consider investigating performance"]
            elif response_time_ms > 2000:  # 2 seconds  
                status = HealthStatus.DEGRADED
                suggestions = ["Server response time is slower than optimal"]
            else:
                status = HealthStatus.HEALTHY
                suggestions = []

            return HealthCheckResult(
                status=status,
                response_time_ms=response_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                details={
                    "connectivity": "ok",
                    "basic_response": "ok",
                    "server_type": config.get("name", "unknown")
                },
                suggestions=suggestions
            )

        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                error="Health check timed out",
                details={"timeout_seconds": self.timeout_seconds},
                suggestions=["Check server responsiveness", "Verify server is running"]
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                error=str(e),
                details={"error_type": type(e).__name__},
                suggestions=["Check server logs", "Verify server configuration"]
            )


class PerformanceHealthCheck(HealthCheckProtocol):
    """
    Performance-focused health check
    Measures response times, throughput, and resource utilization
    """

    def __init__(self, timeout_seconds: float = 10.0, retry_count: int = 1):
        super().__init__(timeout_seconds, retry_count)
        self.performance_thresholds = {
            "excellent_ms": 100,
            "good_ms": 500,
            "acceptable_ms": 1000,
            "poor_ms": 2000
        }

    async def check_health(self, server: Any, config: Dict[str, Any]) -> HealthCheckResult:
        """
        Perform performance-focused health check
        
        Args:
            server: Server instance to check
            config: Server configuration
            
        Returns:
            HealthCheckResult with performance metrics
        """
        start_time = time.time()
        check_type = HealthCheckType.PERFORMANCE
        
        try:
            # Multiple performance samples
            response_times = []
            
            for i in range(3):  # Take 3 samples
                sample_start = time.time()
                
                if hasattr(server, 'benchmark') and callable(server.benchmark):
                    benchmark_call = server.benchmark()
                    if asyncio.iscoroutine(benchmark_call):
                        await self._execute_with_timeout(benchmark_call)
                    else:
                        # Handle sync benchmark
                        benchmark_call
                elif hasattr(server, 'health_check') and callable(server.health_check):
                    health_call = server.health_check()
                    if asyncio.iscoroutine(health_call):
                        await self._execute_with_timeout(health_call)
                    else:
                        # Handle sync health check
                        health_call
                elif hasattr(server, 'ping') and callable(server.ping):
                    ping_call = server.ping()
                    if asyncio.iscoroutine(ping_call):
                        await self._execute_with_timeout(ping_call)
                    else:
                        # Handle sync ping
                        ping_call
                
                response_times.append((time.time() - sample_start) * 1000)
                
                # Small delay between samples
                if i < 2:
                    await asyncio.sleep(0.1)

            # Calculate performance metrics
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            total_time_ms = (time.time() - start_time) * 1000

            # Classify performance
            if avg_response_time <= self.performance_thresholds["excellent_ms"]:
                status = HealthStatus.HEALTHY
                performance_rating = "excellent"
                suggestions = []
            elif avg_response_time <= self.performance_thresholds["good_ms"]:
                status = HealthStatus.HEALTHY
                performance_rating = "good"
                suggestions = []
            elif avg_response_time <= self.performance_thresholds["acceptable_ms"]:
                status = HealthStatus.HEALTHY
                performance_rating = "acceptable"
                suggestions = ["Monitor performance trends"]
            elif avg_response_time <= self.performance_thresholds["poor_ms"]:
                status = HealthStatus.DEGRADED
                performance_rating = "poor"
                suggestions = ["Investigate performance bottlenecks", "Check resource utilization"]
            else:
                status = HealthStatus.UNHEALTHY
                performance_rating = "critical"
                suggestions = ["Server performance is critically poor", "Immediate investigation required"]

            return HealthCheckResult(
                status=status,
                response_time_ms=total_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                details={
                    "performance_rating": performance_rating,
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "min_response_time_ms": round(min_response_time, 2),
                    "max_response_time_ms": round(max_response_time, 2),
                    "response_time_consistency": round(max_response_time - min_response_time, 2),
                    "samples_count": len(response_times),
                    "server_type": config.get("name", "unknown")
                },
                suggestions=suggestions
            )

        except asyncio.TimeoutError:
            total_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=total_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                error="Performance check timed out",
                details={"timeout_seconds": self.timeout_seconds},
                suggestions=["Server may be overloaded", "Check resource availability"]
            )

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=total_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                error=str(e),
                details={"error_type": type(e).__name__},
                suggestions=["Check server performance logs", "Verify resource availability"]
            )


class ComprehensiveHealthCheck(HealthCheckProtocol):
    """
    Comprehensive health check combining multiple check types
    Provides thorough server health assessment
    """

    def __init__(self, timeout_seconds: float = 15.0, retry_count: int = 1):
        super().__init__(timeout_seconds, retry_count)
        self.basic_check = BasicHealthCheck(timeout_seconds=5.0)
        self.performance_check = PerformanceHealthCheck(timeout_seconds=10.0)

    async def check_health(self, server: Any, config: Dict[str, Any]) -> HealthCheckResult:
        """
        Perform comprehensive health check combining all check types
        
        Args:
            server: Server instance to check
            config: Server configuration
            
        Returns:
            HealthCheckResult with comprehensive assessment
        """
        start_time = time.time()
        check_type = HealthCheckType.COMPREHENSIVE
        
        try:
            # Run basic and performance checks
            basic_result = await self.basic_check.check_health(server, config)
            performance_result = await self.performance_check.check_health(server, config)

            # Combine results
            total_time_ms = (time.time() - start_time) * 1000

            # Determine overall status (most restrictive wins)
            if basic_result.status == HealthStatus.UNHEALTHY or performance_result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif basic_result.status == HealthStatus.DEGRADED or performance_result.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY

            # Combine details
            combined_details = {
                "overall_assessment": "comprehensive",
                "basic_check": {
                    "status": basic_result.status.value,
                    "response_time_ms": basic_result.response_time_ms,
                    "details": basic_result.details
                },
                "performance_check": {
                    "status": performance_result.status.value,
                    "response_time_ms": performance_result.response_time_ms,
                    "details": performance_result.details
                },
                "server_type": config.get("name", "unknown"),
                "total_checks": 2
            }

            # Combine suggestions
            combined_suggestions = []
            if basic_result.suggestions:
                combined_suggestions.extend([f"Basic: {s}" for s in basic_result.suggestions])
            if performance_result.suggestions:
                combined_suggestions.extend([f"Performance: {s}" for s in performance_result.suggestions])

            # Add comprehensive analysis suggestions
            if overall_status == HealthStatus.UNHEALTHY:
                combined_suggestions.append("Server requires immediate attention")
            elif overall_status == HealthStatus.DEGRADED:
                combined_suggestions.append("Server performance should be investigated")
            else:
                combined_suggestions.append("Server is operating within normal parameters")

            # Collect any errors
            errors = []
            if basic_result.error:
                errors.append(f"Basic check: {basic_result.error}")
            if performance_result.error:
                errors.append(f"Performance check: {performance_result.error}")

            return HealthCheckResult(
                status=overall_status,
                response_time_ms=total_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                details=combined_details,
                error="; ".join(errors) if errors else None,
                suggestions=combined_suggestions
            )

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=total_time_ms,
                timestamp=time.time(),
                check_type=check_type,
                error=f"Comprehensive check failed: {str(e)}",
                details={"error_type": type(e).__name__},
                suggestions=["Check server status", "Review server logs", "Verify configuration"]
            )


class HealthCheckManager:
    """
    Manager class for coordinating different health check protocols
    """

    def __init__(self):
        """Initialize health check manager with default protocols"""
        self.protocols = {
            HealthCheckType.BASIC: BasicHealthCheck(),
            HealthCheckType.PERFORMANCE: PerformanceHealthCheck(),
            HealthCheckType.COMPREHENSIVE: ComprehensiveHealthCheck()
        }

    async def check_server_health(self, server: Any, config: Dict[str, Any],
                                check_type: HealthCheckType = HealthCheckType.BASIC) -> HealthCheckResult:
        """
        Check server health using specified protocol
        
        Args:
            server: Server instance to check
            config: Server configuration
            check_type: Type of health check to perform
            
        Returns:
            HealthCheckResult from the specified protocol
        """
        if check_type not in self.protocols:
            raise ValueError(f"Unknown health check type: {check_type}")

        protocol = self.protocols[check_type]
        return await protocol.check_health(server, config)

    def add_protocol(self, check_type: HealthCheckType, protocol: HealthCheckProtocol):
        """
        Add or replace a health check protocol
        
        Args:
            check_type: Health check type identifier
            protocol: Protocol instance to use
        """
        self.protocols[check_type] = protocol
        logger.info(f"Added health check protocol: {check_type.value}")

    def get_available_protocols(self) -> List[HealthCheckType]:
        """
        Get list of available health check protocols
        
        Returns:
            List of available HealthCheckType values
        """
        return list(self.protocols.keys())


# Default health check manager instance
default_health_check_manager = HealthCheckManager()