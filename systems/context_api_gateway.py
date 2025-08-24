#!/usr/bin/env python3
"""
Context Intelligence Platform - API Gateway
Issue #119: DPIBS Architecture Phase 1

API Gateway with role-based access, request routing, and caching.
Implements the 3-layer integration system's API Gateway component.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import threading
from functools import wraps
import re

# Import platform components
from context_intelligence_platform import (
    ContextIntelligencePlatform, ContextRequest, ContextResponse, 
    AgentType, PerformanceMetrics, ServiceStatus
)

logger = logging.getLogger(__name__)

class AccessLevel(Enum):
    """Access levels for role-based security"""
    PUBLIC = "public"
    AGENT = "agent" 
    ADMIN = "admin"
    SYSTEM = "system"

class RoutePattern(Enum):
    """API route patterns"""
    HEALTH = "/health"
    CONTEXT_REQUEST = "/context/{agent_type}"
    CONTEXT_BATCH = "/context/batch"
    PERFORMANCE_STATS = "/stats"
    CACHE_MANAGEMENT = "/cache/{operation}"
    SERVICE_STATUS = "/services"

@dataclass
class APIRequest:
    """API request with authentication and routing info"""
    request_id: str
    route: str
    method: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    client_ip: str
    user_agent: str
    timestamp: datetime
    auth_token: Optional[str] = None

@dataclass 
class APIResponse:
    """Standardized API response"""
    request_id: str
    status_code: int
    data: Any
    error_message: Optional[str] = None
    response_time_ms: float = 0
    cache_hit: bool = False
    warnings: List[str] = None

class RateLimiter:
    """Rate limiting for API requests"""
    
    def __init__(self):
        self.requests = {}  # client_ip -> [(timestamp, count), ...]
        self.limits = {
            AccessLevel.PUBLIC: (10, 60),    # 10 requests per minute
            AccessLevel.AGENT: (100, 60),    # 100 requests per minute
            AccessLevel.ADMIN: (1000, 60),   # 1000 requests per minute
            AccessLevel.SYSTEM: (10000, 60)  # 10000 requests per minute
        }
        self.lock = threading.RLock()
    
    def is_allowed(self, client_ip: str, access_level: AccessLevel) -> bool:
        """Check if request is within rate limits"""
        with self.lock:
            now = datetime.now()
            limit, window_seconds = self.limits[access_level]
            
            # Clean old requests
            if client_ip in self.requests:
                cutoff = now - timedelta(seconds=window_seconds)
                self.requests[client_ip] = [
                    (ts, count) for ts, count in self.requests[client_ip]
                    if ts > cutoff
                ]
            else:
                self.requests[client_ip] = []
            
            # Count current requests
            current_count = sum(count for _, count in self.requests[client_ip])
            
            if current_count >= limit:
                return False
            
            # Add this request
            self.requests[client_ip].append((now, 1))
            return True

class AuthenticationManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        # Simulated auth tokens - in production, integrate with proper auth system
        self.tokens = {
            "agent_token_123": {"access_level": AccessLevel.AGENT, "agent_type": "rif-implementer"},
            "admin_token_456": {"access_level": AccessLevel.ADMIN, "permissions": ["all"]},
            "system_token_789": {"access_level": AccessLevel.SYSTEM, "service": "context-platform"}
        }
        
        self.public_routes = {RoutePattern.HEALTH.value}
    
    def authenticate(self, request: APIRequest) -> Dict[str, Any]:
        """Authenticate request and return auth context"""
        auth_token = request.auth_token or request.headers.get("Authorization", "").replace("Bearer ", "")
        
        # Check if route is public
        if request.route in self.public_routes:
            return {"access_level": AccessLevel.PUBLIC, "authenticated": False}
        
        # Validate token
        if not auth_token or auth_token not in self.tokens:
            return {"access_level": None, "authenticated": False, "error": "Invalid token"}
        
        token_info = self.tokens[auth_token]
        return {
            "access_level": token_info["access_level"],
            "authenticated": True,
            "token_info": token_info
        }
    
    def authorize(self, auth_context: Dict[str, Any], required_level: AccessLevel) -> bool:
        """Check if authenticated user has required access level"""
        if not auth_context.get("authenticated") and required_level != AccessLevel.PUBLIC:
            return False
        
        user_level = auth_context.get("access_level")
        if not user_level:
            return False
        
        # Access level hierarchy
        level_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.AGENT: 1,
            AccessLevel.ADMIN: 2,
            AccessLevel.SYSTEM: 3
        }
        
        return level_hierarchy.get(user_level, 0) >= level_hierarchy.get(required_level, 0)

class RequestRouter:
    """Intelligent request routing"""
    
    def __init__(self):
        self.routes = {}
        self._register_default_routes()
    
    def _register_default_routes(self):
        """Register default API routes"""
        self.routes = {
            re.compile(r"^/health$"): {
                "handler": "health_check",
                "method": ["GET"],
                "access_level": AccessLevel.PUBLIC,
                "cache_ttl": 30
            },
            re.compile(r"^/context/([a-z-]+)$"): {
                "handler": "context_request",
                "method": ["POST"],
                "access_level": AccessLevel.AGENT,
                "cache_ttl": 0  # Context requests not cached at API level
            },
            re.compile(r"^/context/batch$"): {
                "handler": "batch_context_request", 
                "method": ["POST"],
                "access_level": AccessLevel.AGENT,
                "cache_ttl": 0
            },
            re.compile(r"^/stats$"): {
                "handler": "performance_stats",
                "method": ["GET"],
                "access_level": AccessLevel.ADMIN,
                "cache_ttl": 60
            },
            re.compile(r"^/services$"): {
                "handler": "service_status",
                "method": ["GET"], 
                "access_level": AccessLevel.ADMIN,
                "cache_ttl": 30
            },
            re.compile(r"^/cache/([a-z]+)$"): {
                "handler": "cache_management",
                "method": ["POST", "DELETE"],
                "access_level": AccessLevel.ADMIN,
                "cache_ttl": 0
            }
        }
    
    def match_route(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Match request path to route configuration"""
        for pattern, config in self.routes.items():
            match = pattern.match(path)
            if match and method in config["method"]:
                return {
                    **config,
                    "path_params": match.groups(),
                    "pattern": pattern.pattern
                }
        return None

class ResponseCache:
    """Response caching for API Gateway"""
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.lock = threading.RLock()
    
    def get(self, cache_key: str) -> Optional[APIResponse]:
        """Get cached response"""
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if entry["expires_at"] > datetime.now():
                    self.cache_stats["hits"] += 1
                    response = entry["response"]
                    response.cache_hit = True
                    return response
                else:
                    # Expired
                    del self.cache[cache_key]
            
            self.cache_stats["misses"] += 1
            return None
    
    def set(self, cache_key: str, response: APIResponse, ttl_seconds: int):
        """Cache response with TTL"""
        if ttl_seconds <= 0:
            return
        
        with self.lock:
            self.cache[cache_key] = {
                "response": response,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=ttl_seconds)
            }
    
    def create_key(self, request: APIRequest) -> str:
        """Create cache key from request"""
        key_data = f"{request.route}|{request.method}|{json.dumps(request.body, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

class ContextAPIGateway:
    """API Gateway for Context Intelligence Platform"""
    
    def __init__(self, platform: ContextIntelligencePlatform):
        self.platform = platform
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.router = RequestRouter()
        self.response_cache = ResponseCache()
        self.request_counter = 0
        self.performance_metrics = []
        
        # Gateway status
        self.status = ServiceStatus.RUNNING
        self.startup_time = datetime.now()
    
    async def handle_request(self, route: str, method: str, headers: Dict[str, str],
                            body: Dict[str, Any], client_ip: str = "127.0.0.1",
                            user_agent: str = "unknown") -> APIResponse:
        """Handle incoming API request"""
        
        start_time = time.time()
        self.request_counter += 1
        
        # Create request object
        request = APIRequest(
            request_id=f"gw_{self.request_counter}_{int(time.time())}",
            route=route,
            method=method,
            headers=headers,
            body=body,
            client_ip=client_ip,
            user_agent=user_agent,
            timestamp=datetime.now(),
            auth_token=headers.get("Authorization", "").replace("Bearer ", "") if headers.get("Authorization") else None
        )
        
        try:
            # 1. Authentication
            auth_context = self.auth_manager.authenticate(request)
            if not auth_context.get("authenticated") and route not in {"/health"}:
                return APIResponse(
                    request_id=request.request_id,
                    status_code=401,
                    data=None,
                    error_message="Authentication required",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # 2. Route matching
            route_config = self.router.match_route(route, method)
            if not route_config:
                return APIResponse(
                    request_id=request.request_id,
                    status_code=404,
                    data=None,
                    error_message=f"Route not found: {method} {route}",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # 3. Authorization
            if not self.auth_manager.authorize(auth_context, route_config["access_level"]):
                return APIResponse(
                    request_id=request.request_id,
                    status_code=403,
                    data=None,
                    error_message="Insufficient permissions",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # 4. Rate limiting
            if not self.rate_limiter.is_allowed(client_ip, auth_context.get("access_level", AccessLevel.PUBLIC)):
                return APIResponse(
                    request_id=request.request_id,
                    status_code=429,
                    data=None,
                    error_message="Rate limit exceeded",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # 5. Check response cache
            if route_config["cache_ttl"] > 0:
                cache_key = self.response_cache.create_key(request)
                cached_response = self.response_cache.get(cache_key)
                if cached_response:
                    cached_response.request_id = request.request_id  # Update request ID
                    cached_response.response_time_ms = (time.time() - start_time) * 1000
                    return cached_response
            
            # 6. Route to handler
            response = await self._route_to_handler(request, route_config, auth_context)
            response.response_time_ms = (time.time() - start_time) * 1000
            
            # 7. Cache response if appropriate
            if route_config["cache_ttl"] > 0 and response.status_code == 200:
                cache_key = self.response_cache.create_key(request)
                self.response_cache.set(cache_key, response, route_config["cache_ttl"])
            
            # 8. Record metrics
            self._record_metrics(request, response, auth_context)
            
            return response
            
        except Exception as e:
            logger.error(f"API Gateway error: {e}")
            return APIResponse(
                request_id=request.request_id,
                status_code=500,
                data=None,
                error_message=f"Internal server error: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _route_to_handler(self, request: APIRequest, route_config: Dict[str, Any], 
                               auth_context: Dict[str, Any]) -> APIResponse:
        """Route request to appropriate handler"""
        handler_name = route_config["handler"]
        path_params = route_config.get("path_params", [])
        
        # Call the appropriate handler method
        if handler_name == "health_check":
            return await self._handle_health_check(request)
        elif handler_name == "context_request":
            return await self._handle_context_request(request, path_params[0], auth_context)
        elif handler_name == "batch_context_request":
            return await self._handle_batch_context_request(request, auth_context)
        elif handler_name == "performance_stats":
            return await self._handle_performance_stats(request)
        elif handler_name == "service_status":
            return await self._handle_service_status(request)
        elif handler_name == "cache_management":
            return await self._handle_cache_management(request, path_params[0])
        else:
            return APIResponse(
                request_id=request.request_id,
                status_code=501,
                data=None,
                error_message=f"Handler not implemented: {handler_name}"
            )
    
    async def _handle_health_check(self, request: APIRequest) -> APIResponse:
        """Handle health check requests"""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "version": "1.0.0",
            "services": {
                name: service.status.value 
                for name, service in self.platform.services.items()
            }
        }
        
        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            data=health_data
        )
    
    async def _handle_context_request(self, request: APIRequest, agent_type_str: str, 
                                    auth_context: Dict[str, Any]) -> APIResponse:
        """Handle single context requests"""
        try:
            # Parse agent type
            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                return APIResponse(
                    request_id=request.request_id,
                    status_code=400,
                    data=None,
                    error_message=f"Invalid agent type: {agent_type_str}"
                )
            
            # Validate request body
            task_context = request.body.get("task_context")
            if not task_context:
                return APIResponse(
                    request_id=request.request_id,
                    status_code=400,
                    data=None,
                    error_message="task_context is required"
                )
            
            issue_number = request.body.get("issue_number")
            priority = request.body.get("priority", 1)
            
            # Process context request
            context_response = await self.platform.process_context_request(
                agent_type, task_context, issue_number, priority
            )
            
            # Format response
            formatted_context = self.platform.format_context_for_agent(context_response)
            
            response_data = {
                "agent_type": agent_type.value,
                "context": formatted_context,
                "performance": {
                    "response_time_ms": context_response.total_response_time_ms,
                    "cache_hit": context_response.cache_hit,
                    "source_services": context_response.source_services
                },
                "metadata": {
                    "context_window_utilization": context_response.agent_context.context_window_utilization,
                    "total_size": context_response.agent_context.total_size,
                    "items_provided": len(context_response.agent_context.relevant_knowledge)
                }
            }
            
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                data=response_data,
                cache_hit=context_response.cache_hit
            )
            
        except Exception as e:
            logger.error(f"Context request error: {e}")
            return APIResponse(
                request_id=request.request_id,
                status_code=500,
                data=None,
                error_message=f"Context request failed: {str(e)}"
            )
    
    async def _handle_batch_context_request(self, request: APIRequest, 
                                          auth_context: Dict[str, Any]) -> APIResponse:
        """Handle batch context requests"""
        requests_data = request.body.get("requests", [])
        if not requests_data:
            return APIResponse(
                request_id=request.request_id,
                status_code=400,
                data=None,
                error_message="requests array is required"
            )
        
        # Process all requests concurrently
        tasks = []
        for req_data in requests_data:
            try:
                agent_type = AgentType(req_data["agent_type"])
                task_context = req_data["task_context"]
                issue_number = req_data.get("issue_number")
                priority = req_data.get("priority", 1)
                
                task = self.platform.process_context_request(
                    agent_type, task_context, issue_number, priority
                )
                tasks.append(task)
                
            except Exception as e:
                logger.error(f"Batch request parsing error: {e}")
        
        # Wait for all requests
        start_batch_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        batch_time_ms = (time.time() - start_batch_time) * 1000
        
        # Format batch response
        batch_results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                batch_results.append({
                    "index": i,
                    "success": False,
                    "error": str(response)
                })
            else:
                batch_results.append({
                    "index": i,
                    "success": True,
                    "agent_type": response.agent_context.agent_type.value,
                    "response_time_ms": response.total_response_time_ms,
                    "cache_hit": response.cache_hit
                })
        
        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            data={
                "batch_results": batch_results,
                "batch_performance": {
                    "total_requests": len(requests_data),
                    "successful": len([r for r in batch_results if r.get("success")]),
                    "batch_time_ms": batch_time_ms
                }
            }
        )
    
    async def _handle_performance_stats(self, request: APIRequest) -> APIResponse:
        """Handle performance statistics requests"""
        platform_stats = self.platform.get_performance_stats()
        gateway_stats = self._get_gateway_stats()
        
        combined_stats = {
            "platform": platform_stats,
            "gateway": gateway_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            data=combined_stats
        )
    
    async def _handle_service_status(self, request: APIRequest) -> APIResponse:
        """Handle service status requests"""
        service_status = {
            "gateway": {
                "status": self.status.value,
                "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
                "request_count": self.request_counter
            },
            "platform_services": {
                name: service.status.value 
                for name, service in self.platform.services.items()
            },
            "health_summary": {
                "healthy_services": len([s for s in self.platform.services.values() 
                                       if s.status == ServiceStatus.RUNNING]),
                "total_services": len(self.platform.services) + 1  # +1 for gateway
            }
        }
        
        return APIResponse(
            request_id=request.request_id,
            status_code=200,
            data=service_status
        )
    
    async def _handle_cache_management(self, request: APIRequest, operation: str) -> APIResponse:
        """Handle cache management operations"""
        if operation == "clear":
            # Clear all caches
            self.response_cache.cache.clear()
            self.platform.cache.caches[self.platform.cache.caches.keys()] = {}
            
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                data={"message": "All caches cleared", "operation": operation}
            )
        elif operation == "stats":
            cache_stats = {
                "gateway_cache": self.response_cache.cache_stats,
                "platform_cache": self.platform.cache.cache_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                data=cache_stats
            )
        else:
            return APIResponse(
                request_id=request.request_id,
                status_code=400,
                data=None,
                error_message=f"Unknown cache operation: {operation}"
            )
    
    def _record_metrics(self, request: APIRequest, response: APIResponse, 
                       auth_context: Dict[str, Any]):
        """Record request metrics"""
        metric = {
            "timestamp": request.timestamp.isoformat(),
            "request_id": request.request_id,
            "route": request.route,
            "method": request.method,
            "status_code": response.status_code,
            "response_time_ms": response.response_time_ms,
            "cache_hit": response.cache_hit,
            "client_ip": request.client_ip,
            "user_agent": request.user_agent,
            "access_level": auth_context.get("access_level", "").value if auth_context.get("access_level") else "none"
        }
        
        self.performance_metrics.append(metric)
        
        # Keep only last 1000 metrics in memory
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
    
    def _get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway-specific performance statistics"""
        if not self.performance_metrics:
            return {"status": "no_data"}
        
        recent_metrics = [
            m for m in self.performance_metrics 
            if datetime.fromisoformat(m["timestamp"]) > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return {"status": "no_recent_data"}
        
        response_times = [m["response_time_ms"] for m in recent_metrics]
        successful_requests = [m for m in recent_metrics if m["status_code"] < 400]
        cache_hits = [m for m in recent_metrics if m["cache_hit"]]
        
        return {
            "total_requests": len(recent_metrics),
            "successful_requests": len(successful_requests),
            "success_rate": len(successful_requests) / len(recent_metrics),
            "cache_hit_rate": len(cache_hits) / len(recent_metrics),
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "p95_response_time_ms": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
            "requests_by_route": self._group_metrics_by_field(recent_metrics, "route"),
            "requests_by_status": self._group_metrics_by_field(recent_metrics, "status_code"),
            "cache_stats": self.response_cache.cache_stats
        }
    
    def _group_metrics_by_field(self, metrics: List[Dict[str, Any]], field: str) -> Dict[str, int]:
        """Group metrics by field for statistics"""
        groups = {}
        for metric in metrics:
            value = str(metric.get(field, "unknown"))
            groups[value] = groups.get(value, 0) + 1
        return groups

# Testing and CLI Interface
async def main():
    """Main function for testing the API Gateway"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context API Gateway")
    parser.add_argument("--test", action="store_true", help="Run API tests")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Initialize platform and gateway
    platform = ContextIntelligencePlatform()
    gateway = ContextAPIGateway(platform)
    
    if args.test:
        print("=== API Gateway Test ===\n")
        
        # Test health check
        response = await gateway.handle_request("/health", "GET", {}, {})
        print(f"Health Check - Status: {response.status_code}, Time: {response.response_time_ms:.1f}ms")
        print(json.dumps(response.data, indent=2))
        print()
        
        # Test context request (will fail without auth)
        response = await gateway.handle_request("/context/rif-implementer", "POST", {}, {
            "task_context": {"description": "Test context request"},
            "issue_number": 119
        })
        print(f"Context Request (no auth) - Status: {response.status_code}")
        print(f"Error: {response.error_message}")
        print()
        
        # Test context request (with auth)
        response = await gateway.handle_request("/context/rif-implementer", "POST", {
            "Authorization": "Bearer agent_token_123"
        }, {
            "task_context": {"description": "Test context request with auth"},
            "issue_number": 119
        })
        print(f"Context Request (with auth) - Status: {response.status_code}, Time: {response.response_time_ms:.1f}ms")
        if response.data:
            print(f"Cache Hit: {response.cache_hit}")
            print(f"Performance: {response.data['performance']}")
        print()
        
        # Test performance stats
        response = await gateway.handle_request("/stats", "GET", {
            "Authorization": "Bearer admin_token_456"
        }, {})
        print(f"Performance Stats - Status: {response.status_code}")
        if response.data:
            print("Gateway Stats:", json.dumps(response.data["gateway"], indent=2))
        
    elif args.benchmark:
        print("=== API Gateway Benchmark ===\n")
        
        # Run concurrent requests
        tasks = []
        auth_header = {"Authorization": "Bearer agent_token_123"}
        test_body = {
            "task_context": {"description": "Benchmark test request"},
            "issue_number": 119
        }
        
        # Create 20 concurrent requests
        for i in range(20):
            task = gateway.handle_request("/context/rif-implementer", "POST", auth_header, test_body)
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful = [r for r in responses if r.status_code == 200]
        response_times = [r.response_time_ms for r in responses]
        
        print(f"Completed {len(successful)}/{len(responses)} requests in {total_time:.2f}s")
        print(f"Average response time: {sum(response_times)/len(response_times):.1f}ms")
        print(f"P95 response time: {sorted(response_times)[int(len(response_times)*0.95)]:.1f}ms")
        print(f"Cache hits: {len([r for r in responses if r.cache_hit])}")
        
        # Get final stats
        stats_response = await gateway.handle_request("/stats", "GET", {
            "Authorization": "Bearer admin_token_456"
        }, {})
        
        if stats_response.data:
            gateway_stats = stats_response.data["gateway"]
            print(f"\nGateway Performance:")
            print(f"- Success rate: {gateway_stats.get('success_rate', 0)*100:.1f}%")
            print(f"- Cache hit rate: {gateway_stats.get('cache_hit_rate', 0)*100:.1f}%")
            print(f"- Average response time: {gateway_stats.get('avg_response_time_ms', 0):.1f}ms")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())