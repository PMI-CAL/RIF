#!/usr/bin/env python3
"""
Context Intelligence Platform - Foundation Implementation
Issue #119: DPIBS Architecture Phase 1

Implements the comprehensive Context Intelligence Platform with:
- 4 core microservices architecture
- 3-layer integration system 
- Sub-200ms performance targets
- 100% backward compatibility

Based on RIF-Architect design specifications.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import os
import subprocess

# Import existing context optimization engine  
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("context_optimization_engine", "context-optimization-engine.py")
coe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coe_module)

# Import the classes we need
ContextOptimizer = coe_module.ContextOptimizer
AgentType = coe_module.AgentType
ContextType = coe_module.ContextType
ContextItem = coe_module.ContextItem
SystemContext = coe_module.SystemContext
AgentContext = coe_module.AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status tracking"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

class CacheLevel(Enum):
    """Multi-layer cache levels"""
    L1_EDGE = "l1_edge"      # Agent-specific, <10ms
    L2_QUERY = "l2_query"    # Processed results, <50ms
    L3_SOURCE = "l3_source"  # Raw data, <100ms

@dataclass
class PerformanceMetrics:
    """Performance tracking for sub-200ms targets"""
    request_id: str
    service_name: str
    start_time: float
    end_time: float
    cache_level: Optional[CacheLevel]
    agent_type: AgentType
    success: bool
    error_message: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

@dataclass
class ContextRequest:
    """Context request with optimization parameters"""
    request_id: str
    agent_type: AgentType
    task_context: Dict[str, Any]
    issue_number: Optional[int]
    priority: int = 1  # 1=high, 2=medium, 3=low
    max_response_time_ms: int = 200
    require_real_time: bool = False

@dataclass
class ContextResponse:
    """Context response with performance tracking"""
    request_id: str
    agent_context: AgentContext
    performance_metrics: PerformanceMetrics
    cache_hit: bool
    source_services: List[str]
    total_response_time_ms: float

class ContextCache:
    """Multi-layer intelligent caching system"""
    
    def __init__(self, base_path: str = "/Users/cal/DEV/RIF/systems/context"):
        self.base_path = base_path
        self.caches = {
            CacheLevel.L1_EDGE: {},    # In-memory agent-specific cache
            CacheLevel.L2_QUERY: {},   # In-memory processed results cache
            CacheLevel.L3_SOURCE: {}   # Persistent source data cache
        }
        self.cache_stats = {level: {"hits": 0, "misses": 0} for level in CacheLevel}
        self.lock = threading.RLock()
        self._initialize_persistent_cache()
    
    def _initialize_persistent_cache(self):
        """Initialize persistent L3 cache"""
        os.makedirs(self.base_path, exist_ok=True)
        self.l3_db_path = os.path.join(self.base_path, "context_cache.db")
        
        # Ensure directory exists and is writable
        try:
            with open(os.path.join(self.base_path, '.test'), 'w') as f:
                f.write('test')
            os.remove(os.path.join(self.base_path, '.test'))
        except (OSError, IOError) as e:
            # Fall back to /tmp if base_path is not writable
            self.base_path = "/tmp/context_cache"
            os.makedirs(self.base_path, exist_ok=True)
            self.l3_db_path = os.path.join(self.base_path, "context_cache.db")
        
        with sqlite3.connect(self.l3_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON context_cache(expires_at)
            """)
    
    def get(self, cache_key: str, level: CacheLevel) -> Optional[Any]:
        """Get cached item with intelligent level selection"""
        with self.lock:
            if level == CacheLevel.L3_SOURCE:
                return self._get_l3_persistent(cache_key)
            else:
                cache_data = self.caches[level].get(cache_key)
                if cache_data:
                    if cache_data["expires_at"] > datetime.now():
                        self.cache_stats[level]["hits"] += 1
                        return cache_data["data"]
                    else:
                        # Expired, remove from cache
                        del self.caches[level][cache_key]
                
                self.cache_stats[level]["misses"] += 1
                return None
    
    def set(self, cache_key: str, data: Any, level: CacheLevel, ttl_seconds: int):
        """Set cached item with TTL"""
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        with self.lock:
            if level == CacheLevel.L3_SOURCE:
                self._set_l3_persistent(cache_key, data, expires_at)
            else:
                self.caches[level][cache_key] = {
                    "data": data,
                    "created_at": datetime.now(),
                    "expires_at": expires_at
                }
    
    def _get_l3_persistent(self, cache_key: str) -> Optional[Any]:
        """Get from L3 persistent cache"""
        try:
            with sqlite3.connect(self.l3_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data FROM context_cache 
                    WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, datetime.now()))
                
                result = cursor.fetchone()
                if result:
                    # Update access stats
                    cursor.execute("""
                        UPDATE context_cache 
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE cache_key = ?
                    """, (datetime.now(), cache_key))
                    conn.commit()
                    
                    self.cache_stats[CacheLevel.L3_SOURCE]["hits"] += 1
                    return json.loads(result[0])
                else:
                    self.cache_stats[CacheLevel.L3_SOURCE]["misses"] += 1
                    return None
        except Exception as e:
            logger.error(f"L3 cache get error: {e}")
            return None
    
    def _set_l3_persistent(self, cache_key: str, data: Any, expires_at: datetime):
        """Set in L3 persistent cache"""
        try:
            with sqlite3.connect(self.l3_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO context_cache 
                    (cache_key, data, created_at, expires_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, 0, ?)
                """, (cache_key, json.dumps(data), datetime.now(), expires_at, datetime.now()))
                conn.commit()
        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
    
    def cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        
        with self.lock:
            # Clean memory caches
            for level in [CacheLevel.L1_EDGE, CacheLevel.L2_QUERY]:
                expired_keys = [
                    key for key, value in self.caches[level].items()
                    if value["expires_at"] <= current_time
                ]
                for key in expired_keys:
                    del self.caches[level][key]
            
            # Clean L3 persistent cache
            try:
                with sqlite3.connect(self.l3_db_path) as conn:
                    conn.execute("DELETE FROM context_cache WHERE expires_at <= ?", (current_time,))
                    conn.commit()
            except Exception as e:
                logger.error(f"L3 cache cleanup error: {e}")

class BaseService(ABC):
    """Base class for Context Intelligence Platform services"""
    
    def __init__(self, service_name: str, cache: ContextCache):
        self.service_name = service_name
        self.cache = cache
        self.status = ServiceStatus.INITIALIZING
        self.metrics = []
        self.performance_target_ms = 200
        
    @abstractmethod
    async def process_request(self, request: ContextRequest) -> Any:
        """Process service-specific request"""
        pass
    
    def _create_cache_key(self, *args) -> str:
        """Create cache key from arguments"""
        key_data = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _record_performance(self, request_id: str, start_time: float, 
                          success: bool, cache_level: Optional[CacheLevel] = None,
                          error_message: Optional[str] = None):
        """Record performance metrics"""
        metrics = PerformanceMetrics(
            request_id=request_id,
            service_name=self.service_name,
            start_time=start_time,
            end_time=time.time(),
            cache_level=cache_level,
            agent_type=AgentType.IMPLEMENTER,  # Will be set from request
            success=success,
            error_message=error_message
        )
        
        self.metrics.append(metrics)
        
        # Log if performance target missed
        if metrics.duration_ms > self.performance_target_ms:
            logger.warning(f"{self.service_name} exceeded target: {metrics.duration_ms:.1f}ms > {self.performance_target_ms}ms")
        
        return metrics

class ContextOptimizationService(BaseService):
    """Enhanced Context Optimization Engine Service"""
    
    def __init__(self, cache: ContextCache):
        super().__init__("context-optimization", cache)
        self.optimizer = ContextOptimizer()
        self.status = ServiceStatus.RUNNING
        
    async def process_request(self, request: ContextRequest) -> AgentContext:
        """Process context optimization request with caching"""
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key(
            request.agent_type.value,
            str(request.task_context),
            request.issue_number
        )
        
        # Try L1 cache first (agent-specific)
        l1_key = f"agent_{request.agent_type.value}_{cache_key}"
        cached_context = self.cache.get(l1_key, CacheLevel.L1_EDGE)
        
        if cached_context:
            metrics = self._record_performance(request.request_id, start_time, True, CacheLevel.L1_EDGE)
            return AgentContext(**cached_context)
        
        # Try L2 cache (processed results)
        cached_context = self.cache.get(cache_key, CacheLevel.L2_QUERY)
        
        if cached_context:
            metrics = self._record_performance(request.request_id, start_time, True, CacheLevel.L2_QUERY)
            agent_context = AgentContext(**cached_context)
            
            # Store in L1 for faster future access
            self.cache.set(l1_key, asdict(agent_context), CacheLevel.L1_EDGE, ttl_seconds=30)
            return agent_context
        
        try:
            # Generate new context
            agent_context = self.optimizer.optimize_for_agent(
                request.agent_type, 
                request.task_context, 
                request.issue_number
            )
            
            # Cache the results
            context_dict = asdict(agent_context)
            self.cache.set(cache_key, context_dict, CacheLevel.L2_QUERY, ttl_seconds=300)  # 5 minutes
            self.cache.set(l1_key, context_dict, CacheLevel.L1_EDGE, ttl_seconds=30)  # 30 seconds
            
            metrics = self._record_performance(request.request_id, start_time, True)
            return agent_context
            
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            metrics = self._record_performance(request.request_id, start_time, False, error_message=str(e))
            raise

class SystemContextMaintenanceService(BaseService):
    """Real-time system context maintenance service"""
    
    def __init__(self, cache: ContextCache):
        super().__init__("system-context-maintenance", cache)
        self.system_context_cache = {}
        self.last_update = datetime.now()
        self.status = ServiceStatus.RUNNING
        
    async def process_request(self, request: ContextRequest) -> SystemContext:
        """Get live system context with real-time updates"""
        start_time = time.time()
        
        # Check if system context needs refresh (5 minute max age)
        if (datetime.now() - self.last_update).seconds > 300:
            await self._refresh_system_context()
        
        try:
            # Get cached system context
            cache_key = "live_system_context"
            cached_context = self.cache.get(cache_key, CacheLevel.L2_QUERY)
            
            if cached_context:
                metrics = self._record_performance(request.request_id, start_time, True, CacheLevel.L2_QUERY)
                return SystemContext(**cached_context)
            
            # Generate fresh system context
            system_context = await self._generate_system_context()
            
            # Cache for future requests
            self.cache.set(cache_key, asdict(system_context), CacheLevel.L2_QUERY, ttl_seconds=300)
            
            metrics = self._record_performance(request.request_id, start_time, True)
            return system_context
            
        except Exception as e:
            logger.error(f"System context maintenance failed: {e}")
            metrics = self._record_performance(request.request_id, start_time, False, error_message=str(e))
            raise
    
    async def _refresh_system_context(self):
        """Refresh system context from live sources"""
        # This would integrate with git hooks, issue tracking, etc.
        self.last_update = datetime.now()
        logger.info("System context refreshed")
    
    async def _generate_system_context(self) -> SystemContext:
        """Generate fresh system context"""
        # Get live system state
        return SystemContext(
            overview="RIF Context Intelligence Platform - Live System Context",
            purpose="Provide real-time system understanding for intelligent agent context delivery",
            design_goals=[
                "Sub-200ms context query response",
                "Real-time system state tracking",
                "Multi-agent concurrent support",
                "Intelligent context optimization"
            ],
            constraints=[
                "Performance budget <200ms P95",
                "100% backward compatibility",
                "Multi-layer caching required",
                "Event-driven updates"
            ],
            dependencies={
                "context_optimization": "Core optimization engine",
                "mcp_servers": "Knowledge integration",
                "github_api": "Issue and PR tracking",
                "git_hooks": "Code change detection"
            },
            architecture_summary="Event-driven microservices with intelligent context aggregation and multi-layer caching",
            last_updated=datetime.now()
        )

class AgentContextDeliveryService(BaseService):
    """Role-specific context delivery service"""
    
    def __init__(self, cache: ContextCache):
        super().__init__("agent-context-delivery", cache)
        self.context_optimizer = ContextOptimizationService(cache)
        self.system_maintenance = SystemContextMaintenanceService(cache)
        self.status = ServiceStatus.RUNNING
        
    async def process_request(self, request: ContextRequest) -> ContextResponse:
        """Deliver optimized context to RIF agents"""
        start_time = time.time()
        
        try:
            # Get optimized agent context
            agent_context_task = self.context_optimizer.process_request(request)
            
            # Get system context in parallel
            system_context_task = self.system_maintenance.process_request(request)
            
            # Wait for both with timeout
            agent_context, system_context = await asyncio.gather(
                agent_context_task, 
                system_context_task
            )
            
            # Update agent context with live system context
            agent_context.system_context = system_context
            
            # Create performance metrics
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            performance_metrics = PerformanceMetrics(
                request_id=request.request_id,
                service_name=self.service_name,
                start_time=start_time,
                end_time=end_time,
                cache_level=None,
                agent_type=request.agent_type,
                success=True
            )
            
            # Create response
            response = ContextResponse(
                request_id=request.request_id,
                agent_context=agent_context,
                performance_metrics=performance_metrics,
                cache_hit=len([m for m in self.context_optimizer.metrics if m.cache_level]) > 0,
                source_services=["context-optimization", "system-context-maintenance"],
                total_response_time_ms=response_time
            )
            
            # Log performance
            if response_time > request.max_response_time_ms:
                logger.warning(f"Response time exceeded target: {response_time:.1f}ms > {request.max_response_time_ms}ms")
            else:
                logger.info(f"Context delivered to {request.agent_type.value} in {response_time:.1f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Agent context delivery failed: {e}")
            end_time = time.time()
            
            performance_metrics = PerformanceMetrics(
                request_id=request.request_id,
                service_name=self.service_name,
                start_time=start_time,
                end_time=end_time,
                cache_level=None,
                agent_type=request.agent_type,
                success=False,
                error_message=str(e)
            )
            
            raise

class KnowledgeIntegrationService(BaseService):
    """MCP Knowledge Server integration service"""
    
    def __init__(self, cache: ContextCache):
        super().__init__("knowledge-integration", cache)
        self.mcp_endpoints = {
            "rif-knowledge": "mcp__rif-knowledge__query_knowledge",
            "claude-docs": "mcp__rif-knowledge__get_claude_documentation"
        }
        self.status = ServiceStatus.RUNNING
        
    async def process_request(self, request: ContextRequest) -> List[ContextItem]:
        """Integrate knowledge from MCP servers"""
        start_time = time.time()
        
        # Create cache key for knowledge query
        cache_key = self._create_cache_key(
            request.agent_type.value,
            str(request.task_context.get("description", "")),
            request.issue_number
        )
        
        # Try L3 cache first (longer-lived knowledge)
        cached_knowledge = self.cache.get(cache_key, CacheLevel.L3_SOURCE)
        
        if cached_knowledge:
            metrics = self._record_performance(request.request_id, start_time, True, CacheLevel.L3_SOURCE)
            return [ContextItem(**item) for item in cached_knowledge]
        
        try:
            # Query knowledge from MCP servers
            knowledge_items = await self._query_mcp_knowledge(request)
            
            # Cache the knowledge (longer TTL for stable knowledge)
            knowledge_dicts = [asdict(item) for item in knowledge_items]
            self.cache.set(cache_key, knowledge_dicts, CacheLevel.L3_SOURCE, ttl_seconds=1800)  # 30 minutes
            
            metrics = self._record_performance(request.request_id, start_time, True)
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Knowledge integration failed: {e}")
            metrics = self._record_performance(request.request_id, start_time, False, error_message=str(e))
            raise
    
    async def _query_mcp_knowledge(self, request: ContextRequest) -> List[ContextItem]:
        """Query knowledge from MCP servers (simulated for now)"""
        # This would use actual MCP server calls
        # For now, return simulated knowledge items
        
        knowledge_items = []
        
        # Simulated Claude Code capabilities query
        claude_item = ContextItem(
            id="claude-code-mcp-integration",
            type=ContextType.CLAUDE_CODE_CAPABILITIES,
            content="Claude Code MCP integration provides access to knowledge servers, "
                   "tool integrations, and real-time information retrieval. "
                   "Sub-200ms response times achieved through intelligent caching.",
            relevance_score=0.9,
            last_updated=datetime.now(),
            source="mcp-knowledge-server",
            agent_relevance={request.agent_type: 0.9},
            size_estimate=200
        )
        knowledge_items.append(claude_item)
        
        # Simulated implementation patterns
        if "implement" in request.task_context.get("description", "").lower():
            impl_item = ContextItem(
                id="context-platform-patterns",
                type=ContextType.IMPLEMENTATION_PATTERNS,
                content="Context Intelligence Platform patterns: Microservices architecture, "
                       "event-driven updates, multi-layer caching, backward compatibility preservation.",
                relevance_score=0.8,
                last_updated=datetime.now(),
                source="mcp-knowledge-server",
                agent_relevance={request.agent_type: 0.8},
                size_estimate=150
            )
            knowledge_items.append(impl_item)
        
        return knowledge_items

class ContextIntelligencePlatform:
    """Main Context Intelligence Platform orchestrator"""
    
    def __init__(self, base_path: str = "/Users/cal/DEV/RIF/systems"):
        self.base_path = base_path
        self.cache = ContextCache(os.path.join(base_path, "context"))
        
        # Initialize services
        self.services = {
            "context-optimization": ContextOptimizationService(self.cache),
            "system-context-maintenance": SystemContextMaintenanceService(self.cache),
            "agent-context-delivery": AgentContextDeliveryService(self.cache),
            "knowledge-integration": KnowledgeIntegrationService(self.cache)
        }
        
        self.request_counter = 0
        self.performance_history = []
        
        # Start background tasks
        self._start_background_tasks()
        
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Cache cleanup task
        def cleanup_task():
            while True:
                time.sleep(300)  # Every 5 minutes
                self.cache.cleanup_expired()
                logger.info("Cache cleanup completed")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        
    async def process_context_request(self, agent_type: AgentType, 
                                    task_context: Dict[str, Any],
                                    issue_number: Optional[int] = None,
                                    priority: int = 1) -> ContextResponse:
        """Process context request through the platform"""
        
        # Create request
        self.request_counter += 1
        request = ContextRequest(
            request_id=f"req_{self.request_counter}_{int(time.time())}",
            agent_type=agent_type,
            task_context=task_context,
            issue_number=issue_number,
            priority=priority
        )
        
        start_time = time.time()
        
        try:
            # Process through agent context delivery service
            response = await self.services["agent-context-delivery"].process_request(request)
            
            # Track performance
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "request_id": request.request_id,
                "agent_type": agent_type.value,
                "response_time_ms": response.total_response_time_ms,
                "cache_hit": response.cache_hit,
                "success": True
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Context request failed: {e}")
            
            # Track failure
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "request_id": request.request_id,
                "agent_type": agent_type.value,
                "response_time_ms": (time.time() - start_time) * 1000,
                "cache_hit": False,
                "success": False,
                "error": str(e)
            })
            
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get platform performance statistics"""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_requests = [h for h in self.performance_history 
                          if datetime.fromisoformat(h["timestamp"]) > datetime.now() - timedelta(hours=1)]
        
        if not recent_requests:
            return {"status": "no_recent_data"}
        
        response_times = [r["response_time_ms"] for r in recent_requests if r["success"]]
        cache_hits = [r for r in recent_requests if r["cache_hit"]]
        
        stats = {
            "total_requests": len(recent_requests),
            "successful_requests": len(response_times),
            "cache_hit_rate": len(cache_hits) / len(recent_requests) if recent_requests else 0,
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "p95_response_time_ms": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
            "sub_200ms_compliance": len([t for t in response_times if t < 200]) / len(response_times) if response_times else 0,
            "service_status": {name: service.status.value for name, service in self.services.items()},
            "cache_stats": {level.value: self.cache.cache_stats[level] for level in CacheLevel}
        }
        
        return stats
    
    def format_context_for_agent(self, response: ContextResponse) -> str:
        """Format context response for agent consumption"""
        optimizer = ContextOptimizer()
        return optimizer.format_context_for_agent(response.agent_context)

# CLI and Testing Interface
async def main():
    """Main function for testing the Context Intelligence Platform"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context Intelligence Platform")
    parser.add_argument("--agent", type=str, choices=[a.value for a in AgentType],
                       help="Agent type for context request")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--issue", type=int, help="GitHub issue number")
    parser.add_argument("--test", action="store_true", help="Run comprehensive test")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Initialize platform
    platform = ContextIntelligencePlatform()
    
    if args.test:
        print("=== Context Intelligence Platform Test ===\n")
        
        # Test all agent types
        test_task = {
            "description": "Implement Context Intelligence Platform with microservices architecture",
            "complexity": "very_high",
            "type": "implementation"
        }
        
        for agent_type in [AgentType.ANALYST, AgentType.IMPLEMENTER, AgentType.VALIDATOR]:
            try:
                response = await platform.process_context_request(agent_type, test_task, 119)
                formatted = platform.format_context_for_agent(response)
                
                print(f"## {agent_type.value.upper()} - Response Time: {response.total_response_time_ms:.1f}ms")
                print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
                print("\n" + "="*50 + "\n")
                
            except Exception as e:
                print(f"Error testing {agent_type.value}: {e}")
        
        # Print performance stats
        stats = platform.get_performance_stats()
        print("=== Performance Statistics ===")
        print(json.dumps(stats, indent=2))
        
    elif args.benchmark:
        print("=== Performance Benchmark ===\n")
        
        # Run concurrent requests to test performance
        tasks = []
        test_task = {"description": "Performance benchmark test", "type": "benchmark"}
        
        for i in range(10):  # 10 concurrent requests
            for agent_type in [AgentType.IMPLEMENTER, AgentType.VALIDATOR]:
                task = platform.process_context_request(agent_type, test_task)
                tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful = [r for r in responses if isinstance(r, ContextResponse)]
        
        print(f"Completed {len(successful)}/{len(tasks)} requests in {total_time:.2f}s")
        
        if successful:
            response_times = [r.total_response_time_ms for r in successful]
            print(f"Average response time: {sum(response_times)/len(response_times):.1f}ms")
            print(f"P95 response time: {sorted(response_times)[int(len(response_times)*0.95)]:.1f}ms")
            print(f"Sub-200ms compliance: {len([t for t in response_times if t < 200])/len(response_times)*100:.1f}%")
        
        stats = platform.get_performance_stats()
        print(json.dumps(stats, indent=2))
        
    elif args.agent and args.task:
        # Single context request
        agent_type = AgentType(args.agent)
        task_context = {"description": args.task}
        
        try:
            response = await platform.process_context_request(agent_type, task_context, args.issue)
            formatted = platform.format_context_for_agent(response)
            
            print(f"Response Time: {response.total_response_time_ms:.1f}ms")
            print(f"Cache Hit: {response.cache_hit}")
            print("\n" + formatted)
            
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())