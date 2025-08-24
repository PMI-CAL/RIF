"""
MCP Context Aggregator

Intelligent context aggregation system that queries multiple MCP servers in parallel,
merges responses with conflict resolution, and provides optimized results for agent consumption.

Issue: #85 - Implement MCP context aggregator
Agent: RIF-Implementer
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cachetools import TTLCache
import hashlib
import json

# Import existing components for integration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from knowledge.context.optimizer import ContextOptimizer
from ..loader.dynamic_loader import DynamicMCPLoader
from ..registry.server_registry import MCPServerRegistry
from ..security.security_gateway import SecurityGateway

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServerResponse:
    """Response from a single MCP server"""
    server_id: str
    server_name: str
    response_data: Any
    response_time_ms: int
    status: str  # success, failed, timeout
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregationResult:
    """Result of context aggregation across multiple servers"""
    merged_response: Any
    server_responses: List[ServerResponse]
    optimization_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    cache_info: Dict[str, Any]
    total_time_ms: int
    query: str
    successful_servers: int
    failed_servers: int


@dataclass
class QueryOptimizationResult:
    """Result of query optimization analysis"""
    optimized_queries: Dict[str, str]  # server_id -> optimized_query
    server_capabilities: Dict[str, List[str]]
    priority_servers: List[str]
    expected_response_time_ms: int
    cache_strategy: str


class MockHealthMonitor:
    """
    Mock health monitor interface for immediate implementation.
    Will be replaced with real HealthMonitor when Issue #84 completes.
    """
    
    def __init__(self):
        self.server_health: Dict[str, str] = {}
        self.health_cache_ttl = 30  # 30 seconds
        self.last_health_check = {}
        
    async def get_server_health(self, server_id: str) -> str:
        """Get health status of a server"""
        # Simple mock implementation - always return healthy for now
        # Real implementation would check actual server health
        return "healthy"
    
    async def get_healthy_servers(self, server_ids: List[str]) -> List[str]:
        """Filter to only healthy servers"""
        healthy = []
        for server_id in server_ids:
            health = await self.get_server_health(server_id)
            if health == "healthy":
                healthy.append(server_id)
        return healthy
    
    async def register_server(self, server_id: str):
        """Register server for health monitoring"""
        self.server_health[server_id] = "healthy"
        logger.debug(f"Registered server {server_id} with mock health monitor")


class QueryOptimizer:
    """
    Multi-dimensional query analysis and server-specific optimization.
    Integrates with ContextOptimizer for performance characteristics.
    """
    
    def __init__(self, context_optimizer: ContextOptimizer, server_registry: MCPServerRegistry):
        self.context_optimizer = context_optimizer
        self.server_registry = server_registry
        self.performance_history: Dict[str, List[float]] = {}  # server_id -> response_times
        
    async def optimize_query(self, query: str, available_servers: List[str], 
                           context: Optional[Dict[str, Any]] = None) -> QueryOptimizationResult:
        """
        Optimize query for multiple servers based on capabilities and performance
        
        Args:
            query: Original query string
            available_servers: List of available server IDs
            context: Query context for optimization
            
        Returns:
            Query optimization result with server-specific optimizations
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze server capabilities
            server_capabilities = await self._analyze_server_capabilities(available_servers)
            
            # Step 2: Generate optimized queries for each server
            optimized_queries = await self._optimize_queries_by_server(query, server_capabilities, context)
            
            # Step 3: Prioritize servers based on performance and capabilities
            priority_servers = await self._prioritize_servers(available_servers, query, context)
            
            # Step 4: Estimate response time
            expected_time = self._estimate_response_time(priority_servers)
            
            # Step 5: Determine cache strategy
            cache_strategy = self._determine_cache_strategy(query, server_capabilities)
            
            optimization_time_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"Query optimization completed in {optimization_time_ms}ms for {len(available_servers)} servers")
            
            return QueryOptimizationResult(
                optimized_queries=optimized_queries,
                server_capabilities=server_capabilities,
                priority_servers=priority_servers,
                expected_response_time_ms=expected_time,
                cache_strategy=cache_strategy
            )
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            # Fallback to basic optimization
            return QueryOptimizationResult(
                optimized_queries={server_id: query for server_id in available_servers},
                server_capabilities={},
                priority_servers=available_servers,
                expected_response_time_ms=1000,  # Conservative estimate
                cache_strategy="ttl"
            )
    
    async def _analyze_server_capabilities(self, server_ids: List[str]) -> Dict[str, List[str]]:
        """Analyze capabilities of each server"""
        capabilities = {}
        for server_id in server_ids:
            try:
                server_info = await self.server_registry.get_server(server_id)
                if server_info:
                    capabilities[server_id] = server_info.get('capabilities', [])
                else:
                    capabilities[server_id] = ['general']  # Default capability
            except Exception as e:
                logger.warning(f"Failed to get capabilities for {server_id}: {e}")
                capabilities[server_id] = ['general']
        
        return capabilities
    
    async def _optimize_queries_by_server(self, query: str, server_capabilities: Dict[str, List[str]], 
                                        context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Generate optimized queries for each server based on capabilities"""
        optimized = {}
        
        for server_id, capabilities in server_capabilities.items():
            optimized_query = query  # Base query
            
            # Optimize based on server capabilities
            if 'semantic_search' in capabilities:
                optimized_query = f"semantic:{query}"
            elif 'full_text_search' in capabilities:
                optimized_query = f"fulltext:{query}"
            elif 'pattern_matching' in capabilities:
                optimized_query = f"pattern:{query}"
            
            # Context-specific optimization
            if context:
                if context.get('agent_type') == 'rif-implementer' and 'code_search' in capabilities:
                    optimized_query = f"code:{query}"
                elif context.get('agent_type') == 'rif-analyst' and 'analysis_tools' in capabilities:
                    optimized_query = f"analyze:{query}"
            
            optimized[server_id] = optimized_query
        
        return optimized
    
    async def _prioritize_servers(self, server_ids: List[str], query: str, 
                                context: Optional[Dict[str, Any]]) -> List[str]:
        """Prioritize servers based on performance history and relevance"""
        server_scores = {}
        
        for server_id in server_ids:
            score = 1.0  # Base score
            
            # Performance history factor
            if server_id in self.performance_history:
                avg_response_time = sum(self.performance_history[server_id]) / len(self.performance_history[server_id])
                # Lower response time = higher score
                performance_factor = max(0.1, 1.0 - (avg_response_time / 1000.0))  # Normalize to 1 second
                score *= performance_factor
            
            # TODO: Add capability relevance scoring
            # TODO: Add server reliability scoring
            
            server_scores[server_id] = score
        
        # Sort by score (highest first)
        return sorted(server_ids, key=lambda sid: server_scores.get(sid, 0.5), reverse=True)
    
    def _estimate_response_time(self, priority_servers: List[str]) -> int:
        """Estimate expected response time based on server performance history"""
        if not priority_servers:
            return 1000  # Conservative default
        
        # Use best performing server's history
        best_server = priority_servers[0]
        if best_server in self.performance_history and self.performance_history[best_server]:
            avg_time = sum(self.performance_history[best_server]) / len(self.performance_history[best_server])
            return int(avg_time * 1.2)  # Add 20% buffer
        
        return 500  # Default estimate
    
    def _determine_cache_strategy(self, query: str, server_capabilities: Dict[str, List[str]]) -> str:
        """Determine optimal cache strategy based on query and server capabilities"""
        # Simple heuristics for cache strategy
        if len(query) > 100:
            return "ttl_extended"  # Complex queries cached longer
        elif any('real_time' in caps for caps in server_capabilities.values()):
            return "ttl_short"  # Real-time data cached briefly
        else:
            return "ttl"  # Standard TTL caching


class CacheManager:
    """
    Multi-layer intelligent caching with health-based invalidation.
    Integrates with MockHealthMonitor for server health coordination.
    """
    
    def __init__(self, health_monitor: MockHealthMonitor, 
                 default_ttl: int = 300, max_size: int = 1000):
        self.health_monitor = health_monitor
        self.cache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def generate_cache_key(self, query: str, servers: List[str], 
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Generate intelligent cache key for query and server combination"""
        key_parts = [
            query.strip().lower(),
            "|".join(sorted(servers)),  # Server order doesn't matter for caching
        ]
        
        if context:
            # Include relevant context parts
            context_parts = []
            if 'agent_type' in context:
                context_parts.append(f"agent:{context['agent_type']}")
            if 'issue_id' in context:
                context_parts.append(f"issue:{context['issue_id']}")
            if context_parts:
                key_parts.append("|".join(context_parts))
        
        # Create hash for consistent key length
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    async def get(self, cache_key: str) -> Optional[AggregationResult]:
        """Get cached result with health validation"""
        try:
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                
                # Validate that servers used in cached result are still healthy
                if await self._validate_cached_result_health(cached_result):
                    self.hit_count += 1
                    logger.debug(f"Cache hit for key {cache_key}")
                    return cached_result
                else:
                    # Remove invalid cached result
                    del self.cache[cache_key]
                    if cache_key in self.cache_metadata:
                        del self.cache_metadata[cache_key]
                    logger.debug(f"Invalidated cached result for key {cache_key} due to server health")
            
            self.miss_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for key {cache_key}: {e}")
            return None
    
    async def put(self, cache_key: str, result: AggregationResult, ttl_override: Optional[int] = None):
        """Store result in cache with metadata"""
        try:
            # Store the result
            if ttl_override:
                # Create temporary cache with custom TTL
                temp_cache = TTLCache(maxsize=1, ttl=ttl_override)
                temp_cache[cache_key] = result
                # This is a simplified approach - in production would need proper TTL management
            
            self.cache[cache_key] = result
            
            # Store metadata for cache management
            self.cache_metadata[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'servers_used': [r.server_id for r in result.server_responses],
                'query': result.query,
                'ttl_override': ttl_override
            }
            
            logger.debug(f"Cached result for key {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache put failed for key {cache_key}: {e}")
    
    async def _validate_cached_result_health(self, cached_result: AggregationResult) -> bool:
        """Validate that servers in cached result are still healthy"""
        server_ids = [r.server_id for r in cached_result.server_responses]
        healthy_servers = await self.health_monitor.get_healthy_servers(server_ids)
        
        # Require at least 80% of original servers to be healthy
        health_ratio = len(healthy_servers) / max(len(server_ids), 1)
        return health_ratio >= 0.8
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.cache.maxsize
        }


class MCPContextAggregator:
    """
    Main MCP context aggregator that orchestrates parallel queries, response merging,
    and intelligent caching for optimal context delivery to agents.
    
    Integrates with:
    - ContextOptimizer for response merging and relevance scoring
    - DynamicMCPLoader for server discovery and management
    - MockHealthMonitor for server health coordination  
    - SecurityGateway for authentication and validation
    """
    
    def __init__(self,
                 context_optimizer: Optional[ContextOptimizer] = None,
                 mcp_loader: Optional[DynamicMCPLoader] = None,
                 server_registry: Optional[MCPServerRegistry] = None,
                 security_gateway: Optional[SecurityGateway] = None,
                 health_monitor: Optional[MockHealthMonitor] = None,
                 max_concurrent_servers: int = 4,
                 query_timeout_seconds: int = 10,
                 cache_ttl_seconds: int = 300):
        """
        Initialize MCP Context Aggregator
        
        Args:
            context_optimizer: ContextOptimizer for response merging
            mcp_loader: DynamicMCPLoader for server management
            server_registry: MCPServerRegistry for server discovery
            security_gateway: SecurityGateway for authentication
            health_monitor: Health monitor (mock for now)
            max_concurrent_servers: Maximum parallel server queries
            query_timeout_seconds: Timeout for individual server queries
            cache_ttl_seconds: Default cache TTL
        """
        # Initialize core components
        self.context_optimizer = context_optimizer or ContextOptimizer()
        self.mcp_loader = mcp_loader or DynamicMCPLoader()
        self.server_registry = server_registry or MCPServerRegistry()
        self.security_gateway = security_gateway or SecurityGateway()
        self.health_monitor = health_monitor or MockHealthMonitor()
        
        # Initialize sub-components
        self.query_optimizer = QueryOptimizer(self.context_optimizer, self.server_registry)
        self.cache_manager = CacheManager(self.health_monitor, cache_ttl_seconds)
        
        # Configuration
        self.max_concurrent_servers = max_concurrent_servers
        self.query_timeout_seconds = query_timeout_seconds
        self.query_semaphore = asyncio.Semaphore(max_concurrent_servers)
        
        # Performance tracking
        self.query_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time_ms': 0.0,
            'avg_servers_queried': 0.0,
            'avg_cache_hit_rate': 0.0,
            'successful_aggregations': 0,
            'failed_aggregations': 0
        }
        
        logger.info(f"MCPContextAggregator initialized with {max_concurrent_servers} concurrent servers, "
                   f"{query_timeout_seconds}s timeout, {cache_ttl_seconds}s cache TTL")
    
    async def get_context(self, query: str, 
                         required_servers: Optional[List[str]] = None,
                         agent_type: str = 'default',
                         context: Optional[Dict[str, Any]] = None,
                         use_cache: bool = True,
                         min_servers: int = 2,
                         explain: bool = False) -> AggregationResult:
        """
        Get aggregated context from multiple MCP servers
        
        Args:
            query: Context query string
            required_servers: Specific servers to query (None for auto-discovery)
            agent_type: Agent type for optimization
            context: Query context for optimization
            use_cache: Whether to use caching
            min_servers: Minimum number of servers to query
            explain: Include detailed explanation
            
        Returns:
            Aggregated context result
        """
        start_time = time.time()
        self.performance_metrics['total_queries'] += 1
        
        try:
            logger.info(f"Getting context for query: {query[:50]}..." if len(query) > 50 else query)
            
            # Step 1: Server discovery and health filtering
            available_servers = await self._discover_available_servers(required_servers, context)
            if len(available_servers) < min_servers:
                logger.warning(f"Only {len(available_servers)} servers available, minimum {min_servers} required")
            
            # Step 2: Check cache if enabled
            cache_key = None
            if use_cache and available_servers:
                cache_key = self.cache_manager.generate_cache_key(query, available_servers, context)
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    logger.debug("Returning cached result")
                    return cached_result
            
            # Step 3: Query optimization
            optimization_result = await self.query_optimizer.optimize_query(
                query, available_servers, context
            )
            
            # Step 4: Parallel server queries
            server_responses = await self._query_servers_parallel(
                optimization_result.optimized_queries
            )
            
            # Step 5: Response merging and optimization
            merged_response = await self._merge_responses(server_responses, query, context, agent_type)
            
            # Step 6: Create aggregation result
            total_time_ms = int((time.time() - start_time) * 1000)
            successful_servers = len([r for r in server_responses if r.status == "success"])
            failed_servers = len(server_responses) - successful_servers
            
            aggregation_result = AggregationResult(
                merged_response=merged_response,
                server_responses=server_responses,
                optimization_info=self._generate_optimization_info(optimization_result, server_responses),
                performance_metrics=self._generate_performance_metrics(server_responses, total_time_ms),
                cache_info=self.cache_manager.get_cache_stats(),
                total_time_ms=total_time_ms,
                query=query,
                successful_servers=successful_servers,
                failed_servers=failed_servers
            )
            
            # Step 7: Cache result if successful
            if use_cache and cache_key and successful_servers > 0:
                await self.cache_manager.put(cache_key, aggregation_result)
            
            # Step 8: Update performance metrics
            self._update_performance_metrics(aggregation_result)
            
            # Step 9: Record query history
            self._record_query_history(aggregation_result, context)
            
            if successful_servers > 0:
                self.performance_metrics['successful_aggregations'] += 1
                logger.info(f"Successfully aggregated context from {successful_servers}/{len(server_responses)} servers in {total_time_ms}ms")
            else:
                self.performance_metrics['failed_aggregations'] += 1
                logger.error(f"Failed to get context from any servers in {total_time_ms}ms")
            
            return aggregation_result
            
        except Exception as e:
            self.performance_metrics['failed_aggregations'] += 1
            logger.error(f"Context aggregation failed: {e}")
            
            # Return error result
            total_time_ms = int((time.time() - start_time) * 1000)
            return AggregationResult(
                merged_response={"error": str(e), "fallback_used": True},
                server_responses=[],
                optimization_info={"error": str(e)},
                performance_metrics={"error": str(e)},
                cache_info=self.cache_manager.get_cache_stats(),
                total_time_ms=total_time_ms,
                query=query,
                successful_servers=0,
                failed_servers=0
            )
    
    async def _discover_available_servers(self, required_servers: Optional[List[str]], 
                                        context: Optional[Dict[str, Any]]) -> List[str]:
        """Discover available servers for querying"""
        try:
            if required_servers:
                # Use specified servers, filter for health
                return await self.health_monitor.get_healthy_servers(required_servers)
            
            # Auto-discover from registry
            all_servers = await self.server_registry.list_servers()
            server_ids = [server['server_id'] for server in all_servers if server.get('status') == 'active']
            
            # Filter for healthy servers
            healthy_servers = await self.health_monitor.get_healthy_servers(server_ids)
            
            # Limit to max concurrent
            return healthy_servers[:self.max_concurrent_servers]
            
        except Exception as e:
            logger.error(f"Server discovery failed: {e}")
            return []
    
    async def _query_servers_parallel(self, optimized_queries: Dict[str, str]) -> List[ServerResponse]:
        """Query multiple servers in parallel with resource management"""
        async def query_single_server(server_id: str, query: str) -> ServerResponse:
            async with self.query_semaphore:
                return await self._query_single_server(server_id, query)
        
        # Create tasks for parallel execution
        tasks = [query_single_server(server_id, query) 
                for server_id, query in optimized_queries.items()]
        
        try:
            # Execute with timeout
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.query_timeout_seconds
            )
            
            # Process results and handle exceptions
            processed_responses = []
            for i, response in enumerate(responses):
                server_id = list(optimized_queries.keys())[i]
                if isinstance(response, Exception):
                    processed_responses.append(ServerResponse(
                        server_id=server_id,
                        server_name=f"server_{server_id}",
                        response_data=None,
                        response_time_ms=self.query_timeout_seconds * 1000,
                        status="failed",
                        error_message=str(response)
                    ))
                else:
                    processed_responses.append(response)
            
            return processed_responses
            
        except asyncio.TimeoutError:
            logger.error(f"Server queries timed out after {self.query_timeout_seconds} seconds")
            return [ServerResponse(
                server_id=server_id,
                server_name=f"server_{server_id}",
                response_data=None,
                response_time_ms=self.query_timeout_seconds * 1000,
                status="timeout",
                error_message="Query timeout"
            ) for server_id in optimized_queries.keys()]
    
    async def _query_single_server(self, server_id: str, query: str) -> ServerResponse:
        """Query a single MCP server"""
        start_time = time.time()
        
        try:
            # Get server info
            server_info = await self.server_registry.get_server(server_id)
            server_name = server_info.get('name', f'server_{server_id}') if server_info else f'server_{server_id}'
            
            # Security validation
            if not await self.security_gateway.validate_query_permission(server_id, query):
                return ServerResponse(
                    server_id=server_id,
                    server_name=server_name,
                    response_data=None,
                    response_time_ms=int((time.time() - start_time) * 1000),
                    status="failed",
                    error_message="Query permission denied"
                )
            
            # Mock server query - in real implementation would call actual MCP server
            await asyncio.sleep(0.1 + (hash(server_id) % 100) / 1000)  # Simulate variable response time
            
            # Mock response data
            mock_response = {
                "results": [
                    {"content": f"Mock result from {server_name} for query: {query}", "relevance": 0.8},
                    {"content": f"Additional context from {server_name}", "relevance": 0.6}
                ],
                "metadata": {
                    "server_id": server_id,
                    "server_name": server_name,
                    "query_processed": query,
                    "result_count": 2
                }
            }
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Update performance history for optimization
            if server_id not in self.query_optimizer.performance_history:
                self.query_optimizer.performance_history[server_id] = []
            self.query_optimizer.performance_history[server_id].append(response_time_ms)
            # Keep only last 100 response times
            if len(self.query_optimizer.performance_history[server_id]) > 100:
                self.query_optimizer.performance_history[server_id] = \
                    self.query_optimizer.performance_history[server_id][-100:]
            
            return ServerResponse(
                server_id=server_id,
                server_name=server_name,
                response_data=mock_response,
                response_time_ms=response_time_ms,
                status="success"
            )
            
        except Exception as e:
            return ServerResponse(
                server_id=server_id,
                server_name=f"server_{server_id}",
                response_data=None,
                response_time_ms=int((time.time() - start_time) * 1000),
                status="failed",
                error_message=str(e)
            )
    
    async def _merge_responses(self, server_responses: List[ServerResponse], 
                             query: str, context: Optional[Dict[str, Any]], 
                             agent_type: str) -> Dict[str, Any]:
        """Merge responses from multiple servers using ContextOptimizer"""
        try:
            # Extract successful responses
            successful_responses = [r for r in server_responses if r.status == "success" and r.response_data]
            
            if not successful_responses:
                return {"error": "No successful responses to merge", "fallback_used": True}
            
            # Convert server responses to ContextOptimizer format
            optimizer_results = []
            for response in successful_responses:
                if response.response_data and "results" in response.response_data:
                    for result in response.response_data["results"]:
                        optimizer_result = {
                            "content": result.get("content", ""),
                            "metadata": {
                                "source": response.server_name,
                                "server_id": response.server_id,
                                "relevance": result.get("relevance", 0.5),
                                "response_time_ms": response.response_time_ms,
                                "type": "mcp_result"
                            }
                        }
                        optimizer_results.append(optimizer_result)
            
            # Use ContextOptimizer to merge and optimize results
            optimization_result = self.context_optimizer.optimize_for_agent(
                results=optimizer_results,
                query=query,
                agent_type=agent_type,
                context=context,
                preserve_context=True,
                min_results=1,
                explain=False
            )
            
            # Structure the merged response
            merged_response = {
                "optimized_results": optimization_result["optimized_results"],
                "server_summary": {
                    "successful_servers": len(successful_responses),
                    "total_results": len(optimizer_results),
                    "sources": [r.server_name for r in successful_responses]
                },
                "optimization_applied": True,
                "optimization_info": optimization_result.get("optimization_info", {}),
                "performance_stats": optimization_result.get("performance_stats", {})
            }
            
            return merged_response
            
        except Exception as e:
            logger.error(f"Response merging failed: {e}")
            # Fallback to simple concatenation
            all_results = []
            for response in server_responses:
                if response.status == "success" and response.response_data:
                    if "results" in response.response_data:
                        all_results.extend(response.response_data["results"])
            
            return {
                "results": all_results,
                "error": f"Advanced merging failed: {e}",
                "fallback_used": True
            }
    
    def _generate_optimization_info(self, optimization_result: QueryOptimizationResult, 
                                  server_responses: List[ServerResponse]) -> Dict[str, Any]:
        """Generate optimization information for the aggregation result"""
        return {
            "query_optimization": {
                "servers_queried": len(optimization_result.optimized_queries),
                "priority_servers": optimization_result.priority_servers,
                "cache_strategy": optimization_result.cache_strategy,
                "expected_response_time_ms": optimization_result.expected_response_time_ms
            },
            "server_performance": {
                "successful_queries": len([r for r in server_responses if r.status == "success"]),
                "failed_queries": len([r for r in server_responses if r.status == "failed"]),
                "timeout_queries": len([r for r in server_responses if r.status == "timeout"]),
                "avg_response_time_ms": sum(r.response_time_ms for r in server_responses) / max(len(server_responses), 1)
            }
        }
    
    def _generate_performance_metrics(self, server_responses: List[ServerResponse], 
                                    total_time_ms: int) -> Dict[str, Any]:
        """Generate performance metrics for the aggregation"""
        successful_responses = [r for r in server_responses if r.status == "success"]
        
        return {
            "aggregation_time_ms": total_time_ms,
            "server_query_times": {r.server_id: r.response_time_ms for r in server_responses},
            "successful_servers": len(successful_responses),
            "total_servers_queried": len(server_responses),
            "success_rate": len(successful_responses) / max(len(server_responses), 1) * 100,
            "avg_server_response_time_ms": sum(r.response_time_ms for r in successful_responses) / max(len(successful_responses), 1)
        }
    
    def _update_performance_metrics(self, result: AggregationResult):
        """Update cumulative performance metrics"""
        n = self.performance_metrics['total_queries']
        
        # Update average response time
        current_avg = self.performance_metrics['avg_response_time_ms']
        self.performance_metrics['avg_response_time_ms'] = (
            (current_avg * (n-1) + result.total_time_ms) / n
        )
        
        # Update average servers queried
        servers_queried = len(result.server_responses)
        current_avg = self.performance_metrics['avg_servers_queried']
        self.performance_metrics['avg_servers_queried'] = (
            (current_avg * (n-1) + servers_queried) / n
        )
        
        # Update cache hit rate
        cache_hit_rate = result.cache_info.get('hit_rate_percent', 0)
        current_avg = self.performance_metrics['avg_cache_hit_rate']
        self.performance_metrics['avg_cache_hit_rate'] = (
            (current_avg * (n-1) + cache_hit_rate) / n
        )
    
    def _record_query_history(self, result: AggregationResult, context: Optional[Dict[str, Any]]):
        """Record query in history for analysis"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': result.query,
            'total_time_ms': result.total_time_ms,
            'successful_servers': result.successful_servers,
            'failed_servers': result.failed_servers,
            'context': context,
            'cache_hit_rate': result.cache_info.get('hit_rate_percent', 0)
        }
        
        self.query_history.append(record)
        
        # Keep only last 1000 queries
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
    
    # Public API methods
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of the aggregator and its components"""
        try:
            # Check component health
            available_servers = await self._discover_available_servers(None, None)
            
            return {
                "status": "healthy",
                "components": {
                    "context_optimizer": "healthy",
                    "query_optimizer": "healthy", 
                    "cache_manager": "healthy",
                    "health_monitor": "healthy (mock)"
                },
                "available_servers": len(available_servers),
                "performance_metrics": self.performance_metrics,
                "cache_stats": self.cache_manager.get_cache_stats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "metrics": self.performance_metrics,
            "cache_stats": self.cache_manager.get_cache_stats(),
            "recent_queries": len(self.query_history),
            "query_history_sample": self.query_history[-10:] if self.query_history else []
        }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear the response cache"""
        try:
            cache_size = len(self.cache_manager.cache)
            self.cache_manager.cache.clear()
            self.cache_manager.cache_metadata.clear()
            self.cache_manager.hit_count = 0
            self.cache_manager.miss_count = 0
            
            return {
                "status": "success",
                "cleared_entries": cache_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }