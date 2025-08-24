#!/usr/bin/env python3
"""
Context Optimization Engine Integration
Issue #137: DPIBS Sub-Issue 1 - Core API Framework + Context Optimization Engine

Integrates existing ContextOptimizer with API framework for agent-specific context delivery.
Provides <200ms response time with multi-factor relevance scoring.
"""

import os
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import json
from dataclasses import dataclass, asdict

# Import existing context optimization components
import sys
sys.path.append('/Users/cal/DEV/RIF/systems')

# Import from context optimization engine (with hyphens in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("context_optimization_engine", "/Users/cal/DEV/RIF/systems/context-optimization-engine.py")
context_optimization_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context_optimization_engine)
ContextOptimizer = context_optimization_engine.ContextOptimizer
AgentType = context_optimization_engine.AgentType
ContextItem = context_optimization_engine.ContextItem
AgentContext = context_optimization_engine.AgentContext
SystemContext = context_optimization_engine.SystemContext

class ContextOptimizationAPI:
    """
    API integration layer for context optimization engine.
    Provides high-performance agent-specific context delivery with caching and monitoring.
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge",
                 cache_ttl: int = 300,  # 5 minutes
                 performance_monitoring: bool = True):
        
        self.knowledge_base_path = knowledge_base_path
        self.cache_ttl = cache_ttl
        self.performance_monitoring = performance_monitoring
        
        # Initialize core context optimizer
        self.context_optimizer = ContextOptimizer(knowledge_base_path)
        
        # Performance tracking
        self.optimization_metrics = []
        self.cache_metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
        # Enhanced caching with TTL and performance tracking
        self.context_cache = {}
        self.cache_access_times = {}
        
        # Agent request queues for concurrent processing
        self.active_optimizations = {}
        
    async def optimize_agent_context(self, 
                                   agent_type: str,
                                   task_context: Dict[str, Any],
                                   issue_number: Optional[int] = None,
                                   use_cache: bool = True) -> Dict[str, Any]:
        """
        Optimize context for specific agent with performance monitoring.
        Target: <200ms response time
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Convert agent type string to enum
            agent_enum = AgentType(agent_type)
            
            # Check cache first if enabled
            cache_key = self._generate_cache_key(agent_type, task_context, issue_number)
            cached_result = None
            
            if use_cache:
                cached_result = await self._get_cached_context(cache_key)
                if cached_result:
                    self._update_cache_metrics("hit")
                    optimization_time = (time.time() - start_time) * 1000
                    
                    # Add performance metadata to cached result
                    cached_result["performance"] = {
                        "response_time_ms": optimization_time,
                        "cache_hit": True,
                        "target_met": optimization_time < 200,
                        "request_id": request_id
                    }
                    
                    self._record_optimization_metric({
                        "request_id": request_id,
                        "agent_type": agent_type,
                        "optimization_time_ms": optimization_time,
                        "cache_hit": True,
                        "context_size": cached_result["metadata"]["context_size"]
                    })
                    
                    return cached_result
            
            self._update_cache_metrics("miss")
            
            # Perform context optimization
            agent_context = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.context_optimizer.optimize_for_agent,
                agent_enum, 
                task_context, 
                issue_number
            )
            
            # Format context for API response
            formatted_context = self.context_optimizer.format_context_for_agent(agent_context)
            optimization_time = (time.time() - start_time) * 1000
            
            # Build comprehensive response
            response = {
                "request_id": request_id,
                "agent_type": agent_type,
                "context": formatted_context,
                "metadata": {
                    "total_items": len(agent_context.relevant_knowledge),
                    "context_size": agent_context.total_size,
                    "window_utilization": agent_context.context_window_utilization,
                    "optimization_factors": self._extract_optimization_factors(agent_context)
                },
                "performance": {
                    "response_time_ms": optimization_time,
                    "target_met": optimization_time < 200,
                    "cache_hit": False,
                    "request_id": request_id
                },
                "quality_indicators": {
                    "relevance_score": self._calculate_overall_relevance(agent_context),
                    "completeness_score": self._calculate_completeness_score(agent_context),
                    "freshness_score": self._calculate_freshness_score(agent_context)
                }
            }
            
            # Cache the result if optimization was successful and fast enough
            if use_cache and optimization_time < 500:  # Only cache relatively fast optimizations
                await self._cache_context(cache_key, response, self.cache_ttl)
            
            # Record performance metrics
            self._record_optimization_metric({
                "request_id": request_id,
                "agent_type": agent_type,
                "optimization_time_ms": optimization_time,
                "cache_hit": False,
                "context_size": agent_context.total_size,
                "relevance_score": self._calculate_overall_relevance(agent_context),
                "window_utilization": agent_context.context_window_utilization
            })
            
            return response
            
        except ValueError as e:
            raise ValueError(f"Invalid agent type '{agent_type}': {str(e)}")
        except Exception as e:
            optimization_time = (time.time() - start_time) * 1000
            self._record_optimization_metric({
                "request_id": request_id,
                "agent_type": agent_type,
                "optimization_time_ms": optimization_time,
                "cache_hit": False,
                "error": str(e)
            })
            raise Exception(f"Context optimization failed: {str(e)}")
    
    async def batch_optimize_contexts(self, 
                                    optimization_requests: List[Dict[str, Any]],
                                    max_concurrent: int = 4) -> List[Dict[str, Any]]:
        """
        Optimize multiple agent contexts concurrently for improved throughput.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def optimize_with_semaphore(request_data):
            async with semaphore:
                return await self.optimize_agent_context(
                    agent_type=request_data["agent_type"],
                    task_context=request_data["task_context"],
                    issue_number=request_data.get("issue_number"),
                    use_cache=request_data.get("use_cache", True)
                )
        
        tasks = [optimize_with_semaphore(req) for req in optimization_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "request_index": i,
                    "agent_type": optimization_requests[i]["agent_type"]
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_agent_context_history(self, 
                                      agent_type: str,
                                      hours_back: int = 24) -> Dict[str, Any]:
        """
        Get context optimization history for specific agent type.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter metrics for the specific agent
        agent_metrics = [
            metric for metric in self.optimization_metrics
            if metric.get("agent_type") == agent_type
            and metric.get("timestamp", datetime.min) > cutoff_time
        ]
        
        if not agent_metrics:
            return {
                "agent_type": agent_type,
                "history_period_hours": hours_back,
                "total_requests": 0,
                "message": "No optimization history found for this period"
            }
        
        # Calculate statistics
        response_times = [m["optimization_time_ms"] for m in agent_metrics if "optimization_time_ms" in m]
        cache_hits = sum(1 for m in agent_metrics if m.get("cache_hit", False))
        errors = sum(1 for m in agent_metrics if "error" in m)
        
        return {
            "agent_type": agent_type,
            "history_period_hours": hours_back,
            "total_requests": len(agent_metrics),
            "performance_stats": {
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                "min_response_time_ms": min(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                "target_compliance_rate": sum(1 for t in response_times if t < 200) / len(response_times) if response_times else 0
            },
            "cache_stats": {
                "cache_hits": cache_hits,
                "cache_misses": len(agent_metrics) - cache_hits,
                "cache_hit_rate": cache_hits / len(agent_metrics) if agent_metrics else 0
            },
            "reliability_stats": {
                "successful_requests": len(agent_metrics) - errors,
                "failed_requests": errors,
                "success_rate": (len(agent_metrics) - errors) / len(agent_metrics) if agent_metrics else 0
            }
        }
    
    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """
        Analyze and optimize cache performance.
        """
        start_time = time.time()
        
        # Analyze cache hit patterns
        current_time = datetime.now()
        expired_keys = []
        access_patterns = {}
        
        for cache_key, cache_data in self.context_cache.items():
            if cache_data["expires_at"] < current_time:
                expired_keys.append(cache_key)
            else:
                # Analyze access patterns
                access_time = self.cache_access_times.get(cache_key, current_time)
                age_minutes = (current_time - access_time).total_seconds() / 60
                access_patterns[cache_key] = age_minutes
        
        # Remove expired entries
        for key in expired_keys:
            del self.context_cache[key]
            self.cache_access_times.pop(key, None)
        
        self.cache_metrics["evictions"] += len(expired_keys)
        
        # Identify least recently used entries for potential eviction
        sorted_by_access = sorted(access_patterns.items(), key=lambda x: x[1], reverse=True)
        lru_candidates = sorted_by_access[-5:] if len(sorted_by_access) > 10 else []
        
        optimization_time = (time.time() - start_time) * 1000
        
        return {
            "optimization_timestamp": current_time.isoformat(),
            "cache_stats": {
                "total_entries": len(self.context_cache),
                "expired_removed": len(expired_keys),
                "active_entries": len(self.context_cache) - len(expired_keys),
                "lru_candidates": len(lru_candidates)
            },
            "performance_metrics": dict(self.cache_metrics),
            "optimization_time_ms": optimization_time,
            "recommendations": self._generate_cache_recommendations()
        }
    
    def get_optimization_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive optimization analytics.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_metrics = [
            metric for metric in self.optimization_metrics
            if metric.get("timestamp", datetime.min) > cutoff_time
        ]
        
        if not recent_metrics:
            return {
                "period_hours": hours_back,
                "total_optimizations": 0,
                "message": "No optimization data available for this period"
            }
        
        # Group by agent type
        by_agent = {}
        for metric in recent_metrics:
            agent_type = metric.get("agent_type", "unknown")
            if agent_type not in by_agent:
                by_agent[agent_type] = []
            by_agent[agent_type].append(metric)
        
        # Calculate aggregate statistics
        all_response_times = [m["optimization_time_ms"] for m in recent_metrics if "optimization_time_ms" in m]
        cache_hits = sum(1 for m in recent_metrics if m.get("cache_hit", False))
        
        agent_summaries = {}
        for agent_type, agent_metrics in by_agent.items():
            agent_response_times = [m["optimization_time_ms"] for m in agent_metrics if "optimization_time_ms" in m]
            agent_summaries[agent_type] = {
                "total_requests": len(agent_metrics),
                "avg_response_time_ms": sum(agent_response_times) / len(agent_response_times) if agent_response_times else 0,
                "target_compliance_rate": sum(1 for t in agent_response_times if t < 200) / len(agent_response_times) if agent_response_times else 0,
                "cache_hit_rate": sum(1 for m in agent_metrics if m.get("cache_hit", False)) / len(agent_metrics)
            }
        
        return {
            "period_hours": hours_back,
            "total_optimizations": len(recent_metrics),
            "overall_performance": {
                "avg_response_time_ms": sum(all_response_times) / len(all_response_times) if all_response_times else 0,
                "target_compliance_rate": sum(1 for t in all_response_times if t < 200) / len(all_response_times) if all_response_times else 0,
                "overall_cache_hit_rate": cache_hits / len(recent_metrics)
            },
            "agent_performance": agent_summaries,
            "cache_performance": dict(self.cache_metrics),
            "system_health": {
                "active_cache_entries": len(self.context_cache),
                "active_optimizations": len(self.active_optimizations),
                "memory_efficiency": self._calculate_memory_efficiency()
            }
        }
    
    # Private helper methods
    
    def _generate_cache_key(self, agent_type: str, task_context: Dict[str, Any], issue_number: Optional[int]) -> str:
        """Generate cache key for context optimization request"""
        key_components = [
            agent_type,
            str(issue_number) if issue_number else "no_issue",
            str(hash(json.dumps(task_context, sort_keys=True)))[:8]
        ]
        return ":".join(key_components)
    
    async def _get_cached_context(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached context if available and not expired"""
        if cache_key in self.context_cache:
            cache_entry = self.context_cache[cache_key]
            if cache_entry["expires_at"] > datetime.now():
                self.cache_access_times[cache_key] = datetime.now()
                return cache_entry["data"].copy()
            else:
                # Remove expired entry
                del self.context_cache[cache_key]
                self.cache_access_times.pop(cache_key, None)
        return None
    
    async def _cache_context(self, cache_key: str, context_data: Dict[str, Any], ttl_seconds: int):
        """Cache context optimization result"""
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        self.context_cache[cache_key] = {
            "data": context_data.copy(),
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        self.cache_access_times[cache_key] = datetime.now()
        
        # Prevent cache from growing too large
        if len(self.context_cache) > 1000:
            await self._evict_old_cache_entries()
    
    async def _evict_old_cache_entries(self, keep_count: int = 800):
        """Evict oldest cache entries to maintain performance"""
        if len(self.context_cache) <= keep_count:
            return
            
        # Sort by creation time and remove oldest entries
        sorted_entries = sorted(
            self.context_cache.items(),
            key=lambda x: x[1]["created_at"]
        )
        
        entries_to_remove = len(self.context_cache) - keep_count
        for i in range(entries_to_remove):
            cache_key = sorted_entries[i][0]
            del self.context_cache[cache_key]
            self.cache_access_times.pop(cache_key, None)
        
        self.cache_metrics["evictions"] += entries_to_remove
    
    def _update_cache_metrics(self, operation: str):
        """Update cache performance metrics"""
        self.cache_metrics["total_requests"] += 1
        if operation == "hit":
            self.cache_metrics["hits"] += 1
        elif operation == "miss":
            self.cache_metrics["misses"] += 1
    
    def _record_optimization_metric(self, metric_data: Dict[str, Any]):
        """Record optimization performance metric"""
        metric_data["timestamp"] = datetime.now()
        self.optimization_metrics.append(metric_data)
        
        # Keep only recent metrics in memory
        if len(self.optimization_metrics) > 10000:
            self.optimization_metrics = self.optimization_metrics[-5000:]
    
    def _extract_optimization_factors(self, agent_context: AgentContext) -> Dict[str, Any]:
        """Extract optimization factors from agent context"""
        return {
            "agent_type": agent_context.agent_type.value,
            "knowledge_items_considered": len(agent_context.relevant_knowledge),
            "window_utilization": agent_context.context_window_utilization,
            "context_size_bytes": agent_context.total_size
        }
    
    def _calculate_overall_relevance(self, agent_context: AgentContext) -> float:
        """Calculate overall relevance score for context"""
        if not agent_context.relevant_knowledge:
            return 0.0
        
        relevance_scores = [item.relevance_score for item in agent_context.relevant_knowledge]
        return sum(relevance_scores) / len(relevance_scores)
    
    def _calculate_completeness_score(self, agent_context: AgentContext) -> float:
        """Calculate completeness score based on context coverage"""
        # Simple heuristic: higher utilization = more complete
        return min(1.0, agent_context.context_window_utilization * 1.2)
    
    def _calculate_freshness_score(self, agent_context: AgentContext) -> float:
        """Calculate freshness score based on context recency"""
        if not agent_context.relevant_knowledge:
            return 0.0
        
        now = datetime.now()
        freshness_scores = []
        
        for item in agent_context.relevant_knowledge:
            age_hours = (now - item.last_updated).total_seconds() / 3600
            # Fresher is better, decay over 48 hours
            freshness = max(0.0, 1.0 - (age_hours / 48))
            freshness_scores.append(freshness)
        
        return sum(freshness_scores) / len(freshness_scores)
    
    def _generate_cache_recommendations(self) -> List[str]:
        """Generate cache optimization recommendations"""
        recommendations = []
        
        if self.cache_metrics["total_requests"] > 0:
            hit_rate = self.cache_metrics["hits"] / self.cache_metrics["total_requests"]
            if hit_rate < 0.3:
                recommendations.append("Cache hit rate below 30% - consider increasing TTL or improving cache key strategy")
            elif hit_rate > 0.8:
                recommendations.append("Excellent cache hit rate - current strategy is working well")
        
        if len(self.context_cache) > 500:
            recommendations.append("Cache size large - monitor memory usage and consider more aggressive eviction")
        
        return recommendations
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        if not self.context_cache:
            return 1.0
        
        # Simple heuristic based on cache size and hit rate
        cache_size_score = max(0.0, 1.0 - (len(self.context_cache) / 1000))
        hit_rate = (self.cache_metrics["hits"] / max(1, self.cache_metrics["total_requests"]))
        
        return (cache_size_score + hit_rate) / 2


# Factory function for easy API integration  
def create_context_optimization_api(**kwargs) -> ContextOptimizationAPI:
    """Create and configure context optimization API instance"""
    return ContextOptimizationAPI(**kwargs)

# Async context manager for resource cleanup
class ContextOptimizationManager:
    """Context manager for context optimization API with automatic cleanup"""
    
    def __init__(self, **kwargs):
        self.api = None
        self.kwargs = kwargs
    
    async def __aenter__(self) -> ContextOptimizationAPI:
        self.api = create_context_optimization_api(**self.kwargs)
        return self.api
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.api:
            # Cleanup resources
            await self.api.optimize_cache_performance()

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import asyncio
    
    async def test_optimization_api():
        """Test the context optimization API"""
        async with ContextOptimizationManager() as api:
            # Test agent context optimization
            test_request = {
                "agent_type": "rif-implementer",
                "task_context": {
                    "description": "Test context optimization for DPIBS implementation",
                    "complexity": "medium",
                    "issue_type": "feature"
                },
                "issue_number": 137
            }
            
            print("Testing context optimization API...")
            start_time = time.time()
            
            result = await api.optimize_agent_context(**test_request)
            
            optimization_time = (time.time() - start_time) * 1000
            print(f"Optimization completed in {optimization_time:.2f}ms")
            print(f"Target met (<200ms): {optimization_time < 200}")
            print(f"Context items: {result['metadata']['total_items']}")
            print(f"Window utilization: {result['metadata']['window_utilization']:.2%}")
            
            # Test analytics
            analytics = api.get_optimization_analytics(1)  # Last hour
            print(f"\nAnalytics: {analytics['total_optimizations']} optimizations")
    
    if __name__ == "__main__":
        asyncio.run(test_optimization_api())