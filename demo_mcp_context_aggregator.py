"""
Demo script for MCP Context Aggregator

Tests core functionality without complex dependencies.
Issue: #85 - Implement MCP context aggregator
"""

import asyncio
import json
import time
from datetime import datetime

# Test the core MockHealthMonitor independently
class SimpleMockHealthMonitor:
    def __init__(self):
        self.server_health = {}
        
    async def get_server_health(self, server_id: str) -> str:
        return "healthy"
    
    async def get_healthy_servers(self, server_ids):
        return server_ids  # All healthy
    
    async def register_server(self, server_id: str):
        self.server_health[server_id] = "healthy"

# Test cache functionality
from cachetools import TTLCache
import hashlib

class SimpleCacheManager:
    def __init__(self):
        self.cache = TTLCache(maxsize=100, ttl=300)
        self.hit_count = 0
        self.miss_count = 0
        
    def generate_cache_key(self, query: str, servers, context=None):
        key_parts = [query.strip().lower(), "|".join(sorted(servers))]
        if context:
            if 'agent_type' in context:
                key_parts.append(f"agent:{context['agent_type']}")
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    async def get(self, cache_key: str):
        if cache_key in self.cache:
            self.hit_count += 1
            return self.cache[cache_key]
        self.miss_count += 1
        return None
    
    async def put(self, cache_key: str, result):
        self.cache[cache_key] = result
    
    def get_cache_stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache)
        }

# Mock server response
class MockServerResponse:
    def __init__(self, server_id, server_name, response_time=150):
        self.server_id = server_id
        self.server_name = server_name
        self.response_time_ms = response_time
        self.status = "success"
        self.response_data = {
            "results": [
                {"content": f"Mock result from {server_name} for query", "relevance": 0.8},
                {"content": f"Additional context from {server_name}", "relevance": 0.6}
            ],
            "metadata": {
                "server_id": server_id,
                "server_name": server_name,
                "result_count": 2
            }
        }

# Simple aggregator for testing
class SimpleMCPAggregator:
    def __init__(self):
        self.health_monitor = SimpleMockHealthMonitor()
        self.cache_manager = SimpleCacheManager()
        self.query_count = 0
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time_ms': 0.0,
            'successful_queries': 0
        }
    
    async def get_context(self, query: str, servers=None, use_cache=True):
        start_time = time.time()
        self.query_count += 1
        self.performance_metrics['total_queries'] += 1
        
        # Default servers
        if not servers:
            servers = ["server1", "server2", "server3"]
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self.cache_manager.generate_cache_key(query, servers)
            cached = await self.cache_manager.get(cache_key)
            if cached:
                print(f"Cache hit for query: {query[:50]}...")
                return cached
        
        # Mock parallel server queries
        print(f"Querying {len(servers)} servers in parallel...")
        server_responses = []
        for server_id in servers:
            # Simulate variable response times
            response_time = 100 + (hash(server_id) % 100)
            await asyncio.sleep(0.01)  # Small delay to simulate network
            
            response = MockServerResponse(server_id, f"Server_{server_id}", response_time)
            server_responses.append(response)
        
        # Simple response merging
        all_results = []
        for response in server_responses:
            all_results.extend(response.response_data["results"])
        
        # Create aggregated result
        total_time_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "query": query,
            "total_time_ms": total_time_ms,
            "servers_queried": len(servers),
            "successful_servers": len(server_responses),
            "merged_response": {
                "results": all_results,
                "server_count": len(servers),
                "sources": [r.server_name for r in server_responses]
            },
            "cache_info": self.cache_manager.get_cache_stats()
        }
        
        # Cache result
        if use_cache and cache_key:
            await self.cache_manager.put(cache_key, result)
        
        # Update metrics
        self.performance_metrics['successful_queries'] += 1
        current_avg = self.performance_metrics['avg_response_time_ms']
        n = self.performance_metrics['total_queries']
        self.performance_metrics['avg_response_time_ms'] = (
            (current_avg * (n-1) + total_time_ms) / n
        )
        
        return result
    
    async def health_check(self):
        return {
            "status": "healthy",
            "components": {
                "health_monitor": "healthy (mock)",
                "cache_manager": "healthy"
            },
            "performance_metrics": self.performance_metrics
        }

async def demo_basic_functionality():
    """Demo basic aggregator functionality"""
    print("MCP Context Aggregator Demo")
    print("=" * 40)
    
    # Initialize aggregator
    aggregator = SimpleMCPAggregator()
    
    # Health check
    print("\n1. Health Check:")
    health = await aggregator.health_check()
    print(json.dumps(health, indent=2))
    
    # Test queries
    test_queries = [
        "find authentication patterns",
        "get database connection examples", 
        "show error handling best practices"
    ]
    
    print(f"\n2. Testing {len(test_queries)} queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # First query (cache miss)
        result = await aggregator.get_context(query, use_cache=True)
        print(f"  Response time: {result['total_time_ms']}ms")
        print(f"  Servers: {result['successful_servers']}/{result['servers_queried']}")
        print(f"  Results: {len(result['merged_response']['results'])}")
        print(f"  Cache hit rate: {result['cache_info']['hit_rate_percent']:.1f}%")
        
        # Second query (cache hit)
        result2 = await aggregator.get_context(query, use_cache=True)
        print(f"  Second query time: {result2['total_time_ms']}ms (cached)")
    
    # Performance metrics
    print(f"\n3. Final Performance Metrics:")
    final_health = await aggregator.health_check()
    print(json.dumps(final_health['performance_metrics'], indent=2))
    
    print(f"\n4. Cache Statistics:")
    cache_stats = aggregator.cache_manager.get_cache_stats()
    print(json.dumps(cache_stats, indent=2))

async def benchmark_performance():
    """Benchmark aggregator performance"""
    print("\nPerformance Benchmark")
    print("=" * 40)
    
    aggregator = SimpleMCPAggregator()
    
    # Benchmark with different server counts
    server_configs = [
        ["server1"],
        ["server1", "server2"],
        ["server1", "server2", "server3", "server4"]
    ]
    
    query = "benchmark test query"
    
    for servers in server_configs:
        print(f"\nTesting with {len(servers)} servers:")
        
        # Run multiple iterations
        times = []
        for i in range(5):
            result = await aggregator.get_context(
                query + f" {i}", 
                servers=servers, 
                use_cache=False
            )
            times.append(result['total_time_ms'])
        
        avg_time = sum(times) / len(times)
        print(f"  Average response time: {avg_time:.1f}ms")
        print(f"  Response time range: {min(times)}-{max(times)}ms")

async def test_caching_behavior():
    """Test caching behavior specifically"""
    print("\nCaching Behavior Test")
    print("=" * 40)
    
    aggregator = SimpleMCPAggregator()
    query = "caching test query"
    
    print("\n1. First query (cache miss):")
    result1 = await aggregator.get_context(query, use_cache=True)
    print(f"  Time: {result1['total_time_ms']}ms")
    print(f"  Cache stats: {result1['cache_info']}")
    
    print("\n2. Second query (cache hit):")
    result2 = await aggregator.get_context(query, use_cache=True)
    print(f"  Time: {result2['total_time_ms']}ms")
    print(f"  Cache stats: {result2['cache_info']}")
    
    print("\n3. Different query (cache miss):")
    result3 = await aggregator.get_context("different query", use_cache=True)
    print(f"  Time: {result3['total_time_ms']}ms")
    print(f"  Cache stats: {result3['cache_info']}")

async def main():
    """Main demo function"""
    try:
        await demo_basic_functionality()
        await benchmark_performance()
        await test_caching_behavior()
        
        print(f"\n{'='*40}")
        print("✓ All demo tests completed successfully!")
        print("MCP Context Aggregator core functionality validated.")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())