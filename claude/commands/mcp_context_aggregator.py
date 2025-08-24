"""
MCP Context Aggregator Command Interface

CLI interface for the MCP Context Aggregator system, providing context queries
across multiple MCP servers with intelligent merging and caching.

Issue: #85 - Implement MCP context aggregator
Agent: RIF-Implementer
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import the core aggregator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from mcp.aggregator.context_aggregator import MCPContextAggregator
from knowledge.context.optimizer import ContextOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPContextAggregatorCLI:
    """
    Command-line interface for MCP Context Aggregator.
    
    Provides easy access to context aggregation functionality with
    monitoring, performance metrics, and debugging capabilities.
    """
    
    def __init__(self):
        """Initialize the CLI with aggregator instance"""
        self.aggregator = None
        self._initialize_aggregator()
    
    def _initialize_aggregator(self):
        """Initialize the MCP Context Aggregator with default settings"""
        try:
            # Initialize with production-ready settings
            context_optimizer = ContextOptimizer()
            
            self.aggregator = MCPContextAggregator(
                context_optimizer=context_optimizer,
                max_concurrent_servers=4,
                query_timeout_seconds=10,
                cache_ttl_seconds=300
            )
            
            logger.info("MCP Context Aggregator CLI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize aggregator: {e}")
            self.aggregator = None
    
    async def query_context(self, query: str, 
                          agent_type: str = 'default',
                          servers: Optional[List[str]] = None,
                          use_cache: bool = True,
                          explain: bool = False,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query context across multiple MCP servers
        
        Args:
            query: Context query string
            agent_type: Agent type for optimization (rif-analyst, rif-implementer, etc.)
            servers: Specific servers to query (None for auto-discovery)
            use_cache: Whether to use response caching
            explain: Include detailed explanation
            context: Additional query context
            
        Returns:
            Aggregated context result
        """
        if not self.aggregator:
            return {"error": "Aggregator not initialized", "status": "failed"}
        
        try:
            logger.info(f"Querying context: {query[:50]}...")
            
            result = await self.aggregator.get_context(
                query=query,
                required_servers=servers,
                agent_type=agent_type,
                context=context,
                use_cache=use_cache,
                explain=explain
            )
            
            return self._format_result(result, explain)
            
        except Exception as e:
            logger.error(f"Context query failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _format_result(self, result, explain: bool = False) -> Dict[str, Any]:
        """Format aggregation result for CLI output"""
        formatted = {
            "status": "success" if result.successful_servers > 0 else "failed",
            "query": result.query,
            "total_time_ms": result.total_time_ms,
            "servers_queried": len(result.server_responses),
            "successful_servers": result.successful_servers,
            "failed_servers": result.failed_servers,
            "cache_info": result.cache_info,
            "merged_response": result.merged_response
        }
        
        if explain:
            formatted["optimization_info"] = result.optimization_info
            formatted["performance_metrics"] = result.performance_metrics
            formatted["server_responses"] = [
                {
                    "server_id": r.server_id,
                    "server_name": r.server_name,
                    "status": r.status,
                    "response_time_ms": r.response_time_ms,
                    "error": r.error_message
                }
                for r in result.server_responses
            ]
        
        return formatted
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of the aggregator system"""
        if not self.aggregator:
            return {"status": "failed", "error": "Aggregator not initialized"}
        
        try:
            return await self.aggregator.health_check()
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.aggregator:
            return {"error": "Aggregator not initialized"}
        
        try:
            return await self.aggregator.get_performance_summary()
        except Exception as e:
            return {"error": str(e)}
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear the response cache"""
        if not self.aggregator:
            return {"error": "Aggregator not initialized"}
        
        try:
            return await self.aggregator.clear_cache()
        except Exception as e:
            return {"error": str(e)}
    
    async def benchmark_performance(self, queries: List[str], 
                                  iterations: int = 5) -> Dict[str, Any]:
        """
        Benchmark aggregator performance with multiple queries
        
        Args:
            queries: List of test queries
            iterations: Number of iterations per query
            
        Returns:
            Benchmark results
        """
        if not self.aggregator:
            return {"error": "Aggregator not initialized"}
        
        try:
            benchmark_results = {
                "start_time": datetime.now().isoformat(),
                "queries_tested": len(queries),
                "iterations_per_query": iterations,
                "results": []
            }
            
            for query in queries:
                query_results = []
                
                for i in range(iterations):
                    logger.info(f"Benchmarking query {queries.index(query) + 1}/{len(queries)}, iteration {i + 1}/{iterations}")
                    
                    result = await self.aggregator.get_context(
                        query=query,
                        use_cache=(i > 0)  # First iteration bypasses cache
                    )
                    
                    query_results.append({
                        "iteration": i + 1,
                        "total_time_ms": result.total_time_ms,
                        "successful_servers": result.successful_servers,
                        "cache_hit": i > 0 and result.cache_info.get('hit_rate_percent', 0) > 0
                    })
                
                avg_time = sum(r["total_time_ms"] for r in query_results) / iterations
                cache_hits = sum(1 for r in query_results if r.get("cache_hit", False))
                
                benchmark_results["results"].append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "avg_response_time_ms": avg_time,
                    "cache_hits": cache_hits,
                    "cache_hit_rate": (cache_hits / iterations) * 100,
                    "detailed_results": query_results
                })
            
            benchmark_results["end_time"] = datetime.now().isoformat()
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"error": str(e)}


async def main():
    """
    Main CLI entry point for MCP Context Aggregator.
    Provides interactive mode and command execution.
    """
    print("MCP Context Aggregator CLI")
    print("=" * 50)
    
    # Initialize CLI
    cli = MCPContextAggregatorCLI()
    
    # Health check on startup
    health = await cli.health_check()
    print(f"System Status: {health.get('status', 'unknown')}")
    
    if health.get('status') == 'healthy':
        print(f"Available Servers: {health.get('available_servers', 0)}")
        print("")
    
    # Interactive mode
    print("Available commands:")
    print("  query <text> - Query context across MCP servers")
    print("  health - Check system health")  
    print("  metrics - Show performance metrics")
    print("  cache clear - Clear response cache")
    print("  benchmark - Run performance benchmark")
    print("  help - Show this help")
    print("  exit - Exit the CLI")
    print("")
    
    while True:
        try:
            command = input("mcp-aggregator> ").strip()
            
            if not command:
                continue
            elif command == "exit":
                break
            elif command == "help":
                print("Commands:")
                print("  query <text> - Query context")
                print("  health - System health check")
                print("  metrics - Performance metrics")
                print("  cache clear - Clear cache")
                print("  benchmark - Performance benchmark")
                print("  exit - Exit")
            elif command == "health":
                result = await cli.health_check()
                print(json.dumps(result, indent=2))
            elif command == "metrics":
                result = await cli.get_performance_metrics()
                print(json.dumps(result, indent=2))
            elif command == "cache clear":
                result = await cli.clear_cache()
                print(json.dumps(result, indent=2))
            elif command == "benchmark":
                test_queries = [
                    "find authentication patterns",
                    "get database connection examples", 
                    "show error handling best practices",
                    "list testing frameworks"
                ]
                print("Running benchmark with test queries...")
                result = await cli.benchmark_performance(test_queries, iterations=3)
                print(json.dumps(result, indent=2))
            elif command.startswith("query "):
                query_text = command[6:].strip()
                if query_text:
                    result = await cli.query_context(query_text, explain=True)
                    print(json.dumps(result, indent=2))
                else:
                    print("Please provide query text: query <text>")
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("MCP Context Aggregator CLI terminated.")


# Demo function for testing
async def demo_aggregation():
    """
    Demo function showing MCP Context Aggregator capabilities
    """
    print("MCP Context Aggregator Demo")
    print("=" * 40)
    
    # Initialize CLI
    cli = MCPContextAggregatorCLI()
    
    # Health check
    print("1. Health Check:")
    health = await cli.health_check()
    print(json.dumps(health, indent=2))
    print()
    
    # Test queries for different agent types
    test_queries = [
        {
            "query": "find authentication implementation patterns",
            "agent_type": "rif-implementer",
            "context": {"issue_id": "85", "component": "auth"}
        },
        {
            "query": "analyze security vulnerabilities in user management",
            "agent_type": "rif-analyst", 
            "context": {"issue_id": "85", "domain": "security"}
        },
        {
            "query": "get database migration best practices",
            "agent_type": "rif-architect",
            "context": {"issue_id": "85", "component": "database"}
        }
    ]
    
    # Execute test queries
    for i, test in enumerate(test_queries, 1):
        print(f"{i}. Query Test - {test['agent_type']}:")
        print(f"Query: {test['query']}")
        
        result = await cli.query_context(
            query=test['query'],
            agent_type=test['agent_type'],
            context=test['context'],
            explain=True
        )
        
        print(f"Status: {result.get('status')}")
        print(f"Response Time: {result.get('total_time_ms')}ms")
        print(f"Servers: {result.get('successful_servers')}/{result.get('servers_queried')}")
        print(f"Cache Hit Rate: {result.get('cache_info', {}).get('hit_rate_percent', 0):.1f}%")
        print()
    
    # Performance metrics
    print("4. Performance Metrics:")
    metrics = await cli.get_performance_metrics()
    print(json.dumps(metrics, indent=2))
    print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Context Aggregator CLI")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--query", type=str, help="Execute single query")
    parser.add_argument("--agent-type", type=str, default="default", help="Agent type for query")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_aggregation())
    elif args.query:
        async def run_single_query():
            cli = MCPContextAggregatorCLI()
            result = await cli.query_context(args.query, agent_type=args.agent_type, explain=True)
            print(json.dumps(result, indent=2))
        asyncio.run(run_single_query())
    else:
        asyncio.run(main())