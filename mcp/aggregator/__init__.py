"""
MCP Context Aggregator

Advanced context aggregation system that queries multiple MCP servers in parallel,
intelligently merges responses, and optimizes context for agent consumption.

Issue: #85 - Implement MCP context aggregator
Component: Core aggregation system with 85% component reuse
"""

from .context_aggregator import MCPContextAggregator, AggregationResult, ServerResponse, QueryOptimizationResult

__all__ = [
    'MCPContextAggregator',
    'AggregationResult',
    'ServerResponse',
    'QueryOptimizationResult'
]