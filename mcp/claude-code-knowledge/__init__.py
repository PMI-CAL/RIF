"""
Claude Code Knowledge MCP Server - Phase 2 Implementation.

A Model Context Protocol server providing Claude Code capability knowledge
through the existing RIF knowledge graph system.

This module provides:
- MCP server implementation with 5 core tools
- Query engine for optimized knowledge graph access  
- Safety features including validation and graceful degradation
- Integration with existing RIF DuckDB knowledge graph
- Comprehensive error handling and monitoring

Phase 2 delivers a production-ready MCP server that acts as a read-only
query interface over the Claude Code knowledge graph extensions from Phase 1.
"""

__version__ = "1.0.0"
__author__ = "RIF Team"
__description__ = "Claude Code Knowledge MCP Server"

# Core components
from .server import ClaudeCodeKnowledgeServer
from .config import ServerConfig, load_server_config
from .query_engine import ClaudeKnowledgeQueryEngine, QueryResult
from .safety import (
    InputValidator, 
    OutputSanitizer, 
    RateLimiter, 
    GracefulDegradation,
    HealthMonitor
)

# Data structures
from .server import (
    MCPRequest,
    MCPResponse, 
    CompatibilityCheck,
    PatternRecommendation,
    ArchitectureValidation
)

__all__ = [
    # Core server
    "ClaudeCodeKnowledgeServer",
    
    # Configuration
    "ServerConfig", 
    "load_server_config",
    
    # Query engine
    "ClaudeKnowledgeQueryEngine",
    "QueryResult",
    
    # Safety components
    "InputValidator",
    "OutputSanitizer", 
    "RateLimiter",
    "GracefulDegradation",
    "HealthMonitor",
    
    # Data structures
    "MCPRequest",
    "MCPResponse",
    "CompatibilityCheck", 
    "PatternRecommendation",
    "ArchitectureValidation",
]


def create_server(config_path=None):
    """
    Convenience function to create and initialize MCP server.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ClaudeCodeKnowledgeServer instance
    """
    config = load_server_config(config_path)
    return ClaudeCodeKnowledgeServer(config.__dict__)


async def main():
    """Main entry point for running the server."""
    import asyncio
    
    server = create_server()
    
    if not await server.initialize():
        print("Failed to initialize server")
        return 1
    
    try:
        print("Claude Code Knowledge MCP Server started")
        await server.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await server.shutdown()
    
    return 0


if __name__ == "__main__":
    import sys
    import asyncio
    
    sys.exit(asyncio.run(main()))