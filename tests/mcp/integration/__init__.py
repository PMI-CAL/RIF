"""
MCP Integration Testing Framework

Comprehensive testing framework for MCP server integration with mock infrastructure,
performance benchmarking, and automated test scenarios.

Issue: #86 - Build MCP integration tests
Component: Integration testing framework
Phase: 1 - Enhanced Mock Framework
"""

__version__ = "1.0.0"
__author__ = "RIF-Implementer"

# Import key classes for easy access
from .enhanced_mock_server import EnhancedMockMCPServer
from .test_base import MCPIntegrationTestBase
from .performance_metrics import PerformanceMetrics

__all__ = [
    "EnhancedMockMCPServer",
    "MCPIntegrationTestBase", 
    "PerformanceMetrics"
]