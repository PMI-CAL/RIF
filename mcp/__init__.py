"""
MCP (Model Context Protocol) Integration Package

This package provides comprehensive MCP server integration for the RIF system,
including dynamic loading, security management, and performance optimization.

Components:
- loader: Dynamic MCP server loading based on project requirements
- registry: MCP server catalog and capability tracking
- security: Security gateway and credential management
- monitor: Health monitoring and performance tracking
"""

from .loader.dynamic_loader import DynamicMCPLoader
from .registry.server_registry import MCPServerRegistry
from .security.gateway import MCPSecurityGateway

__version__ = "1.0.0"
__all__ = ["DynamicMCPLoader", "MCPServerRegistry", "MCPSecurityGateway"]