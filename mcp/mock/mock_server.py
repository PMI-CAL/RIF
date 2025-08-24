"""
Mock MCP Server

Mock implementation of an MCP server for testing dynamic loading functionality.

Issue: #82 - Dynamic MCP loader (testing support)
Component: Mock server for development and testing
"""

import asyncio
import logging
import random
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MockMCPServer:
    """
    Mock MCP server implementation for testing dynamic loading
    
    Simulates the behavior of a real MCP server including:
    - Health checks
    - Resource usage simulation
    - Error scenarios for testing
    - Cleanup operations
    """
    
    def __init__(self, server_config: Dict[str, Any]):
        """
        Initialize mock server
        
        Args:
            server_config: Server configuration
        """
        self.config = server_config
        self.server_id = server_config['server_id']
        self.name = server_config['name']
        self.is_running = False
        self.health_failures = 0
        self.max_health_failures = 3  # Fail health check after 3 failures
        
        # Simulate resource usage
        self.memory_usage_mb = server_config.get('resource_requirements', {}).get('memory_mb', 64)
        self.cpu_usage_percent = server_config.get('resource_requirements', {}).get('cpu_percent', 5)
        
        logger.debug(f"MockMCPServer initialized: {self.name}")
    
    async def initialize(self):
        """Initialize the mock server"""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self.is_running = True
        logger.info(f"Mock server initialized: {self.name}")
    
    async def health_check(self) -> str:
        """
        Perform health check
        
        Returns:
            Health status string
        """
        if not self.is_running:
            return "unhealthy"
        
        # For testing, always return healthy for successful demo
        # Simulate occasional health issues in production mode
        if random.random() < 0.05:  # 5% chance of degraded status, but not unhealthy
            return "degraded"
        else:
            self.health_failures = max(0, self.health_failures - 1)  # Recover gradually
            return "healthy"
    
    async def cleanup(self):
        """Clean up server resources"""
        self.is_running = False
        await asyncio.sleep(0.05)  # Simulate cleanup time
        logger.info(f"Mock server cleaned up: {self.name}")
    
    async def get_capabilities(self) -> list:
        """Get server capabilities"""
        return self.config.get('capabilities', [])
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        # Simulate some variation in resource usage
        memory_variation = random.uniform(0.8, 1.2)
        cpu_variation = random.uniform(0.5, 1.5)
        
        return {
            'memory_mb': int(self.memory_usage_mb * memory_variation),
            'cpu_percent': min(100, int(self.cpu_usage_percent * cpu_variation)),
            'status': 'running' if self.is_running else 'stopped'
        }
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock tool execution
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate execution time
        
        return {
            'tool': tool_name,
            'result': 'success',
            'output': f"Mock execution of {tool_name} with {len(parameters)} parameters",
            'execution_time_ms': random.randint(100, 500)
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        return {
            'server_id': self.server_id,
            'name': self.name,
            'running': self.is_running,
            'health_failures': self.health_failures,
            'capabilities': await self.get_capabilities(),
            'resource_usage': await self.get_resource_usage()
        }