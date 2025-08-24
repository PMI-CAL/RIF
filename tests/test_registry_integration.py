"""
Integration Tests for MCP Server Registry

Tests integration between the server registry and other MCP components.

Issue: #81 - Create MCP server registry
Component: Integration test suite
"""

import pytest
import asyncio
import tempfile
from typing import Dict, Any

# Add the mcp module to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp.registry.server_registry import MCPServerRegistry
from mcp.monitor.health_monitor import HealthMonitor, HealthStatus


class MockServer:
    """Mock server for testing health monitor integration"""
    
    def __init__(self, server_id: str, health_status: str = HealthStatus.HEALTHY):
        self.server_id = server_id
        self._health_status = health_status
    
    async def health_check(self):
        """Mock health check"""
        return self._health_status
    
    def set_health_status(self, status: str):
        """Set mock health status"""
        self._health_status = status


class TestRegistryIntegration:
    """Test registry integration with other components"""
    
    @pytest.fixture
    def registry(self):
        """Create a registry instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = os.path.join(temp_dir, "integration_registry.json")
            registry = MCPServerRegistry(registry_file=registry_file, auto_save=True)
            yield registry
    
    @pytest.fixture
    def health_monitor(self):
        """Create a health monitor for testing"""
        return HealthMonitor(check_interval_seconds=1)
    
    @pytest.mark.asyncio
    async def test_registry_health_monitor_integration(self, registry, health_monitor):
        """Test integration between registry and health monitor"""
        # Register server in registry
        server_config = {
            "server_id": "integration-test-server",
            "name": "Integration Test Server",
            "capabilities": ["integration_testing"],
            "resource_requirements": {"memory_mb": 64, "cpu_percent": 5}
        }
        
        success = await registry.register_server(server_config)
        assert success is True
        
        # Create mock server for health monitoring
        mock_server = MockServer("integration-test-server", HealthStatus.HEALTHY)
        
        # Set up health callback to update registry
        health_updates = []
        
        def health_callback(server_id: str, new_status: str):
            health_updates.append((server_id, new_status))
        
        registry.set_health_callback(health_callback)
        
        # Register server with health monitor
        await health_monitor.register_server(mock_server, server_config)
        
        # Let health monitor run one check cycle
        await asyncio.sleep(0.1)
        
        # Update registry based on health monitor results
        server_status = await health_monitor.get_server_status("integration-test-server")
        if server_status:
            await registry.update_server_health(
                "integration-test-server", 
                server_status["status"],
                {
                    "response_time_ms": server_status["average_response_time_ms"],
                    "uptime_percent": server_status["uptime_percent"]
                }
            )
        
        # Verify registry was updated
        server_info = await registry.get_server("integration-test-server", include_metrics=True)
        assert server_info is not None
        assert server_info["health_status"] == HealthStatus.HEALTHY
        assert server_info["total_health_checks"] > 0
        
        # Test health status change
        mock_server.set_health_status(HealthStatus.UNHEALTHY)
        
        # Let health monitor detect the change
        await asyncio.sleep(1.2)  # Wait for next health check
        
        # Update registry with new status
        server_status = await health_monitor.get_server_status("integration-test-server")
        if server_status:
            await registry.update_server_health(
                "integration-test-server",
                server_status["status"]
            )
        
        # Verify health callback was triggered
        assert len(health_updates) > 0
        last_update = health_updates[-1]
        assert last_update[0] == "integration-test-server"
        assert last_update[1] == HealthStatus.UNHEALTHY
        
        # Clean up
        await health_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_registry_capability_discovery(self, registry):
        """Test registry's capability discovery for dynamic loading"""
        # Register multiple servers with different capabilities
        servers = [
            {
                "server_id": "github-test-server",
                "name": "GitHub Test Server",
                "capabilities": ["github_api", "issue_management"]
            },
            {
                "server_id": "database-test-server", 
                "name": "Database Test Server",
                "capabilities": ["database_operations", "sql_queries"]
            },
            {
                "server_id": "multi-capability-server",
                "name": "Multi-Capability Server",
                "capabilities": ["github_api", "database_operations", "file_operations"]
            }
        ]
        
        # Register all servers
        for server_config in servers:
            await registry.register_server(server_config)
        
        # Test capability-based discovery (simulating dynamic loader requirements)
        
        # Find servers for GitHub operations
        github_servers = await registry.find_servers_by_capability("github_api")
        assert len(github_servers) >= 2  # At least github-test-server + multi-capability-server (+ default server)
        
        # Verify our specific servers are in the results
        server_ids = [s["server_id"] for s in github_servers]
        assert "github-test-server" in server_ids
        assert "multi-capability-server" in server_ids
        
        # Find servers for database operations
        db_servers = await registry.find_servers_by_capability("database_operations")
        assert len(db_servers) == 2  # database-test-server + multi-capability-server
        
        # Get complete capability catalog
        catalog = await registry.get_capability_catalog()
        assert "github_api" in catalog
        assert "database_operations" in catalog
        assert len(catalog["github_api"]) >= 2  # At least our servers + default
        assert len(catalog["database_operations"]) == 2
        
        # Test resource-based filtering for server selection
        all_servers = await registry.list_servers()
        healthy_servers = await registry.list_servers(health_status="unknown")  # Default status
        
        assert len(all_servers) >= 3  # At least our test servers + defaults
        assert len(healthy_servers) >= 3
    
    @pytest.mark.asyncio
    async def test_registry_dependency_chain_validation(self, registry):
        """Test dependency chain validation for complex server configurations"""
        # Create dependency chain: A -> B -> C
        servers = [
            {
                "server_id": "base-server-c",
                "name": "Base Server C",
                "capabilities": ["base_capability"]
            },
            {
                "server_id": "middle-server-b",
                "name": "Middle Server B", 
                "capabilities": ["middle_capability"],
                "dependencies": ["base-server-c"]
            },
            {
                "server_id": "top-server-a",
                "name": "Top Server A",
                "capabilities": ["top_capability"],
                "dependencies": ["middle-server-b"]
            }
        ]
        
        # Register servers in order
        for server_config in servers:
            await registry.register_server(server_config)
        
        # Make all servers healthy
        for server in servers:
            await registry.update_server_health(server["server_id"], "healthy")
        
        # Validate dependency chain - should pass
        validation = await registry.validate_server_dependencies("top-server-a")
        assert validation["valid"] is True
        
        validation = await registry.validate_server_dependencies("middle-server-b") 
        assert validation["valid"] is True
        
        # Break the chain by making base server unhealthy
        await registry.update_server_health("base-server-c", "unhealthy")
        
        # Validate dependency chain - middle server should fail
        validation = await registry.validate_server_dependencies("middle-server-b")
        assert validation["valid"] is False
        assert "base-server-c" in validation["unhealthy_dependencies"]
        
        # Top server dependency validation should still work
        # (it only checks direct dependencies, not transitive ones)
        validation = await registry.validate_server_dependencies("top-server-a")
        assert validation["valid"] is True  # middle-server-b is still healthy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])