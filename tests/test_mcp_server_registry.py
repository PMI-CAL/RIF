"""
Tests for MCP Server Registry

Comprehensive test suite for the enhanced MCP server registry implementation.

Issue: #81 - Create MCP server registry
Component: Test suite for server registry
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the mcp module to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp.registry.server_registry import MCPServerRegistry, ServerRecord


class TestMCPServerRegistry:
    """Test suite for MCPServerRegistry"""
    
    @pytest.fixture
    def registry(self):
        """Create a registry instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = os.path.join(temp_dir, "test_registry.json")
            registry = MCPServerRegistry(registry_file=registry_file, auto_save=True)
            yield registry
    
    @pytest.fixture
    def sample_server_config(self):
        """Create a sample server configuration"""
        return {
            "server_id": "test-server-v1.0.0",
            "name": "Test MCP Server",
            "version": "1.0.0",
            "capabilities": ["test_capability", "sample_feature"],
            "resource_requirements": {"memory_mb": 128, "cpu_percent": 10},
            "description": "Test server for unit testing",
            "tags": ["test", "development"],
            "dependencies": [],
            "configuration": {"debug": True}
        }
    
    @pytest.mark.asyncio
    async def test_registry_initialization(self, registry):
        """Test registry initialization with default servers"""
        # Registry should initialize with default servers
        servers = await registry.list_servers()
        assert len(servers) == 5  # 5 default servers
        
        # Check that capability index is built
        catalog = await registry.get_capability_catalog()
        assert len(catalog) > 0
        assert "github_api" in catalog
        assert "git_operations" in catalog
    
    @pytest.mark.asyncio
    async def test_server_registration(self, registry, sample_server_config):
        """Test server registration functionality"""
        # Register new server
        success = await registry.register_server(sample_server_config)
        assert success is True
        
        # Verify server is registered
        server = await registry.get_server("test-server-v1.0.0")
        assert server is not None
        assert server["name"] == "Test MCP Server"
        assert server["version"] == "1.0.0"
        assert "test_capability" in server["capabilities"]
        assert server["description"] == "Test server for unit testing"
        assert "test" in server["tags"]
    
    @pytest.mark.asyncio
    async def test_server_registration_validation(self, registry):
        """Test server registration validation"""
        # Test missing required fields
        incomplete_config = {"name": "Incomplete Server"}
        success = await registry.register_server(incomplete_config)
        assert success is False
        
        # Test duplicate registration
        valid_config = {
            "server_id": "duplicate-test",
            "name": "Duplicate Test",
            "capabilities": ["test"]
        }
        
        success1 = await registry.register_server(valid_config)
        assert success1 is True
        
        success2 = await registry.register_server(valid_config)
        assert success2 is False  # Should fail due to duplicate
    
    @pytest.mark.asyncio
    async def test_capability_indexing(self, registry, sample_server_config):
        """Test capability-based server discovery"""
        # Register server with specific capabilities
        await registry.register_server(sample_server_config)
        
        # Find servers by capability
        servers = await registry.find_servers_by_capability("test_capability")
        assert len(servers) == 1
        assert servers[0]["server_id"] == "test-server-v1.0.0"
        
        # Test non-existent capability
        servers = await registry.find_servers_by_capability("non_existent")
        assert len(servers) == 0
        
        # Test existing capability from default servers
        servers = await registry.find_servers_by_capability("github_api")
        assert len(servers) == 1
        assert "github" in servers[0]["server_id"].lower()
    
    @pytest.mark.asyncio
    async def test_tag_indexing(self, registry, sample_server_config):
        """Test tag-based server discovery"""
        # Register server with tags
        await registry.register_server(sample_server_config)
        
        # Find servers by tag
        servers = await registry.find_servers_by_tag("test")
        assert len(servers) == 1
        assert servers[0]["server_id"] == "test-server-v1.0.0"
        
        # Test non-existent tag
        servers = await registry.find_servers_by_tag("non_existent")
        assert len(servers) == 0
    
    @pytest.mark.asyncio
    async def test_version_indexing(self, registry, sample_server_config):
        """Test version-based server discovery"""
        # Register server with specific version
        await registry.register_server(sample_server_config)
        
        # Find servers by version
        servers = await registry.find_servers_by_version("1.0.0")
        assert len(servers) >= 1  # At least our test server + default servers
        
        # Verify our test server is in the results
        test_server = next((s for s in servers if s["server_id"] == "test-server-v1.0.0"), None)
        assert test_server is not None
    
    @pytest.mark.asyncio
    async def test_health_status_updates(self, registry, sample_server_config):
        """Test health status tracking"""
        # Register server
        await registry.register_server(sample_server_config)
        server_id = "test-server-v1.0.0"
        
        # Update health status
        success = await registry.update_server_health(server_id, "healthy")
        assert success is True
        
        # Verify health status update
        server = await registry.get_server(server_id)
        assert server["health_status"] == "healthy"
        assert server["total_health_checks"] == 1
        
        # Update with health check result
        health_result = {"response_time_ms": 50, "uptime_percent": 99.5}
        success = await registry.update_server_health(server_id, "degraded", health_result)
        assert success is True
        
        # Verify metrics update
        server = await registry.get_server(server_id, include_metrics=True)
        assert server["health_status"] == "degraded"
        assert server["total_health_checks"] == 2
        assert server["failed_health_checks"] == 1
        assert server["metrics"]["last_response_time_ms"] == 50
    
    @pytest.mark.asyncio
    async def test_resource_based_filtering(self, registry, sample_server_config):
        """Test resource-based server filtering"""
        # Register server with specific resource requirements
        await registry.register_server(sample_server_config)
        
        # Find servers within resource constraints
        servers = await registry.find_servers_by_resource_requirements(
            max_memory_mb=200, max_cpu_percent=15
        )
        
        # Should include our test server and some default servers
        assert len(servers) > 0
        test_server = next((s for s in servers if s["server_id"] == "test-server-v1.0.0"), None)
        assert test_server is not None
        
        # Test strict constraints that exclude our server
        servers = await registry.find_servers_by_resource_requirements(
            max_memory_mb=50, max_cpu_percent=5
        )
        test_server = next((s for s in servers if s["server_id"] == "test-server-v1.0.0"), None)
        assert test_server is None  # Should be excluded
    
    @pytest.mark.asyncio
    async def test_server_unregistration(self, registry, sample_server_config):
        """Test server unregistration and index cleanup"""
        # Register server
        await registry.register_server(sample_server_config)
        server_id = "test-server-v1.0.0"
        
        # Verify registration
        server = await registry.get_server(server_id)
        assert server is not None
        
        # Verify capability index
        servers = await registry.find_servers_by_capability("test_capability")
        assert len(servers) == 1
        
        # Unregister server
        success = await registry.unregister_server(server_id)
        assert success is True
        
        # Verify removal
        server = await registry.get_server(server_id)
        assert server is None
        
        # Verify index cleanup
        servers = await registry.find_servers_by_capability("test_capability")
        assert len(servers) == 0
    
    @pytest.mark.asyncio
    async def test_dependency_validation(self, registry):
        """Test server dependency validation"""
        # Register dependency server
        dep_config = {
            "server_id": "dependency-server",
            "name": "Dependency Server",
            "capabilities": ["dep_capability"]
        }
        await registry.register_server(dep_config)
        
        # Make dependency healthy first
        await registry.update_server_health("dependency-server", "healthy")
        
        # Register server with dependency
        server_config = {
            "server_id": "dependent-server",
            "name": "Dependent Server",
            "capabilities": ["main_capability"],
            "dependencies": ["dependency-server"]
        }
        await registry.register_server(server_config)
        
        # Validate dependencies - should pass now that dependency is healthy
        validation = await registry.validate_server_dependencies("dependent-server")
        assert validation["valid"] is True
        
        # Make dependency unhealthy
        await registry.update_server_health("dependency-server", "unhealthy")
        
        # Validate dependencies - should fail
        validation = await registry.validate_server_dependencies("dependent-server")
        assert validation["valid"] is False
        assert "dependency-server" in validation["unhealthy_dependencies"]
    
    @pytest.mark.asyncio
    async def test_registry_statistics(self, registry, sample_server_config):
        """Test registry statistics collection"""
        # Get initial statistics
        stats = await registry.get_registry_statistics()
        initial_servers = stats["total_servers"]
        
        # Register additional server
        await registry.register_server(sample_server_config)
        
        # Update health status
        await registry.update_server_health("test-server-v1.0.0", "healthy")
        
        # Make some queries to generate statistics
        await registry.get_server("test-server-v1.0.0")
        await registry.find_servers_by_capability("test_capability")
        await registry.find_servers_by_tag("test")
        await registry.list_servers()
        
        # Get updated statistics
        stats = await registry.get_registry_statistics()
        assert stats["total_servers"] == initial_servers + 1
        assert stats["health_distribution"]["healthy"] >= 1
        assert stats["total_capabilities"] > 0
        assert stats["query_statistics"]["total_queries"] > 0
        assert stats["query_statistics"]["capability_queries"] > 0
    
    @pytest.mark.asyncio
    async def test_capability_catalog(self, registry, sample_server_config):
        """Test capability catalog generation"""
        # Register server with unique capabilities
        await registry.register_server(sample_server_config)
        
        # Get capability catalog
        catalog = await registry.get_capability_catalog()
        
        # Verify our test capabilities are in catalog
        assert "test_capability" in catalog
        assert "sample_feature" in catalog
        assert "test-server-v1.0.0" in catalog["test_capability"]
        
        # Verify default server capabilities
        assert "github_api" in catalog
        assert len(catalog["github_api"]) > 0
    
    @pytest.mark.asyncio
    async def test_persistence(self, sample_server_config):
        """Test registry persistence and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = os.path.join(temp_dir, "persistence_test.json")
            
            # Create registry and register server
            registry1 = MCPServerRegistry(registry_file=registry_file, auto_save=True)
            await registry1.register_server(sample_server_config)
            
            # Verify persistence file exists
            assert Path(registry_file).exists()
            
            # Create new registry instance (should load from file)
            registry2 = MCPServerRegistry(registry_file=registry_file, auto_save=True)
            
            # Verify server was loaded
            server = await registry2.get_server("test-server-v1.0.0")
            assert server is not None
            assert server["name"] == "Test MCP Server"
            
            # Verify capability index was rebuilt
            servers = await registry2.find_servers_by_capability("test_capability")
            assert len(servers) == 1
    
    @pytest.mark.asyncio
    async def test_export_functionality(self, registry, sample_server_config):
        """Test registry export functionality"""
        # Register server
        await registry.register_server(sample_server_config)
        
        # Export as JSON
        json_export = await registry.export_registry("json")
        assert isinstance(json_export, str)
        
        # Parse and validate JSON export
        export_data = json.loads(json_export)
        assert "registry_export" in export_data
        assert "servers" in export_data["registry_export"]
        assert "statistics" in export_data["registry_export"]
        assert export_data["registry_export"]["total_servers"] > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_stale_servers(self, registry, sample_server_config):
        """Test cleanup of stale servers"""
        # Register server
        await registry.register_server(sample_server_config)
        
        # Manually set old update time
        server_id = "test-server-v1.0.0"
        old_date = datetime.utcnow() - timedelta(days=35)
        registry.servers[server_id].last_updated = old_date
        
        # Run cleanup
        cleaned_count = await registry.cleanup_stale_servers(max_age_days=30)
        assert cleaned_count == 1
        
        # Verify server was removed
        server = await registry.get_server(server_id)
        assert server is None
    
    @pytest.mark.asyncio
    async def test_health_callback(self, registry, sample_server_config):
        """Test health status change callbacks"""
        callback_calls = []
        
        def health_callback(server_id: str, new_status: str):
            callback_calls.append((server_id, new_status))
        
        # Set callback
        registry.set_health_callback(health_callback)
        
        # Register server
        await registry.register_server(sample_server_config)
        server_id = "test-server-v1.0.0"
        
        # Update health status
        await registry.update_server_health(server_id, "healthy")
        
        # Verify callback was called
        assert len(callback_calls) == 1
        assert callback_calls[0] == (server_id, "healthy")
    
    @pytest.mark.asyncio
    async def test_thread_safety(self, registry, sample_server_config):
        """Test thread-safe operations"""
        # Create multiple concurrent registration tasks
        tasks = []
        
        for i in range(10):
            config = sample_server_config.copy()
            config["server_id"] = f"concurrent-server-{i}"
            config["name"] = f"Concurrent Server {i}"
            tasks.append(registry.register_server(config))
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All registrations should succeed
        assert all(results)
        
        # Verify all servers are registered
        servers = await registry.list_servers()
        concurrent_servers = [s for s in servers if s["server_id"].startswith("concurrent-server-")]
        assert len(concurrent_servers) == 10
    
    def test_server_record_creation(self):
        """Test ServerRecord dataclass creation and serialization"""
        record = ServerRecord(
            server_id="test-id",
            name="Test Server",
            version="1.0.0",
            capabilities=["test"],
            resource_requirements={"memory_mb": 64},
            description="Test description",
            tags=["tag1", "tag2"],
            dependencies=["dep1"],
            configuration={"key": "value"}
        )
        
        assert record.server_id == "test-id"
        assert record.name == "Test Server"
        assert record.capabilities == ["test"]
        assert record.tags == ["tag1", "tag2"]
        assert record.dependencies == ["dep1"]
        assert record.configuration == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])