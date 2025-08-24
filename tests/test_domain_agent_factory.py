#!/usr/bin/env python3
"""
Test Suite for Domain Agent Factory - Issue #71
Comprehensive tests for factory pattern, configuration validation, resource allocation, and registry management.
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import threading
import time

# Import the classes we're testing
from claude.commands.domain_agent_factory import (
    DomainAgentFactory, DomainConfiguration, DomainAgent, AgentType, AgentStatus,
    DomainCapability, AgentResources, AgentRegistry, ResourceAllocator,
    create_simple_agent, create_web_development_team
)

class TestDomainCapability(unittest.TestCase):
    """Test DomainCapability class"""
    
    def test_basic_capability_creation(self):
        """Test basic capability creation"""
        capability = DomainCapability(
            name="test_capability",
            description="Test capability description"
        )
        self.assertEqual(capability.name, "test_capability")
        self.assertEqual(capability.description, "Test capability description")
        self.assertEqual(capability.complexity, "medium")
        self.assertEqual(capability.resource_requirement, "standard")
    
    def test_capability_with_all_parameters(self):
        """Test capability with all parameters"""
        capability = DomainCapability(
            name="advanced_capability",
            description="Advanced capability",
            complexity="high",
            resource_requirement="intensive",
            dependencies=["dep1", "dep2"],
            tools=["tool1", "tool2"]
        )
        self.assertEqual(capability.complexity, "high")
        self.assertEqual(capability.resource_requirement, "intensive")
        self.assertEqual(capability.dependencies, ["dep1", "dep2"])
        self.assertEqual(capability.tools, ["tool1", "tool2"])

class TestDomainConfiguration(unittest.TestCase):
    """Test DomainConfiguration class"""
    
    def test_valid_configuration(self):
        """Test valid configuration creation"""
        config = DomainConfiguration(
            name="Test Agent",
            domain_type=AgentType.FRONTEND,
            expertise=["react", "javascript"],
            tools=["react", "webpack"]
        )
        self.assertEqual(config.name, "Test Agent")
        self.assertEqual(config.domain_type, AgentType.FRONTEND)
        self.assertEqual(config.priority, 50)  # default
        self.assertEqual(config.config_version, "1.0")
    
    def test_configuration_validation(self):
        """Test configuration validation in __post_init__"""
        # Test empty name
        with self.assertRaises(ValueError):
            DomainConfiguration(name="", domain_type=AgentType.FRONTEND)
        
        # Test invalid domain type
        with self.assertRaises(ValueError):
            DomainConfiguration(name="Test", domain_type="invalid")
        
        # Test invalid priority
        with self.assertRaises(ValueError):
            DomainConfiguration(name="Test", domain_type=AgentType.FRONTEND, priority=150)
        
        with self.assertRaises(ValueError):
            DomainConfiguration(name="Test", domain_type=AgentType.FRONTEND, priority=-10)

class TestAgentRegistry(unittest.TestCase):
    """Test AgentRegistry class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = AgentRegistry()
        # Override registry file to use temp directory
        self.registry.registry_file = Path(self.temp_dir) / "test_registry.json"
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_agent(self, name="TestAgent", domain_type=AgentType.BACKEND):
        """Helper to create test agent"""
        return DomainAgent(
            agent_id=f"TEST-{name.upper()}-12345",
            name=name,
            domain_type=domain_type
        )
    
    def test_agent_registration(self):
        """Test agent registration"""
        agent = self.create_test_agent()
        
        success = self.registry.register(agent)
        self.assertTrue(success)
        self.assertEqual(agent.status, AgentStatus.REGISTERED)
        self.assertIn(agent.agent_id, self.registry.agents)
        self.assertIn(agent.agent_id, self.registry.agents_by_type[AgentType.BACKEND])
    
    def test_duplicate_registration(self):
        """Test duplicate agent registration"""
        agent = self.create_test_agent()
        
        # First registration should succeed
        success1 = self.registry.register(agent)
        self.assertTrue(success1)
        
        # Second registration should fail
        success2 = self.registry.register(agent)
        self.assertFalse(success2)
    
    def test_agent_unregistration(self):
        """Test agent unregistration"""
        agent = self.create_test_agent()
        
        # Register then unregister
        self.registry.register(agent)
        success = self.registry.unregister(agent.agent_id)
        
        self.assertTrue(success)
        self.assertNotIn(agent.agent_id, self.registry.agents)
        self.assertNotIn(agent.agent_id, self.registry.agents_by_type[AgentType.BACKEND])
    
    def test_get_agents_by_type(self):
        """Test getting agents by type"""
        frontend_agent = self.create_test_agent("FrontendAgent", AgentType.FRONTEND)
        backend_agent = self.create_test_agent("BackendAgent", AgentType.BACKEND)
        
        self.registry.register(frontend_agent)
        self.registry.register(backend_agent)
        
        frontend_agents = self.registry.get_agents_by_type(AgentType.FRONTEND)
        backend_agents = self.registry.get_agents_by_type(AgentType.BACKEND)
        
        self.assertEqual(len(frontend_agents), 1)
        self.assertEqual(len(backend_agents), 1)
        self.assertEqual(frontend_agents[0].name, "FrontendAgent")
        self.assertEqual(backend_agents[0].name, "BackendAgent")

class TestResourceAllocator(unittest.TestCase):
    """Test ResourceAllocator class"""
    
    def setUp(self):
        """Set up test environment"""
        self.allocator = ResourceAllocator()
    
    def test_resource_allocation_success(self):
        """Test successful resource allocation"""
        requirements = {
            "cpu_cores": 1,
            "memory_mb": 512,
            "disk_mb": 100
        }
        
        success, resources, message = self.allocator.allocate_resources("test-agent-1", requirements)
        
        self.assertTrue(success)
        self.assertIsNotNone(resources)
        self.assertEqual(resources.cpu_cores, 1)
        self.assertEqual(resources.memory_mb, 512)
        self.assertIn("test-agent-1", self.allocator.allocated_resources)
    
    def test_duplicate_allocation(self):
        """Test duplicate resource allocation fails"""
        requirements = {"cpu_cores": 1, "memory_mb": 512}
        
        # First allocation should succeed
        success1, _, _ = self.allocator.allocate_resources("test-agent-1", requirements)
        self.assertTrue(success1)
        
        # Second allocation for same agent should fail
        success2, _, message = self.allocator.allocate_resources("test-agent-1", requirements)
        self.assertFalse(success2)
        self.assertIn("already has resources", message)
    
    @patch('psutil.cpu_count', return_value=4)
    @patch('psutil.virtual_memory')
    def test_resource_limits(self, mock_memory, mock_cpu):
        """Test resource allocation respects limits"""
        # Mock system with limited resources
        mock_memory.return_value.total = 2 * 1024 * 1024 * 1024  # 2GB
        
        # Try to allocate more than allowed
        excessive_requirements = {
            "cpu_cores": 10,  # More than available
            "memory_mb": 10000  # More than available
        }
        
        success, resources, message = self.allocator.allocate_resources("greedy-agent", excessive_requirements)
        self.assertFalse(success)
        self.assertIn("Insufficient", message)
    
    def test_resource_deallocation(self):
        """Test resource deallocation"""
        requirements = {"cpu_cores": 1, "memory_mb": 512}
        
        # Allocate then deallocate
        self.allocator.allocate_resources("test-agent-1", requirements)
        success = self.allocator.deallocate_resources("test-agent-1")
        
        self.assertTrue(success)
        self.assertNotIn("test-agent-1", self.allocator.allocated_resources)
    
    def test_resource_usage_stats(self):
        """Test resource usage statistics"""
        # Allocate some resources
        self.allocator.allocate_resources("agent-1", {"cpu_cores": 1, "memory_mb": 512})
        self.allocator.allocate_resources("agent-2", {"cpu_cores": 2, "memory_mb": 1024})
        
        usage = self.allocator.get_resource_usage()
        
        self.assertEqual(usage["allocated_agents"], 2)
        self.assertEqual(usage["cpu_utilization"]["allocated"], 3)
        self.assertEqual(usage["memory_utilization"]["allocated_mb"], 1536)

class TestDomainAgentFactory(unittest.TestCase):
    """Test DomainAgentFactory class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.factory = DomainAgentFactory()
        # Override registry file to use temp directory
        self.factory.registry.registry_file = Path(self.temp_dir) / "test_registry.json"
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up any created agents
        for agent_id in list(self.factory.created_agents.keys()):
            self.factory.cleanup_agent(agent_id)
    
    def create_test_config(self, name="TestAgent", domain_type=AgentType.BACKEND):
        """Helper to create test configuration"""
        return DomainConfiguration(
            name=name,
            domain_type=domain_type,
            capabilities=[DomainCapability("test_capability", "Test capability")],
            expertise=["testing"],
            tools=["test_tool"]
        )
    
    def test_configuration_validation_success(self):
        """Test successful configuration validation"""
        config = self.create_test_config()
        
        is_valid, message = self.factory.validate_config(config)
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Configuration is valid")
    
    def test_configuration_validation_failures(self):
        """Test configuration validation failures"""
        # Empty name - this will be caught by __post_init__ validation
        with self.assertRaises(ValueError):
            DomainConfiguration(name="", domain_type=AgentType.BACKEND)
        
        # Test invalid capability format
        config_invalid_cap = DomainConfiguration(
            name="TestAgent", 
            domain_type=AgentType.BACKEND,
            capabilities=["invalid_capability"]  # Should be DomainCapability objects
        )
        is_valid, message = self.factory.validate_config(config_invalid_cap)
        self.assertFalse(is_valid)
        self.assertIn("Invalid capability format", message)
    
    def test_agent_creation_success(self):
        """Test successful agent creation"""
        config = self.create_test_config("SuccessfulAgent")
        
        success, agent, message = self.factory.create_agent(config)
        
        self.assertTrue(success)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "SuccessfulAgent")
        self.assertEqual(agent.status, AgentStatus.REGISTERED)
        self.assertIsNotNone(agent.resources)
        self.assertIn("successfully", message)
    
    def test_agent_creation_duplicate_name(self):
        """Test agent creation with duplicate name fails"""
        config1 = self.create_test_config("DuplicateAgent")
        config2 = self.create_test_config("DuplicateAgent")
        
        # First agent should succeed
        success1, agent1, _ = self.factory.create_agent(config1)
        self.assertTrue(success1)
        
        # Second agent with same name should fail
        success2, agent2, message = self.factory.create_agent(config2)
        self.assertFalse(success2)
        self.assertIn("already exists", message)
    
    def test_team_creation(self):
        """Test specialist team creation"""
        project_requirements = {
            "project_name": "TestProject",
            "type": "web",
            "technologies": ["react", "nodejs", "postgresql"],
            "features": ["authentication", "api"]
        }
        
        success, team, message = self.factory.create_specialist_team(project_requirements)
        
        self.assertTrue(success)
        self.assertGreater(len(team), 0)
        self.assertIn("specialist team", message)
        
        # Verify team has different domain types
        domain_types = {agent.domain_type for agent in team}
        self.assertGreater(len(domain_types), 1)  # Should have multiple domain types
    
    def test_domain_identification(self):
        """Test domain identification from project requirements"""
        # Web project
        web_requirements = {
            "type": "web",
            "technologies": ["react", "nodejs"],
            "features": ["authentication"]
        }
        
        domains = self.factory.identify_required_domains(web_requirements)
        
        self.assertIn(AgentType.FRONTEND, domains)
        self.assertIn(AgentType.BACKEND, domains)
        self.assertIn(AgentType.SECURITY, domains)  # Due to authentication
        self.assertIn(AgentType.TESTING, domains)   # Always included
    
    def test_factory_metrics(self):
        """Test factory metrics tracking"""
        initial_metrics = self.factory.get_factory_metrics()
        initial_created = initial_metrics["factory_metrics"]["agents_created"]
        
        # Create an agent
        config = self.create_test_config("MetricsTestAgent")
        self.factory.create_agent(config)
        
        # Check metrics updated
        updated_metrics = self.factory.get_factory_metrics()
        updated_created = updated_metrics["factory_metrics"]["agents_created"]
        
        self.assertEqual(updated_created, initial_created + 1)
        self.assertGreater(updated_metrics["factory_metrics"]["average_creation_time"], 0)
    
    def test_agent_cleanup(self):
        """Test agent cleanup functionality"""
        config = self.create_test_config("CleanupTestAgent")
        success, agent, _ = self.factory.create_agent(config)
        self.assertTrue(success)
        
        agent_id = agent.agent_id
        
        # Verify agent exists
        self.assertIn(agent_id, self.factory.created_agents)
        self.assertIsNotNone(self.factory.registry.get_agent(agent_id))
        
        # Cleanup agent
        cleanup_success = self.factory.cleanup_agent(agent_id)
        
        self.assertTrue(cleanup_success)
        self.assertNotIn(agent_id, self.factory.created_agents)
        self.assertIsNone(self.factory.registry.get_agent(agent_id))

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_simple_agent(self):
        """Test simple agent creation function"""
        factory = DomainAgentFactory()
        factory.registry.registry_file = Path(self.temp_dir) / "test_registry.json"
        
        success, agent, message = create_simple_agent("SimpleTestAgent", AgentType.FRONTEND, factory)
        
        self.assertTrue(success)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "SimpleTestAgent")
        self.assertEqual(agent.domain_type, AgentType.FRONTEND)
        
        # Cleanup
        factory.cleanup_agent(agent.agent_id)
    
    def test_create_web_development_team(self):
        """Test web development team creation function"""
        factory = DomainAgentFactory()
        factory.registry.registry_file = Path(self.temp_dir) / "test_registry.json"
        
        success, team, message = create_web_development_team("WebTestProject", factory)
        
        self.assertTrue(success)
        self.assertGreater(len(team), 1)
        self.assertIn("team", message.lower())
        
        # Should have frontend and backend at minimum
        domain_types = {agent.domain_type for agent in team}
        self.assertIn(AgentType.FRONTEND, domain_types)
        self.assertIn(AgentType.BACKEND, domain_types)
        
        # Cleanup
        for agent in team:
            factory.cleanup_agent(agent.agent_id)

class TestConcurrentOperations(unittest.TestCase):
    """Test concurrent operations and thread safety"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.factory = DomainAgentFactory()
        self.factory.registry.registry_file = Path(self.temp_dir) / "test_registry.json"
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Cleanup agents
        for agent_id in list(self.factory.created_agents.keys()):
            self.factory.cleanup_agent(agent_id)
    
    def test_concurrent_agent_creation(self):
        """Test concurrent agent creation"""
        def create_agent(thread_id):
            config = DomainConfiguration(
                name=f"ConcurrentAgent_{thread_id}",
                domain_type=AgentType.BACKEND,
                capabilities=[DomainCapability("test", "Test capability")],
                expertise=["testing"],
                tools=["test"]
            )
            return self.factory.create_agent(config)
        
        threads = []
        results = []
        
        # Create multiple threads
        for i in range(5):
            thread = threading.Thread(target=lambda i=i: results.append(create_agent(i)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        successful_creations = sum(1 for success, _, _ in results if success)
        self.assertEqual(successful_creations, 5)  # All should succeed with unique names
    
    def test_concurrent_resource_allocation(self):
        """Test concurrent resource allocation"""
        def allocate_resources(agent_id):
            requirements = {"cpu_cores": 1, "memory_mb": 256}
            return self.factory.resource_allocator.allocate_resources(agent_id, requirements)
        
        threads = []
        results = []
        
        # Create multiple threads trying to allocate resources
        for i in range(3):
            thread = threading.Thread(target=lambda i=i: results.append(allocate_resources(f"agent_{i}")))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should succeed with unique agent IDs
        successful_allocations = sum(1 for success, _, _ in results if success)
        self.assertEqual(successful_allocations, 3)

class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.factory = DomainAgentFactory()
        self.factory.registry.registry_file = Path(self.temp_dir) / "test_registry.json"
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_domain_type(self):
        """Test handling of invalid domain types"""
        with self.assertRaises(ValueError):
            DomainConfiguration(name="Test", domain_type="invalid_type")
    
    def test_empty_project_requirements(self):
        """Test team creation with empty requirements"""
        success, team, message = self.factory.create_specialist_team({})
        
        # Should still create at least a backend agent
        self.assertTrue(success)
        self.assertGreater(len(team), 0)
        
        # Cleanup
        for agent in team:
            self.factory.cleanup_agent(agent.agent_id)
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_count', return_value=1)  # Simulate limited resources
    def test_resource_exhaustion(self, mock_cpu, mock_memory):
        """Test behavior when resources are exhausted"""
        # Mock very limited memory
        mock_memory.return_value.total = 1024 * 1024 * 1024  # 1GB total
        
        # Create factory with limited resources
        limited_factory = DomainAgentFactory()
        limited_factory.registry.registry_file = Path(self.temp_dir) / "limited_registry.json"
        limited_factory.resource_allocator.total_limits["memory_mb"] = 1024  # 1GB limit
        
        # Try to create agents that exceed limits
        configs = []
        for i in range(5):  # Try to create 5 agents with 512MB each (would need 2.5GB)
            config = DomainConfiguration(
                name=f"ResourceHungryAgent_{i}",
                domain_type=AgentType.BACKEND,
                capabilities=[DomainCapability("test", "Test")],
                expertise=["test"],
                tools=["test"],
                resource_requirements={"memory_mb": 512, "cpu_cores": 1}
            )
            configs.append(config)
        
        successful_agents = 0
        failed_agents = 0
        
        for config in configs:
            success, agent, message = limited_factory.create_agent(config)
            if success:
                successful_agents += 1
            else:
                failed_agents += 1
        
        # Some should fail due to resource limits
        self.assertGreater(failed_agents, 0)
        self.assertLessEqual(successful_agents, 2)  # Should be able to create only 1-2 agents
        
        # Cleanup successful agents
        for agent_id in list(limited_factory.created_agents.keys()):
            limited_factory.cleanup_agent(agent_id)

class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.factory = DomainAgentFactory()
        self.factory.registry.registry_file = Path(self.temp_dir) / "integration_registry.json"
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Cleanup all agents
        for agent_id in list(self.factory.created_agents.keys()):
            self.factory.cleanup_agent(agent_id)
    
    def test_full_project_lifecycle(self):
        """Test complete project lifecycle with team creation and management"""
        # Step 1: Create project team
        project_requirements = {
            "project_name": "FullStackApp",
            "type": "web",
            "technologies": ["react", "nodejs", "postgresql", "redis"],
            "features": ["authentication", "api", "real-time", "analytics"],
            "description": "Full-stack application with real-time features and analytics"
        }
        
        success, team, message = self.factory.create_specialist_team(project_requirements)
        self.assertTrue(success, f"Team creation failed: {message}")
        self.assertGreaterEqual(len(team), 4, "Team should have at least 4 specialists")
        
        # Step 2: Verify all agents are properly registered and allocated resources
        for agent in team:
            self.assertEqual(agent.status, AgentStatus.REGISTERED)
            self.assertIsNotNone(agent.resources)
            self.assertIn(agent.agent_id, self.factory.registry.agents)
        
        # Step 3: Check team has appropriate domain coverage
        domain_types = {agent.domain_type for agent in team}
        expected_domains = {AgentType.FRONTEND, AgentType.BACKEND, AgentType.DATABASE, AgentType.SECURITY, AgentType.TESTING}
        
        # Should have most expected domains (may not have all depending on requirements analysis)
        self.assertGreaterEqual(len(domain_types & expected_domains), 3)
        
        # Step 4: Verify factory metrics are updated
        metrics = self.factory.get_factory_metrics()
        self.assertEqual(metrics["factory_metrics"]["agents_created"], len(team))
        self.assertGreater(metrics["registry_stats"]["total_agents"], 0)
        
        # Step 5: Test individual agent retrieval
        for agent in team:
            retrieved_agent = self.factory.registry.get_agent(agent.agent_id)
            self.assertIsNotNone(retrieved_agent)
            self.assertEqual(retrieved_agent.name, agent.name)
        
        # Step 6: Test agents by type retrieval
        for domain_type in domain_types:
            agents_of_type = self.factory.registry.get_agents_by_type(domain_type)
            self.assertGreater(len(agents_of_type), 0)
        
        print(f"✓ Successfully created and managed team of {len(team)} agents with domains: {[d.value for d in domain_types]}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)